#include <vector>
#include <string>
#include <type_traits>
#include <cutlass/bfloat16.h>

// #include "common/commonUtils.h"
// #include "common/logger.h"
#include "common/cutlassUtils.h"

#include "cute/tensor.hpp"
#include <cute/layout.hpp>  // 提供 cute::Shape 和 cute::Int

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"

#include "cutlass/util/mixed_dtype_utils.hpp" // cutlass::DeviceAllocation
#include "cutlass/util/packed_stride.hpp" // cutlass::make_cute_packed_stride
#include <cuda_runtime.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

// namespace W4FP8GroupGemm
namespace {
// using namespace cute;
using MmaType = cutlass::float_e4m3_t;
using QuantType = cutlass::int4b_t;

template <
    typename OutType,
    typename CTAShape,
    typename ClusterShape,
    typename MainloopScheduleType = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
    typename EpilogueScheduleType = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>
struct DeviceGroupGemmInt4Fp8Sm90 {
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// GEMM kernel configurations
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // A matrix configuration
    using         ElementA    = MmaType;
    using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
    static constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Alignment of A matrix in units of elements (up to 16 bytes)

    // B matrix configuration
    using         ElementB    = QuantType;                                      // Element type for B matrix operand
    using         LayoutB     = cutlass::layout::ColumnMajor;                   // Layout type for B matrix operand
    static constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

    // This example manually swaps and transposes, so keep transpose of input layouts
    using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
    using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;

    // Need to pass a pointer type to make the 3rd dimension of Stride be _0
    using StrideA = cute::remove_pointer_t<cutlass::detail::TagToStrideA_t<LayoutA*>>;
    using StrideB = cute::remove_pointer_t<cutlass::detail::TagToStrideB_t<LayoutB*>>;

    // Scale configuration
    // must align to 64 bits.
    using ElementScalePacked = cutlass::Array<MmaType, 8>;
    using LayoutScale = cutlass::layout::RowMajor;

    // C/D matrix configuration
    using         ElementC    = OutType;                                // Element type for C and D matrix operands
    using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
    static constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

    // D matrix configuration
    using         ElementD    = ElementC;
    using         LayoutD     = LayoutC;
    static constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;

    // Core kernel configurations
    using ElementAccumulator  = float;                                          // Element type for internal accumulation
    using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
    using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
    // TODO: optimize the tileShape according running time gemm shape
    using StageCountType = cutlass::gemm::collective::StageCountAuto; // Stage count maximized based on the tile size


    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, 
        CTAShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, typename cutlass::layout::LayoutTranspose<LayoutC>::type*, AlignmentC, 
        ElementD, typename cutlass::layout::LayoutTranspose<LayoutD>::type*, AlignmentD,
        EpilogueScheduleType>::CollectiveOp;
    // =========================================================== MIXED INPUT WITH SCALES
    // =========================================================================== The Scale information must get paired
    // with the operand that will be scaled. In this example, B is scaled so we make a tuple of B's information and the
    // scale information.
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        cute::tuple<ElementB, ElementScalePacked>, LayoutB_Transpose*, AlignmentB,
        ElementA, LayoutA_Transpose *, AlignmentA,
        ElementAccumulator, CTAShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        MainloopScheduleType
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>,
        CollectiveMainloop, CollectiveEpilogue>;

    using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideC = typename GemmKernel::InternalStrideC;
    using StrideD = typename GemmKernel::InternalStrideD;
    using StrideS = typename CollectiveMainloop::StrideScale;
};

template <typename GemmConfig>
struct GroupedGemmInput {
    int _num_experts;
    std::vector<std::tuple<int, int, int>> _mnk_sizes;
    int64_t _topk;
    int _n_total, _k_total;
    int _group_quant_size = 128;
    cudaStream_t _stream;
    using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
    typename cutlass::DeviceAllocation<typename ProblemShape::UnderlyingProblemShape> _problem_shapes;
    // grouped address
    cutlass::DeviceAllocation<const typename GemmConfig::ElementA *> ptr_A; // fp8
    cutlass::DeviceAllocation<const typename GemmConfig::ElementB *> ptr_B; // int4
    // cutlass::DeviceAllocation<const cutlass::int8_t *> ptr_B; // int8
    cutlass::DeviceAllocation<const typename GemmConfig::ElementScalePacked *> ptr_B_scale; // fp8
    cutlass::DeviceAllocation<const float *> ptr_alpha; // float
    cutlass::DeviceAllocation<typename GemmConfig::ElementD *> ptr_D; // bf16
    // stride
    cutlass::DeviceAllocation<typename GemmConfig::StrideA> _stride_A;
    cutlass::DeviceAllocation<typename GemmConfig::StrideB> _stride_B;
    cutlass::DeviceAllocation<typename GemmConfig::StrideS> _stride_B_Scale;
    // to confirm: StrideD is right
    cutlass::DeviceAllocation<typename GemmConfig::StrideD> _stride_D;

    std::vector<cutlass::DeviceAllocation<cutlass::int4b_t>> _weight_int4_modifieds;
    // must be fp8 * 8 = 64 bits
    std::vector<cutlass::DeviceAllocation<cutlass::Array<cutlass::float_e4m3_t, 8>>> _weight_scale_fp8_packeds;

    GroupedGemmInput(
        torch::Tensor& out_tensors, // [m, n]
        torch::Tensor const& a_tensors, // [m, k]
        torch::Tensor const& b_tensors, // [E, n, k // 2]
        torch::Tensor const& a_scales, // scalar
        torch::Tensor const& b_scales, // [E, k//512, n * 4], group_size = 128
        torch::Tensor const& expert_offsets, // [E，]
        torch::Tensor const& problem_sizes, // [E, 3], [n, m, k]
        int64_t top_k) {
        _topk = _topk;
        _num_experts = static_cast<int>(expert_offsets.size(0));
        _n_total = static_cast<int>(b_tensors.size(1));
        _k_total = static_cast<int>(a_tensors.size(1));
        _stream = at::cuda::getCurrentCUDAStream(a_tensors.device().index());
        // alloc
        _weight_int4_modifieds.resize(_num_experts);
        _weight_scale_fp8_packeds.resize(_num_experts);
        // init
        ptr_A.reset(_num_experts);
        ptr_B.reset(_num_experts);
        ptr_B_scale.reset(_num_experts);
        ptr_alpha.reset(_num_experts);
        ptr_D.reset(_num_experts);
        // problem shape
        _problem_shapes.reset(_num_experts);
        _mnk_sizes.resize(_num_experts);
        // stride
        _stride_A.reset(_num_experts);
        _stride_B.reset(_num_experts);
        _stride_B_Scale.reset(_num_experts);
        _stride_D.reset(_num_experts);
        // init problem shape, stride
        std::vector<typename ProblemShape::UnderlyingProblemShape> problem_shapes_host(_num_experts);
        std::vector<typename GemmConfig::StrideA> stride_A_host(_num_experts);
        std::vector<typename GemmConfig::StrideB> stride_B_host(_num_experts);
        std::vector<typename GemmConfig::StrideS> stride_B_Scale_host(_num_experts);
        std::vector<typename GemmConfig::StrideD> stride_D_host(_num_experts);

        torch::Tensor problem_sizes_cpu = problem_sizes.cpu().contiguous();
        auto problem_sizes_cpu_ptr = problem_sizes_cpu.data_ptr<int32_t>();

        for (int32_t i=0; i<_num_experts; i++) {
            int _m, _n, _k;
            // std::tie(_m, _n, _k) = _mnk_sizes[i];
            _n = problem_sizes_cpu_ptr[i * 3];
            _m = problem_sizes_cpu_ptr[i * 3 + 1];
            _k = problem_sizes_cpu_ptr[i * 3 + 2];
            _mnk_sizes[i] = std::make_tuple(_m, _n, _k);
            assert(_k_total == _k && _n_total == _n &&
                   static_cast<int>(b_tensors.size(0)) == _num_experts &&
                   static_cast<int>(b_tensors.size(2)) == _k_total / 2);
            problem_shapes_host[i] = typename ProblemShape::UnderlyingProblemShape{_n, _m, _k};
            stride_A_host[i] = cutlass::make_cute_packed_stride(typename GemmConfig::StrideA{}, {_m, _k, 1});
            stride_B_host[i] = cutlass::make_cute_packed_stride(typename GemmConfig::StrideB{}, {_n, _k, 1});
            stride_B_Scale_host[i] = cutlass::make_cute_packed_stride(typename GemmConfig::StrideS{}, {_n, _k / _group_quant_size, 1});
            stride_D_host[i] = cutlass::make_cute_packed_stride(typename GemmConfig::StrideD{}, {_n, _m, 1});
        }
        _problem_shapes.copy_from_host(problem_shapes_host.data());
        _stride_A.copy_from_host(stride_A_host.data());
        _stride_B.copy_from_host(stride_B_host.data());
        _stride_B_Scale.copy_from_host(stride_B_Scale_host.data());
        _stride_D.copy_from_host(stride_D_host.data());
        // set input ptr
        std::vector<typename GemmConfig::ElementA *> ptr_A_host(_num_experts);
        std::vector<typename GemmConfig::ElementB *> ptr_B_host(_num_experts);
        // std::vector<cutlass::int8_t*> ptr_B_host(_num_experts);
        std::vector<typename GemmConfig::ElementScalePacked *> ptr_B_scale_host(_num_experts);
        std::vector<float *> ptr_alpha_host(_num_experts);
        std::vector<typename GemmConfig::ElementD *> ptr_D_host(_num_experts);

        torch::Tensor expert_offsets_cpu = expert_offsets.cpu().contiguous();
        auto expert_offsets_cpu_ptr = expert_offsets_cpu.data_ptr<int32_t>();

        cutlass::float_e4m3_t* a_base_ptr = static_cast<cutlass::float_e4m3_t*>(a_tensors.data_ptr());
        cutlass::int8_t* b_base_ptr = static_cast<cutlass::int8_t*>(b_tensors.data_ptr());
        cutlass::float_e4m3_t* b_scales_base_ptr = static_cast<cutlass::float_e4m3_t*>(b_scales.data_ptr());
        typename GemmConfig::ElementD * out_base_ptr = static_cast<typename GemmConfig::ElementD *>(out_tensors.data_ptr());

        for (int32_t i=0; i<_num_experts; i++) {
            int _m, _n, _k;
            // std::tie(_m, _n, _k) = _mnk_sizes[i];
            _n = problem_sizes_cpu_ptr[i * 3];
            _m = problem_sizes_cpu_ptr[i * 3 + 1];
            _k = problem_sizes_cpu_ptr[i * 3 + 2];
            int w_size = _n * _k;
            int w_scale_size = _n * _k / _group_quant_size;
            _weight_int4_modifieds[i].reset(w_size);
            _weight_scale_fp8_packeds[i].reset(w_scale_size);

            cutlass::int4b_t* weight_int4_ptr = (cutlass::int4b_t*)(b_base_ptr + i * w_size / 2);
            cutlass::float_e4m3_t* weight_scale_ptr = b_scales_base_ptr + i * w_scale_size;
            // modify int4 uni-code
            cutlass::unified_encode_int4b(static_cast<cutlass::int4b_t const*>(weight_int4_ptr),
                                         _weight_int4_modifieds[i].get(), w_size);
            // scale to packed
            cutlass::pack_scale_fp8(weight_scale_ptr, _weight_scale_fp8_packeds[i].get(), w_scale_size);
        }

        for (int i=0; i<_num_experts; i++) {
            // compute address
            int32_t expert_offset = expert_offsets_cpu_ptr[i];
            ptr_A_host[i] = a_base_ptr + expert_offset * _k_total;
            ptr_B_host[i] = _weight_int4_modifieds[i].get();
            ptr_B_scale_host[i] = _weight_scale_fp8_packeds[i].get();
            ptr_alpha_host[i] = static_cast<float*>(a_scales.data_ptr());
            ptr_D_host[i] = out_base_ptr + expert_offset * _n_total;
        }
        // copy to device
        ptr_A.copy_from_host(ptr_A_host.data());
        ptr_B.copy_from_host(ptr_B_host.data());
        ptr_B_scale.copy_from_host(ptr_B_scale_host.data());
        ptr_alpha.copy_from_host(ptr_alpha_host.data());
        ptr_D.copy_from_host(ptr_D_host.data());
    }
};

template <typename GemmConfig>
typename GemmConfig::GemmGrouped::Arguments prepare_sm90_int4_fp8_args(GroupedGemmInput<GemmConfig>& inputs) {
    typename GemmConfig::GemmGrouped::Arguments arguments;
    decltype(arguments.epilogue.thread) fusion_args;
    fusion_args.alpha = 0;
    fusion_args.beta = 0;
    fusion_args.alpha_ptr = nullptr;
    fusion_args.beta_ptr = nullptr;
    fusion_args.alpha_ptr_array = inputs.ptr_alpha.get();
    fusion_args.beta_ptr_array = nullptr;
    // One alpha and beta per each group
    fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 1};
    fusion_args.dBeta = {cute::_0{}, cute::_0{}, 1};

    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);;

    // arguments
    arguments = typename GemmConfig::GemmGrouped::Arguments {cutlass::gemm::GemmUniversalMode::kGrouped,
        {inputs._num_experts, inputs._problem_shapes.get(), nullptr},
        {reinterpret_cast<typename GemmConfig::ElementB const**>(inputs.ptr_B.get()), inputs._stride_B.get(),
            reinterpret_cast<typename GemmConfig::ElementA const**>(inputs.ptr_A.get()), inputs._stride_A.get(),
            reinterpret_cast<typename GemmConfig::ElementScalePacked const**>(inputs.ptr_B_scale.get()),
            inputs._stride_B_Scale.get(), inputs._group_quant_size},
        {fusion_args, nullptr, nullptr,
            reinterpret_cast<typename GemmConfig::ElementD**>(inputs.ptr_D.get()),
            inputs._stride_D.get()},
        hw_info};
    return arguments;
}

template <typename GemmConfig>
void run_sm90_int4_fp8_grouped_gemm(GroupedGemmInput<GemmConfig>& inputs) {
    auto arguments = prepare_sm90_int4_fp8_args<GemmConfig>(inputs);
    typename GemmConfig::GemmGrouped gemm_op;

    size_t workspace_size = gemm_op.get_workspace_size(arguments);
    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    auto can_implement = gemm_op.can_implement(arguments);
    if (can_implement != cutlass::Status::kSuccess)
    {
        std::string err_msg = "mixed dtype WS grouped cutlass kernel will fail for params. Error: "
            + std::string(cutlassGetStatusString(can_implement));
        std::cout << err_msg << std::endl;
        throw std::runtime_error("[Mixed dtype WS grouped GEMM] " + err_msg);
    }

    auto init_status = gemm_op.initialize(arguments, workspace.get(), inputs._stream);
    if (init_status != cutlass::Status::kSuccess)
    {
        std::string err_msg = "Failed to initialize cutlass mixed dtype WS grouped gemm. Error: "
            + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[Mixed dtype WS grouped GEMM] " + err_msg);
    }

    auto run_status = gemm_op.run(inputs._stream);
    if (run_status != cutlass::Status::kSuccess)
    {
        std::string err_msg = "Failed to run cutlass mixed dtype WS grouped gemm. Error: "
            + std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[Mixed dtype WS grouped GEMM] " + err_msg);
    }

    // profile
    int warmup = 10;
    int loop = 100;
    HAI::ProfileResult result;
    grouped_mixed_dtype_profiling(result, gemm_op, inputs._mnk_sizes, warmup, loop, "run_sm90_int4_fp8_grouped_gemm");
    return;
}

void sm90_int4_fp8_group_gemm_dispatch(torch::Tensor& out_tensors, // [m, n]
        torch::Tensor const& a_tensors, // [m, k]
        torch::Tensor const& b_tensors, // [E, n, k // 2]
        torch::Tensor const& a_scales, // scalar
        torch::Tensor const& b_scales, // [E, k//512, n * 4], group_size = 128
        torch::Tensor const& expert_offsets, // [E，]
        torch::Tensor const& problem_sizes, // [E, 3], [n, m, k]
        int64_t top_k) {
    // const int packedNum = 2; // if K % 256 == 0, 
    int PackedScalesNum = 1; // diff tileShape the precision of result is different.
    // TODO: optimize TileShape
    // TODO: optimize ClusterShape with input shape: <2,1,1>, <1,2,1>, <2,2,1>
    using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;
    // TODO: optimize kernelSchedule the optional setting: cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative
    // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
    using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
    using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative; // Epilogue to launch
    if (PackedScalesNum == 1) {
        // printf("[cxx] dispatch to TileShape: [128, 16, 128]\n");
        using TileShape           = cute::Shape<cute::Int<128>, cute::Int<16>, cute::Int<128>>;
        using GroupGemmConfig = DeviceGroupGemmInt4Fp8Sm90<
            cutlass::half_t, // cutlass::bfloat16_t,
            TileShape,
            ClusterShape,
            KernelSchedule,
            EpilogueSchedule>;
        // construct grouped gemm input
        GroupedGemmInput<GroupGemmConfig> inputs(out_tensors, a_tensors, b_tensors,
            a_scales, b_scales, expert_offsets, problem_sizes, top_k);
        return run_sm90_int4_fp8_grouped_gemm<GroupGemmConfig>(inputs);
    }
    else if (PackedScalesNum == 2) {
        // printf("dispatch to TileShape: [128, 16, 256]\n");
        using TileShape           = cute::Shape<cute::Int<128>, cute::Int<16>, cute::Int<256>>;
        using GroupGemmConfig = DeviceGroupGemmInt4Fp8Sm90<
            cutlass::half_t, // cutlass::bfloat16_t,
            TileShape,
            ClusterShape,
            KernelSchedule,
            EpilogueSchedule>;
        // construct grouped gemm input
        GroupedGemmInput<GroupGemmConfig> inputs(out_tensors, a_tensors, b_tensors,
            a_scales, b_scales, expert_offsets, problem_sizes, top_k);
        return run_sm90_int4_fp8_grouped_gemm<GroupGemmConfig>(inputs);
    }
    else if (PackedScalesNum == 4) {
        // printf("dispatch to TileShape: [128, 16, 512]\n");
        using TileShape           = cute::Shape<cute::Int<128>, cute::Int<16>, cute::Int<512>>;
        using GroupGemmConfig = DeviceGroupGemmInt4Fp8Sm90<
            cutlass::half_t, // cutlass::bfloat16_t,
            TileShape,
            ClusterShape,
            KernelSchedule,
            EpilogueSchedule>;
        // construct grouped gemm input
        GroupedGemmInput<GroupGemmConfig> inputs(out_tensors, a_tensors, b_tensors,
            a_scales, b_scales, expert_offsets, problem_sizes, top_k);
        return run_sm90_int4_fp8_grouped_gemm<GroupGemmConfig>(inputs);
    }
}

} // namespace W4FP8GroupGemm

void sm90_int4_fp8_group_gemm_simple(
    torch::Tensor& d_tensors,
    torch::Tensor const& a_tensors,
    torch::Tensor const& b_tensors,
    torch::Tensor const& a_scales,
    torch::Tensor const& b_scales,
    torch::Tensor const& expert_offsets,
    torch::Tensor const& problem_sizes,
    int64_t topk) {
    // HAI::show_info(a_tensors, 10, "a_tensors");
    // HAI::show_info(b_tensors, 10, "b_tensors");
    // HAI::show_info(a_scales, 10, "a_scales");
    // HAI::show_info(b_scales, 10, "b_scales");
    sm90_int4_fp8_group_gemm_dispatch(d_tensors, // [m, n]
        a_tensors, // [m, k]
        b_tensors, // [E, n, k // 2]
        a_scales, // scalar
        b_scales, // [E, k//512, n * 4], group_size = 128
        expert_offsets, // [E，]
        problem_sizes, // [E, 3], [n, m, k]
        topk);
    // HAI::show_info(d_tensors, 10, "out_tensors");
    return;
}