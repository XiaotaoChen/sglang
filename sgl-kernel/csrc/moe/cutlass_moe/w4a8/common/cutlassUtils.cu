#include <vector>
#include <string>
#include <cutlass/bfloat16.h>
#include <cutlass/half.h>
#include <cutlass/array.h>
#include "logger.h"
#include "cutlassUtils.h"
#include "logger.h"
#include "commonUtils.h"

namespace HAI
{

void unpack_to_int8(int8_t& low, int8_t& high, const int8_t& packed) {
    low = (int8_t(packed << 4) >> 4); // The double shift here is to ensure sign extension
    high = packed >> 4;
    return;
}

void pack_to_int8(int8_t& packed, const int8_t& low, const int8_t& high) {
    int8_t packed_int4s = 0;
    packed_int4s |= ((low & 0x0F));
    packed_int4s |= int8_t(high << 4);
    packed = packed_int4s;
    return;
}

bool unpack_int4_int8_to_int8x2(int8_t* unpacked, const int8_t* packed, int count) {
    int packed_size = count / 2;
    for (int i=0; i<packed_size; i++) {
        unpack_to_int8(unpacked[2*i], unpacked[2*i+1], packed[i]);
    }
    return true;
}

bool unpack_int4_int8_to_int8x2(int8_t* unpacked, const cutlass::int4b_t* packed, int count) {
    const int8_t* src_ptr = reinterpret_cast<const int8_t*>(packed);
    return unpack_int4_int8_to_int8x2(unpacked, src_ptr, count);
}

bool pack_int4_int8x2_to_int8(int8_t* packed, const int8_t* unpacked, int count) {
    int packed_size = count / 2;
    for (int i=0; i<packed_size; i++) {
        pack_to_int8(packed[i], unpacked[2*i], unpacked[2*i+1]);
    }
    return true;
}

bool pack_int4_int8x2_to_int8(cutlass::int4b_t* packed, const int8_t* unpacked, int count) {
    int8_t* dst_ptr = reinterpret_cast<int8_t*>(packed);
    return pack_int4_int8x2_to_int8(dst_ptr, unpacked, count);
}

cutlass::float_e4m3_t dequant_int8_to_fp8(const int8_t src, const cutlass::float_e4m3_t scale) {
    float val = src * fp8_to_fp32(scale);
    return fp32_to_fp8(val);
}

cutlass::float_e4m3_t dequant_int8_to_fp8(const int8_t src, const cutlass::bfloat16_t scale) {
    float scale_fp32 = 0;
    float16_to_fp32(scale_fp32, scale);
    float val = src * scale_fp32;
    return fp32_to_fp8(val);
}

bool int8_to_int4bx2_array(cutlass::int4b_t* dst, const int8_t* packed, size_t packed_count) {
    int8_t* dst_ptr = reinterpret_cast<int8_t*>(dst);
    // for (size_t i=0; i<packed_count; i++) {
    //     int8_to_int4bx2(low, high, packed[i]);
    //     dst[2*i] = low;
    //     dst[2*i+1] = high;
    // }
    std::memcpy(dst_ptr, packed, packed_count);
    return true;
}

float fp8_to_fp32(cutlass::float_e4m3_t value) {
    return static_cast<float>(value);
}
void fp8_to_fp32_array(float* dst, const cutlass::float_e4m3_t* src, size_t count) {
    for (size_t i=0; i<count; i++) {
        dst[i] = fp8_to_fp32(src[i]);
    }
}

cutlass::float_e4m3_t fp32_to_fp8(float value) {
    return static_cast<cutlass::float_e4m3_t>(std::fmin(std::fmax(value, -448.0), 448.0));
}
void fp32_to_fp8_array(cutlass::float_e4m3_t* dst, const float* src, size_t count) {
    for (size_t i=0; i<count; i++) {
        dst[i] = fp32_to_fp8(src[i]);
    }
}

template<typename dtype>
void float16_to_fp32(float& dst, dtype value) {
    dst = static_cast<float>(value);
    return;
}
template void float16_to_fp32<cutlass::bfloat16_t>(float& dst, cutlass::bfloat16_t value);
template void float16_to_fp32<cutlass::half_t>(float& dst, cutlass::half_t value);

template<typename dtype>
void fp32_to_float16(dtype& dst, float value) {
    dst = static_cast<dtype>(value);
    return;
}
template void fp32_to_float16<cutlass::bfloat16_t>(cutlass::bfloat16_t& dst, float value);
template void fp32_to_float16<cutlass::half_t>(cutlass::half_t& dst, float value);

template<typename dtype>
void fp32_to_float16_array(dtype* dst, const float* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        fp32_to_float16<dtype>(dst[i], src[i]);
    }
}
template void fp32_to_float16_array<cutlass::bfloat16_t>(cutlass::bfloat16_t* dst, const float* src, size_t count);
template void fp32_to_float16_array<cutlass::half_t>(cutlass::half_t* dst, const float* src, size_t count);

template<typename dtype>
void float16_to_fp32_array(float* dst, const dtype* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        float16_to_fp32(dst[i], src[i]);
    }
}
template void float16_to_fp32_array<cutlass::bfloat16_t>(float* dst, const cutlass::bfloat16_t* src, size_t count);
template void float16_to_fp32_array<cutlass::half_t>(float* dst, const cutlass::half_t* src, size_t count);

template<typename dtype>
void float16_to_fp8(cutlass::float_e4m3_t& dst, dtype value, dtype scale_inv) {
    dst = fp32_to_fp8(static_cast<float>(value) * static_cast<float>(scale_inv));
    return;
}
template void float16_to_fp8(cutlass::float_e4m3_t& dst, cutlass::bfloat16_t value, cutlass::bfloat16_t scale_inv);
template void float16_to_fp8(cutlass::float_e4m3_t& dst, cutlass::half_t value, cutlass::half_t scale_inv);
template<typename dtype>
void float16_to_fp8_array(cutlass::float_e4m3_t* dst, const dtype* src, size_t count, dtype scale_inv) {
    for (size_t i=0; i<count; i++) {
        float16_to_fp8(dst[i], src[i], scale_inv);
    }
}
template void float16_to_fp8_array<cutlass::bfloat16_t>(cutlass::float_e4m3_t* dst, const cutlass::bfloat16_t* src, size_t count, cutlass::bfloat16_t scale_inv);
template void float16_to_fp8_array<cutlass::half_t>(cutlass::float_e4m3_t* dst, const cutlass::half_t* src, size_t count, cutlass::half_t scale_inv);

template<typename dtype>
void fp8_to_float16(dtype& dst, cutlass::float_e4m3_t value, dtype scale) {
    fp32_to_float16<dtype>(dst, static_cast<float>(value) * static_cast<float>(scale));
    return;
}
template void fp8_to_float16(cutlass::bfloat16_t& dst, cutlass::float_e4m3_t value, cutlass::bfloat16_t scale);
template void fp8_to_float16(cutlass::half_t& dst, cutlass::float_e4m3_t value, cutlass::half_t scale);

template<typename dtype>
void fp8_to_float16_array(dtype* dst, const cutlass::float_e4m3_t* src, size_t count, dtype scale) {
    for (size_t i=0; i<count; i++) {
        fp8_to_float16(dst[i], src[i], scale);
    }
}
template void fp8_to_float16_array<cutlass::bfloat16_t>(cutlass::bfloat16_t* dst, const cutlass::float_e4m3_t* src, size_t count, cutlass::bfloat16_t scale);
template void fp8_to_float16_array<cutlass::half_t>(cutlass::half_t* dst, const cutlass::float_e4m3_t* src, size_t count, cutlass::half_t scale);

template<typename dtype>
bool dumpFile(const cutlass::DeviceAllocation<dtype>& block, const std::string& out_path) {
    static_assert(std::is_same<dtype, float>::value ||
                cute::is_same_v<dtype, cutlass::int4b_t> ||
                cute::is_same_v<dtype, cutlass::float_e4m3_t> ||
                cute::is_same_v<dtype, cutlass::bfloat16_t> ||
                cute::is_same_v<dtype, cutlass::half_t>,
                "only 8 bit arithmetic types are supported.");
    std::vector<dtype> block_host(block.size());
    cutlass::device_memory::copy_to_host(block_host.data(), block.get(), block.size());
    if (std::is_same<dtype, float>::value) {
        Engine::CommonUtils::dumpFile(block_host.data(), block_host.size(), out_path);
    }
    else if constexpr (cute::is_same_v<dtype, cutlass::int4b_t>) {
        // special process
        std::vector<int8_t> block_int8(block_host.size());
        unpack_int4_int8_to_int8x2(block_int8.data(), (const int8_t*)block_host.data(), block_int8.size());
        Engine::CommonUtils::dumpFile(block_int8.data(), block_int8.size(), out_path);
        Engine::CommonUtils::dumpFile((const int8_t*)block_host.data(), block_host.size() / 2, out_path + ".packed");
    }
    else if constexpr (cute::is_same_v<dtype, cutlass::float_e4m3_t>) {
        std::vector<float> block_fp32(block_host.size());
        fp8_to_fp32_array(block_fp32.data(), block_host.data(), block_host.size());
        Engine::CommonUtils::dumpFile(block_fp32.data(), block_fp32.size(), out_path);
    }
    else if constexpr (cute::is_same_v<dtype, cutlass::bfloat16_t> ||
             cute::is_same_v<dtype, cutlass::half_t>) {
        std::vector<float> block_fp32(block_host.size());
        float16_to_fp32_array(block_fp32.data(), block_host.data(), block_host.size());
        Engine::CommonUtils::dumpFile(block_fp32.data(), block_fp32.size(), out_path);
    }
    else {
        HAI_PRINT("HAI::dumpFile only support float, int4, fp8, fp16, bf16 data type currently!\n");
        return false;
    }
    return true;
}

template bool dumpFile<float>(const cutlass::DeviceAllocation<float>& block, const std::string& out_path);
template bool dumpFile<cutlass::half_t>(const cutlass::DeviceAllocation<cutlass::half_t>& block, const std::string& out_path);
template bool dumpFile<cutlass::bfloat16_t>(const cutlass::DeviceAllocation<cutlass::bfloat16_t>& block, const std::string& out_path);
template bool dumpFile<cutlass::float_e4m3_t>(const cutlass::DeviceAllocation<cutlass::float_e4m3_t>& block, const std::string& out_path);
template bool dumpFile<cutlass::int4b_t>(const cutlass::DeviceAllocation<cutlass::int4b_t>& block, const std::string& out_path);

template<typename dtype>
bool load_from_file(cutlass::DeviceAllocation<dtype>& block, const std::string& out_path, size_t count) {
    static_assert(std::is_same<dtype, float>::value ||
                cute::is_same_v<dtype, cutlass::int4b_t> ||
                cute::is_same_v<dtype, cutlass::float_e4m3_t> ||
                cute::is_same_v<dtype, cutlass::bfloat16_t> ||
                cute::is_same_v<dtype, cutlass::half_t>,
                "only float, int4, fp8, float16, half types are supported.");
    assert(block.size() == count);
    std::vector<dtype> block_host(count);
    if (std::is_same<dtype, float>::value) {
        Engine::CommonUtils::read_data(block_host.data(), count, out_path);
    }
    else if constexpr (cute::is_same_v<dtype, cutlass::int4b_t>) {
        std::vector<int8_t> block_int8(block_host.size());
        Engine::CommonUtils::read_data(block_int8.data(), count, out_path);
        pack_int4_int8x2_to_int8((int8_t*)block_host.data(), block_int8.data(), block_int8.size());
    }
    else if constexpr (cute::is_same_v<dtype, cutlass::float_e4m3_t>) {
        std::vector<float> block_fp32(block_host.size());
        Engine::CommonUtils::read_data(block_fp32.data(), count, out_path);
        fp32_to_fp8_array(block_host.data(), block_fp32.data(), count);
    }
    else if constexpr (cute::is_same_v<dtype, cutlass::bfloat16_t> ||
             cute::is_same_v<dtype, cutlass::half_t>) {
        std::vector<float> block_fp32(block_host.size());
        Engine::CommonUtils::read_data(block_fp32.data(), count, out_path);
        fp32_to_float16_array(block_host.data(), block_fp32.data(), count);
    }
    else {
        HAI_PRINT("HAI::load_from_file only support float, int4, fp8, fp16, bf16 data type currently!\n");
        return false;
    }
    block.copy_from_host(block_host.data());
    return true;
}

template bool load_from_file<float>(cutlass::DeviceAllocation<float>& block, const std::string& out_path, size_t count);
template bool load_from_file<cutlass::half_t>(cutlass::DeviceAllocation<cutlass::half_t>& block, const std::string& out_path, size_t count);
template bool load_from_file<cutlass::bfloat16_t>(cutlass::DeviceAllocation<cutlass::bfloat16_t>& block, const std::string& out_path, size_t count);
template bool load_from_file<cutlass::float_e4m3_t>(cutlass::DeviceAllocation<cutlass::float_e4m3_t>& block, const std::string& out_path, size_t count);
template bool load_from_file<cutlass::int4b_t>(cutlass::DeviceAllocation<cutlass::int4b_t>& block, const std::string& out_path, size_t count);

template<typename dtype>
void show_info(const dtype* block_ptr, size_t block_size, size_t row, size_t column, size_t cnt_per_rows, const std::string& name) {
    static_assert(std::is_same<dtype, float>::value ||
                cute::is_same_v<dtype, cutlass::int4b_t> ||
                cute::is_same_v<dtype, cutlass::float_e4m3_t> ||
                cute::is_same_v<dtype, cutlass::bfloat16_t> ||
                cute::is_same_v<dtype, cutlass::half_t>,
                "only float, int4, fp8, float16, half types are supported.");
    assert(block_size == row * column);
    if (cute::is_same_v<dtype, cutlass::int4b_t>) {
        block_size /= 2;
        column /= 2;
    }
    size_t show_row = std::min(row, (size_t)5);
    size_t show_count = std::min(column, cnt_per_rows);
    HAI_PRINT("============= %s total: [%zu, %zu] [:%zu]=============\n", name.c_str(), row, column, show_count);
    for (size_t i=0; i<show_row; i++) {
        for (size_t j=0; j<show_count; j++) {
            if (cute::is_same_v<dtype, cutlass::int4b_t>) {
                const int8_t* ptr = (const int8_t*)(block_ptr);
                int8_t low, high;
                unpack_to_int8(low, high, ptr[i*column+j]);
                HAI_PRINT("%d %d ", low, high);
            }
            else if constexpr (std::is_same<dtype, float>::value) {
                HAI_PRINT("%f ", block_ptr[i*column+j]);
            }
            else if constexpr (cute::is_same_v<dtype, cutlass::float_e4m3_t>) {
                float val = fp8_to_fp32(block_ptr[i*column+j]);
                HAI_PRINT("%f ", val);
            }
            else if constexpr (cute::is_same_v<dtype, cutlass::bfloat16_t> ||
                    cute::is_same_v<dtype, cutlass::half_t>) {
                float val;
                float16_to_fp32(val, block_ptr[i*column+j]);
                HAI_PRINT("%f ", val);
            }
        }
        HAI_PRINT("\n");
    }
    HAI_PRINT("\n");
    return;
}

template<typename dtype>
void show_info(const std::vector<dtype>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name) {
    show_info(block.data(), block.size(), row, column, cnt_per_rows, name);
    return;
}

template<typename dtype>
void show_info(const cutlass::DeviceAllocation<dtype>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name) {
    std::vector<dtype> block_host(block.size());
    block.copy_to_host(block_host.data());
    show_info(block_host, row, column, cnt_per_rows, name);
    return;
}

template void show_info<float>(const float* block_ptr, size_t block_size, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::int4b_t>(const cutlass::int4b_t* block_ptr, size_t block_size, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::float_e4m3_t>(const cutlass::float_e4m3_t* block_ptr, size_t block_size, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::bfloat16_t>(const cutlass::bfloat16_t* block_ptr, size_t block_size, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::half_t>(const cutlass::half_t* block_ptr, size_t block_size, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);

template void show_info<float>(const std::vector<float>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::int4b_t>(const std::vector<cutlass::int4b_t>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::float_e4m3_t>(const std::vector<cutlass::float_e4m3_t>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::bfloat16_t>(const std::vector<cutlass::bfloat16_t>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::half_t>(const std::vector<cutlass::half_t>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);

template void show_info<float>(const cutlass::DeviceAllocation<float>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::int4b_t>(const cutlass::DeviceAllocation<cutlass::int4b_t>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::float_e4m3_t>(const cutlass::DeviceAllocation<cutlass::float_e4m3_t>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::bfloat16_t>(const cutlass::DeviceAllocation<cutlass::bfloat16_t>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);
template void show_info<cutlass::half_t>(const cutlass::DeviceAllocation<cutlass::half_t>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name);

double gflops(const std::vector<std::tuple<int, int, int>>& problem_sizes) {
    uint64_t fmas = std::accumulate(problem_sizes.begin(), problem_sizes.end(), 0ULL,
        [](uint64_t sum, const std::tuple<int, int, int>& problem) {
            int m, n, k;
            std::tie(m, n, k) = problem;
            return sum + static_cast<uint64_t>(m) *
                            static_cast<uint64_t>(n) *
                            static_cast<uint64_t>(k);
        });
    return (2.0 * fmas) / 1e9; // Convert to GFLOPS
}

void show_info(torch::Tensor const& tensor, size_t cnt_per_rows, const std::string& name) {
    if (tensor.numel() == 0) {
        HAI_PRINT("Tensor %s is empty.\n", name.c_str());
        return;
    }
    std::ostringstream shape_ss;
    shape_ss << "[";
    for (size_t i = 0; i < tensor.sizes().size(); ++i) {
        shape_ss << tensor.sizes()[i];
        if (i + 1 < tensor.sizes().size()) shape_ss << ", ";
    }
    shape_ss << "]";
    HAI_PRINT("Tensor %s shape: %s\n", name.c_str(), shape_ss.str().c_str());
    // Print tensor dtype
    std::string dtype_str = std::string(tensor.dtype().name());
    HAI_PRINT("Tensor %s dtype: %s\n", name.c_str(), dtype_str.c_str());
    
    auto tensor_cpu = tensor.to(torch::kCPU);
    auto tensor_sizes = tensor_cpu.sizes();
    size_t row = 1;
    size_t column = 1;
    if (tensor_sizes.size() == 0) {
        HAI_PRINT("Tensor %s is empty.\n", name.c_str());
        return;
    }
    else if (tensor_sizes.size() == 1) {
        column = tensor_sizes[0];
    }
    else if (tensor_sizes.size() == 2) {
        row = tensor_sizes[0];
        column = tensor_sizes[1];
    }
    else if (tensor_sizes.size() == 3) {
        row = tensor_sizes[1];
        column = tensor_sizes[2];
    }

    size_t show_row = std::min(row, (size_t)5);
    size_t show_count = std::min(column, cnt_per_rows);

    auto dtype = tensor_cpu.dtype();
    if (dtype == torch::kFloat32) {
        show_info<float>(tensor_cpu.data_ptr<float>(), tensor_cpu.numel(), row, column, show_count, name);
    }
    else if (dtype == torch::kBFloat16) {
        show_info<cutlass::bfloat16_t>(reinterpret_cast<const cutlass::bfloat16_t*>(tensor_cpu.data_ptr()), tensor_cpu.numel(), row, column, show_count, name);
    }
    else if (dtype == torch::kFloat16) {
        show_info<cutlass::half_t>(reinterpret_cast<const cutlass::half_t*>(tensor_cpu.data_ptr()), tensor_cpu.numel(), row, column, show_count, name);
    }
    else if (dtype == torch::kInt8) {
        show_info<cutlass::int4b_t>(reinterpret_cast<const cutlass::int4b_t*>(tensor_cpu.data_ptr()), tensor_cpu.numel()*2, row, column*2, show_count, name);
    } else if (dtype == torch::kFloat8_e4m3fn) {
        show_info<cutlass::float_e4m3_t>(reinterpret_cast<const cutlass::float_e4m3_t*>(tensor_cpu.data_ptr()), tensor_cpu.numel(), row, column, show_count, name);
    } else {
        HAI_PRINT("Unsupported tensor dtype: %s\n", std::string(dtype.name()).c_str());
        return;
    }
}

} // namespace HAI
