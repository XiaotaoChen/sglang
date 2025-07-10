#pragma onece

#include <vector>
#include <string>
#include <cutlass/float8.h>
#include <cutlass/integer_subbyte.h>
#include <cutlass/util/device_memory.h>

#include "cuda_runtime.h"
#include <iostream>
#include <torch/all.h>


/**
 * Panic wrapper for unwinding CUTLASS errors
 */
#define CUTLASS_CHECK(status)                                                                    \
  {                                                                                              \
    cutlass::Status error = status;                                                              \
    if (error != cutlass::Status::kSuccess) {                                                    \
      std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
                << std::endl;                                                                    \
      exit(EXIT_FAILURE);                                                                        \
    }                                                                                            \
  }


/**
 * Panic wrapper for unwinding CUDA runtime errors
 */
#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

namespace HAI
{

void unpack_to_int8(int8_t& low, int8_t& high, const int8_t& packed);
void pack_to_int8(int8_t& packed, const int8_t& low, const int8_t& high);

bool unpack_int4_int8_to_int8x2(int8_t* unpacked, const int8_t* packed, int count);
bool unpack_int4_int8_to_int8x2(int8_t* unpacked, const cutlass::int4b_t* packed, int count);

bool pack_int4_int8x2_to_int8(int8_t* packed, const int8_t* unpacked, int count);
bool pack_int4_int8x2_to_int8(cutlass::int4b_t* packed, const int8_t* unpacked, int count);


cutlass::float_e4m3_t dequant_int8_to_fp8(const int8_t src, const cutlass::float_e4m3_t scale);
cutlass::float_e4m3_t dequant_int8_to_fp8(const int8_t src, const cutlass::bfloat16_t scale);

float fp8_to_fp32(cutlass::float_e4m3_t value);
void fp8_to_fp32_array(float* dst, const cutlass::float_e4m3_t* src, size_t count);
cutlass::float_e4m3_t fp32_to_fp8(float value);
void fp32_to_fp8_array(cutlass::float_e4m3_t* dst, const float* src, size_t count);

template<typename dtype>
void float16_to_fp32(float& dst, dtype value);

template<typename dtype>
void fp32_to_float16(dtype& dst, float value);

template<typename dtype>
void fp32_to_float16_array(dtype* dst, const float* src, size_t count);

template<typename dtype>
void float16_to_fp32_array(float* dst, const dtype* src, size_t count);

template<typename dtype>
void float16_to_fp8(cutlass::float_e4m3_t& dst, dtype value, dtype scale_inv);
template<typename dtype>
void float16_to_fp8_array(cutlass::float_e4m3_t* dst, const dtype* src, size_t count, dtype scale_inv);

template<typename dtype>
void fp8_to_float16(dtype& dst, cutlass::float_e4m3_t value, dtype scale);
template<typename dtype>
void fp8_to_float16_array(dtype* dst, const cutlass::float_e4m3_t* src, size_t count, dtype scale);

template<typename dtype>
bool dumpFile(const cutlass::DeviceAllocation<dtype>& block, const std::string& out_path);
template<typename dtype>
bool load_from_file(cutlass::DeviceAllocation<dtype>& block, const std::string& out_path, size_t count);

template<typename dtype>
void show_info(const std::vector<dtype>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name="block");

template<typename dtype>
void show_info(const dtype* block_ptr, size_t block_size, size_t row, size_t column, size_t cnt_per_rows, const std::string& name="block");

template<typename dtype>
void show_info(const cutlass::DeviceAllocation<dtype>& block, size_t row, size_t column, size_t cnt_per_rows, const std::string& name="block");

/// Result structure
struct ProfileResult
{
  double avg_runtime_ms = 0.0;
  double gflops = 0.0;
  // unused currently
  cutlass::Status status = cutlass::Status::kSuccess;
  cudaError_t error = cudaSuccess;
  bool passed = false;
};

double gflops(const std::vector<std::tuple<int, int, int>>& problem_sizes);

void show_info(torch::Tensor const& tensor, size_t cnt_per_rows, const std::string& name);


template <class Gemm>
void grouped_mixed_dtype_profiling(
    HAI::ProfileResult& result, Gemm& gemm,
    const std::vector<std::tuple<int, int, int>>& problem_sizes,
    int warmup, int loop, const std::string& name = "GEMM") {

    if (loop <= 0) return;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::vector<float> runtimes;
    runtimes.reserve(loop); 

    for (int iter = 0; iter < warmup + loop; ++iter) {
        cudaEventRecord(start);
        CUTLASS_CHECK(gemm.run());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        if (iter >= warmup) {
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            runtimes.push_back(milliseconds);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    result.avg_runtime_ms = std::accumulate(runtimes.begin(), runtimes.end(), 0.0f) / runtimes.size();
    result.gflops = HAI::gflops(problem_sizes);
    float gflops_per_sec = 1000.0f * result.gflops / result.avg_runtime_ms;
    printf("========== %s Profiling Result ==========\n", name.c_str());
    printf("  Groups      : %zu\n", problem_sizes.size());
    for (int i = 0; i < problem_sizes.size(); ++i) {
        auto& size = problem_sizes[i];
        int m, n, k;
        std::tie(m, n, k) = size;
        if (m > 0) {
          printf("          %d:   %d x %d x %d\n", i, m, n, k);
        }
    }
    printf("  Avg runtime : %f ms\n", result.avg_runtime_ms);
    printf("  GFLOPS      : %.3f\n", result.gflops);
    printf("  GFLOPS/sec  : %.3f\n", gflops_per_sec);
    return;
}


} // namespace HAI
