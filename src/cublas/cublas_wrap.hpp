#pragma once

#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cublas_api.h>
#include <stdexcept>
#include <vector>

// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/utils/cublas_utils.h
// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

namespace cublas_wrap {
__host__ float Ddot(const std::vector<double>& A, const std::vector<double>& B);

} // namespace cublas_wrap

