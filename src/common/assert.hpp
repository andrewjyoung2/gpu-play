#pragma once

#include <iostream>
#include <stdexcept>

//------------------------------------------------------------------------------
// General purpose error checking
#define ASSERT(cond)                                \
  do {                                              \
    if (!(cond)) {                                  \
      std::cerr << "Assert "  << #cond              \
                << " faied, " << __FILE__           \
                << ":"        << __LINE__           \
                << std::endl;                       \
      throw std::runtime_error("Assertion failed"); \
    }                                               \
  } while (0)

//------------------------------------------------------------------------------
// CUDA API error checking
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/utils/cublas_utils.h
#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t err_ = (err);                                              \
        if (err_ != cudaSuccess) {                                             \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
    } while (0)

//------------------------------------------------------------------------------
// cublas API error checking
#define CUBLAS_CHECK(err)                                                        \
    do {                                                                         \
        cublasStatus_t err_ = (err);                                             \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                     \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("cublas error");                            \
        }                                                                        \
    } while (0)

