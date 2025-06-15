#include <cassert>
#include <limits>
#include "src/cublas/cublas_wrap.hpp"

namespace cublas_wrap {

__host__ float Ddot(const std::vector<double>& A, const std::vector<double>& B)
{
  assert(A.size() == B.size());

  cublasHandle_t cublasH { nullptr };
  cudaStream_t   stream  { nullptr };

  double result = std::numeric_limits<double>::min();

  double* d_A { nullptr };
  double* d_B { nullptr };

  // creat cuBLAS handle and bind to stream
  CUBLAS_CHECK(cublasCreate(&cublasH));
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CUBLAS_CHECK(cublasSetStream(cublasH, stream));

  // Allocate device memory, copy to device
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                        sizeof(double) * A.size()));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                        sizeof(double) * B.size()));

  CUDA_CHECK(cudaMemcpyAsync(d_A,
                             A.data(),
                             sizeof(double) * A.size(),
                             cudaMemcpyHostToDevice,
                             stream));
  CUDA_CHECK(cudaMemcpyAsync(d_B,
                             B.data(),
                             sizeof(double) * B.size(),
                             cudaMemcpyHostToDevice,
                             stream));
  // Compute the dot product
  const int incx = 1;
  const int incy = 1;
  CUBLAS_CHECK(cublasDdot(cublasH, A.size(), d_A, incx, d_B, incy, &result));

  // Synchronize stream to copy result to host
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Cleanup
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));

  CUBLAS_CHECK(cublasDestroy(cublasH));

  CUDA_CHECK(cudaStreamDestroy(stream));

  CUDA_CHECK(cudaDeviceReset());

  return result;
}

} // namespace cublas_wrap

