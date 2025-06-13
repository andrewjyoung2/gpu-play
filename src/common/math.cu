#include <cassert>
#include <stdexcept>

namespace math {

static const size_t MAX_CUDA_THREADS { 256 };

//------------------------------------------------------------------------------
__global__ void VectorMultiply(float* C,
                               float* A,
                               float* B)
{
  const int idx = threadIdx.x;

  C[idx] = A[idx] * B[idx];
}

//------------------------------------------------------------------------------
__host__ void VectorMultiplyDevice(float* C,
                                   float* A,
                                   float* B,
                                   const size_t len)
{
  assert(len < MAX_CUDA_THREADS); // TODO: throw exception

  VectorMultiply<<<1, len>>>(C, A, B);
}

//------------------------------------------------------------------------------
__host__ void VectorMultiplyHost(float* C,
                                 float* A,
                                 float* B,
                                 const size_t len)
{
  const size_t bytes = len * sizeof(float);

  float* d_C { nullptr };
  float* d_A { nullptr };
  float* d_B { nullptr };

  // Allocate device memory
  if (cudaSuccess != cudaMalloc(&d_C, bytes)) {
    throw std::runtime_error("cudaMalloc failed");
  }
  if (cudaSuccess != cudaMalloc(&d_A, bytes)) {
    throw std::runtime_error("cudaMalloc failed");
  }
  if (cudaSuccess != cudaMalloc(&d_B, bytes)) {
    throw std::runtime_error("cudaMalloc failed");
  }

  // Copy inputs to device
  if (cudaSuccess != cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice)) {
    throw std::runtime_error("Failed transfer from host to device");
  }
  if (cudaSuccess != cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice)) {
    throw std::runtime_error("Failed transfer from host to device");
  }

  // Execute kernel
  VectorMultiplyDevice(d_C, d_A, d_B, len);

  // Copy output to host
  if (cudaSuccess != cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost)) {
    throw std::runtime_error("Failed transfer from device to host");
  }

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

} // namespace math

