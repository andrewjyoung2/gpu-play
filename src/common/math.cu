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
__global__ void Accumulate(float* d_result, float* d_A)
{
  *d_result = 10.0f;
}

//------------------------------------------------------------------------------
__host__ void AccumulateDevice(float* d_result, float* d_A, const size_t len)
{
  if (len >= MAX_CUDA_THREADS) {
    throw std::runtime_error("len must be less than MAX_CUDA_THREADS");
  }

  Accumulate<<<1, len>>>(d_result, d_A);
}

//------------------------------------------------------------------------------
__host__ float AccumulateHost(float* A, const size_t len)
{
  if (len >= MAX_CUDA_THREADS) {
    throw std::runtime_error("len must be less than MAX_CUDA_THREADS");
  }

  const size_t bytes = len * sizeof(float);

  // Allocate device memory
  float* d_A { nullptr };
  if (cudaSuccess != cudaMalloc(&d_A, bytes)) {
    throw std::runtime_error("cudaMalloc failed");
  }
  float* d_result { nullptr };
  if (cudaSuccess != cudaMalloc(&d_result, sizeof(float))) {
    throw std::runtime_error("cudaMalloc failed");
  }

  // Copy inputs to device
  if (cudaSuccess != cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice)) {
    throw std::runtime_error("Failed transfer from host to device");
  }

  AccumulateDevice(d_result, d_A, len);

  // Copy result to host
  float result { 0.7734f };
  if (cudaSuccess != cudaMemcpy(&result,
                                d_result,
                                sizeof(float),
                                cudaMemcpyDeviceToHost)) {
    throw std::runtime_error("Failed transfer from device to host");
  }

  // Cleanup
  cudaFree(d_A);
  cudaFree(d_result);

  return result;
}

//------------------------------------------------------------------------------
__host__ void VectorMultiplyDevice(float* C,
                                   float* A,
                                   float* B,
                                   const size_t len)
{
  if (len >= MAX_CUDA_THREADS) {
    throw std::runtime_error("len must be less than MAX_CUDA_THREADS");
  }

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

