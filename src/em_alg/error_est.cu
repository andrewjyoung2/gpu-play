#include "src/common/assert.hpp"
#include "src/em_alg/error_est.hpp"

namespace EM { namespace CUDA {

//------------------------------------------------------------------------------
__global__ void ErrorEstKernel(float* d_error_est,
                               float* d_mean_new,
                               float* d_mean_old,
                               float* d_covar_new,
                               float* d_covar_old,
                               float* d_prior_new,
                               float* d_prior_old,
                               const int dimension,
                               const int numClasses)
{
  __shared__ float scratch[12];

  const int idx = threadIdx.x;

  if (idx < numClasses * dimension) {
    scratch[idx] = abs(d_mean_new[idx] - d_mean_old[idx]);
  }
  else if (idx < numClasses * (dimension + 1)) {
    scratch[idx] = abs(d_covar_new[idx - 6] - d_covar_old[idx - 6]);
  }
  else if (idx < numClasses * (dimension + 2)) {
    scratch[idx] = abs(d_prior_new[idx - 9] - d_prior_old[idx - 9]);
  }

  __syncthreads();

  // Add them up
  if (0 == idx) {
    float err { 0 };
    for (int i = 0; i < 12; ++i) {
      err += scratch[i];
    }

    *d_error_est = err;
  }
}

//------------------------------------------------------------------------------
__host__ float ErrorEstHost(const common::Matrix<float>& mean_new,
                            const common::Matrix<float>& mean_old,
                            const common::Vector<float>& covar_new,
                            const common::Vector<float>& covar_old,
                            const common::Vector<float>& prior_new,
                            const common::Vector<float>& prior_old)
{
  const int numClasses = mean_new.rows();
  const int dimension  = mean_new.cols();

  ASSERT(mean_old.rows()  == numClasses);
  ASSERT(mean_old.cols()  == dimension);
  ASSERT(covar_new.size() == numClasses);
  ASSERT(covar_old.size() == numClasses);
  ASSERT(prior_new.size() == numClasses);
  ASSERT(prior_old.size() == numClasses);

  // Allocate device memory
  float* d_error_est { nullptr };
  float* d_mean_new  { nullptr };
  float* d_mean_old  { nullptr };
  float* d_covar_new { nullptr };
  float* d_covar_old { nullptr };
  float* d_prior_new { nullptr };
  float* d_prior_old { nullptr };

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_error_est),
             sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mean_new),
             mean_new.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mean_old),
             mean_old.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_covar_new),
             covar_new.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_covar_old),
             covar_old.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prior_new),
             prior_new.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prior_old),
             prior_old.size() * sizeof(float)));

  // Transfer data from host to device
  CUDA_CHECK(cudaMemcpy(d_mean_new,
                        mean_new.data(),
                        mean_new.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mean_old,
                        mean_old.data(),
                        mean_old.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_covar_new,
                        covar_new.data(),
                        covar_new.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_covar_old,
                        covar_old.data(),
                        covar_old.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_prior_new,
                        prior_new.data(),
                        prior_new.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_prior_old,
                        prior_old.data(),
                        prior_old.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Run the calculation
  EM::CUDA::ErrorEstDevice(d_error_est,
                           d_mean_new,
                           d_mean_old,
                           d_covar_new,
                           d_covar_old,
                           d_prior_new,
                           d_prior_old,
                           dimension,
                           numClasses);

  cudaDeviceSynchronize();

  // Transfer results from device to host
  float error_est { 0.7734 };
  CUDA_CHECK(cudaMemcpy(&error_est,
                        d_error_est,
                        sizeof(float),
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaDeviceReset());

  return error_est;
}

//------------------------------------------------------------------------------
__host__ void ErrorEstDevice(float* d_error_est,
                             float* d_mean_new,
                             float* d_mean_old,
                             float* d_covar_new,
                             float* d_covar_old,
                             float* d_prior_new,
                             float* d_prior_old,
                             const int dimension,
                             const int numClasses)
{
  ASSERT(nullptr != d_error_est);
  ASSERT(nullptr != d_mean_new);
  ASSERT(nullptr != d_mean_old);
  ASSERT(nullptr != d_covar_new);
  ASSERT(nullptr != d_covar_old);

  const int numBlocks { 1 };
  const int threadsPerBlock = numClasses * dimension // mean
                            + numClasses             // covar
                            + numClasses;            // prior

  // Run kernel
  ErrorEstKernel<<< numBlocks, threadsPerBlock >>>(d_error_est,
                                                   d_mean_new,
                                                   d_mean_old,
                                                   d_covar_new,
                                                   d_covar_old,
                                                   d_prior_new,
                                                   d_prior_old,
                                                   dimension,
                                                   numClasses);

}

} // namespace CUDA
} // namspace EM

