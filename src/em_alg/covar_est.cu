#include <chrono>
#include "src/common/assert.hpp"
#include "src/em_alg/covar_est.hpp"

namespace EM { namespace CUDA {

//------------------------------------------------------------------------------
__global__ void CovarEstKernel(float*    d_covar_est,
                               float*    d_mean_est,
                               float*    d_posteriors,
                               float*    d_observations,
                               const int dimension,
                               const int numClasses,
                               const int numObs)
{
  const int j = threadIdx.x;

  if (j < numClasses) {
    float num { 0 };
    float den { 0 };

    // point to j-th mean vector
    float* m = d_mean_est + j * dimension;

    for (int k = 0; k < numObs; ++k) {
      // point to k-th observation vector
      float*      x           = d_observations + k * dimension;
      const float normSquared = pow(x[0] - m[0], 2) + pow(x[0] - m[0], 2);
      num += d_posteriors[k + j * numObs] * normSquared;
      den += d_posteriors[k + j * numObs];
    }

    d_covar_est[j] = num / (dimension * den);
  }
}

//------------------------------------------------------------------------------
__host__ void CovarEstHost(common::Vector<float>&       covar_est,
                           const common::Matrix<float>& mean_est,
                           const common::Matrix<float>& posteriors,
                           const common::Matrix<float>& observations)
{
  const int numClasses = covar_est.size();
  const int numObs     = observations.rows();
  const int dimension  = observations.cols();

  ASSERT(dimension         == 2);
  ASSERT(mean_est.rows()   == numClasses);
  ASSERT(mean_est.cols()   == dimension);
  ASSERT(posteriors.rows() == numClasses);
  ASSERT(posteriors.cols() == numObs);

  // Allocate device memory
  float* d_covar_est    { nullptr };
  float* d_mean_est     { nullptr };
  float* d_posteriors   { nullptr };
  float* d_observations { nullptr };

  auto start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_covar_est),
             covar_est.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mean_est),
             mean_est.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_posteriors),
             posteriors.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_observations),
             observations.size() * sizeof(float)));

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to allocate device memory = " << duration.count()
            << " microseconds"                     << std::endl;

  // Transfer data from host to device
  start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(d_mean_est,
                        mean_est.data(),
                        mean_est.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_posteriors,
                        posteriors.data(),
                        posteriors.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_observations,
                        observations.data(),
                        observations.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to execute transfer from host to device = " << duration.count()
            << " microseconds"                                   << std::endl;

  // Run the calculation
  start = std::chrono::high_resolution_clock::now();

  CovarEstDevice(d_covar_est,
                 d_mean_est,
                 d_posteriors,
                 d_observations,
                 dimension,
                 numClasses,
                 numObs);

  cudaDeviceSynchronize();

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to execute EM::CUDA::CovarEstDevice = " << duration.count()
            << " microseconds"                               << std::endl;


  // Transfer results from device to host
  start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(covar_est.data(),
                        d_covar_est,
                        covar_est.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to transfer from device to host = " << duration.count()
            << " microseconds"                           << std::endl;

  CUDA_CHECK(cudaDeviceReset());
}

//------------------------------------------------------------------------------
__host__ void CovarEstDevice(float*    d_covar_est,
                             float*    d_mean_est,
                             float*    d_posteriors,
                             float*    d_observations,
                             const int dimension,
                             const int numClasses,
                             const int numObs)
{
  ASSERT(nullptr != d_covar_est);
  ASSERT(nullptr != d_mean_est);
  ASSERT(nullptr != d_posteriors);
  ASSERT(nullptr != d_observations);

  const int xDim { numClasses };
  const int yDim { 64 };
  ASSERT(xDim * yDim < 256);

  const dim3 threadsPerBlock(xDim, yDim);
  const int  numBlocks = (numObs + yDim - 1) / yDim;

  CovarEstKernel<<< numBlocks, threadsPerBlock >>>(d_covar_est,
                                                   d_mean_est,
                                                   d_posteriors,
                                                   d_observations,
                                                   dimension,
                                                   numClasses,
                                                   numObs);
}

} // namespace CUDA
} // namspace EM

