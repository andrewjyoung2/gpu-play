#include <chrono>
#include "src/common/assert.hpp"
#include "src/em_alg/posterior.hpp"

namespace EM { namespace CUDA {

//------------------------------------------------------------------------------
__global__ void PosteriorKernel(float*    d_posteriors,
                                float*    d_densities,
                                float*    d_denominators,
                                float*    d_observations,
                                float*    d_means,
                                float*    d_covariances,
                                float*    d_priors,
                                const int dimension,
                                const int numClasses,
                                const int numObs)
{
  // Step 1: Calculate matrix of Gaussian densities
  // dens(k, j) = j-th Gaussian evaluated at k-th observation
  {
    // class ID is threadIdx.x
    // observation ID is threadIdx.y modulo block size

    // size of density matrix
    const int numWrites = numObs * numClasses;
    // offset for entry (k, j) of density matrix
    const int writeIdx  = threadIdx.x
                        + threadIdx.y * blockDim.x
                        + blockIdx.x  * blockDim.x * blockDim.y;

    if (writeIdx < numWrites) {

      // point to k-th row of observation matrix
      float* x = d_observations + dimension * (threadIdx.y + blockIdx.x * blockDim.y);
      // point to j-th row of means matrix
      float* m = d_means + threadIdx.x * dimension;

      const float normSquared = pow(x[0] - m[0], 2) + pow(x[1] - m[1], 2);
      const float s           = d_covariances[threadIdx.x];
      const float c           = 1 / (2 * M_PI * s);
      d_densities[writeIdx]   = c * exp(-normSquared / (2 * s));
    }
  }

  __syncthreads();

  // Step 2: Caclulate the vector of denominators
  // denom(k) = sum_j dens(k,j) * prior(j)
  {
    const int numWrites = numObs;
    const int writeIdx  = threadIdx.y + blockIdx.x * blockDim.y;

    if (writeIdx < numWrites) {
      // point to k-th row of density matrix
      float* x = d_densities
               + threadIdx.y * blockDim.x
               + blockIdx.x  * blockDim.x * blockDim.y;
      // dot product
      float tmp { 0 };
      for (int j = 0; j < numClasses; ++j) {
        tmp += x[j] * d_priors[j];
      }

      d_denominators[writeIdx] = tmp;
    }
  }

  __syncthreads();

  // Step 3: Calculate matrix of posterior probabilities
  {
    // size of posterior matrix = size of density matrix
    const int numWrites = numObs * numClasses;
    // offset for k-th entry of denominator vector
    const int obsIdx  = threadIdx.y + blockIdx.x * blockDim.y;
    // offset for entry (k, j) of density matrix
    const int densIdx = threadIdx.x + obsIdx * blockDim.x;
    // offset for entry (j, k) of posterior matrix
    const int postIdx = threadIdx.x * numObs + obsIdx;

    if ((obsIdx < numObs) && (densIdx < numWrites)) {
      float denom = d_denominators[obsIdx];
      d_posteriors[postIdx] = d_densities[densIdx] * d_priors[threadIdx.x] / denom;
    }
  }
}

//------------------------------------------------------------------------------
__host__ void PosteriorHost(common::Matrix<float>&       posteriors,
                            common::Matrix<float>&       densities,
                            common::Vector<float>&       denominators,
                            const common::Matrix<float>& observations,
                            const common::Matrix<float>& means,
                            const common::Vector<float>& covariances,
                            const common::Vector<float>& priors)
{
  const int numObs     = observations.rows();
  const int dimension  = observations.cols();
  const int numClasses = means.rows();

  ASSERT(dimension == 2); // limit of the current implementation

  ASSERT(means.cols()        == dimension);
  ASSERT(covariances.size()  == numClasses);
  ASSERT(priors.size()       == numClasses);
  ASSERT(posteriors.rows()   == numClasses);
  ASSERT(posteriors.cols()   == numObs);
  ASSERT(densities.rows()    == numObs);
  ASSERT(densities.cols()    == numClasses);
  ASSERT(denominators.size() == numObs);

  // Allocate device memory
  float* d_posteriors   { nullptr };
  float* d_densities    { nullptr };
  float* d_denominators { nullptr };
  float* d_observations { nullptr };
  float* d_means        { nullptr };
  float* d_covariances  { nullptr };
  float* d_priors       { nullptr };

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_posteriors),
             posteriors.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_densities),
             densities.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_denominators),
             denominators.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_observations),
             observations.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_means),
             means.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_covariances),
             covariances.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_priors),
             priors.size() * sizeof(float)));

  // Transfer data from host to device
  auto start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(d_observations,
                        observations.data(),
                        observations.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_means,
                        means.data(),
                        means.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_covariances,
                        covariances.data(),
                        covariances.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_priors,
                        priors.data(),
                        priors.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to transfer from host to device = " << duration.count()
            << " microseconds"                           << std::endl;

  // Run the calculation
  start = std::chrono::high_resolution_clock::now();

  PosteriorDevice(d_posteriors,
                  d_densities,
                  d_denominators,
                  d_observations,
                  d_means,
                  d_covariances,
                  d_priors,
                  dimension,
                  numClasses,
                  numObs);

  cudaDeviceSynchronize();

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to execute EM::CUDA::PosteriorDevice = " << duration.count()
            << " microseconds"                          << std::endl;

  // Transfer results from device to host
  start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(posteriors.data(),
                        d_posteriors,
                        posteriors.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(densities.data(),
                        d_densities,
                        densities.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(denominators.data(),
                        d_denominators,
                        denominators.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to transfer from device to host = " << duration.count()
            << " microseconds"                           << std::endl;

  // Cleanup
  CUDA_CHECK(cudaFree(d_posteriors));
  CUDA_CHECK(cudaFree(d_densities));
  CUDA_CHECK(cudaFree(d_denominators));
  CUDA_CHECK(cudaFree(d_observations));
  CUDA_CHECK(cudaFree(d_means));
  CUDA_CHECK(cudaFree(d_covariances));
  CUDA_CHECK(cudaFree(d_priors));

  CUDA_CHECK(cudaDeviceReset());
}

//------------------------------------------------------------------------------
__host__ void PosteriorDevice(float*    d_posteriors,
                              float*    d_densities,
                              float*    d_denominators,
                              float*    d_observations,
                              float*    d_means,
                              float*    d_covariances,
                              float*    d_priors,
                              const int dimension,
                              const int numClasses,
                              const int numObs)
{
  ASSERT(nullptr != d_posteriors);
  ASSERT(nullptr != d_densities);
  ASSERT(nullptr != d_denominators);
  ASSERT(nullptr != d_observations);
  ASSERT(nullptr != d_means);
  ASSERT(nullptr != d_covariances);
  ASSERT(nullptr != d_priors);

  // Run kernel
  const int xDim = numClasses;
  const int yDim = 64;
  ASSERT(xDim * yDim < 256);

  const dim3 threadsPerBlock(xDim, yDim);

  // Round Up: ceil((numClasses * numObs) / (xDim * yDim))
  const int  numBlocks = (numClasses * numObs + xDim * yDim - 1) / (xDim * yDim);

  PosteriorKernel<<<numBlocks, threadsPerBlock>>>(d_posteriors,
                                                  d_densities,
                                                  d_denominators,
                                                  d_observations,
                                                  d_means,
                                                  d_covariances,
                                                  d_priors,
                                                  dimension,
                                                  numClasses,
                                                  numObs);
}

} // namespace CUDA
} // namespace EM

