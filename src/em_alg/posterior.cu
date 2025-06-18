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
  // Sanity check - copy observations to densities
  const int idx = threadIdx.x
                + threadIdx.y * blockDim.x
                + blockIdx.x  * blockDim.x * blockDim.y;
  const int matrixSize = numObs * dimension;

  if (idx < matrixSize) {
    d_densities[idx] = d_observations[idx];
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

  // Run the calculation
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

  // Transfer results from device to host
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
  const int yDim = 32;
  ASSERT(xDim * yDim < 256);

  const dim3 threadsPerBlock(xDim, yDim);
  const int  numBlocks = numClasses * numObs / (xDim * yDim);

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

