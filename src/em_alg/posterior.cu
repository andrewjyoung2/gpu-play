#include "src/em_alg/posterior.hpp"

namespace EM { namespace CUDA {

//------------------------------------------------------------------------------
// TODO: write kernel

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
  const int dim        = observations.cols();
  const int numClasses = means.rows();

  ASSERT(dim == 2); // limit of the current implementation

  ASSERT(means.cols()        == dim);
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

  // Run the calculation

  // Transfer results from device to host

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
}

} // namespace CUDA
} // namespace EM

