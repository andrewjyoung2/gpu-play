#include <chrono>
#include "src/common/assert.hpp"
#include "src/em_alg/covar_est.hpp"
#include "src/em_alg/em_alg.hpp"
#include "src/em_alg/error_est.hpp"
#include "src/em_alg/mean_est.hpp"
#include "src/em_alg/posterior.hpp"

namespace EM { namespace CUDA {

//------------------------------------------------------------------------------
__host__ void EM_IterationHost(float&                       error_est,
                               common::Vector<float>&       covar_est,
                               common::Vector<float>&       prior_est,
                               common::Matrix<float>&       mean_est,
                               common::Matrix<float>&       posteriors,
                               common::Matrix<float>&       densities,
                               common::Vector<float>&       denominators,
                               const common::Matrix<float>& observations,
                               const common::Matrix<float>& mean_init,
                               const common::Vector<float>& covar_init,
                               const common::Vector<float>& prior_init)
{
  const int numObs     = observations.rows();
  const int dimension  = observations.cols();
  const int numClasses = mean_init.rows();

  ASSERT(dimension == 2); // limit of the current implementation

  ASSERT(mean_init.cols()    == dimension);
  ASSERT(covar_init.size()   == numClasses);
  ASSERT(prior_init.size()   == numClasses);
  ASSERT(posteriors.rows()   == numClasses);
  ASSERT(posteriors.cols()   == numObs);
  ASSERT(densities.rows()    == numObs);
  ASSERT(densities.cols()    == numClasses);
  ASSERT(denominators.size() == numObs);

  // TODO: finish validation

  // Allocate device memory
  float* d_error_est    { nullptr };
  float* d_covar_est    { nullptr };
  float* d_prior_est    { nullptr };
  float* d_mean_est     { nullptr };
  float* d_posteriors   { nullptr };
  float* d_densities    { nullptr };
  float* d_denominators { nullptr };
  float* d_observations { nullptr };
  float* d_mean_init    { nullptr };
  float* d_covar_init   { nullptr };
  float* d_prior_init   { nullptr };

  auto start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_error_est),
             sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_covar_est),
             covar_est.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prior_est),
             prior_est.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mean_est),
             mean_est.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_posteriors),
             posteriors.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_densities),
             densities.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_denominators),
             denominators.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_observations),
             observations.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mean_init),
             mean_init.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_covar_init),
             covar_init.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prior_init),
             prior_init.size() * sizeof(float)));

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to allocate device memory = " << duration.count()
            << " microseconds"                     << std::endl;

  // Transfer data from host to device
  start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(d_observations,
                        observations.data(),
                        observations.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mean_init,
                        mean_init.data(),
                        mean_init.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_covar_init,
                        covar_init.data(),
                        covar_init.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_prior_init,
                        prior_init.data(),
                        prior_init.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to transfer from host to device = " << duration.count()
            << " microseconds"                           << std::endl;

  // Run the calculation
  start = std::chrono::high_resolution_clock::now();

  EM_IterationDevice(d_error_est,
                     d_covar_est,
                     d_prior_est,
                     d_mean_est,
                     d_posteriors,
                     d_densities,
                     d_denominators,
                     d_observations,
                     d_mean_init,
                     d_covar_init,
                     d_prior_init,
                     dimension,
                     numClasses,
                     numObs);

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to execute EM::CUDA::IterationDevice = " << duration.count()
            << " microseconds"                                << std::endl;

  // Transfer results from device to host
  start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(&error_est,
                        d_error_est,
                        sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(covar_est.data(),
                        d_covar_est,
                        covar_est.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(prior_est.data(),
                        d_prior_est,
                        prior_est.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mean_est.data(),
                        d_mean_est,
                        mean_est.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
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
  CUDA_CHECK(cudaFree(d_error_est));
  CUDA_CHECK(cudaFree(d_covar_est));
  CUDA_CHECK(cudaFree(d_prior_est));
  CUDA_CHECK(cudaFree(d_mean_est));
  CUDA_CHECK(cudaFree(d_posteriors));
  CUDA_CHECK(cudaFree(d_densities));
  CUDA_CHECK(cudaFree(d_denominators));
  CUDA_CHECK(cudaFree(d_observations));
  CUDA_CHECK(cudaFree(d_mean_init));
  CUDA_CHECK(cudaFree(d_covar_init));
  CUDA_CHECK(cudaFree(d_prior_init));

  CUDA_CHECK(cudaDeviceReset());
}

//------------------------------------------------------------------------------
__host__ void EM_IterationDevice(float* d_error_est,
                                 float* d_covar_est,
                                 float* d_prior_est,
                                 float* d_mean_est,
                                 float* d_posteriors,
                                 float* d_densities,
                                 float* d_denominators,
                                 float* d_observations,
                                 float* d_mean_init,
                                 float* d_covar_init,
                                 float* d_prior_init,
                                 const int dimension,
                                 const int numClasses,
                                 const int numObs)
{
  // TODO: validate input

  PosteriorDevice(d_posteriors,
                  d_densities,
                  d_denominators,
                  d_observations,
                  d_mean_init,
                  d_covar_init,
                  d_prior_init,
                  dimension,
                  numClasses,
                  numObs);

  MeanEstDevice(d_mean_est,
                d_posteriors,
                d_observations,
                dimension,
                numClasses,
                numObs);

  CovarEstDevice(d_covar_est,
                 d_prior_est,
                 d_mean_est,
                 d_posteriors,
                 d_observations,
                 dimension,
                 numClasses,
                 numObs);

  ErrorEstDevice(d_error_est,
                 d_mean_est,
                 d_mean_init,
                 d_covar_est,
                 d_covar_init,
                 d_prior_est,
                 d_prior_init,
                 dimension,
                 numClasses);
}

} // namespace CUDA
} // namspace EM


