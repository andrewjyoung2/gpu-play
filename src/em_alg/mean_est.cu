#include <chrono>
#include "src/common/assert.hpp"
#include "src/em_alg/mean_est.hpp"

namespace EM { namespace CUDA {

//------------------------------------------------------------------------------
// TODO: __global__ void MeanEstKernel

//------------------------------------------------------------------------------
__host__ void MeanEstHost(common::Matrix<float>&       means,
                          const common::Matrix<float>& posteriors,
                          const common::Matrix<float>& observations)
{
  const int numClasses = means.rows();
  const int dimension  = means.cols();
  const int numObs     = posteriors.cols();

  ASSERT(dimension == 2); // limit of the current implementation

  ASSERT(posteriors.rows()    == numClasses);
  ASSERT(observations.rows() == numObs);
  ASSERT(observations.cols() == dimension);

  // Allocate device memory
  float* d_means        { nullptr };
  float* d_posteriors   { nullptr };
  float* d_observations { nullptr };

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_means),
             means.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_posteriors),
             posteriors.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_observations),
             observations.size() * sizeof(float)));

  // Transfer data from host to device
  auto start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(d_observations,
                        observations.data(),
                        observations.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_posteriors,
                        posteriors.data(),
                        posteriors.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to execute EM::CUDA::PosteriorDevice = " << duration.count()
            << " microseconds"                          << std::endl;

  // Run the calculation
  start = std::chrono::high_resolution_clock::now();

  MeanEstDevice(d_means,
                d_posteriors,
                d_observations,
                dimension,
                numClasses,
                numObs);

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to execute EM::CUDA::MeanEstDevice = " << duration.count()
            << " microseconds"                              << std::endl;

  // Transfer results from device to host
  start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMemcpy(means.data(),
                        d_means,
                        means.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to transfer from device to host = " << duration.count()
            << " microseconds"                           << std::endl;

  // Cleanup
  CUDA_CHECK(cudaFree(d_means));
  CUDA_CHECK(cudaFree(d_posteriors));
  CUDA_CHECK(cudaFree(d_observations));

  CUDA_CHECK(cudaDeviceReset());
}

__host__ void MeanEstDevice(float*    d_means,
                            float*    d_posteriors,
                            float*    d_observations,
                            const int dimension,
                            const int numClasses,
                            const int numObs)
{
  ASSERT(nullptr != d_means);
  ASSERT(nullptr != d_posteriors);
  ASSERT(nullptr != d_observations);
}

} // namespace CUDA
} // namespace EM
