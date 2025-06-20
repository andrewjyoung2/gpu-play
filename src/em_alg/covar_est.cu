#include <chrono>
#include "src/common/assert.hpp"
#include "src/em_alg/covar_est.hpp"

namespace EM { namespace CUDA {

//------------------------------------------------------------------------------
__global__ void CovarEstKernel(float*    d_covar_est,
                               float*    d_prior_est,
                               float*    d_mean_est,
                               float*    d_posteriors,
                               float*    d_observations,
                               const int dimension,
                               const int numClasses,
                               const int numObs)
{
  __shared__ float scratch[3][2][64];

  const int j = threadIdx.x;
  const int groupSize = (numObs + blockDim.z - 1) / blockDim.z;

  if (j < numClasses) {
    float* m = d_mean_est + j * dimension;

    if (0 == threadIdx.y) {
      float num { 0 };

      for (int k = 0; k < groupSize; ++k) {
        const int obsIdx = k + threadIdx.z * groupSize;
        // point to k-th observation vector
        if (obsIdx < numObs) {
          float*      x           = d_observations + obsIdx * dimension;
          const float normSquared = pow(x[0] - m[0], 2) + pow(x[1] - m[1], 2);
          num += d_posteriors[j * numObs + obsIdx] * normSquared;
        }
      }

      scratch[j][0][threadIdx.z] = num;
    }
    else if (1 == threadIdx.y) {
      float den { 0 };

      for (int k = 0; k < groupSize; ++k) {
        const int obsIdx = k + threadIdx.z * groupSize;
        if (obsIdx < numObs) {
          den += d_posteriors[j * numObs + obsIdx];
        }
      }

      scratch[j][1][threadIdx.z] = den;
    }

    __syncthreads();

    // Parallel reduction
    // https://leimao.github.io/blog/CUDA-Reduction/
    __shared__ float morescratch[2][2][64];
    if (threadIdx.z < 32) {
      morescratch[j][threadIdx.y][threadIdx.z]
        = scratch[j][threadIdx.y][2 * threadIdx.z]
        + scratch[j][threadIdx.y][2 * threadIdx.z + 1];
    }
    __syncthreads();
    if (threadIdx.z < 16) {
      morescratch[j][threadIdx.y][32 + threadIdx.z]
        = morescratch[j][threadIdx.y][2 * threadIdx.z]
        + morescratch[j][threadIdx.y][2 * threadIdx.z + 1];
    }
    __syncthreads();
    if (threadIdx.z < 8) {
      morescratch[j][threadIdx.y][48 + threadIdx.z]
        = morescratch[j][threadIdx.y][32 + 2 * threadIdx.z]
        + morescratch[j][threadIdx.y][32 + 2 * threadIdx.z + 1];
    }
    __syncthreads();
    if (threadIdx.z < 4) {
      morescratch[j][threadIdx.y][56 + threadIdx.z]
        = morescratch[j][threadIdx.y][48 + 2 * threadIdx.z]
        + morescratch[j][threadIdx.y][48 + 2 * threadIdx.z + 1];
    }
    __syncthreads();
    if (threadIdx.z < 2) {
      morescratch[j][threadIdx.y][60 + threadIdx.z]
        = morescratch[j][threadIdx.y][56 + 2 * threadIdx.z]
        + morescratch[j][threadIdx.y][56 + 2 * threadIdx.z + 1];
    }
    if (0 == threadIdx.z) {
      morescratch[j][threadIdx.y][62]
        = morescratch[j][threadIdx.y][60]
        + morescratch[j][threadIdx.y][61];
    }
    __syncthreads();
    if ((0 == threadIdx.y) && (0 == threadIdx.z)) {
      d_covar_est[j] = morescratch[j][0][62] / (dimension * morescratch[j][1][62]);
      d_prior_est[j] = morescratch[j][1][62] / numObs;
    }
  }
}

//------------------------------------------------------------------------------
__host__ void CovarEstHost(common::Vector<float>&       covar_est,
                           common::Vector<float>&       prior_est,
                           const common::Matrix<float>& mean_est,
                           const common::Matrix<float>& posteriors,
                           const common::Matrix<float>& observations)
{
  const int numClasses = covar_est.size();
  const int numObs     = observations.rows();
  const int dimension  = observations.cols();

  ASSERT(dimension         == 2);
  ASSERT(prior_est.size()  == numClasses);
  ASSERT(mean_est.rows()   == numClasses);
  ASSERT(mean_est.cols()   == dimension);
  ASSERT(posteriors.rows() == numClasses);
  ASSERT(posteriors.cols() == numObs);

  // Allocate device memory
  float* d_covar_est    { nullptr };
  float* d_prior_est    { nullptr };
  float* d_mean_est     { nullptr };
  float* d_posteriors   { nullptr };
  float* d_observations { nullptr };

  auto start = std::chrono::high_resolution_clock::now();

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_covar_est),
             covar_est.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_prior_est),
             prior_est.size() * sizeof(float)));
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
                 d_prior_est,
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
  CUDA_CHECK(cudaMemcpy(prior_est.data(),
                        d_prior_est,
                        prior_est.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  end      = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "Time to transfer from device to host = " << duration.count()
            << " microseconds"                           << std::endl;

  CUDA_CHECK(cudaDeviceReset());
}

//------------------------------------------------------------------------------
__host__ void CovarEstDevice(float*    d_covar_est,
                             float*    d_prior_est,
                             float*    d_mean_est,
                             float*    d_posteriors,
                             float*    d_observations,
                             const int dimension,
                             const int numClasses,
                             const int numObs)
{
  ASSERT(nullptr != d_covar_est);
  ASSERT(nullptr != d_prior_est);
  ASSERT(nullptr != d_mean_est);
  ASSERT(nullptr != d_posteriors);
  ASSERT(nullptr != d_observations);

  const int xDim { numClasses };
  //const int yDim { 64 };
  const int yDim { 2 };
  const int zDim { 64 };
  ASSERT(xDim * yDim * zDim < 512);

  const dim3 threadsPerBlock(xDim, yDim, zDim);
  //const int  numBlocks = (numObs + yDim - 1) / yDim;
  const int numBlocks = 1;

  CovarEstKernel<<< numBlocks, threadsPerBlock >>>(d_covar_est,
                                                   d_prior_est,
                                                   d_mean_est,
                                                   d_posteriors,
                                                   d_observations,
                                                   dimension,
                                                   numClasses,
                                                   numObs);
}

} // namespace CUDA
} // namspace EM

