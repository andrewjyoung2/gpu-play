#pragma once

#include <cuda_runtime.h>
#include "src/common/matrix.hpp"
#include "src/common/vector.hpp"

namespace EM { namespace Scalar {

void EM_Iteration(float&                       error_est,
                  common::Vector<float>&       covar_est,
                  common::Vector<float>&       prior_est,
                  common::Matrix<float>&       mean_est,
                  common::Matrix<float>&       posteriors,
                  common::Matrix<float>&       densities,
                  common::Vector<float>&       denominators,
                  const common::Matrix<float>& observations,
                  const common::Matrix<float>& mean_init,
                  const common::Vector<float>& covar_init,
                  const common::Vector<float>& prior_init);

} // namspace Scalar

namespace CUDA {

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
                               const common::Vector<float>& prior_init);

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
                                 const int numObs);

} // namespace CUDA
} // namspace EM

