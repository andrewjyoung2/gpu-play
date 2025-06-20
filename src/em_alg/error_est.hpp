#pragma once

#include <cuda_runtime.h>
#include "src/common/matrix.hpp"
#include "src/common/vector.hpp"

namespace EM { namespace Scalar {

float ErrorEst(const common::Matrix<float>& mean_new,
               const common::Matrix<float>& mean_old,
               const common::Vector<float>& covar_new,
               const common::Vector<float>& covar_old,
               const common::Vector<float>& prior_new,
               const common::Vector<float>& prior_old);

} // namspace Scalar

namespace CUDA {

__host__ float ErrorEstHost(const common::Matrix<float>& mean_new,
                            const common::Matrix<float>& mean_old,
                            const common::Vector<float>& covar_new,
                            const common::Vector<float>& covar_old,
                            const common::Vector<float>& prior_new,
                            const common::Vector<float>& prior_old);

__host__ void ErrorEstDevice(float* error_est,
                             float* d_mean_new,
                             float* d_mean_old,
                             float* d_covar_new,
                             float* d_covar_old,
                             float* d_prior_new,
                             float* d_prior_old,
                             const int dimension,
                             const int numClasses);

} // namespace CUDA
} // namspace EM

