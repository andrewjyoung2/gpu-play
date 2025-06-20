#pragma once

#include <cuda_runtime.h>
#include "src/common/matrix.hpp"
#include "src/common/vector.hpp"

namespace EM { namespace Scalar {

void CovarEst(common::Vector<float>&       covar_est,
              common::Vector<float>&       prior_est,
              const common::Matrix<float>& mean_est,
              const common::Matrix<float>& posteriors,
              const common::Matrix<float>& observations);

} // namspace Scalar

namespace CUDA {

__host__ void CovarEstHost(common::Vector<float>&       covar_est,
                           const common::Matrix<float>& mean_est,
                           const common::Matrix<float>& posteriors,
                           const common::Matrix<float>& observations);

__host__ void CovarEstDevice(float*    d_covar_est,
                             float*    d_mean_est,
                             float*    d_posteriors,
                             float*    d_observations,
                             const int dimension,
                             const int numClasses,
                             const int numObs);

} // namespace CUDA
} // namspace EM

