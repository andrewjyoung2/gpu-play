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

} // namespace CUDA
} // namspace EM


