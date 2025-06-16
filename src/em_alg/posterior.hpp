#pragma once

#include <cuda_runtime.h>
#include "src/common/matrix.hpp"
#include "src/common/vector.hpp"

namespace EM { namespace Scalar {

void Posterior(common::Matrix<float>&       posteriors,
               common::Matrix<float>&       densities,
               common::Vector<float>&       denominators,
               const common::Matrix<float>& observations,
               const common::Matrix<float>& means,
               const common::Vector<float>& covariances,
               const common::Vector<float>& priors);

} // namspace Scalar

namespace CUDA {

__host__ void PosteriorHost(common::Matrix<float>&       posteriors,
                            common::Matrix<float>&       densities,
                            common::Vector<float>&       denominators,
                            const common::Matrix<float>& observations,
                            const common::Matrix<float>& means,
                            const common::Vector<float>& covariances,
                            const common::Vector<float>& priors);

__host__ void PosteriorDevice(float*    d_posteriors,
                              float*    d_densities,
                              float*    d_denominators,
                              float*    d_observations,
                              float*    d_means,
                              float*    d_covariances,
                              float*    d_priors,
                              const int dimension,
                              const int numClasses,
                              const int numObs);
} // namespace CUDA
} // namspace EM

