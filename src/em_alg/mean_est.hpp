#pragma once

#include <cuda_runtime.h>
#include "src/common/matrix.hpp"
#include "src/common/vector.hpp"

namespace EM { namespace Scalar {

void MeanEst(common::Matrix<float>&       means,
             const common::Matrix<float>& posteriors,
             const common::Matrix<float>& observations);

} // namspace Scalar

namespace CUDA {

__host__ void MeanEstHost(common::Matrix<float>&       means,
                          const common::Matrix<float>& posteriors,
                          const common::Matrix<float>& observations);

__host__ void MeanEstDevice(float*    d_means,
                            float*    d_posteriors,
                            float*    d_observations,
                            const int dimension,
                            const int numClasses,
                            const int numObs);

} // namespace CUDA
} // namspace EM

