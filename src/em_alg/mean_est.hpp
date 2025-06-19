#pragma once

#include <cuda_runtime.h>
#include "src/common/matrix.hpp"
#include "src/common/vector.hpp"

namespace EM { namespace Scalar {

void MeanEst(common::Matrix<float>&       mean_est,
             const common::Matrix<float>& posterior,
             const common::Matrix<float>& observations);

} // namspace Scalar

namespace CUDA {

} // namespace CUDA
} // namspace EM
