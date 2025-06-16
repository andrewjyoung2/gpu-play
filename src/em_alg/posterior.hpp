#pragma once

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
} // namspace EM

