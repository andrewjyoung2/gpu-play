#include <cmath>
#include "src/common/assert.hpp"
#include "src/em_alg/posterior.hpp"

namespace EM { namespace Scalar {

void Posterior(common::Matrix<float>&       posteriors,
               common::Matrix<float>&       densities,
               common::Vector<float>&       denominators,
               const common::Matrix<float>& observations,
               const common::Matrix<float>& means,
               const common::Vector<float>& covariances,
               const common::Vector<float>& priors)
{
  const int numObs     = observations.rows();
  const int dim        = observations.cols();
  const int numClasses = means.rows();

  ASSERT(dim == 2); // limit of the current implementation

  ASSERT(means.cols()        == dim);
  ASSERT(covariances.size()  == numClasses);
  ASSERT(priors.size()       == numClasses);
  ASSERT(posteriors.rows()   == numClasses);
  ASSERT(posteriors.cols()   == numObs);
  ASSERT(densities.rows()    == numObs);
  ASSERT(densities.cols()    == numClasses);
  ASSERT(denominators.size() == numObs);

  // Step 1: Calculate matrix of Gaussian densities
  // dens(k, j) = j-th Gaussian evaluated at k-th observation
  for (int j = 0; j < numClasses; ++j) {
    for (int k = 0; k < numObs; ++k) {
      const auto  x = observations.get_row(k);
      const auto  m = means.get_row(j);
      const auto  s = covariances[j];
      const float c = 1 / (2 * M_PI * s);

      const float normSquared = std::pow(x[0] - m[0], 2)
                              + std::pow(x[1] - m[1], 2);

      densities(k, j) = c * std::exp(-normSquared / (2 * s));
    }
  }
}

} // namspace Scalar
} // namspace EM

