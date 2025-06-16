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

  ASSERT(means.cols()        == dim);
  ASSERT(covariances.size()  == numClasses);
  ASSERT(priors.size()       == numClasses);
  ASSERT(posteriors.rows()   == numClasses);
  ASSERT(posteriors.cols()   == numObs);
  ASSERT(densities.rows()    == numObs);
  ASSERT(densities.cols()    == numClasses);
  ASSERT(denominators.size() == numObs);

}

} // namspace Scalar
} // namspace EM

