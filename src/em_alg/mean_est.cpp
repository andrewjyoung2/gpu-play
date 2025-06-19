#include "src/common/assert.hpp"
#include "src/em_alg/mean_est.hpp"

namespace EM { namespace Scalar {

void MeanEst(common::Matrix<float>&       mean_est,
             const common::Matrix<float>& posterior,
             const common::Matrix<float>& observations)
{
  const int numClasses      = mean_est.rows();
  const int dimension       = mean_est.cols();
  const int numObservations = posterior.cols();

  ASSERT(dimension == 2); // limit of the current implementation

  ASSERT(posterior.rows()    == numClasses);
  ASSERT(observations.rows() == numObservations);
  ASSERT(observations.cols() == dimension);
}

} // namspace Scalar
} // namespace EM

