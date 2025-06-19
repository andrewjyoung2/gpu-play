#include "src/common/assert.hpp"
#include "src/em_alg/mean_est.hpp"

namespace EM { namespace Scalar {

void MeanEst(common::Matrix<float>&       means,
             const common::Matrix<float>& posteriors,
             const common::Matrix<float>& observations)
{
  const int numClasses = means.rows();
  const int dimension  = means.cols();
  const int numObs     = posteriors.cols();

  ASSERT(dimension == 2); // limit of the current implementation

  ASSERT(posteriors.rows()   == numClasses);
  ASSERT(observations.rows() == numObs);
  ASSERT(observations.cols() == dimension);

  for (int j = 0; j < numClasses; ++j) {
    float denom { 0 };
    for (int k = 0; k < numObs; ++k) {
      denom+= posteriors(j, k);
    }

    for (int n = 0; n < dimension; ++n) {
      float tmp { 0 };
      for (int k = 0; k < numObs; ++k) {
        tmp += posteriors(j, k) * observations(k, n);
      }
      means(j, n) = tmp / denom;
    }
  }
}

} // namspace Scalar
} // namespace EM

