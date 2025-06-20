#include <cmath>
#include "src/common/assert.hpp"
#include "src/em_alg/covar_est.hpp"

namespace EM { namespace Scalar {

void CovarEst(common::Vector<float>&       covar_est,
              common::Vector<float>&       prior_est,
              const common::Matrix<float>& mean_est,
              const common::Matrix<float>& posteriors,
              const common::Matrix<float>& observations)
{
  const int numClasses = covar_est.size();
  const int numObs     = observations.rows();
  const int dimension  = observations.cols();

  ASSERT(dimension         == 2);
  ASSERT(mean_est.rows()   == numClasses);
  ASSERT(mean_est.cols()   == dimension);
  ASSERT(posteriors.rows() == numClasses);
  ASSERT(posteriors.cols() == numObs);

  for (int j = 0; j < numClasses; ++j) {
    const auto m = mean_est.get_row(j);

    float num { 0 };
    float den { 0 };

    for (int k = 0; k < numObs; ++k) {
      const auto  x           = observations.get_row(k);
      const float normSquared = std::pow(x[0] - m[0], 2) + std::pow(x[1] - m[1], 2);
      num += posteriors(j, k) * normSquared;
      den += posteriors(j, k);
    }

    covar_est[j] = num / (dimension * den);
    prior_est[j] = den / numObs;
  }
}

} // namspace Scalar
} // namspace EM

