#include <cmath>
#include "src/common/assert.hpp"
#include "src/em_alg/error_est.hpp"

namespace EM { namespace Scalar {

float ErrorEst(const common::Matrix<float>& mean_new,
               const common::Matrix<float>& mean_old,
               const common::Vector<float>& covar_new,
               const common::Vector<float>& covar_old,
               const common::Vector<float>& prior_new,
               const common::Vector<float>& prior_old)
{
  const int numClasses = mean_new.rows();
  const int dimension  = mean_new.cols();

  ASSERT(mean_old.rows()  == numClasses);
  ASSERT(mean_old.cols()  == dimension);
  ASSERT(covar_new.size() == numClasses);
  ASSERT(covar_old.size() == numClasses);
  ASSERT(prior_new.size() == numClasses);
  ASSERT(prior_old.size() == numClasses);

  float err { 0 };

  for (int i = 0; i < numClasses; ++i) {
    for (int j = 0; j < dimension; ++j) {
      err += std::abs(mean_new(i, j) - mean_old(i, j));
    }
  }

  for (int i = 0; i < numClasses; ++i) {
    err += std::abs(covar_new[i] - covar_old[i]);
    err += std::abs(prior_new[i] - prior_old[i]);
  }

  return err;
}

} // namspace Scalar
} // namspace EM

