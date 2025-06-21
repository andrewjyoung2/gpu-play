#include <cmath>
#include "src/common/assert.hpp"
#include "src/em_alg/covar_est.hpp"
#include "src/em_alg/em_alg.hpp"
#include "src/em_alg/error_est.hpp"
#include "src/em_alg/mean_est.hpp"
#include "src/em_alg/posterior.hpp"

namespace EM { namespace Scalar {

void EM_Iteration(float&                       error_est,
                  common::Vector<float>&       covar_est,
                  common::Vector<float>&       prior_est,
                  common::Matrix<float>&       mean_est,
                  common::Matrix<float>&       posteriors,
                  common::Matrix<float>&       densities,
                  common::Vector<float>&       denominators,
                  const common::Matrix<float>& observations,
                  const common::Matrix<float>& mean_init,
                  const common::Vector<float>& covar_init,
                  const common::Vector<float>& prior_init)
{
  // TODO: validate inputs
  Posterior(posteriors,
            densities,
            denominators,
            observations,
            mean_init,
            covar_init,
            prior_init);

  MeanEst(mean_est,
          posteriors,
          observations);

  CovarEst(covar_est,
           prior_est,
           mean_est,
           posteriors,
           observations);

  error_est = ErrorEst(mean_est,
                       mean_init,
                       covar_est,
                       covar_init,
                       prior_est,
                       prior_init);
}

} // namspace Scalar
} // namspace EM

