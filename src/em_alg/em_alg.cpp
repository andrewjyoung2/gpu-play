#include <cmath>
#include <cstring>
#include <limits>
#include <iostream>
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
  // TODO: validate input dimensions

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

void EM_Workflow(float&                       error_est,
                 common::Vector<float>&       covar_est,
                 common::Vector<float>&       prior_est,
                 common::Matrix<float>&       mean_est,
                 const float                  error_tol,
                 const common::Matrix<float>& observations,
                 const common::Matrix<float>& mean_init,
                 const common::Vector<float>& covar_init,
                 const common::Vector<float>& prior_init)
{
  ASSERT(covar_est.size() == covar_init.size());
  ASSERT(prior_est.size() == prior_init.size());
  ASSERT(mean_est.rows()  == mean_init.rows());
  ASSERT(mean_est.cols()  == mean_init.cols());
  // TODO: validate the rest

  const size_t max_iter { 100 };
  // TODO: should the number of iterations be a return value?
  size_t          iter  { 0 };

  const int numObs     = observations.rows();
  const int numClasses = mean_init.rows();

  // scratch memory
  common::Matrix<float> posteriors(numClasses, numObs);
  common::Matrix<float> densities(numObs, numClasses);
  common::Vector<float> denominators(numObs);

  error_est = std::numeric_limits<float>::max();

  while ((error_est >= error_tol) && (iter < max_iter)) {
    if (0 != iter) {
      // feed the previous result into the next iteration
      std::memcpy(mean_init.data(),
                  mean_est.data(),
                  mean_init.size() * sizeof(float));
      std::memcpy(covar_init.data(),
                  covar_est.data(),
                  covar_init.size() * sizeof(float));
      std::memcpy(prior_init.data(),
                  prior_est.data(),
                  prior_init.size() * sizeof(float));
    }

    EM::Scalar::EM_Iteration(error_est,
                             covar_est,
                             prior_est,
                             mean_est,
                             posteriors,
                             densities,
                             denominators,
                             observations,
                             mean_init,
                             covar_init,
                             prior_init);
    ++iter;
  }
}

} // namspace Scalar
} // namspace EM

