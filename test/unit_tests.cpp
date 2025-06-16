#include <cstring>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include "src/common/file_io.hpp"
#include "src/common/math.hpp"
#include "src/common/matrix.hpp"
#include "src/common/random_float_vector.hpp"
#include "src/common/scalar.hpp"
#include "src/common/vector.hpp"
#include "src/cublas/cublas_wrap.hpp"
#include "src/em_alg/posterior.hpp"
#include "src/welcome.hpp"

//------------------------------------------------------------------------------
const float eps { 1.0e-6f }; // tolerance for EXPECT_NEAR

//------------------------------------------------------------------------------
TEST(Example, welcome)
{
    const std::string msg { "Welcome to LeetGPU!" };

    const auto result = welcome::execute_kernel(msg);

    std::cout << result.data() << "\n";
    
    EXPECT_EQ(0, std::strcmp(msg.c_str(), result.data()));
}

TEST(Math, VectorMultiply)
{
  const size_t len { 16 };

  common::RandomFloatVector A(len);
  common::RandomFloatVector B(len);
  std::vector<float>        C(len);
  std::vector<float>        expected(len);

  math::VectorMultiplyHost(C.data(), A.data(), B.data(), len);
  math::scalar::VectorMultiply(expected.data(), A.data(), B.data(), len);

  EXPECT_THAT(C, testing::Pointwise(testing::FloatEq(), expected));
}

TEST(Scalar, Accumulate)
{
  std::vector<float> A { 1.0f, 2.0f, 3.0f, 4.0f };
  const auto result = math::scalar::Accumulate(A.data(), A.size());
  EXPECT_FLOAT_EQ(10.0f, result);
}

TEST(Scalar, AccumulateProto)
{
  for (size_t len = 1; len <= 128; ++len) {
    std::vector<float> A(len, 1.0f);
    const auto result = math::scalar::AccumulateProto(A.data(), A.size());
    EXPECT_FLOAT_EQ(len * 1.0f, result);
  }

}

TEST(Math, Accumulate)
{
  for (size_t len = 1; len <= 64; ++len) {
    std::vector<float> A(len, 1.0f);
    const auto result = math::AccumulateHost(A.data(), A.size());
    EXPECT_FLOAT_EQ(len * 1.0f, result);
  }
}

TEST(cuBLAS, Ddot)
{
  std::vector<double> A(10, 2.0);
  std::vector<double> B(10, 3.0);

  const auto res = cublas_wrap::Ddot(A, B);
  EXPECT_FLOAT_EQ(res, 60.0);
}

TEST(Scalar, FileIO)
{
  const std::string dirpath {
    "../test/data/test1"
  };
  const std::string filepath { "../test/data/test1/observations.txt" };

  EXPECT_TRUE(common::IsDirectory(dirpath));
  EXPECT_TRUE(common::IsFile(filepath));

  common::Matrix<float> A = common::ReadMatrix<float>(filepath);
  EXPECT_EQ(500, A.rows());
  EXPECT_EQ(2,   A.cols());

  EXPECT_FLOAT_EQ(A(0, 0),   8.20282895e-01f);
  EXPECT_FLOAT_EQ(A(0, 1),   5.07612718e+00f);
  EXPECT_FLOAT_EQ(A(499, 0), 3.28176162e+00f);
  EXPECT_FLOAT_EQ(A(499, 1), 2.99595594e+00f);

  common::Vector<float> v = A.get_row(3);

  EXPECT_EQ(2, v.size());
  EXPECT_FLOAT_EQ(v[0], 8.79061146e-01f);
  EXPECT_FLOAT_EQ(v[0], A(3, 0));
  EXPECT_FLOAT_EQ(v[1], 1.05496685e+00f);
  EXPECT_FLOAT_EQ(v[1], A(3, 1));
}

TEST(Scalar, Vector)
{
  // vector that owns its memory
  common::Vector<int> v(10);
  EXPECT_EQ(10, v.size());

  for (int idx = 0; idx < v.size(); ++idx) {
    v[idx] = idx;
  }

  auto raw = v.data();

  for (int idx = 0; idx < v.size(); ++idx) {
    EXPECT_EQ(raw[idx], idx);
  }

  // vector that does not own its memory
  common::Vector<int> w(v.data(), 5);

  for (int idx = 0; idx < v.size(); ++ idx) {
    EXPECT_EQ(v[idx], idx);
  }
}

TEST(Scalar, Matrix)
{
  common::Matrix<int> A(2, 3);
  EXPECT_EQ(2, A.rows());
  EXPECT_EQ(3, A.cols());

  for (int idx = 0; idx < A.size(); ++idx) {
    A[idx] = idx;
  }

  auto raw = A.data();

  for (int idx = 0; idx < A.size(); ++idx) {
    EXPECT_EQ(raw[idx], idx);
  }

  EXPECT_EQ(A[0], A(0, 0));
  EXPECT_EQ(A[1], A(0, 1));
  EXPECT_EQ(A[2], A(0, 2));
  EXPECT_EQ(A[3], A(1, 0));
  EXPECT_EQ(A[4], A(1, 1));
  EXPECT_EQ(A[5], A(1, 2));
}

TEST(Scalar, Posterior)
{
  const auto obs = common::ReadMatrix<float>("../test/data/test1/observations.txt" );
  const auto m   = common::ReadMatrix<float>("../test/data/test1/initial_mean.txt");
  const auto cov = common::ReadMatrix<float>("../test/data/test1/initial_covariance.txt");
  const auto pr  = common::ReadMatrix<float>("../test/data/test1/initial_priors.txt");

  const int numObs     = obs.rows();
  const int numClasses = m.rows();
  EXPECT_EQ(2, obs.cols());
  EXPECT_EQ(2, m.cols());

  EXPECT_EQ(1,          cov.rows());
  EXPECT_EQ(numClasses, cov.cols());

  EXPECT_EQ(1,          pr.rows());
  EXPECT_EQ(numClasses, pr.cols());

  common::Matrix<float> post(numClasses, numObs);
  common::Matrix<float> dens(numObs, numClasses);
  common::Vector<float> denom(numObs);

  EM::Scalar::Posterior(post,           // out: posteriors
                        dens,           // out: densities
                        denom,          // out: denominators
                        obs,            // in:  observations
                        m,              // in:  means
                        cov.get_row(0), // in:  covariances
                        pr.get_row(0)); // in:  priors

  // compare outputs to Octave
  const auto exp_dens = common::ReadMatrix<float>("../test/data/test1/densities.txt");

  for (int k = 0; k < numObs; ++k) {
    for (int j = 0; j < numClasses; ++j) {
      EXPECT_NEAR(dens(k, j), exp_dens(k, j), eps);
    }
  }

  const auto exp_denom = common::ReadMatrix<float>("../test/data/test1/denominators.txt");
  EXPECT_EQ(exp_denom.rows(), 1);
  EXPECT_EQ(exp_denom.cols(), numObs);

  for (int k = 0; k < numObs; ++k) {
    EXPECT_NEAR(denom[k], exp_denom(0, k), eps);
  }

  const auto exp_post = common::ReadMatrix<float>("../test/data/test1/posteriors.txt");
  EXPECT_EQ(exp_post.rows(), 3);
  EXPECT_EQ(exp_post.cols(), 500);

  for (int j = 0; j < numClasses; ++j) {
    for (int k = 0; k < numObs; ++k) {
      EXPECT_NEAR(post(j, k), exp_post(j, k), eps);
    }
  }
}

