#include <cstring>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <iostream>
#include "src/common/file_io.hpp"
#include "src/common/math.hpp"
#include "src/common/matrix.hpp"
#include "src/common/random_float_vector.hpp"
#include "src/common/scalar.hpp"
#include "src/cublas/cublas_wrap.hpp"
#include "src/welcome.hpp"

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

  common::Matrix<float> A = common::ReadMatrix(filepath);
  EXPECT_EQ(500, A.rows());
  EXPECT_EQ(2,   A.cols());

  EXPECT_FLOAT_EQ(A(0, 0),   8.20282895e-01f);
  EXPECT_FLOAT_EQ(A(0, 1),   5.07612718e+00f);
  EXPECT_FLOAT_EQ(A(499, 0), 3.28176162e+00f);
  EXPECT_FLOAT_EQ(A(499, 1), 2.99595594e+00f);
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

