//#include <cuda_runtime.h>
#include <cstring>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include "src/welcome.hpp"
#include "src/common/math.hpp"
#include "src/common/random_float_vector.hpp"
#include "src/common/scalar.hpp"
#include "src/cublas/cublas_wrap.hpp"

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
