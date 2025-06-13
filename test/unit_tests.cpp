//#include <cuda_runtime.h>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include "src/welcome.hpp"
#include "src/common/math.hpp"
#include "src/common/random_float_vector.hpp"

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
  common::RandomFloatVector C(len);

  math::VectorMultiplyHost(C.data(), A.data(), B.data(), len);

  for (size_t idx = 0; idx < 5; ++idx) {
    std::cout << C[idx] << ", ";
  }
  std::cout << std::endl;

  EXPECT_EQ(0, 0);
}
