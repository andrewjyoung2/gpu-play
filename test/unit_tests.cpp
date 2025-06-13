//#include <cuda_runtime.h>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include "src/welcome.hpp"
#include "src/common/math.hpp"

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

  std::vector<float> A(len);
  std::vector<float> B(len);
  std::vector<float> C(len);

  math::VectorMultiplyHost(C.data(), A.data(), B.data(), len);

  EXPECT_EQ(0, 0);
}
