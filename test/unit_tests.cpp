//#include <cuda_runtime.h>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include "src/welcome.hpp"

TEST(Example, welcome)
{
    const std::string msg { "Welcome to LeetGPU!" };

    const auto result = welcome::execute_kernel(msg);

    std::cout << result.data() << "\n";
    
    EXPECT_EQ(0, std::strcmp(msg.c_str(), result.data()));
}
