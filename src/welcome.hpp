#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace welcome {

__constant__ char d_message[20];

__host__ std::vector<char> execute_kernel(const std::string& msg);

} // namespace welcome

