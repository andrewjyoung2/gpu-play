#include <iostream>
#include "src/welcome.hpp"

namespace welcome {

__global__ void welcome(char* msg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    msg[idx] = d_message[idx];
}

__host__ std::vector<char> execute_kernel(const std::string& msg)
{
  const size_t length = msg.size() + 1;

  // Copy message to constant memory
  cudaMemcpyToSymbol(d_message, msg.c_str(), length * sizeof(char));
  
  // Allocate device memory
  char* d_msg { nullptr };
  cudaMalloc(&d_msg, length * sizeof(char)); // TODO: error handling

  // Launch welcome kernel
  welcome<<<1, length>>>(d_msg);
    
  // Copy result back to host
  std::vector<char> result(length);
  cudaMemcpy(result.data(),
             d_msg,
             length * sizeof(char),
             cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_msg);

  return result;
}

} // namespace welcome

