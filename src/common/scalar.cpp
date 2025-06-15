#include <iostream>
#include <vector>
#include "src/common/scalar.hpp"

namespace math { namespace scalar {

//------------------------------------------------------------------------------
float Accumulate(float* A, const size_t len)
{
  float result { 0 };

  for (size_t idx = 0; idx < len; ++idx) {
    result += A[idx];
  }

  return result;
}

//------------------------------------------------------------------------------
float AccumulateProto(float* A, const size_t len)
{
  const size_t blockDim = (len + 1) >> 1;

  // TODO: how much scratch is actually necessary?
  std::vector<float> scratch(2 * len); // __shared__

  size_t numThreads = blockDim;
  float* readPtr    = A;
  float* readEnd    = A + len;
  float* writePtr   = scratch.data();

  while (numThreads) {
    for (size_t idx = 0; idx < numThreads; ++idx) {
      writePtr[idx] = readPtr[2 * idx];
      if (readPtr + 2 * idx + 1 < readEnd) {
        writePtr[idx] += readPtr[2 * idx + 1];
      }
    }

    readPtr    =   writePtr;
    readEnd    =   readPtr + numThreads;
    writePtr   +=  numThreads;

    if (1 == numThreads) {
      break;
    }
    numThreads = (numThreads + 1) >> 1;
  }

  return readPtr[0];
}

//------------------------------------------------------------------------------
void VectorMultiply(float* C,
                    float* A,
                    float* B,
                    const size_t len)
{
  for (size_t idx = 0; idx < len; ++idx) {
    C[idx] = A[idx] * B[idx];
  }
}

} // namespace scalar
} // namespace math

