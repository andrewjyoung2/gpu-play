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
  std::vector<float> scratch(len);

  size_t numThreads = (len + 1) >> 1;
  float* readPtr    = A;
  float* writePtr   = scratch.data();

  while (numThreads) {
    for (size_t idx = 0; idx < numThreads; ++idx) {
      // TODO: if len is not a power of 2, how do we know the data is valid?
      writePtr[idx] = readPtr[2 * idx] + readPtr[2 * idx + 1];
    }
    readPtr    =   writePtr;
    writePtr   +=  numThreads;
    numThreads >>= 1;
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

