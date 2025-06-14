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

