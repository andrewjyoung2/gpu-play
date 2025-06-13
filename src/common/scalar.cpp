#include "src/common/scalar.hpp"

namespace math { namespace scalar {

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

