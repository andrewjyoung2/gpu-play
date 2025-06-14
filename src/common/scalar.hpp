#pragma once

#include <cstddef>

namespace math { namespace scalar {

float Accumulate(float* A, const size_t len);

float AccumulateProto(float* A, const size_t len);

void VectorMultiply(float* C,
                    float* A,
                    float* B,
                    const size_t len);

} // namespace scalar
} // namespace math

