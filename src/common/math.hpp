#pragma once

namespace math {

__host__ void VectorMultiplyDevice(float* C,
                                   float* A,
                                   float* B,
                                   const size_t len);

__host__ void VectorMultiplyHost(float* C,
                                 float* A,
                                 float* B,
                                 const size_t len);

} // namespace math

