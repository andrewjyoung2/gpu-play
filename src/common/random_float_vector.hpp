#pragma once

#include <algorithm>
#include <random>
#include <vector>

namespace common {

class RandomFloatVector : public std::vector<float> {
public:
  RandomFloatVector(const size_t len)
    :
    std::vector<float>(len)
  {
    // TODO: Make the range configurable
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    std::generate_n(begin(), len, [&](){ return dis(gen); });
  }
};

} // namespace common

