#pragma once

#include <iostream>
#include <stdexcept>

#define ASSERT(cond) \
  do {               \
    if (!(cond)) {   \
      std::cerr << "Assert " << #cond << " faied, " << __FILE__ << ":" << __LINE__ << std::endl; \
      throw std::runtime_error("Assert failed"); \
    } \
  } while (0)
