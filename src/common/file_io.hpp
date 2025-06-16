#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include "src/common/assert.hpp"
#include "src/common/matrix.hpp"

namespace common {

bool IsDirectory(const std::string& path);

bool IsFile(const std::string& path);

Matrix<float> ReadMatrix(const std::string& path);

} // namespace common

