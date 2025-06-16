#include <fstream>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#include "src/common/assert.hpp"
#include "src/common/matrix.hpp"
#include "src/common/file_io.hpp"

namespace common {

bool IsDirectory(const std::string& path)
{
  struct stat stats;
  stat(path.c_str(), &stats);
  return S_ISDIR(stats.st_mode) ? true : false;
}

bool IsFile(const std::string& path)
{
  struct stat st;
  stat(path.c_str(), &st);
  return static_cast<bool>(S_ISREG(st.st_mode));
}

Matrix<float> ReadMatrix(const std::string& path)
{
  ASSERT(IsFile(path));

  std::ifstream ifs(path);
  ASSERT(ifs.is_open());

  int rows { 0 };
  int cols { 0 };

  std::vector<float> numbers;

  std::string line;
  while (std::getline(ifs, line)) {
    ++rows;

    std::stringstream ss(line);
    float val;
    cols = 0;

    while (ss >> val) {
      ++cols;
      numbers.push_back(val);
    }
  }

  ASSERT(numbers.size() <= std::numeric_limits<int>::max());
  ASSERT(static_cast<size_t>(rows * cols) == numbers.size());

  common::Matrix<float> Mat(rows, cols);

  std::memcpy(Mat.data(), numbers.data(), numbers.size() * sizeof(float));
  return Mat;
}

} // namespace common

