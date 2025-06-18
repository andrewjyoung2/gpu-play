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

template <typename T>
Matrix<T> ReadMatrix(const std::string& path)
{
  ASSERT(IsFile(path));

  std::ifstream ifs(path);
  ASSERT(ifs.is_open());

  int rows { 0 };
  int cols { 0 };

  std::vector<T> numbers;

  std::string line;
  while (std::getline(ifs, line)) {
    ++rows;

    std::stringstream ss(line);
    T val;
    cols = 0;

    while (ss >> val) {
      ++cols;
      numbers.push_back(val);
    }
  }

  ASSERT(static_cast<size_t>(rows * cols) == numbers.size());

  common::Matrix<T> Mat(rows, cols);

  std::memcpy(Mat.data(), numbers.data(), numbers.size() * sizeof(T));
  return Mat;
}

template<typename T>
void WriteMatrix(const std::string& filename, const Matrix<T>& Mat)
{
  // file should not exist yet
  ASSERT(!IsFile(filename));
  ASSERT(!IsDirectory(filename));

  std::ofstream ofs(filename);
  ASSERT(ofs.is_open());

  for (int i = 0; i < Mat.rows(); ++i) {
    for (int j = 0; j < Mat.cols(); ++j) {
      ofs << Mat(i, j) << ' ';
    }
    ofs << std::endl;
  }
}
} // namespace common

