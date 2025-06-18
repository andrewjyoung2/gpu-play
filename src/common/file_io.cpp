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
  return static_cast<bool>(S_ISDIR(stats.st_mode));
}

bool IsFile(const std::string& path)
{
  struct stat st;
  stat(path.c_str(), &st);
  return static_cast<bool>(S_ISREG(st.st_mode));
}

} // namespace common

