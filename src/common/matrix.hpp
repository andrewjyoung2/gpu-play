#pragma once

#include "src/common/assert.hpp"

namespace common {

template <typename T>
class Matrix {
public:
  Matrix(const int rows, const int cols)
    : m_rows(rows), m_cols(cols)
  {
    ASSERT(m_rows > 0);
    ASSERT(m_cols > 0);

    m_data = new T[m_rows * m_cols];
  }

  virtual ~Matrix()
  {
    delete [] m_data;
  }

  int rows() const
  {
    return m_rows;
  }

  int cols() const
  {
    return m_cols;
  }

  int size() const
  {
    return m_rows * m_cols;
  }

  T* data() const
  {
    return m_data;
  }

  inline T& operator[](int n)
  {
    ASSERT(n >= 0);
    ASSERT(n < size());
    return m_data[n];
  }

  inline T& operator()(int x, int y)
  {
    ASSERT(x < m_rows);
    ASSERT(y < m_cols);
    return m_data[y + m_cols * x];
  }

private:
  int m_rows { -1 };
  int m_cols { -1 };
  T*  m_data { nullptr };
};

} // namespace common

