#pragma once

#include "src/common/assert.hpp"
#include "src/common/vector.hpp"

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

    ASSERT(nullptr != m_data);
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

  inline T& operator[](const int n)
  {
    ASSERT(n >= 0);
    ASSERT(n < size());
    return m_data[n];
  }

  inline const T& operator[](const int n) const
  {
    ASSERT(n >= 0);
    ASSERT(n < size());
    return m_data[n];
  }

  inline T& operator()(const int x, const int y)
  {
    ASSERT(x < m_rows);
    ASSERT(y < m_cols);
    return m_data[y + m_cols * x];
  }

  inline const T& operator()(const int x, const int y) const
  {
    ASSERT(x < m_rows);
    ASSERT(y < m_cols);
    return m_data[y + m_cols * x];
  }

  Vector<T> get_row(const int r) const
  {
    return Vector<T>(m_data + r * m_cols, m_cols);
  }

private:
  int m_rows { -1 };
  int m_cols { -1 };
  T*  m_data { nullptr };
};

} // namespace common

