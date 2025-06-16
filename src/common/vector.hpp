#pragma once

#include "src/common/assert.hpp"

namespace common {

template <typename T>
class Vector {
public:
  Vector(const int size)
    : m_size(size), m_ownsData(true)
  {
    ASSERT(m_size > 0);

    m_data = new T[m_size];

    ASSERT(nullptr != m_data);
  }

  Vector(T* data, const int size)
    : m_data(data), m_size(size), m_ownsData(false)
  {
    ASSERT(nullptr != m_data);
    ASSERT(size > 0);
  }

  virtual ~Vector()
  {
    if (m_ownsData) {
      delete[] m_data;
    }
  }

  int size() const
  {
    return m_size;
  }

  T* data() const
  {
    return m_data;
  }

  inline T& operator[](int n)
  {
    ASSERT(n >= 0);
    ASSERT(n < m_size);
    return m_data[n];
  }

  inline const T& operator[](int n) const
  {
    ASSERT(n >= 0);
    ASSERT(n < m_size);
    return m_data[n];
  }

private:
  T*  m_data      { nullptr };
  int m_size      { -1 };
  bool m_ownsData { false };
};

} // namespace common

