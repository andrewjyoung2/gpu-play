#include <gtest/gtest.h>

// https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#cite_note-10
// Ralston method
TEST(RungeKutta, example)
{
  double t[5];
  double y[5];
  
  // initialize t
  const double delta = 0.025;
  t[0]               = 1.0;

  for (size_t idx = 1; idx < 5; ++idx) {
    t[idx] = t[idx - 1] + delta;
  }

  EXPECT_DOUBLE_EQ(t[0], 1.0);
  EXPECT_DOUBLE_EQ(t[1], 1.025);
  EXPECT_DOUBLE_EQ(t[2], 1.05);
  EXPECT_DOUBLE_EQ(t[3], 1.075);
  EXPECT_DOUBLE_EQ(t[4], 1.1);

  y[0] = 1;

}
