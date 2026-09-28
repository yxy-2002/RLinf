// Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
#include "spline.hpp"

#include <gtest/gtest.h>
TEST(Spline, SeedAndLinearReference) {
  Spline s;
  Joints seed{};
  seed.fill(0.25);
  s.reset(seed);
  EXPECT_DOUBLE_EQ(s.sample(1)[0], 0.25);
  for (int i = 0; i < 4; ++i) {
    Joints q{};
    q.fill(i);
    s.push(i, q);
  }
  EXPECT_DOUBLE_EQ(s.sample(1.5)[0], 1.5);
  EXPECT_DOUBLE_EQ(s.sample(5)[0], 3);
  s.reset(seed);
  EXPECT_DOUBLE_EQ(s.sample(5)[0], 0.25);
}
int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
