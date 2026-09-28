// Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
// Catmull-Rom equations ported from
// psi_glove_ros2/wujihand_spline_forwarder.py.
#pragma once
#include <algorithm>
#include <array>
#include <deque>
#include <utility>

using Joints = std::array<double, 20>;
class Spline {
 public:
  void reset(const Joints& q) {
    points_.clear();
    seed_ = q;
  }
  void push(double t, const Joints& q) {
    points_.emplace_back(t, q);
    while (points_.size() > 256) points_.pop_front();
  }
  Joints sample(double t) const {
    if (points_.size() < 4) return seed_;
    if (t <= points_.front().first) return points_.front().second;
    if (t >= points_.back().first) return points_.back().second;
    size_t i = points_.size() - 2;
    while (i > 0 && points_[i].first > t) --i;
    const auto& p0 = points_[i > 0 ? i - 1 : 0].second;
    const auto& p1 = points_[i].second;
    const auto& p2 = points_[i + 1].second;
    const auto& p3 = points_[std::min(i + 2, points_.size() - 1)].second;
    const double dt = points_[i + 1].first - points_[i].first;
    const double u =
        dt > 0 ? std::clamp((t - points_[i].first) / dt, 0.0, 1.0) : 0;
    Joints q{};
    for (size_t j = 0; j < 20; ++j)
      q[j] = 0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * u +
                    (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * u * u +
                    (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * u * u * u);
    return q;
  }

 private:
  Joints seed_{};
  std::deque<std::pair<double, Joints>> points_;
};
