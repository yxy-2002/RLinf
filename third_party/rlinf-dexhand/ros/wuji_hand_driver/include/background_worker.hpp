// Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
#pragma once
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

// One in-flight job, no backlog. Owners must stop before destroying job state.
class BackgroundWorker {
 public:
  ~BackgroundWorker() { stop(); }
  void start(std::chrono::milliseconds period, std::function<void()> job) {
    thread_ = std::thread([this, period, job]() {
      auto next = std::chrono::steady_clock::now() + period;
      std::unique_lock<std::mutex> lock(mutex_);
      while (!wake_.wait_until(lock, next, [this]() { return stopping_; })) {
        lock.unlock();
        job();  // The owner handles job errors and updates its fault state.
        lock.lock();
        next += period;
        const auto end = std::chrono::steady_clock::now();
        if (next <= end) next = end + period;
      }
    });
  }
  void stop() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    wake_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
 private:
  std::mutex mutex_;
  std::condition_variable wake_;
  bool stopping_ = false;
  std::thread thread_;
};
