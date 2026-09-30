// Copyright 2025 Wuji Robotics; Copyright 2026 The RLinf Authors.
// SPDX-License-Identifier: Apache-2.0
// SDK calls adapted from wujihandros2/wujihand_driver (Wuji Robotics).
#include <diagnostic_msgs/DiagnosticArray.h>
#include <diagnostic_msgs/KeyValue.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <std_srvs/SetBool.h>
#include <std_srvs/Trigger.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "background_worker.hpp"
#include "spline.hpp"
#ifdef WUJI_WITH_SDK
#include <wujihandcpp/device/hand.hpp>
#include <wujihandcpp/filter/low_pass.hpp>
#endif


static double now() { return ros::SteadyTime::now().toSec(); }
class Driver {
 public:
  explicit Driver(ros::NodeHandle nh) : nh_(nh) {
    nh_.param("fake_hardware", fake_, false);
    nh_.param("side", side_, std::string("left"));
    nh_.param("serial_number", serial_, std::string());
    nh_.param("timeout", timeout_, 0.5);
    nh_.param("lag_sec", lag_, 0.07);
    double rate, cutoff, state_rate;
    nh_.param("output_rate_hz", rate, 1000.0);
    nh_.param("filter_cutoff_hz", cutoff, 10.0);
    nh_.param("state_rate_hz", state_rate, 100.0);
    if (side_ != "left" && side_ != "right")
      throw std::runtime_error("Invalid side");
    for (double value : {rate, cutoff, state_rate, timeout_})
      if (!std::isfinite(value) || value <= 0)
        throw std::runtime_error("Invalid timing parameter");
    if (!std::isfinite(lag_) || lag_ < 0)
      throw std::runtime_error("Invalid lag");
    if (!nh_.getParam("lower", lower_) || !nh_.getParam("upper", upper_) ||
        lower_.size() != 20 || upper_.size() != 20)
      throw std::runtime_error("20 model limits are required");
    for (int i = 0; i < 20; ++i) {
      if (!std::isfinite(lower_[i]) || !std::isfinite(upper_[i]) ||
          lower_[i] >= upper_[i])
        throw std::runtime_error("Invalid limits");
      names_.push_back(side_ + "_finger" + std::to_string(i / 4 + 1) +
                       "_joint" + std::to_string(i % 4 + 1));
      actual_[i] = std::clamp(0.0, lower_[i], upper_[i]);
    }
    if (!fake_) {
#ifdef WUJI_WITH_SDK
      if (serial_.empty())
        throw std::runtime_error("Explicit USB serial_number required");
      hand_ = std::make_unique<wujihandcpp::device::Hand>(serial_.c_str());
      hand_->disable_thread_safe_check();
      auto side = hand_->read<wujihandcpp::data::hand::Handedness>();
      if ((side == 1 ? "left" : "right") != side_)
        throw std::runtime_error("Hardware side mismatch");
      controller_ = hand_->realtime_controller<true>(
          wujihandcpp::filter::LowPass(cutoff));
      const auto feedback = read_hardware();
      actual_ = feedback.actual;
      error_codes_ = feedback.errors;
      motor_error_ = feedback.motor_error;
#else
      throw std::runtime_error(
          "Built without SDK: only fake_hardware:=true is available");
#endif
    }
    target_ = actual_;
    spline_.reset(target_);
    last_health_ = now();
    state_pub_ = nh_.advertise<sensor_msgs::JointState>("joint_states", 1);
    target_pub_ = nh_.advertise<sensor_msgs::JointState>("joint_targets", 1);
    diag_pub_ =
        nh_.advertise<diagnostic_msgs::DiagnosticArray>("diagnostics", 1, true);
    command_sub_ = nh_.subscribe("joint_commands", 1, &Driver::command, this);
    teleop_sub_ = nh_.subscribe("teleop_commands", 1, &Driver::teleop_command, this);
    teleop_srv_ = nh_.advertiseService("set_teleop", &Driver::set_teleop, this);
    enabled_srv_ = nh_.advertiseService("set_enabled", &Driver::enable, this);
    hold_srv_ = nh_.advertiseService("hold", &Driver::hold, this);
    resume_srv_ = nh_.advertiseService("resume", &Driver::resume, this);
    clear_srv_ = nh_.advertiseService("clear_trajectory", &Driver::clear, this);
    error_srv_ =
        nh_.advertiseService("reset_error", &Driver::reset_error, this);
    control_timer_ =
        nh_.createWallTimer(ros::WallDuration(1 / rate), &Driver::tick, this);
    state_timer_ = nh_.createWallTimer(ros::WallDuration(1 / state_rate),
                                       &Driver::publish, this);
    ROS_INFO("Wuji driver ready; side=%s fake=%d (motors disabled)",
             side_.c_str(), fake_);
    // Start last: all callback state is initialized before the worker runs.
    health_worker_.start(std::chrono::milliseconds(100), [this]() { health(); });
  }
  ~Driver() {
    control_timer_.stop();
    state_timer_.stop();
    health_worker_.stop();  // Join before disabling motors or destroying SDK state.
#ifdef WUJI_WITH_SDK
    if (hand_) {
      try {
        hand_->write<wujihandcpp::data::joint::Enabled>(false);
      } catch (...) {
      }
    }
    controller_.reset();
    hand_.reset();
#endif
  }

 private:
  struct HardwareState {
    Joints actual{};
    std::array<uint32_t, 20> errors{};
    bool motor_error = false;
  };
  HardwareState read_hardware() {
    HardwareState feedback;
    // Serialize explicit SDK requests with enable/reset services, never tick.
    std::lock_guard<std::mutex> hardware_lock(hardware_mutex_);
#ifdef WUJI_WITH_SDK
    if (hand_) {
      // A bounded explicit read validates USB/all-joint feedback. Cached TPDO
      // getters alone cannot establish feedback freshness in SDK 1.5.1.
      wujihandcpp::device::Latch completion;
      hand_->read_async<wujihandcpp::data::joint::ActualPosition>(
          completion, std::chrono::milliseconds(50));
      hand_->read_async<wujihandcpp::data::joint::ErrorCode>(
          completion, std::chrono::milliseconds(50));
      completion.wait();
      for (int i = 0; i < 20; ++i) {
        double value = hand_->finger(i / 4)
                           .joint(i % 4)
                           .get<wujihandcpp::data::joint::ActualPosition>();
        if (!std::isfinite(value))
          throw std::runtime_error("Nonfinite hardware position");
        feedback.actual[i] = value;
        feedback.errors[i] = hand_->finger(i / 4)
                              .joint(i % 4)
                              .get<wujihandcpp::data::joint::ErrorCode>();
        feedback.motor_error = feedback.motor_error || feedback.errors[i] != 0;
      }
    }
#endif
    return feedback;
  }
  void latch(const std::string& reason) {
    held_ = true;
    armed_ = false;
    reason_ = reason;
    spline_.reset(target_);
  }
  bool service_error(std_srvs::Trigger::Response& res,
                     const std::string& message) {
    res.success = false;
    res.message = message;
    return true;
  }
  bool set_teleop(std_srvs::SetBool::Request& req,
                  std_srvs::SetBool::Response& res) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (req.data && (fatal_ || held_ || !enabled_ ||
                     now() - last_health_ > timeout_)) {
      res.success = false;
      res.message = "Fresh enabled feedback and explicit fault recovery required";
      return true;
    }
    teleop_mode_ = req.data;
    target_ = actual_;
    spline_.reset(target_);
    armed_ = false;
    res.success = true;
    return true;
  }
  void command(const sensor_msgs::JointState::ConstPtr& msg) {
    accept_command(msg, false);
  }
  void teleop_command(const sensor_msgs::JointState::ConstPtr& msg) {
    accept_command(msg, true);
  }
  void accept_command(const sensor_msgs::JointState::ConstPtr& msg, bool teleop) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (teleop != teleop_mode_ || !enabled_ || held_ || fatal_) return;
    if (msg->name != names_ || msg->position.size() != 20) {
      latch("Invalid joint order/dimension");
      return;
    }
    const double age = (ros::Time::now() - msg->header.stamp).toSec();
    if (!std::isfinite(age) || age < -0.01 || age > timeout_) {
      latch("Stale command");
      return;
    }
    Joints q;
    for (int i = 0; i < 20; ++i) {
      q[i] = msg->position[i];
      if (!std::isfinite(q[i]) || q[i] < lower_[i] - 1e-8 ||
          q[i] > upper_[i] + 1e-8) {
        latch("Invalid target");
        return;
      }
    }
    last_command_ = now();
    armed_ = true;
    spline_.push(last_command_, q);
  }
  bool enable(std_srvs::SetBool::Request& req,
              std_srvs::SetBool::Response& res) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
      if (req.data &&
          (fatal_ || motor_error_ || now() - last_health_ > timeout_))
        throw std::runtime_error("Feedback unavailable");
#ifdef WUJI_WITH_SDK
      if (hand_) {
        std::lock_guard<std::mutex> hardware_lock(hardware_mutex_);
        if (req.data) send(actual_);
        hand_->write<wujihandcpp::data::joint::Enabled>(req.data);
      }
#endif
      enabled_ = req.data;
      target_ = actual_;
      spline_.reset(target_);
      armed_ = false;
      res.success = true;
    } catch (const std::exception& e) {
      res.success = false;
      res.message = e.what();
    }
    return true;
  }
  bool hold(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    std::lock_guard<std::mutex> lock(mutex_);
    latch("Input interrupted");
    res.success = true;
    return true;
  }
  bool resume(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fatal_ || motor_error_ || !enabled_ || now() - last_health_ > timeout_)
      return service_error(
          res, "Enabled motors and fresh hardware feedback required");
    target_ = actual_;
    spline_.reset(target_);
    held_ = false;
    armed_ = false;
    reason_.clear();
    res.success = true;
    return true;
  }
  bool clear(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (held_ || fatal_) return service_error(res, "Explicit resume required");
    target_ = actual_;
    spline_.reset(target_);
    armed_ = false;
    res.success = true;
    return true;
  }
  bool reset_error(std_srvs::Trigger::Request&,
                   std_srvs::Trigger::Response& res) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
#ifdef WUJI_WITH_SDK
      if (hand_) {
        std::lock_guard<std::mutex> hardware_lock(hardware_mutex_);
        hand_->write<wujihandcpp::data::joint::ResetError>(1);
      }
#endif
      res.success = true;  // Does not resume or clear a USB fault.
    } catch (const std::exception& e) {
      res.success = false;
      res.message = e.what();
    }
    return true;
  }
  void send(const Joints& q) {
#ifdef WUJI_WITH_SDK
    if (controller_) {
      double p[5][4];
      for (int i = 0; i < 20; ++i) p[i / 4][i % 4] = q[i];
      controller_->set_joint_target_position(p);
    }
#endif
  }
  void tick(const ros::WallTimerEvent&) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!enabled_ || fatal_) return;
    if (now() - last_health_ > timeout_) {
      fatal_ = true;
      latch("Hardware feedback timeout");
      return;
    }
    if (armed_ && now() - last_command_ > timeout_) latch("Command timeout");
    const double play_time = now() - lag_;
    if (!held_) {
      target_ = spline_.sample(play_time);
    }
    for (int i = 0; i < 20; ++i)
      target_[i] = std::clamp(target_[i], lower_[i], upper_[i]);
    try {
      send(target_);
      if (fake_) actual_ = target_;
    } catch (const std::exception& e) {
      fatal_ = true;
      latch(e.what());
    }
  }
  void health() {
    HardwareState feedback;
    std::string error;
    bool succeeded = false;
    try {
      // No control-state lock while requesting or waiting for hardware.
      feedback = read_hardware();
      succeeded = true;
    } catch (const std::exception& e) {
      error = e.what();
    } catch (...) {
      error = "Unknown hardware read failure";
    }
    const double completed = now();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!succeeded) {
        fatal_ = true;
        latch(error);
      } else {
        if (!fake_) {
          actual_ = feedback.actual;
          error_codes_ = feedback.errors;
          motor_error_ = feedback.motor_error;
        }
        last_health_ = completed;
        if (motor_error_)
          latch("Motor error; reset_error and explicit resume required");
      }
    }
  }
  void publish(const ros::WallTimerEvent&) {
    std::lock_guard<std::mutex> lock(mutex_);
    sensor_msgs::JointState msg;
    msg.name = names_;
    msg.header.stamp = ros::Time::now();
    // These are explicit reads at 10 Hz, republished at 100 Hz. Publish the
    // original read age in diagnostics; do not imply a new physical sample.
    if (!fatal_ && now() - last_health_ <= timeout_) {
      msg.header.stamp -= ros::Duration(now() - last_health_);
      msg.position.assign(actual_.begin(), actual_.end());
      state_pub_.publish(msg);
    }
    msg.header.stamp = ros::Time::now();
    msg.position.assign(target_.begin(), target_.end());
    target_pub_.publish(msg);
    diagnostic_msgs::DiagnosticArray array;
    array.header.stamp = ros::Time::now();
    diagnostic_msgs::DiagnosticStatus status;
    status.name = nh_.getNamespace();
    status.hardware_id = serial_;
    status.level = fatal_ ? 2 : held_ ? 1 : 0;
    status.message = reason_.empty() ? "ready" : reason_;
    for (const auto& pair : std::vector<std::pair<std::string, std::string>>{
             {"enabled", enabled_ ? "true" : "false"},
             {"held", held_ ? "true" : "false"},
             {"teleop_mode", teleop_mode_ ? "true" : "false"},
             {"fatal", fatal_ ? "true" : "false"},
             {"side", side_},
             {"feedback_age_s", std::to_string(now() - last_health_)},
             {"fake_hardware", fake_ ? "true" : "false"}}) {
      diagnostic_msgs::KeyValue kv;
      kv.key = pair.first;
      kv.value = pair.second;
      status.values.push_back(kv);
    }
    for (size_t i = 0; i < 20; ++i) {
      diagnostic_msgs::KeyValue value;
      value.key = names_[i] + "/error_code";
      value.value = std::to_string(error_codes_[i]);
      status.values.push_back(value);
    }
    array.status.push_back(status);
    diag_pub_.publish(array);
  }
  ros::NodeHandle nh_;
  ros::Publisher state_pub_, target_pub_, diag_pub_;
  ros::Subscriber command_sub_, teleop_sub_;
  ros::ServiceServer enabled_srv_, hold_srv_, resume_srv_, clear_srv_,
      error_srv_, teleop_srv_;
  ros::WallTimer control_timer_, state_timer_;
  BackgroundWorker health_worker_;
  std::mutex hardware_mutex_;
  std::mutex mutex_;
  bool fake_ = false, enabled_ = false, held_ = false, fatal_ = false,
       armed_ = false;
  bool motor_error_ = false;
  bool teleop_mode_ = false;
  std::array<uint32_t, 20> error_codes_{};
  double timeout_, lag_, last_health_ = 0, last_command_ = 0;
  std::string side_, serial_, reason_;
  std::vector<std::string> names_;
  std::vector<double> lower_, upper_;
  Joints actual_{}, target_{};
  Spline spline_;
#ifdef WUJI_WITH_SDK
  std::unique_ptr<wujihandcpp::device::Hand> hand_;
  std::unique_ptr<wujihandcpp::device::IController> controller_;
#endif
};
int main(int argc, char** argv) {
  ros::init(argc, argv, "wuji_hand_driver");
  try {
    Driver driver(ros::NodeHandle("~"));
    ros::spin();
  } catch (const std::exception& e) {
    ROS_FATAL("%s", e.what());
    return 1;
  }
  return 0;
}
