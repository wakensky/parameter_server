#pragma once

#include "proto/heartbeat.pb.h"
#include "util/resource_usage.h"
#include "util/common.h"

namespace PS {
class HeartbeatCollector {
public:
  HeartbeatCollector();
  ~HeartbeatCollector();
  HeartbeatCollector(const HeartbeatCollector& other) = delete;
  HeartbeatCollector& operator= (const HeartbeatCollector& rhs) = delete;

  HeartbeatReport produceReport();

  // set network interface which is under use
  //   such as "eth0"
  // set hostname
  void init(const string& interface, const string& hostname);

  void increaseTime(const size_t delta) { Lock l(mu_); milli_seconds_ += delta; }
  void increaseInBytes(const size_t delta) { Lock l(mu_); in_bytes_ += delta; }
  void increaseOutBytes(const size_t delta) { Lock l(mu_); out_bytes_ += delta; }

private:
  size_t milli_seconds_;
  size_t in_bytes_;
  size_t out_bytes_;

  string interface_;
  string hostname_;

  // snapshot of performance counters
  struct Snapshot {
    uint64 process_user;
    uint64 process_sys;

    uint64 host_user;
    uint64 host_sys;

    // the sum of all kinds of cpu statistic, who acts as denominator
    uint64 host_cpu_base;

    uint64 host_in_bytes;
    uint64 host_out_bytes;

    // the time snapshot produced
    std::chrono::system_clock::time_point time_point;

    Snapshot() :
      process_user(0),
      process_sys(0),
      host_user(0),
      host_sys(0),
      host_cpu_base(0),
      host_in_bytes(0),
      host_out_bytes(0) {
        // do nothing
    }

    string shortDebugString() {
      std::stringstream ss;
      ss << "{";
      ss << "process_user: " << process_user << ", ";
      ss << "process_sys: " << process_sys << ", ";
      ss << "host_user: " << host_user << ", ";
      ss << "host_sys: " << host_sys << ", ";
      ss << "host_cpu_base: " << host_cpu_base << ", ";
      ss << "host_in_bytes: " << host_in_bytes << ", ";
      ss << "host_out_bytes: " << host_out_bytes;
      ss << "}";

      return ss.str();
    }
  }; // struct Snapshot

  Snapshot snapshot_last_;
  Snapshot snapshot();

  std::mutex mu_;
  size_t cpu_core_number_;
}; // class Heartbeatinfo
}; // namespace PS
