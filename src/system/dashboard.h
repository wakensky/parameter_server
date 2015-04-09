#pragma once
#include "system/message.h"
#include "proto/heartbeat.pb.h"

namespace PS {

DECLARE_int32(dead_interval);

struct NodeIDCmp {
  void splitNodeID(const NodeID& in, string& primary, string& secondary);
  bool operator()(const NodeID& a, const NodeID& b);
};

class Dashboard {
 public:
  void addTask(const NodeID& node, int task_id);
  void addReport(const NodeID& node, const HeartbeatReport& report);
  // Return dashboard via string
  // Add dead nodes who are down unexpectedly out to dead_nodes
  string report(std::vector<string>& dead_nodes);
 private:
  string title();
  string report(
    const NodeID& node,
    const HeartbeatReport& report,
    std::vector<string>& dead_nodes);
  std::mutex mu_;
  std::map<NodeID, HeartbeatReport, NodeIDCmp> data_;
};

} // namespace PS
