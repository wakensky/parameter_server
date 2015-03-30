#pragma once
#include "util/common.h"
#include "system/message.h"
#include "system/yellow_pages.h"
#include "system/heartbeat_collector.h"
#include "system/validation.h"
#include "util/threadsafe_queue.h"
#include "dashboard.h"

namespace PS {

DECLARE_int32(num_servers);
DECLARE_int32(num_workers);
DECLARE_int32(num_unused);
DECLARE_string(node_file);

class Postoffice {
 public:
  SINGLETON(Postoffice);
  ~Postoffice();
  // Run the system
  void run();
  // Queue a message into the sending buffer, which will be sent by the sending
  // thread.
  void queue(const MessagePtr& msg);
  // reply *task* from *recver* with *reply_msg*
  void reply(const NodeID& recver, const Task& task, const string& reply_msg = string());
  // reply message *msg* with protocal message *proto*
  template <class P> void replyProtocalMessage(const MessagePtr& msg, const P& proto) {
    string str; proto.SerializeToString(&str);
    reply(msg->sender, msg->task, str);
    msg->replied = true;
  }

  // add the nodes in _pt_ into the system
  void manageNode(const Task& pt);

  // accessors and mutators
  YellowPages& yp() { return yellow_pages_; }
  Node& myNode() { return yellow_pages_.van().myNode(); }
  Node& scheduler() { return yellow_pages_.van().scheduler(); }

  HeartbeatCollector& hb_collector() { return heartbeat_collector_; };

  Ocean& ocean() { return ocean_; }
  Validation& validation() { return validation_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(Postoffice);
  Postoffice():
    manage_task_done_(false) {
    // do nothing
  }

  void manageApp(const Task& pt);
  void send();
  void recv();
  // heartbeat thread function
  void heartbeat();
  // monitor thread function only used by scheduler
  void monitor();

  std::mutex mutex_;
  bool done_ = false;

  std::promise<void> nodes_are_ready_;
  std::unique_ptr<std::thread> recving_;
  std::unique_ptr<std::thread> sending_;
  std::unique_ptr<std::thread> heartbeating_;
  std::unique_ptr<std::thread> monitoring_;
  threadsafe_queue<MessagePtr> sending_queue_;

  // yp_ should stay behind sending_queue_ so it will be destroied earlier
  YellowPages yellow_pages_;

  // heartbeat reporter for workers/servers
  HeartbeatCollector heartbeat_collector_;
  // If I have finished MANAGE task, I have connected to the scheduler certainly
  bool manage_task_done_;
  Dashboard dashboard_;

  Ocean ocean_;
  Validation validation_;
};

} // namespace PS
