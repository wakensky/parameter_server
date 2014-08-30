#pragma once
#include "util/common.h"
#include "system/message.h"
#include "system/yellow_pages.h"
#include "util/threadsafe_queue.h"
#include "system/running_status.h"

namespace PS {

DECLARE_int32(num_servers);
DECLARE_int32(num_workers);
DECLARE_int32(num_unused);
DECLARE_int32(num_replicas);
DECLARE_string(node_file);

class Postoffice {
 public:
  SINGLETON(Postoffice);
  ~Postoffice();

  // run the system
  void run(const string &net_interface);

  // queue a message into the sending buffer, which will be sent by the sending
  // thread.
  void queue(const Message& msg);

  // send the reply message for the _task_ from _recver_
  void reply(const NodeID& recver,
             const Task& task,
             const string& reply_msg = string());

  // reply message *msg* with protocal message *proto*
  template <class P> void replyProtocalMessage(Message* msg, const P& proto) {
      string str; proto.SerializeToString(&str);
      reply(msg->sender, msg->task, str);
      msg->replied = true;
  }

void reply(const Message& msg, const string& reply_msg = string());

  // add the nodes in _pt_ into the system
  void manageNode(const Task& pt);

  // accessors and mutators
  YellowPages& yp() { return yp_; }
  Node& myNode() { return yp_.van().myNode(); }
  RunningStatus& runningStatus() { return running_status_; }

 private:
  DISALLOW_COPY_AND_ASSIGN(Postoffice);
  Postoffice() { }

  void manage_app(const Task& pt);
  void send();
  void recv();
  void heartbeat();
  void monitor();

  string printDashboardTopRow();
  string printRunningStatusReport(
    const string &node_id,
    const RunningStatusReport &report);

  std::mutex mutex_;
  bool done_ = false;

  std::unique_ptr<std::thread> recving_;
  std::unique_ptr<std::thread> sending_;
  std::unique_ptr<std::thread> heartbeating_;
  std::unique_ptr<std::thread> monitoring_;

  threadsafe_queue<Message> sending_queue_;

  // yp_ should stay behind sending_queue_ so it will be destroied earlier
  YellowPages yp_;

  RunningStatus running_status_;

  // records running status for all workers/servers
  std::map<
    NodeID, RunningStatusReport,
    bool (*)(const NodeID &a, const NodeID &b)> dashboard_{
      [](const NodeID &a, const NodeID &b)->bool {
        // lambda: split NodeID into primary and secondary
        auto splitNodeID = [] (const NodeID &in, string &primary, string &secondary) {
          size_t last_alpha_idx = in.find_last_not_of("0123456789");
          if (std::string::npos == last_alpha_idx) {
            primary = in;
            secondary = "";
          } else {
            primary = in.substr(0, last_alpha_idx + 1);
            secondary = in.substr(last_alpha_idx + 1);
          }
          return;
        };

        // split
        string a_primary, a_secondary;
        splitNodeID(a, a_primary, a_secondary);
        string b_primary, b_secondary;
        splitNodeID(b, b_primary, b_secondary);

        // compare
        if (a_primary != b_primary) {
          return a_primary < b_primary;
        } else {
          return std::stoul(a_secondary) < std::stoul(b_secondary);
        }
      }
    };
  std::mutex mu_dashboard_;
};

} // namespace PS
