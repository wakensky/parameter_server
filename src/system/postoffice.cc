#include <iomanip>
#include "system/postoffice.h"
// #include <omp.h>
#include "system/customer.h"
#include "system/app.h"
// #include "system/debug.h"
#include "util/file.h"

namespace PS {

DEFINE_bool(enable_fault_tolerance, false, "enable fault tolerance feature");

DEFINE_int32(num_servers, 1, "number of servers");
DEFINE_int32(num_workers, 1, "number of clients");
DEFINE_int32(num_unused, 0, "number of unused nodes");

DEFINE_int32(num_replicas, 0, "number of replica");

DEFINE_string(app, "../config/rcv1_l1lr.config", "the configuration file of app");
// DEFINE_string(app, "../config/block_prox_grad.config", "the configuration file of app");

DEFINE_string(node_file, "./nodes", "node information");

DEFINE_int32(num_threads, 2, "number of computational threads");

DEFINE_int32(report_interval, 5,
    "Servers/Workers report running status to scheduler "
    "in every report_interval seconds. "
    "default: 5; if set to 0, heartbeat is disabled");

Postoffice::~Postoffice() {
  // sending_->join();
  // yp_.van().destroy();
  recving_->join();
  Message stop; stop.terminate = true; queue(stop);
  sending_->join();
}

void Postoffice::run(const string &net_interface) {
  yp_.init();
  running_status_.setInterface(net_interface);
  recving_ = std::unique_ptr<std::thread>(new std::thread(&Postoffice::recv, this));
  sending_ = std::unique_ptr<std::thread>(new std::thread(&Postoffice::send, this));

  if (FLAGS_report_interval > 0) {
    if (Node::SCHEDULER == myNode().role()) {
      monitoring_ = std::unique_ptr<std::thread>(
        new std::thread(&Postoffice::monitor, this));
      monitoring_->detach();
    } else {
      heartbeating_ = std::unique_ptr<std::thread>(
        new std::thread(&Postoffice::heartbeat, this));
      heartbeating_->detach();
    }
  }

  // omp_set_dynamic(0);
  // omp_set_num_threads(FLAGS_num_threads);
  if (myNode().role() == Node::SCHEDULER) {
    // run the application
    AppConfig config;
    ReadFileToProtoOrDie(FLAGS_app, &config);
    AppPtr app = App::create(config);
    yp_.add(app);
    app->run();
    app->stop();
  } else {
    // run as a daemon
    while (!done_) usleep(300);
    // LL << myNode().uid() << " stopped";
  }
}

void Postoffice::reply(
    const NodeID& recver, const Task& task, const string& reply_msg) {
  if (!task.request()) return;
  Task tk;
  tk.set_customer(task.customer());
  tk.set_request(false);
  tk.set_type(Task::REPLY);
  if (!reply_msg.empty()) tk.set_msg(reply_msg);
  tk.set_time(task.time());

  Message re(tk);
  re.recver = recver;
  queue(re);
}

void Postoffice::reply(const Message& msg, const string& reply_msg) {
  if (!msg.task.request()) return;
  Message re = replyTemplate(msg);
  re.task.set_type(Task::REPLY);
  if (!reply_msg.empty())
    re.task.set_msg(reply_msg);
  re.task.set_time(msg.task.time());
  queue(re);
}

void Postoffice::queue(const Message& msg) {

  if (msg.valid) {
    sending_queue_.push(msg);
  } else {
    // do not send, fake a reply mesage
    Message re = replyTemplate(msg);
    re.task.set_type(Task::REPLY);
    re.task.set_time(msg.task.time());
    yp_.customer(re.task.customer())->exec().accept(re);
  }
}

//  TODO fault tolerance, check if node info has been changed
void Postoffice::send() {
  Message msg;
  // send out all messge in the queue even done
  while (true) {
    sending_queue_.wait_and_pop(msg);
    if (msg.terminate) break;

    size_t send_bytes = 0;
    Status stat = yp_.van().send(msg, send_bytes);
    running_status_.increaseOutBytes(send_bytes);

    if (!stat.ok()) {
      LL << "sending " << msg.debugString() << " failed\n"
         << "error: " << stat.ToString();
    }
  }
}

void Postoffice::recv() {
  Message msg;
  while (true) {
    size_t recv_bytes = 0;
    auto stat = yp_.van().recv(&msg, recv_bytes);
    running_status_.increaseInBytes(recv_bytes);

    // if (!stat.ok()) break;
    CHECK(stat.ok()) << stat.ToString();
    auto& tk = msg.task;
    if (tk.request() && tk.type() == Task::TERMINATE) {
      yp_.van().statistic();
      done_ = true;
      break;
    } else if (tk.request() && tk.type() == Task::MANAGE) {
      if (tk.has_mng_app()) manage_app(tk);
      if (tk.has_mng_node()) manageNode(tk);
    } else if (Task::HEARTBEATING == tk.type()) {
      RunningStatusReport report;
      report.ParseFromString(tk.msg());

      {
        Lock l(mu_dashboard_);
        report.set_task_id(dashboard_[msg.sender].task_id());
        dashboard_[msg.sender] = report;
      }

      continue;
    } else {
      yp_.customer(tk.customer())->exec().accept(msg);

      if (Node::SCHEDULER == myNode().role() && FLAGS_report_interval > 0) {
        Lock l(mu_dashboard_);
        dashboard_[msg.sender].set_task_id(msg.task.time());
      }
      continue;
    }
    auto ptr = yp_.customer(tk.customer());
    if (ptr != nullptr) ptr->exec().finish(msg);
    reply(msg);
  }
}

void Postoffice::heartbeat() {
    while (true && !done_) {
        if (yp_.van().connectivity("H").ok()) {
            // serialize Runningstatusreport
            string report;
            running_status_.get().SerializeToString(&report);

            // pack msg
            Message msg;
            msg.sender = myNode().id();
            msg.recver = "H";
            msg.original_recver = "H";
            msg.valid = true;
            msg.task.set_time(std::numeric_limits<int32>::min());
            msg.task.set_request(false);
            msg.task.set_customer("HB");
            msg.task.set_type(Task::HEARTBEATING);
            msg.task.set_msg(report);

            // push to sending queue
            queue(msg);

            std::this_thread::sleep_for(std::chrono::seconds(FLAGS_report_interval));
        }
    };
}

string Postoffice::printDashboardTopRow() {
    const size_t WIDTH = 10;

    // time_t
    std::time_t now_time = std::chrono::system_clock::to_time_t(
        std::chrono::system_clock::now());
    string time_str = ctime(&now_time);
    time_str.resize(time_str.size() - 1);

    std::stringstream ss;
    ss << std::setiosflags(std::ios::left) <<
        std::setw(WIDTH * 2) << std::setfill('=') << "" << " Dashboard " <<
        time_str + " " << std::setw(WIDTH * 2) << std::setfill('=') << "" << "\n";
    ss << std::setfill(' ') << std::setw(WIDTH) << "Node" <<
        std::setw(WIDTH) << "Task" <<
        std::setw(WIDTH) << "MyCPU(%)" <<
        std::setw(WIDTH) << "MyRSS(M)" <<
        std::setw(WIDTH) << "MyVirt(M)" <<
        std::setw(WIDTH) << "BusyTime" <<
        std::setw(WIDTH) << "InMB" <<
        std::setw(WIDTH) << "OutMB" <<
        std::setw(WIDTH) << "HostCPU" <<
        std::setw(WIDTH) << "HostFree" <<
        std::setw(WIDTH) << "HostInBw" <<
        std::setw(WIDTH) << "HostOutBw" <<
        std::setw(WIDTH * 2) << "HostName";

    return ss.str();
}

string Postoffice::printRunningStatusReport(
    const string &node_id,
    const RunningStatusReport &report) {
    std::stringstream ss;
    const size_t WIDTH = 10;

    ss << std::setiosflags(std::ios::left) << std::setw(WIDTH) << node_id <<
        std::setw(WIDTH) << report.task_id() <<
        std::setw(WIDTH) << static_cast<uint32>(
            report.my_cpu_usage_user() + report.my_cpu_usage_sys()) <<
        std::setw(WIDTH) << report.my_rss() <<
        std::setw(WIDTH) << report.my_virtual() <<
        // busy time
        std::setw(WIDTH) << std::to_string(static_cast<uint32>(report.busy_time_micro() / 1e3)) + "(" +
            std::to_string(static_cast<uint32>(100 * (
                static_cast<float>(report.busy_time_micro()) / report.total_time_micro()))) + ")" <<
        // net in MB
        std::setw(WIDTH) << std::to_string(report.in_bytes() >> 20) + "(" +
            std::to_string(static_cast<uint32>(report.in_bytes() / (report.total_time_micro() / 1e3) / 1e3)) + ")" <<
        // net out MB
        std::setw(WIDTH) << std::to_string(report.out_bytes() >> 20) + "(" +
            std::to_string(static_cast<uint32>(report.out_bytes() / (report.total_time_micro() / 1e3) / 1e3)) + ")" <<
        // host cpu
        std::setw(WIDTH) << static_cast<uint32>(
            report.host_cpu_usage_user() + report.host_cpu_usage_sys()) <<
        // host free memory
        std::setw(WIDTH) << report.host_free_memory() <<
        // host net in/out bandwidth usage (MB/s)
        std::setw(WIDTH) << (report.host_net_in_bw_usage() < 1e4 ?
            std::to_string(report.host_net_in_bw_usage()) :
            "INIT") <<
        std::setw(WIDTH) << (report.host_net_out_bw_usage() < 1e4 ?
            std::to_string(report.host_net_out_bw_usage()) :
            "INIT") <<
        std::setw(WIDTH * 2) << yp_.getNode(node_id).hostname();

    return ss.str();
}

void Postoffice::monitor() {
    while (true) {
        if (!dashboard_.empty()) {
            Lock l(mu_dashboard_);

            // print progress for all Servers/Workers
            std::stringstream ss;
            ss << printDashboardTopRow() << "\n";
            for (const auto& item : dashboard_) {
                ss << printRunningStatusReport(item.first, item.second) << "\n";
            }

            // output
            std::cerr << "\n\n" << ss.str();
        }

        std::this_thread::sleep_for(std::chrono::seconds(FLAGS_report_interval));
    };
}

void Postoffice::manage_app(const Task& tk) {
  CHECK(tk.has_mng_app());
  auto& mng = tk.mng_app();
  if (mng.cmd() == ManageApp::ADD) {
    yp_.add(std::static_pointer_cast<Customer>(App::create(mng.app_config())));
  }
}

void Postoffice::manageNode(const Task& tk) {
  // LL << tk.DebugString();
  CHECK(tk.has_mng_node());
  auto& mng = tk.mng_node();
  std::vector<Node> nodes;
  for (int i = 0; i < mng.nodes_size(); ++i)
    nodes.push_back(mng.nodes(i));

  auto obj = yp_.customer(tk.customer());
  switch (mng.cmd()) {
    case ManageNode::INIT:
      for (auto n : nodes) yp_.add(n);
      if (obj != nullptr) {
        obj->exec().init(nodes);
        for (auto c : obj->children())
          yp_.customer(c)->exec().init(nodes);
      }
      break;

    case ManageNode::REPLACE:
      CHECK_EQ(nodes.size(), 2);
      obj->exec().replace(nodes[0], nodes[1]);
      for (auto c : obj->children())
        yp_.customer(c)->exec().replace(nodes[0], nodes[1]);
      break;

    default:
      CHECK(false) << " unknow command " << mng.cmd();
  }

}

// Ack Postoffice::send(Mail pkg) {
//   CHECK(pkg.label().has_request());
//   if (!pkg.label().request()) {
//     sending_queue_.push(std::move(pkg));
//     return Ack();
//   }
//   std::promise<std::string> pro;
//   {
//     Lock l(mutex_);
//     pkg.label().set_tracking_num(tracking_num_);
//     promises_[tracking_num_++] = std::move(pro);
//   }
//   sending_queue_.push(std::move(pkg));
//   return pro.get_future();
// }

// void Postoffice::sendThread() {

//   Mail pkg;
//   while (!done_) {
//     if (sending_queue_.try_pop(pkg)) {
//      auto& label = pkg.label();
//     // int cust_id = label.customer_id();
//     int recver = label.recver();
//     label.set_sender(yp_.myNode().uid());
//     if (!NodeGroup::Valid(recver)) {
//       // the receiver is a single node
//       Status stat = yp_.van().Send(pkg);
//       // TODO fault tolerance
//       CHECK(stat.ok()) << stat.ToString();
//     }
//     } else {
//       std::this_thread::yield();
//     }

//     //     yellow_pages_.GetCustomer(cust_id)->Notify(pkg.label());
//   }
// }


// void Postoffice::SendExpress() {
//   while(true) {
//     Express cmd = express_sending_queue_.Take();
//     cmd.set_sender(postman_.my_uid());
//     Status stat = postman_.express_van()->Send(cmd);
//     CHECK(stat.ok()) << stat.ToString();
//   }
// }

// void Postoffice::RecvExpress() {
//   Express cmd;
//   while(true) {
//     Status stat = postman_.express_van()->Recv(&cmd);
//     postmaster_.ProcessExpress(cmd);
//   }
// }

//     // check if is transfer packets
//     // if (head.type() == Header_Type_BACKUP) {
//     //   // LOG(WARNING) << "Header_Type_BACKUP send";
//     //   head.set_sender(postmaster_->my_uid());
//     //   CHECK(package_van_->Send(mail).ok());
//     //   continue;
//     // }
// // 1, fetch a mail, 2) divide the mail into several ones according to the
// // destination machines. caches keys if necessary. 3) send one-by-one 4) notify
// // the mail
// void Postoffice::SendPackage() {
//   while (1) {
//     Package pkg = package_sending_queue_.Take();
//     auto& label = pkg.label();
//     int32 cust_id = label.customer_id();
//     int32 recver = label.recver();
//     Workload *wl = yellow_pages_.GetWorkload(cust_id, recver);
//     CHECK(label.has_key());
//     KeyRange kr(label.key().start(), label.key().end());
//     // CHECK(kr.Valid()); // we may send invalid key range
//     // first check whether the key list is cached
//     bool hit = false;
//     if (FLAGS_enable_key_cache) {
//       if (wl->GetCache(kr, pkg.keys().ComputeCksum()))
//         hit = true;
//       else
//         wl->SetCache(kr, pkg.keys().cksum(), pkg.keys());
//     }
//     // now send the package
//     if (!NodeGroup::Valid(recver)) {
//       // the receiver is a single node
//       label.set_sender(postman_.my_uid());
//       label.mutable_key()->set_empty(hit);
//       label.mutable_key()->set_cksum(pkg.keys().cksum());
//       Status stat = postman_.package_van()->Send(pkg);
//       // TODO fault tolerance
//       CHECK(stat.ok()) << stat.ToString();
//     } else {
//       // the receiver is a group of nodes, fetch the node list
//       const NodeList& recvers = yellow_pages_.GetNodeGroup(cust_id).Get(recver);
//       CHECK(!recvers->empty()) << "no nodes associated with " << recver;
//       // divide the keys according to the key range a node maintaining
//       for (auto node_id : *recvers) {
//         Workload *wl2 = yellow_pages_.GetWorkload(cust_id, node_id);
//         KeyRange kr2 = kr.Limit(wl2->key_range());
//         RawArray key2, value2;
//         bool hit2 = hit;
//         // try to fetch the cached keys, we do not compute the checksum here to
//         // save the computational time. but it may be not safe.
//         if (hit && !wl2->GetCache(kr2, 0, &key2)) {
//           hit2 = false;
//         }
//         if (!hit2) {
//           // slice the according keys and then store in cache
//           key2 = Slice(pkg.keys(), kr2);
//           if (FLAGS_enable_key_cache)
//             wl2->SetCache(kr2, key2.ComputeCksum(), key2);
//         }
//         if (label.has_value() && !label.value().empty())
//           value2 = Slice(pkg.keys(), pkg.vals(), key2);
//         label.set_recver(node_id);
//         label.set_sender(postman_.my_uid());
//         label.mutable_key()->set_start(kr2.start());
//         label.mutable_key()->set_end(kr2.end());
//         label.mutable_key()->set_cksum(key2.cksum());
//         label.mutable_key()->set_empty(hit2);
//         Package pkg2(label, key2, value2);
//         Status stat = postman_.package_van()->Send(pkg2);
//         // TODO fault tolerance
//         CHECK(stat.ok()) << stat.ToString();
//       }
//     }
//     // notify the customer that package has been sent
//     yellow_pages_.GetCustomer(cust_id)->Notify(pkg.label());
//   }
// }

//     // distinguish node types
//     // normal mail send it to container
//     // back up key-value mail send it to replica nodes
//     // put it in the replica manager queue
//     // replica key-value mail send it to replica manager
//     // node management info send it to postmaster queue
//     // rescue mail, send it to the replica manager
//     // check if is a backup mail or a rescue mail
//     // if (FLAGS_enable_fault_tolerance) {
//     //   if (head.type() == Header_Type_BACKUP
//     //       || head.type() == Header_Type_NODE_RESCUE) {
//     //     replica_manager_->Put(mail);
//     //     continue;
//     //   }
//     // }

//     // if (FLAGS_enable_fault_tolerance && !postmaster_->IamClient()) {
//     //   replica_manager_->Put(pkg);
//     // }
// // if mail does not have key, fetch the cached keys. otherwise, cache the keys
// // in mail.
// void Postoffice::RecvPackage() {
//   Package pkg;
//   while(1) {
//     Status stat = postman_.package_van()->Recv(&pkg);
//     // TODO fault tolerance
     // CHECK(stat.ok()) << stat.ToString();
//     const auto& label = pkg.label();
//     int32 cust_id = label.customer_id();
//     auto cust = yellow_pages_.GetCustomer(cust_id);
//     // waiting is necessary
//     cust->WaitInited();
//     // deal with key caches
//     CHECK(label.has_key());
//     KeyRange kr(label.key().start(), label.key().end());
//     CHECK(kr.Valid());
//     Workload *wl = yellow_pages_.GetWorkload(cust_id, label.sender());
//     auto cksum = label.key().cksum();
//     if (!label.key().empty()) {
//       // there are key lists
//       CHECK_EQ(cksum, pkg.keys().ComputeCksum());
//       // ensure the keys() has proper length when giving to the customer
//       pkg.keys().ResetEntrySize(sizeof(Key));
//       if (FLAGS_enable_key_cache && !wl->GetCache(kr, cksum, NULL))
//         wl->SetCache(kr, cksum, pkg.keys());
//     } else {
//       // the sender believe I have cached the key lists. If it is not true,
//       // TODO a fault tolerance way is just as the sender to resend the keys
//       RawArray keys;
//       CHECK(wl->GetCache(kr, cksum, &keys))
//           << "keys" << kr.ToString() << " of " << cust_id << " are not cached";
//       pkg.set_keys(keys);
//     }
//     if (!pkg.keys().empty()) {
//       pkg.vals().ResetEntrySize(pkg.vals().size() / pkg.keys().entry_num());
//     }
//     cust->Accept(pkg);
//   }
// }

} // namespace PS
