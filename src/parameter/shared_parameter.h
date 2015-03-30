#pragma once
#include "system/customer.h"
#include "system/validation.h"
#include "parameter/frequency_filter.h"
namespace PS {

template <typename K> class SharedParameter;
template <typename K> using SharedParameterPtr = std::shared_ptr<SharedParameter<K>>;

// the base class of shared parameters
template <typename K>
class SharedParameter : public Customer {
 public:
  // convenient wrappers of functions in remote_node.h
  int sync(MessagePtr msg) {
    CHECK(msg->task.shared_para().has_cmd()) << msg->debugString();
    if (!msg->task.has_key_range()) {
      Range<Key> key_range(0, std::numeric_limits<Key>::max());
      key_range.to(msg->task.mutable_key_range());
    }
    return taskpool(msg->recver)->submit(msg);
  }
  int push(MessagePtr msg) {
    set(msg)->set_cmd(CallSharedPara::PUSH);
    return sync(msg);
  }
  int pull(MessagePtr msg) {
    set(msg)->set_cmd(CallSharedPara::PULL);
    return sync(msg);
  }
  void waitInMsg(const NodeID& node, int time) {
    taskpool(node)->waitIncomingTask(time);
  }
  bool tryWaitInMsg(const NodeID& node, int time) {
    return taskpool(node)->tryWaitIncomingTask(time);
  }

  void waitOutMsg(const NodeID& node, int time) {
    taskpool(node)->waitOutgoingTask(time);
  }

  bool tryWaitOutMsg(const NodeID& node, int time) {
    return taskpool(node)->tryWaitOutgoingTask(time);
  }

  void finish(const NodeID& node, int time) {
    taskpool(node)->finishIncomingTask(time);
  }

  FreqencyFilter<K>& keyFilter(int chl) { return key_filter_[chl]; }
  void setKeyFilterIgnoreChl(bool flag) { key_filter_ignore_chl_ = flag; }
  // void clearKeyFilter(int chl) { key_filter_[chl].clear(); }

  // process a received message, will called by the thread of executor
  void process(const MessagePtr& msg);

  CallSharedPara* set(MessagePtr msg) {
    msg->task.set_type(Task::CALL_CUSTOMER);
    return msg->task.mutable_shared_para();
  }
  CallSharedPara get(const MessagePtr& msg) {
    CHECK_EQ(msg->task.type(), Task::CALL_CUSTOMER);
    CHECK(msg->task.has_shared_para());
    return msg->task.shared_para();
  }
 protected:
  // fill the values specified by the key lists in msg
  virtual void getValue(const MessagePtr& msg) = 0;
  // set the received KV pairs into my data strcuture
  virtual void setValue(const MessagePtr& msg) = 0;
  // set the received KV pairs into my data strcuture for validation
  virtual void setValidationValue(const MessagePtr& msg) = 0;

  Range<K> myKeyRange() {
    return keyRange(Customer::myNodeID());
  }
  // query the key range of a node
  Range<K> keyRange(const NodeID& id) {
    return Range<K>(exec_.rnode(id)->keyRange());
  }

 private:
  std::unordered_map<int, FreqencyFilter<K>> key_filter_;
  bool key_filter_ignore_chl_ = false;


  // add key_range in the future, it is not necessary now
  std::unordered_map<NodeID, std::vector<int> > clock_replica_;

  // owner task_id -> task_over_count
  // That is how many TASK-OVER notification on a task
  std::unordered_map<int, int> task_over_count_;
};

template <typename K>
void SharedParameter<K>::process(const MessagePtr& msg) {
  bool req = msg->task.request();
  int chl = msg->task.key_channel();
  Range<K> g_key_range(msg->task.key_range());
  auto call = get(msg);
  bool push = call.cmd() == CallSharedPara::PUSH;
  bool pull = call.cmd() == CallSharedPara::PULL;
  MessagePtr reply;
  if (pull && req) {
    reply = MessagePtr(new Message(*msg));
    reply->task.set_request(false);
    std::swap(reply->sender, reply->recver);
  }

  MilliTimer milli_timer; milli_timer.start();
  // process
  if (call.insert_key_freq() || call.has_query_key_freq()) {
    // deal with tail features
    if (key_filter_ignore_chl_) chl = 0;
    if (call.insert_key_freq() && req && !msg->value.empty()) {
      auto& filter = key_filter_[chl];
      if (filter.empty() && !msg->key.empty()) {
        double w = (double)FLAGS_num_workers;
        int scale = static_cast<int>(
          SArray<K>(msg->key).size() * call.countmin_n());
        scale = w * scale / log(w+1);
        if (scale < 0) {
          LL << "bloomfilter size overflow " << myNodeID() << " chl:" << chl <<
            " keycount: " << SArray<K>(msg->key).size() <<
            " countmin_n: " << call.countmin_n() <<
            " w: " << w << " log(w+1): " << log(w+1) <<
            " scale: " << scale <<
            " [bloomfilter size will be restricted to 64, which leaves a lot of filtered features]";
        }
        filter.resize(std::max(scale, 64), call.countmin_k());
      }
      filter.insertKeys(SArray<K>(msg->key), SArray<uint8>(msg->value[0]));
    }
    if (call.has_query_key_freq()) {
      if (req) {
        reply->clearData();
        reply->setKey(key_filter_[chl].queryKeys(
            SArray<K>(msg->key), call.query_key_freq()));
      } else {
        setValue(msg);
      }
    }
  } else if (IamServer() && call.task_over()) {
    if (++task_over_count_[msg->task.owner_time()] >= FLAGS_num_workers) {
      this->ocean().drop(chl, g_key_range, msg->task.owner_time());
    }
  } else {
    if (push && req) {
      setValue(msg);
    } else if (pull && !req) {
      if (call.is_validation()) {
        setValidationValue(msg);
      } else {
        setValue(msg);
      }
    } else if (pull && req) {
      getValue(reply);
    }
  }
  milli_timer.stop();
  this->sys_.hb_collector().increaseTime(milli_timer.get());

  // reply if necessary
  if (pull && req) {
    taskpool(reply->recver)->encodeFilter(reply);
    sys_.queue(reply);
    msg->replied = true;
  }
}

#define USING_SHARED_PARAMETER                  \
  using Customer::taskpool;                     \
  using Customer::myNodeID;                     \
  using SharedParameter<K>::get;                \
  using SharedParameter<K>::set;                \
  using SharedParameter<K>::myKeyRange;         \
  using SharedParameter<K>::keyRange;           \
  using SharedParameter<K>::sync
} // namespace PS
