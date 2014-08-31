#pragma once

#include "Eigen/Dense"
#include "parameter/shared_parameter.h"

namespace PS {

// key-value vector, the (global) keys are sorted and unique. Both keys and
// values are stored in arrays.
template <typename K, typename V>
class KVVector : public SharedParameter<K,V> {
 public:

  SArray<K>& key() { return key_; }
  SArray<V>& value() { return val_; }
  // find the local positions of a global key range
  SizeR localRange (const Range<K>& global_range) {
    return key_.findRange(global_range);
  }
  // slice a segment of the value using the local positions
  SArray<V> segment(const SizeR& local_range) {
    return val_.segment(local_range);
  }

  // fetch the values of *key()* from the servers
  void fetchValueFromServers();

  // push values into servers, waiting the servers aggregated the data, and then
  // pull the values and call *callback*
  typedef std::function<void(int)> Fn;
  void roundTripForWorker (
      int time, const Range<K>& range = Range<K>::all(),
      const SArrayList<V>& push_value = SArrayList<V>(), Fn callback = Fn());

  // aggregate the data from all worker, call *update*
  void roundTripForServer(
      int time, const Range<K>& range = Range<K>::all(), Fn update = Fn(),
      const bool verbose = false);

  // return the data received at time t, then *delete* it
  AlignedArrayList<V> received(int t);

  // backup to replica nodes, will use the current time as the backup time,
  // which means, if you modified the data, you should call this function before
  // any other push and pull have been called. this function will increment the
  // clock by 1,
  void backup(const NodeID& sender, int time, const Range<K>& range);

  // # of keys, or the length of the vector
  size_t size() const { return key_.size(); }
  // # of nnz entries
  size_t nnz() const { return val_.nnz(); }

  // implement the virtual functions of SharedParameter
  std::vector<Message> decomposeTemplate(
      const Message& msg, const std::vector<K>& partition);
  void getValue(Message* msg);
  void setValue(Message* msg);
  void getReplica(Range<K> range, Message *msg);
  void setReplica(Message *msg);
  void recoverFrom(Message *msg);

  using Customer::taskpool;
  using Customer::myNodeID;
  using SharedParameter<K,V>::getCall;
  using SharedParameter<K,V>::setCall;
  using SharedParameter<K,V>::myKeyRange;
  using SharedParameter<K,V>::keyRange;
  using SharedParameter<K,V>::sync;
 private:
  SArray<K> key_;
  SArray<V> val_;

  std::unordered_map<int, AlignedArrayList<V> > recved_val_;
  std::mutex recved_val_mu_;

  struct Replica {
    SArray<K> key;
    SArray<V> value;
    std::vector<std::pair<NodeID, int> > clock;
    void addTime(const Timestamp& t) {
      clock.push_back(std::make_pair(t.sender(), t.time()));
    }
  };
  std::unordered_map<Range<K>, Replica> replica_;
};

template <typename K, typename V>
void KVVector<K, V>::getReplica(Range<K> range, Message *msg) {
  // Range<K> kr(sp_->getCall(*msg).key());
  // auto kr = sp_->keyRange(msg->recver);
  auto it = replica_.find(range);
  CHECK(it != replica_.end())
      << myNodeID() << " does not backup for " << msg->sender
      << " with key range " << range;

  // data
  const auto& rep = it->second;
  if (range == myKeyRange()) {
    msg->key = key_;
    msg->addValue(val_);
  } else {
    msg->key = rep.key;
    msg->addValue(rep.value);
  }

  // timestamp
  auto arg = setCall(msg);
  for (auto& v : rep.clock) {
    auto sv = arg->add_backup();
    sv->set_sender(v.first);
    sv->set_time(v.second);
  }
}

template <typename K, typename V>
void KVVector<K, V>::setReplica(Message *msg) {
  return; // wakensky
#if 0
  auto recved_key = SArray<K>(msg->key);
  auto recved_val = SArray<V>(msg->value[0]);
  auto kr = keyRange(msg->sender);
  auto it = replica_.find(kr);

  // data
  if (it == replica_.end()) {
    replica_[kr].key = recved_key;
    replica_[kr].value = recved_val;
    it = replica_.find(kr);
  } else {
    size_t n;
    auto aligned = match(
        it->second.key, recved_key, recved_val.data(), recved_key.range(), &n);
    CHECK_EQ(n, recved_key.size()) << "my key: " << it->second.key
                                   << "\n received key " << recved_key;
    it->second.value.segment(aligned.first).eigenVector() = aligned.second.eigenVector();
  }

  // LL << myNodeID() << " backup for " << msg->sender << " W: "
  //    << norm(replica_[kr].value.vec(), 1) << ", "
  //    << norm(recved_val.vec(), 1) << " " << recved_val.size();

  // timestamp
  auto arg = getCall(*msg);
  for (int i = 0; i < arg.backup_size(); ++i)
    it->second.addTime(arg.backup(i));
#endif
}

template <typename K, typename V>
void KVVector<K, V>::recoverFrom(Message *msg) {
  // data
  auto arg = getCall(*msg);
  Range<K> range(msg->task.key_range());
  CHECK_EQ(range, myKeyRange());
  key_ = msg->key;
  val_ = msg->value[0];

  LL << myNodeID() << " has recovered W from " << msg->sender << " "
     << key_.size(); // << " keys with |W|_1 " << norm(vec(), 1);

  // timestamp
  for (int i = 0; i < arg.backup_size(); ++i) {
    auto w = taskpool(arg.backup(i).sender());
    w->finishIncomingTask(arg.backup(i).time());
    // LL << "replay " << arg.backup(i).sender() << " time " << arg.backup(i).time();
  }
}


template <typename K, typename V>
void KVVector<K, V>::roundTripForWorker (
    int time, const Range<K>& range, const SArrayList<V>& push_value, Fn callback) {
  auto lr = localRange(range);
  Message push_msg, pull_msg;

  // time 0 : push
  push_msg.key = key_.segment(lr);
  for (auto& val : push_value) {
    if (val.size() == 0) break;
    CHECK_EQ(lr.size(), val.size());
    push_msg.value.push_back(SArray<char>(val));
  }
  time = sync(CallSharedPara::PUSH, kServerGroup, range, push_msg, time);

  // time 1 : servers do update

  // time 2 : pull
  pull_msg.key = key_.segment(lr);
  sync(CallSharedPara::PULL, kServerGroup, range, pull_msg, time+2,
       time+1, []{}, [time, callback]{ callback(time+2); });
}

template <typename K, typename V>
void KVVector<K, V>::roundTripForServer(
    int time, const Range<K>& range, Fn update, const bool verbose) {
  auto wk = taskpool(kWorkerGroup);
  // none of my bussiness
#if 0
  if (!key_.empty() && range.setIntersection(key_.range()).empty()) {
    // cl->finishInTask(cl->incrClock(3));
    if (verbose) {
        LI << "[" << KVVector<K, V>::myNodePrintable() << "] " <<
            "skipped task [" << time << "]";
    }
    return;
  }
#endif

  if (verbose) {
    LI << "[" << KVVector<K, V>::myNodePrintable() << "] " <<
        "is waiting IncomingTask [" << time << "]";
  }

  wk->waitIncomingTask(time);

  if (verbose) {
    LI << "[" << KVVector<K, V>::myNodePrintable() << "] " <<
        "leaves waiting IncomingTask; updating [" << time << "]";
  }

  this->sys_.runningStatus().startTimer(TimerType::BUSY);
  update(time);
  this->sys_.runningStatus().stopTimer(TimerType::BUSY);

  backup(kWorkerGroup, time+1, range);

  // time 1: mark it as finished so that all blocked pulls can be started
  wk->finishIncomingTask(time+1);
}

template <typename K, typename V>
void KVVector<K, V>::backup(const NodeID& sender, int time, const Range<K>& range) {
  if (FLAGS_num_replicas == 0) return;

  // data
  auto local_range = localRange(range);
  Message msg;
  msg.key = key_.segment(local_range);
  msg.addValue(val_.segment(local_range));

  // timestamp
  auto arg = setCall(&msg);
  auto vc = arg->add_backup();
  vc->set_time(time);
  vc->set_sender(sender);

  time = sync(CallSharedPara::PUSH_REPLICA, kReplicaGroup, range, msg);
  taskpool(kReplicaGroup)->waitOutgoingTask(time);

  // also save the timestamp in local, in case if my replica node dead, he
  // will request theose timestampes
  replica_[myKeyRange()].addTime(*vc);
}

template <typename K, typename V>
void KVVector<K, V>::fetchValueFromServers() {
  Message pull_msg; pull_msg.key = key_;
  int time = sync(
      CallSharedPara::PULL, kServerGroup, Range<Key>::all(), pull_msg);
  taskpool(kServerGroup)->waitOutgoingTask(time);
  auto recv = received(time);
  val_ = recv[0].second;
  CHECK_EQ(val_.size(), key_.size());
}

template <typename K, typename V>
AlignedArrayList<V> KVVector<K, V>::received(int t) {
  Lock l(recved_val_mu_);
  auto it = recved_val_.find(t);
  CHECK(it != recved_val_.end()) << myNodeID() << " hasn't received data at time " << t;
  auto ret = it->second;
  recved_val_.erase(it);
  return ret;
}

template <typename K, typename V>
void KVVector<K,V>::getValue(Message* msg) {
  // if (msg->key.empty()) return;
  CHECK_EQ(key_.size(), val_.size());
  if (msg->task.has_need_push_pull() && !msg->task.need_push_pull()) {
    return;
  }

  SArray<K> recv_key(msg->key);
  size_t n = 0;
  Range<Key> union_range = recv_key.range().setUnion(key_.range());

  // get range
  SizeR range;
  if (!recv_key.empty() && !key_.empty()) {
    range = recv_key.findRange(union_range);
  }

  // construct new SArray holding updated values
  SArray<V> new_value;
  if (range.size() > 0) {
    new_value = SArray<V>(range.size());
  }
  CHECK_EQ(new_value.size(), recv_key.size()) <<
    recv_key << "\n" << key_;

  // match
  match(range, recv_key, new_value, key_, val_, &n, MatchOperation::ASSIGN);
  CHECK_GE(new_value.size(), n);

  // store in msg
  msg->value.push_back(SArray<char>(new_value));
}

template <typename K, typename V>
void KVVector<K,V>::setValue(Message* msg) {
  Lock l(recved_val_mu_);
  if (msg->task.has_need_push_pull() && !msg->task.need_push_pull()) {
    return;
  }
  SArray<K> recv_key(msg->key);
  Range<K> key_range(msg->task.key_range());

  if (msg->value.empty() && !msg->key.empty()) {
    key_ = key_.setUnion(recv_key);
    // LL << key_.size();
    return;
  }

  int t = msg->task.time();
  bool first = recved_val_.count(t) == 0;

  // LL << "recv push" << msg->debugString();
  for (int i = 0; i < msg->value.size(); ++i) {
    SArray<V> recv_data(msg->value[i]);
    CHECK_EQ(recv_data.size(), recv_key.size());
    size_t n = 0;

    if (first) {
      // get range
      SizeR range;
      if (!key_.empty() && !recv_key.empty()) {
        range = key_.findRange(key_range);
      }

      // construct SArray<V>
      auto aligned = std::make_pair(range, SArray<V>());
      if (range.size() > 0) {
        aligned.second = SArray<V>(range.size());
      }
      CHECK_GE(aligned.second.size(), recv_key.size());
      recved_val_[t].push_back(aligned);

      // match
      match(range, key_, aligned.second,
        recv_key, recv_data, &n, MatchOperation::ASSIGN);
    } else {
      match(recved_val_[t][i].first, key_, recved_val_[t][i].second,
        recv_key, recv_data, &n, MatchOperation::ADD);
    }

    CHECK_EQ(recv_key.size(), n);
  }
}

// partition is a sorted key ranges
template <typename K, typename V>
std::vector<Message> KVVector<K,V>::
decomposeTemplate(const Message& msg, const std::vector<K>& partition) {
  size_t n = partition.size();
  std::vector<size_t> pos(n, -1);

  SArray<K> key(msg.key);
  pos.reserve(n);
  for (auto k : partition)
    pos.push_back(std::lower_bound(key.begin(), key.end(), k) - key.begin());

  // split the message
  Message part = msg;
  std::vector<Message> ret(n-1);
  for (int i = 0; i < n-1; ++i) {
    part.clearData();
    if (Range<K>(partition[i], partition[i+1]).setIntersection(
            Range<K>(msg.task.key_range())).empty()) {
      // mark as do_not_send, because the remote node does not maintain this key range
      // part.valid = false;
      // part.valid = false;

      // I will still send part to remote, while telling remote drop it automaitically
      part.valid = true;
      part.task.set_need_push_pull(false);
      // LL << myNodeID() << " " << part.shortDebugString() << " " << i;
    } else {
      part.valid = true;
      part.task.set_need_push_pull(true);
      if (pos[i] == -1)
        pos[i] = std::lower_bound(key.begin(), key.end(), partition[i]) - key.begin();
      if (pos[i+1] == -1)
        pos[i+1] = std::lower_bound(key.begin(), key.end(), partition[i+1]) - key.begin();
      SizeR lr(pos[i], pos[i+1]);
      part.key = key.segment(lr);
      for (auto& d : msg.value) {
        SArray<V> data(d);
        part.value.push_back(SArray<char>(data.segment(lr)));
      }
    }
    ret[i] = part;
    // LL << ret[i].debugString();
  }
  return ret;
}


} // namespace PS
