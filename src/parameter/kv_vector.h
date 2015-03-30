#pragma once
#include "Eigen/Dense"
#include "parameter/shared_parameter.h"
#include "util/parallel_ordered_match.h"
namespace PS {

template<typename K, typename V> class KVVector;
template<typename K, typename V> using KVVectorPtr = std::shared_ptr<KVVector<K,V>>;

// key-value vector, the (global) keys are sorted and unique. Both keys and
// values are stored in arrays.
template <typename K, typename V>
class KVVector : public SharedParameter<K> {
 public:
  USING_SHARED_PARAMETER;
  SArray<K>& key(int channel) { return key_[channel]; }
  SArray<V>& value(int channel) { return val_[channel]; }
  void clear(int channel) { key_.erase(channel); val_.erase(channel); }

  // find the local positions of a global key range
  SizeR find(int channel, const Range<K>& key_range) {
    return key_[channel].findRange(key_range);
  }

  // return the mareged data received at time t, then *delete* it. If no
  // message, or empty messages are received at time t, then call received(t)
  // will get an error
  typedef std::pair<SizeR, std::vector<SArray<V>>> MergedData;
  MergedData received(int t);

  // implement the virtual functions required
  MessagePtrList slice(const MessagePtr& msg, const KeyList& sep);
  void getValue(const MessagePtr& msg);
  void setValue(const MessagePtr& msg);
  void setValidationValue(const MessagePtr& msg);

 private:
  std::unordered_map<int, SArray<K>> key_;
  std::unordered_map<int, SArray<V>> val_;

  std::unordered_map<int, MergedData> recved_val_;
  std::mutex recved_val_mu_;
};


template <typename K, typename V>
typename KVVector<K,V>::MergedData KVVector<K,V>::received(int t) {
  Lock l(recved_val_mu_);
  auto it = recved_val_.find(t);
  CHECK(it != recved_val_.end()) << myNodeID() << " hasn't received data at time " << t;
  auto ret = it->second;
  recved_val_.erase(it);
  return ret;
}

template <typename K, typename V>
void KVVector<K,V>::setValue(const MessagePtr& msg) {
  SArray<K> recv_key(msg->key);
  if (recv_key.empty()) return;
  int chl = msg->task.key_channel();
  if (msg->value.size() == 0) {
    // only keys, merge these keys, and also clear the values
    key_[chl] = key_[chl].setUnion(recv_key);
    val_[chl].clear();
    return;
  }

  // merge values, and store them in recved_val
  int t = msg->task.time();
  Range<K> key_range(msg->task.key_range());
  SizeR idx_range = this->ocean().fetchAnchor(chl, key_range);

  // load data
  Ocean::DataPack data_pack = this->ocean().get(
    chl, key_range, msg->task.owner_time());
  SArray<Key> parameter_key(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::PARAMETER_KEY)]);
  CHECK_GE(parameter_key.size(), recv_key.size());

  recved_val_mu_.lock();
  auto& matched = recved_val_[t];
  recved_val_mu_.unlock();

  for (int i = 0; i < msg->value.size(); ++i) {
    SArray<V> recv_data(msg->value[i]);
    CHECK_EQ(recv_data.size(), recv_key.size());
    bool first = matched.second.size() <= i;
    if (first) {
      // it is the first node, allocate the memory
      matched.first = idx_range;
      matched.second.push_back(SArray<V>());
      CHECK_EQ(parallelOrderedMatch(
          recv_key, recv_data, parameter_key,
          OpAssign<V>(), FLAGS_num_threads, &matched.second[i]), recv_key.size());
    } else {
      CHECK_EQ(matched.first, idx_range);
      CHECK_EQ(parallelOrderedMatch(
          recv_key, recv_data, parameter_key,
          OpPlus<V>(), FLAGS_num_threads, &matched.second[i]), recv_key.size());
    }
  }
}

template <typename K, typename V>
void KVVector<K,V>::setValidationValue(const MessagePtr& msg) {
  SArray<K> recv_key(msg->key);
  if (recv_key.empty()) return;
  int chl = msg->task.key_channel();

  Range<K> key_range(msg->task.key_range());
  SizeR idx_range = this->validation().fetchAnchor(chl, key_range);

  // load validation keys
  SArray<K> parameter_key = this->validation().getKey(
    chl, key_range, msg->task.owner_time());

  recved_val_mu_.lock();
  auto& matched = recved_val_[msg->task.time()];
  recved_val_mu_.unlock();

  SArray<V> recv_data(msg->value[0]);
  CHECK_EQ(recv_data.size(), recv_key.size());

  if (matched.second.empty()) {
    matched.first = idx_range;
    matched.second.push_back(SArray<V>());
    CHECK_EQ(parallelOrderedMatch(
      recv_key, recv_data, parameter_key,
      OpAssign<V>(), FLAGS_num_threads, &matched.second[0]), recv_key.size());
  } else {
    CHECK_EQ(matched.first, idx_range);
    CHECK_EQ(parallelOrderedMatch(
      recv_key, recv_data, parameter_key,
      OpPlus<V>(), FLAGS_num_threads, &matched.second[0]), recv_key.size());
  }
}

template <typename K, typename V>
void KVVector<K,V>::getValue(const MessagePtr& msg) {
  SArray<K> recv_key(msg->key);
  if (recv_key.empty()) return;
  int chl = msg->task.key_channel();
  SizeR key_range(msg->task.key_range());

  // load data
  Ocean::DataPack data_pack = this->ocean().get(
    chl, key_range, msg->task.owner_time());
  SArray<Key> parameter_key(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::PARAMETER_KEY)]);
  SArray<double> parameter_value(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::PARAMETER_VALUE)]);

  // check
  CHECK_EQ(parameter_key.size(), parameter_value.size());

  SArray<V> val;
  // auto op = [](const V* src, V* dst) { *dst = *src; };
  size_t n = parallelOrderedMatch(
    parameter_key, parameter_value, recv_key, OpAssign<V>(), FLAGS_num_threads, &val);
  if (!get(msg).is_validation()) {
    // for training data, the size of recv_key and the matched count must be identical
    // for validation data, such equality cannot be guaranteed
    CHECK_EQ(n, val.size());
  }

  msg->clearValue();
  msg->addValue(val);
}

// partition is a sorted key ranges
template <typename K, typename V>
MessagePtrList KVVector<K,V>::slice(const MessagePtr& msg, const KeyList& sep) {
  return sliceKeyOrderedMsg<K>(msg, sep);
}

} // namespace PS
