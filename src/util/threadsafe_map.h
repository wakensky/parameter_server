#pragma once

#include <map>
#include <mutex>
#include <condition_variable>
#include <memory>

namespace PS {

template <typename K, typename V>
class ThreadsafeMap {
public:
  ThreadsafeMap();
  ~ThreadsafeMap();
  ThreadsafeMap(const ThreadsafeMap& other) = delete;
  ThreadsafeMap& operator= (const ThreadsafeMap& rhs) = delete;

  // add {K, V} into map
  // modify the old value if key already exists
  void addAndModify(const K& key, const V& val);

  // add {K, V} into map
  // return false if key already exists
  bool addWithoutModify(const K& key, const V& val);

  // get value with specific key
  // return false if key is absence
  bool tryGet(const K& in_key, V& out_val);

  // Will not return until element with in_key comes up
  void waitAndGet(const K& in_key, V& out_val);

  // return true if key exists
  bool test(const K& key);

private: // methods


private: // attributes
  std::map<K, V> map_;
  mutable std::mutex mu_;
  std::condition_variable cond_;

}; // class ThreadsafeMap

template <typename K, typename V>
ThreadsafeMap<K, V>::ThreadsafeMap() {
  // do nothing
}

template <typename K, typename V>
ThreadsafeMap<K, V>::~ThreadsafeMap() {
  // do nothing
}

template <typename K, typename V>
void ThreadsafeMap<K, V>::addAndModify(const K& key, const V& val) {
  Lock l(mu_);
  map_[key] = val;
  cond_.notify_all();
  return;
}

template <typename K, typename V>
bool ThreadsafeMap<K, V>::addWithoutModify(const K& key, const V& val) {
  Lock l(mu_);

  if (map_.end() == map_.find(key)) {
    // insert
    map_[key] = val;
    cond_.notify_all();
    return true;
  } else {
    return false;
  }
}

template <typename K, typename V>
bool ThreadsafeMap<K, V>::tryGet(const K& in_key, V& out_val) {
  Lock l(mu_);

  auto iter = map_.find(key);
  if (map_.end() == iter) {
    return false;
  }

  out_val = iter->second;
  return true;
}

template <typename K, typename V>
void ThreadsafeMap<K, V>::waitAndGet(const K& in_key, V& out_val) {
  Lock l(mu_);
  cond_.wait(l, [this]() {return map_.end() != map_.find(in_key)});

  out_val = map_.at(in_key);
  return;
}

template <typename K, typename V>
bool ThreadsafeMap<K, V>::test(const K& key) {
  Lock l(mu_);
  return map_.end() != map_.find(key);
}

}; // namespace PS
