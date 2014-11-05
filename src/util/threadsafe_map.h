#pragma once

#include <map>
#include <mutex>
#include <condition_variable>
#include <memory>

namespace PS {

template <typename K, typename V, typename Cmp = std::less<K>>
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

  // pop out an element
  // return false if empty
  bool tryPop(K& out_key, V& out_val);

  // pop out an element
  // wait if empty
  void waitAndPop(K& out_key, V& out_val);

  // return immediately no matter whether key exists or not
  void erase(const K& key);

  size_t size();

  // return all keys and values
  std::vector<std::pair<K, V>> all();

private: // attributes
  std::map<K, V, Cmp> map_;
  mutable std::mutex mu_;
  std::condition_variable cond_;

}; // class ThreadsafeMap

template <typename K, typename V, typename Cmp>
ThreadsafeMap<K, V, Cmp>::ThreadsafeMap() {
  // do nothing
}

template <typename K, typename V, typename Cmp>
ThreadsafeMap<K, V, Cmp>::~ThreadsafeMap() {
  // do nothing
}

template <typename K, typename V, typename Cmp>
void ThreadsafeMap<K, V, Cmp>::addAndModify(const K& key, const V& val) {
  Lock l(mu_);
  map_[key] = val;
  cond_.notify_all();
  return;
}

template <typename K, typename V, typename Cmp>
bool ThreadsafeMap<K, V, Cmp>::addWithoutModify(const K& key, const V& val) {
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

template <typename K, typename V, typename Cmp>
bool ThreadsafeMap<K, V, Cmp>::tryGet(const K& in_key, V& out_val) {
  Lock l(mu_);

  auto iter = map_.find(in_key);
  if (map_.end() == iter) {
    return false;
  }

  out_val = iter->second;
  return true;
}

template <typename K, typename V, typename Cmp>
void ThreadsafeMap<K, V, Cmp>::waitAndGet(const K& in_key, V& out_val) {
  std::unique_lock<std::mutex> l(mu_);
  cond_.wait(l, [this, in_key]() { return map_.end() != map_.find(in_key); });

  out_val = map_.at(in_key);
  return;
}

template <typename K, typename V, typename Cmp>
bool ThreadsafeMap<K, V, Cmp>::test(const K& key) {
  Lock l(mu_);
  return map_.end() != map_.find(key);
}

template <typename K, typename V, typename Cmp>
bool ThreadsafeMap<K, V, Cmp>::tryPop(K& out_key, V& out_val) {
  Lock l(mu_);
  if (map_.empty()) {
    return false;
  }

  auto iter = map_.begin();
  out_key = iter->first;
  out_val = iter->second;
  map_.erase(iter);
  return true;
}

template <typename K, typename V, typename Cmp>
void ThreadsafeMap<K, V, Cmp>::waitAndPop(K& out_key, V& out_val) {
  std::unique_lock<std::mutex> lk(mu_);
  cond_.wait(lk, [this]() { return !map_.empty(); });

  auto iter = map_.begin();
  out_key = iter->first;
  out_val = iter->second;
  map_.erase(iter);
  return;
}

template <typename K, typename V, typename Cmp>
void ThreadsafeMap<K, V, Cmp>::erase(const K& key) {
  Lock l(mu_);
  map_.erase(key);
  return;
}

template <typename K, typename V, typename Cmp>
size_t ThreadsafeMap<K, V, Cmp>::size() {
  Lock l(mu_);
  return map_.size();
}

template <typename K, typename V, typename Cmp>
std::vector<std::pair<K, V>> ThreadsafeMap<K, V, Cmp>::all() {
  Lock l(mu_);

  std::vector<std::pair<K, V>> vec;
  for (const auto& item : map_) {
    vec.push_back(std::make_pair(item.first, item.second));
  }
  return vec;
}

}; // namespace PS
