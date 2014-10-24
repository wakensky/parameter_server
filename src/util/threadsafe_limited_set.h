#pragma once

#include <set>
#include <mutex>
#include <condition_variable>
#include <memory>
#include "util/common.h"

namespace PS {
template <typename T>
class ThreadsafeLimitedSet {
public:
  ThreadsafeLimitedSet();
  ~ThreadsafeLimitedSet();
  ThreadsafeLimitedSet(const ThreadsafeLimitedSet& other) = delete;
  ThreadsafeLimitedSet& operator= (const ThreadsafeLimitedSet& rhs) = delete;

  void setMaxCapacity(const size_t capacity);

  // erase val if it exists
  void erase(const T& val);

  // add without modification
  // wait if full
  void waitAndAdd(const T& val, const size_t capacity);

private:
  std::mutex mu_;
  std::condition_variable full_cond_;
  std::map<T, size_t> map_;
  size_t cur_capacity_;
  size_t max_capacity_;
}; // class Threadsafelimitedset

template <typename T>
ThreadsafeLimitedSet<T>::ThreadsafeLimitedSet() :
  cur_capacity_(0),
  max_capacity_(1024) {
  // do nothing
}

template <typename T>
ThreadsafeLimitedSet<T>::~ThreadsafeLimitedSet() {
  // do nothing
}

template <typename T>
void ThreadsafeLimitedSet<T>::setMaxCapacity(const size_t capacity) {
  Lock l(mu_);
  max_capacity_ = capacity;
}

template <typename T>
void ThreadsafeLimitedSet<T>::erase(const T& val) {
  Lock l(mu_);
  map_.erase(val);
  return;
}

template <typename T>
void ThreadsafeLimitedSet<T>::waitAndAdd(const T& val, const size_t capacity) {
  std::unique_lock<std::mutex> l(mu_);
  CHECK_LE(capacity, max_capacity_);

  full_cond_.wait(l, [this, capacity]() {
    return capacity + cur_capacity_ <= max_capacity_;
  });

  if (map_.end() == map_.find(val)) {
    map_[val] = capacity;
  }

  return;
}

}; // namespace PS
