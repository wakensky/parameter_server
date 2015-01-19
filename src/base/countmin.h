#pragma once

#include <math.h>
#include "base/shared_array_inl.h"
namespace PS {

template <typename K, typename V>
class CountMin {
 public:
  // TODO prefetch to accelerate the memory access
  bool empty() { return n_ == 0; }
  void clear() { data_.clear(); n_ = 0; }
  void resize(int n, int k, V v_max) {
    n_ = std::max(n, 64);
    data_.resize(n_);
    data_.setZero();
    k_ = std::min(30, std::max(1, k));
    v_max_ = v_max;
  }

  // void bulkInsert(const SArray<K>& key, const SArray<V>& count) {
  //   CHECK_GT(n_, 0);
  //   CHECK_EQ(key.size(), count.size());
  //   for (size_t i = 0; i < key.size(); ++i) {
  //     uint32 h = hash(key[i]);
  //     const uint32 delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
  //     for (int j = 0; j < k_; ++j) {
  //       data_[h % n_] += count[i];
  //       h += delta;
  //     }
  //   }
  // }

  void insert(const K& key, const V& count) {
    uint32 h = hash(key);
    const uint32 delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    for (int j = 0; j < k_; ++j) {
      V v = data_[h % n_];
      // to avoid overflow
      data_[h % n_] = count > v_max_ - v ? v_max_ : v + count;
      h += delta;
    }
  }

  V query(const K& key) const {
    V res = v_max_;
    uint32 h = hash(key);
    const uint32 delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
    for (int j = 0; j < k_; ++j) {
      res = std::min(res, data_[h % n_]);
      h += delta;
    }
    return res;
  }

 private:
  uint32 hash(const uint64& key) const {
    // similar to murmurhash
    const uint32 seed = 0xbc9f1d34;
    const uint32 m = 0xc6a4a793;
    const uint32 n = 8;  // sizeof uint64
    uint32 h = seed ^ (n * m);

    uint32 w = (uint32) key;
    h += w; h *= m; h ^= (h >> 16);

    w = (uint32) (key >> 32);
    h += w; h *= m; h ^= (h >> 16);
    return h;
  }

  SArray<V> data_;
  int n_ = 0;
  int k_ = 1;
  V v_max_ = 0;
};

} // namespace PS
