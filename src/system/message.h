#pragma once

#include "util/common.h"
#include "util/threadpool.h"
#include "base/shared_array.h"
#include "proto/task.pb.h"

namespace PS {

// typedef size_t NodeID;
typedef string NodeID;

struct Message {
  Message() { }
  explicit Message(const Task& tk) : task(tk) { }

  // content will be sent over network
  Task task;
  SArray<char> key;
  std::vector<SArray<char> > value;

  NodeID sender, recver, original_recver;

  // true if it is the first massege received in the time specified in task
  // bool first = false;

  // true if this message has been replied
  bool replied = false;
  // true if the task asscociated with this message if finished.
  bool finished = true;
  // an inivalid message will not be sent, instead, the postoffice will fake a
  // reply message. see Postoffice::queue()
  bool valid = true;

  // set it to be true to stop the sending thread of Postoffice.
  bool terminate = false;

  // clear the data, but keep all metadata
  void clearData() {
    key = SArray<char>();
    value.clear();
  }
  template <typename V>
  void addValue(const SArray<V>& val) {
    value.push_back(SArray<char>(val));
  }

  string shortDebugString() const {
    std::stringstream ss;
    ss << sender << "=>" << recver;
    if (!original_recver.empty()) ss << "(" << original_recver << ")";
    ss << ", T: " << task.time() << ", wait_T: " << task.wait_time()
       << ", " << key.size() << " keys, [" << value.size() << "] value: ";
    for (const auto& x: value)
      ss << x.size() << " ";
    ss << "[valid]:" << valid << " ";
    ss << "[finished]:" << finished << " ";
    ss << "[task]:" << task.ShortDebugString();
    return ss.str();
  }

  string debugString() const {
    std::stringstream ss;
    ss << "[message]: " << sender << "=>" << recver
       << "(" << original_recver << ")\n"
       << "[task]:" << task.ShortDebugString()
       << "\n[key]:" << key.size()
       << "\n[" << value.size() << " value]: ";
    for (const auto& x: value)
      ss << x.size() << " ";
    return ss.str();
  }

  // add the key list and the lists of values
  template <typename K, typename V>
  void addKV(const SArray<K>& k, const std::initializer_list<SArray<V>>& v) {
    key = SArray<char>(k);
    for (const auto& w : v) addValue(w);
  }
};

// an reply message, with empty body and task
static Message replyTemplate(const Message& msg) {
  Message reply;
  reply.sender = msg.recver;
  reply.recver = msg.sender;
  reply.task.set_customer(msg.task.customer());
  reply.task.set_request(false);
  return reply;
}

template <typename V> using AlignedArray = std::pair<SizeR, SArray<V>>;
template <typename V> using AlignedArrayList = std::vector<AlignedArray<V>>;

enum class MatchOperation : unsigned char {
  ASSIGN = 0,
  ADD,
  NUM
};

template <typename K, typename V>
static void match(
  const SizeR &dst_key_pos_range,
  const SArray<K> &dst_key,
  SArray<V> &dst_val,
  const SArray<K> &src_key,
  const SArray<V> &src_val,
  size_t *matched,
  const MatchOperation op) {
  *matched = 0;
  if (dst_key.empty() || src_key.empty()) {
    return;
  }

  std::unique_ptr<size_t[]> matched_array_ptr(new size_t[FLAGS_num_threads]);
  {
    // threads
    ThreadPool pool(FLAGS_num_threads);
    for (size_t thread_idx = 0; thread_idx < FLAGS_num_threads; ++thread_idx) {
      pool.add([&, thread_idx]() {
        // matched ptr
        size_t *my_matched = &(matched_array_ptr[thread_idx]);
        *my_matched = 0;

        // partition dst_key_pos_range evenly
        SizeR my_dst_key_pos_range = dst_key_pos_range.evenDivide(
          FLAGS_num_threads, thread_idx);
        // take the remainder if dst_key_range is indivisible by threads number
        if (FLAGS_num_threads - 1 == thread_idx) {
          my_dst_key_pos_range.set(
            my_dst_key_pos_range.begin(), dst_key_pos_range.end());
        }

        // iterators for dst
        const K *dst_key_it = dst_key.data() + my_dst_key_pos_range.begin();
        const K* dst_key_end = dst_key.data() + my_dst_key_pos_range.end();
        V *dst_val_it = dst_val.data() + (
          my_dst_key_pos_range.begin() - dst_key_pos_range.begin());

        // iterators for src
        // const K *src_key_it = src_key.data();
        // const V *src_val_it = src_val.data();
        const K *src_key_it = std::lower_bound(src_key.begin(), src_key.end(), *dst_key_it);
        const K *src_key_end = std::upper_bound(src_key.begin(), src_key.end(), *(dst_key_end - 1));
        const V *src_val_it = src_val.begin() + (src_key_it - src_key.begin());

        // clear dst_val if necessary
        if (MatchOperation::ASSIGN == op) {
          memset(dst_val_it, 0, sizeof(V) * (dst_key_end - dst_key_it));
        }

        // traverse
        // TODO src_key.end() could be lowered too
        while (dst_key_end != dst_key_it && src_key_end != src_key_it) {
          if (*src_key_it < *dst_key_it) {
            // forward iterators for src
            ++src_key_it;
            ++src_val_it;
          } else {
            if (!(*dst_key_it < *src_key_it)) {
              // equals
              if (MatchOperation::ASSIGN == op) {
                *dst_val_it = *src_val_it;
              } else if (MatchOperation::ADD == op) {
                *dst_val_it += *src_val_it;
              } else {
                LL << "BAD MatchOperation [" << static_cast<int32>(op) << "]";
                throw std::runtime_error("BAD MatchOperation");
              }

              // forward iterators for src
              ++src_key_it;
              ++src_val_it;
              ++(*my_matched);
            }

            // forward iterators for dst
            ++dst_key_it;
            ++dst_val_it;
          }
        }
      });
    }
    pool.startWorkers();
  }

  // reduce matched count
  for (size_t i = 0; i < FLAGS_num_threads; ++i) {
    *matched += matched_array_ptr[i];
  }

  return;
}

template <typename K , typename V>
static void newerMatch(
  SizeR &out_range,
  SArray<V> &dst_val,
  const SArray<K> &dst_key,
  const SArray<K> &src_key,
  const V *src_val,
  const Range<K> &src_key_range,
  size_t *matched,
  const MatchOperation op) {
  *matched = 0;
  if (dst_key.empty() || src_key.empty()) {
    // return empty range
    // do not modify out_val_array
    out_range = SizeR();
    return;
  }

  // range
  out_range = dst_key.findRange(src_key_range);
  if (dst_val.empty()) {
    return;
  }

  std::unique_ptr<size_t[]> matched_array_ptr(new size_t[FLAGS_num_threads]);
  {
    // multi threads
    ThreadPool pool(FLAGS_num_threads);
    size_t shard_size = src_key.size() / FLAGS_num_threads;
    for (size_t i = 0; i < FLAGS_num_threads; ++i) {
      // partition src_key evenly
      const K *my_src_key_beg = src_key.begin() + shard_size * i;
      const K *my_src_key_end = i < FLAGS_num_threads - 1 ?
        my_src_key_beg + shard_size :
        src_key.end();
      size_t *my_matched = &(matched_array_ptr[i]);
      pool.add([&dst_key, &src_key,
                my_src_key_beg, my_src_key_end, src_val,
                out_range, &dst_val, my_matched, op]() {
        *my_matched = 0;

        // location
        const K *dst_key_it = std::lower_bound(
          dst_key.begin(), dst_key.end(), *my_src_key_beg);
        V *dst_val_it = dst_val.begin() + (
          (dst_key_it - dst_key.begin()) - out_range.begin());
        const K *src_key_it = std::lower_bound(
          my_src_key_beg, my_src_key_end, *dst_key_it);
        const V *src_val_it = src_val + (src_key_it - src_key.begin());

        // traverse
        while (dst_key.end() != dst_key_it && my_src_key_end != src_key_it) {
          if (*src_key_it < *dst_key_it) {
            ++src_key_it;
            ++src_val_it;
          } else {
            if (!(*dst_key_it < *src_key_it)) {
              if (MatchOperation::ASSIGN == op) {
                *dst_val_it = *src_val_it;
              } else if (MatchOperation::ADD == op) {
                *dst_val_it += *src_val_it;
              } else {
                LL << "BAD MatchOperation [" << static_cast<uint32>(op) << "]";
                throw std::runtime_error("BAD MatchOperation");
              }
              ++src_key_it;
              ++src_val_it;
              ++(*my_matched);
            }
            ++dst_key_it;
            ++dst_val_it;
          }
        }

      });
    }
    pool.startWorkers();
  }

  // reduce matched count
  for (size_t i = 0; i < FLAGS_num_threads; ++i) {
    *matched += matched_array_ptr[i];
  }

  return;
}

template <typename K, typename V>
static AlignedArray<V> oldMatch(const SArray<K>& dst_key,
                             const SArray<K>& src_key,
                             V* src_val,
                             Range<K> src_key_range,
                             size_t* matched) {
  // if (src_key_range == Range<K>::all())
  //   src_key_range = src_key.range();
  *matched = 0;
  if (dst_key.empty() || src_key.empty()) {
    return std::make_pair(SizeR(), SArray<V>());
  }

  SizeR range = dst_key.findRange(src_key_range);

  SArray<V> value(range.size());
  V* dst_val = value.data();
  memset(dst_val, 0, sizeof(V)*value.size());

  // optimization, binary search the start point
  const K* dst_key_it = dst_key.begin() + range.begin();
  const K* src_key_it = std::lower_bound(src_key.begin(), src_key.end(), *dst_key_it);
  src_val += src_key_it - src_key.begin();
  while (dst_key_it != dst_key.end() && src_key_it != src_key.end()) {
    if (*src_key_it < *dst_key_it) {
      ++ src_key_it;
      ++ src_val;
    } else {
      if (!(*dst_key_it < *src_key_it)) {
        *dst_val = *src_val;
        ++ src_key_it;
        ++ src_val;
        ++ *matched;
      }
      ++ dst_key_it;
      ++ dst_val;
    }
  }
  return std::make_pair(range, value);
}



template <typename K, typename V>
Message slice(const Message& msg, const Range<K>& gr) {
  SArray<K> key(msg.key);
  SizeR lr = key.findRange(gr);
  // if (lr.empty()) {
  //   Message ret;
  //   ret.valid = false;
  //   return ret;
  // }

  Message ret = msg;
  ret.task.set_has_key(true);
  ret.key = key.segment(lr);
  ret.value.clear();
  for (auto& d : msg.value) {
    SArray<V> data(d);
    ret.value.push_back(SArray<char>(data.segment(lr)));
  }
  if (lr.empty()) ret.valid = false;
  return ret;
}

// template <typename V>
// struct AlignedSArray {
//   SizeR local;
//   std::vector<SArray<V> > data;
// };

// AlignedSArray<V> merge(const SArray<K> key, const Message& msg) {
//   AlignedSArray<V> res;

// }

// template <typename T> void add(const SArray<T> value) {
//   value_.push_back(SArray<char>(value));
// }

inline std::ostream& operator<<(std::ostream& os, const Message& msg) {
  return (os << msg.shortDebugString());
}


} // namespace PS
