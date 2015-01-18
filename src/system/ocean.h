#pragma once
#include <cstdint>
#include <cinttypes>
#include <atomic>
#include <array>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_hash_map.h>
#include "proto/task.pb.h"
#include "util/common.h"
#include "util/threadpool.h"
#include "base/sparse_matrix.h"
#include "system/path_picker.h"

namespace PS {

DECLARE_bool(verbose);
DECLARE_int32(in_memory_unit_limit);

class Ocean {
  public:
    using FullKey = uint64;
    using ShortKey = uint32;
    using Offset = size_t;
    using Value = double;
    using GroupID = int;
    using TaskID = int;

    enum class DataSource: unsigned char {
      FEATURE_KEY = 0,
      FEATURE_OFFSET,
      FEATURE_VALUE,
      PARAMETER_KEY,
      PARAMETER_VALUE,
      DELTA,
      NUM
    };

    // identity of column partition data unit
    struct UnitID {
      GroupID grp_id;
      Range<FullKey> global_range;

      UnitID() :
        grp_id(0),
        global_range() {
        // do nothing
      }
      UnitID(const int in_grp_id, const Range<FullKey>& in_global_range) :
        grp_id(in_grp_id),
        global_range(in_global_range) {
        // do nothing
      }
      UnitID(const UnitID& other) :
        grp_id(other.grp_id),
        global_range(other.global_range) {
        // do nothing
      }

      UnitID& operator= (const UnitID& rhs) {
        grp_id = rhs.grp_id;
        global_range = rhs.global_range;
        return *this;
      }
      bool operator< (const UnitID& rhs) const {
        return (grp_id < rhs.grp_id ||
                (grp_id == rhs.grp_id && global_range.begin() < rhs.global_range.begin()) ||
                (grp_id == rhs.grp_id && global_range.begin() == rhs.global_range.begin() &&
                 global_range.end() < rhs.global_range.end()));
      }
      bool operator== (const UnitID& rhs) const  {
        return (grp_id == rhs.grp_id &&
                global_range == rhs.global_range) ;
      }

      string toString() const {
        std::stringstream ss;
        ss << "grp: " << grp_id << ", global_range: " << global_range.toString();
        return ss.str();
      }
    };

    struct UnitIDHash {
      std::size_t operator() (const UnitID& unit_id) const {
        const std::size_t magic_num = 0x9e3779b9;
        std::size_t hash = 512927377;
        hash ^= std::hash<GroupID>()(unit_id.grp_id) +
          magic_num + (hash << 6) + (hash >> 2);
        hash ^= std::hash<FullKey>()(unit_id.global_range.begin()) +
          magic_num + (hash << 6) + (hash >> 2);
        hash ^= std::hash<FullKey>()(unit_id.global_range.end()) +
          magic_num + (hash << 6) + (hash >> 2);
        return hash;
      }

      std::size_t hash(const UnitID& unit_id) const {
        const std::size_t magic_num = 0x9e3779b9;
        std::size_t hash = 512927377;
        hash ^= std::hash<GroupID>()(unit_id.grp_id) +
          magic_num + (hash << 6) + (hash >> 2);
        hash ^= std::hash<FullKey>()(unit_id.global_range.begin()) +
          magic_num + (hash << 6) + (hash >> 2);
        hash ^= std::hash<FullKey>()(unit_id.global_range.end()) +
          magic_num + (hash << 6) + (hash >> 2);
        return hash;
      }

      bool equal(const UnitID& a, const UnitID& b) const {
        return a == b;
      }
    };

    struct DataPack {
      std::array<SArray<char>, static_cast<size_t>(DataSource::NUM)> arrays;

      void set(const DataSource target, SArray<char> in) {
        CHECK_LT(
          static_cast<size_t>(target),
          static_cast<size_t>(DataSource::NUM));
        arrays[static_cast<size_t>(target)] = in;
      }

      void clear() {
        for (auto& item : arrays) {
          item.clear();
        }
      }
    };

    // disk file path for each DataSource
    struct PathPack {
      std::array<string, static_cast<size_t>(DataSource::NUM)> path;

      void set(const DataSource target, const string& in) {
        CHECK_LT(
          static_cast<size_t>(target),
          static_cast<size_t>(DataSource::NUM));
        path[static_cast<size_t>(target)] = in;
      }
    };

  public:
    SINGLETON(Ocean);
    ~Ocean();
    Ocean(const Ocean& other) = delete;
    Ocean& operator= (const Ocean& rhs) = delete;

    // initialization
    void init(
      const string& identity, const LM::Config& conf,
      const Task& task, PathPicker* path_picker);

    // dump memory image into disk files for the group
    bool dump(
      const GroupID grp_id,
      SArray<FullKey> parameter_key,
      SArray<Value> parameter_value,
      SArray<Value> delta,
      SparseMatrixPtr<ShortKey, Value> matrix);

    // add prefetch job
    // prefetch may not start immediately if in_memory_job_limit reached
    void prefetch(
      const GroupID grp_id,
      const Range<FullKey>& global_range,
      const TaskID task_id);

    // read data from Ocean
    DataPack get(
      const GroupID grp_id,
      const Range<FullKey>& global_range,
      const TaskID task_id);

    // drop in-memory unit out of Ocean
    void drop(
      const GroupID grp_id,
      const Range<FullKey>& global_range,
      const TaskID task_id);

    SizeR fetchAnchor(
      const GroupID grp_id, const Range<FullKey>& global_range);

    // save {parameter_key, parameter_value} pairs to path
    bool saveModel(const string& path);

  private: // internal types
    enum class UnitStatus: unsigned char {
      INIT = 0,
      REGISTERED,
      LOADING,
      LOADED,
      DROPPING,
      DROPPED,
      NUM
    };

    using InUseTaskHashMap = tbb::concurrent_hash_map<TaskID, int>;
    struct UnitBody {
      PathPack path_pack;
      DataPack data_pack;
      UnitStatus status;
      SizeR in_group_anchor;
      // who are using this unit
      InUseTaskHashMap in_use_tasks;

      UnitBody():
        status(UnitStatus::INIT) {
        // do nothing
      }

      void setStatus(const UnitStatus new_status) {
        CHECK_LT(
          static_cast<size_t>(new_status),
          static_cast<size_t>(UnitStatus::NUM));
        if (static_cast<size_t>(new_status) > static_cast<size_t>(status) ||
            UnitStatus::DROPPED == status) {
          status = new_status;
        }
      }
      UnitStatus getStatus() {
        return status;
      }
    };

  private: // methods
    Ocean();
    void prefetchThreadFunc();
    // dump column partitioned SArray
    bool dumpColumnPartitionedSArray(
      SArray<char> in, const UnitID unit_id,
      const SizeR& anchor, const DataSource data_source,
      UnitBody* unit_body);
    // printable DataSource
    string printDataSource(const DataSource data_source);
    // read data from disk synchronously
    // output to corresponding UnitBody's data_pack
    bool loadFromDiskSynchronously(
      const UnitID unit_id,
      const PathPack& path_pack,
      DataPack* data_pack);

  private: // attributes
    string identity_;

    using UnitHashMap = tbb::concurrent_hash_map<UnitID, UnitBody, UnitIDHash>;
    std::unordered_map<GroupID, std::vector<Range<FullKey>>> group_partition_ranges_;
    UnitHashMap units_;

    using UnitPrefetchQueue = tbb::concurrent_queue<std::pair<UnitID, TaskID>>;
    UnitPrefetchQueue prefetch_queue_;
    std::mutex prefetch_queue_mu_;
    std::condition_variable prefetch_queue_not_empty_cv_;
    std::vector<std::thread> prefetch_threads_;

    std::unordered_map<UnitID, SizeR, UnitIDHash> anchors_;

    // switch for asynchronized threads
    std::atomic_bool go_on_;

    std::atomic_size_t in_memory_unit_count_;
    std::condition_variable in_memory_unit_not_full_cv_;
    std::mutex in_memory_limit_mu_;

    PathPicker* path_picker_;
    LM::Config conf_;







};  // class Ocean
};  // namespace PS
