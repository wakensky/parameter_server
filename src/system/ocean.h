#pragma once
#include <cstdint>
#include <cinttypes>
#include <atomic>
#include <array>
#include "util/common.h"
#include "util/threadsafe_queue.h"
#include "util/threadsafe_map.h"
#include "util/threadpool.h"
#include "proto/task.pb.h"

namespace PS {

DECLARE_bool(verbose);
DECLARE_int32(prefetch_job_limit);
DECLARE_bool(less_memory);

class Ocean {
  public:
    typedef uint32 KeyType;
    typedef size_t OffsetType;
    typedef double ValueType;
    typedef int GrpID;
    enum class DataType: unsigned char {
      FEATURE_KEY = 0,
      FEATURE_OFFSET,
      FEATURE_VALUE,
      PARAMETER_KEY,
      PARAMETER_VALUE,
      DELTA,
      NUM
    };

  public:
    SINGLETON(Ocean);
    ~Ocean();
    Ocean(const Ocean& other) = delete;
    Ocean& operator= (const Ocean& rhs) = delete;

    // initialization
    void init(const string& identity, const LM::Config& conf, const Task& task);

    // dump specific SArray data to disk or memory pool
    // return false on failure
    //  ex: disk is full
    //  ex: {grp_id, type} exists
    bool dump(SArray<char>& input, const GrpID grp_id, const Ocean::DataType type);

    // add prefetch job
    // prefetch may not start immediately if prefetch_job_limit reached
    void prefetch(const GrpID grp_id, const Range<KeyType>& key_range);

    // get needed SArray from memory pool
    SArray<KeyType> getParameterKey(const GrpID grp_id, const Range<KeyType>& range);
    SArray<ValueType> getParameterValue(const GrpID grp_id, const Range<KeyType>& range);
    SArray<ValueType> getDelta(const GrpID grp_id, const Range<KeyType>& range);
    SArray<KeyType> getFeatureKey(const GrpID grp_id, const Range<KeyType>& range);
    SArray<OffsetType> getFeatureOffset(const GrpID grp_id, const Range<KeyType>& range);
    SArray<ValueType> getFeatureValue(const GrpID grp_id, const Range<KeyType>& range);

    // notify Ocean that corresponding memory block could be released if its reference count
    //   decreases to zero
    void drop(const GrpID grp_id, const Range<KeyType>& range);

  private: // internal types
    enum class JobStatus: unsigned char {
      PENDING = 0,
      LOADING = 1,
      FAILED,
      LOADED,
      NUM
    };

    struct JobID {
      GrpID grp_id;
      Range<KeyType> range;

      JobID():
        grp_id(0),
        range() {
        // do nothing
      }

      JobID(const int in_grp_id, const Range<KeyType>& in_range) :
        grp_id(in_grp_id),
        range(in_range) {
        // do nothing
      }

      JobID(const JobID& other) :
        grp_id(other.grp_id),
        range(other.range) {
        // do nothing
      }

      JobID& operator= (const JobID& rhs) {
        grp_id = rhs.grp_id;
        range = rhs.range;
        return *this;
      }

      bool operator< (const JobID& rhs) const {
        if (grp_id < rhs.grp_id) {
          return true;
        } else if (range.begin() < rhs.range.begin()) {
          return true;
        } else if (range.end() < rhs.range.end()) {
          return true;
        }
        return false;
      }

      bool operator== (const JobID& rhs) const {
        return grp_id == rhs.grp_id && range == rhs.range;
      }
    };

    struct JobIDHash {
      std::size_t operator() (const JobID& job_id) const {
        const std::size_t magic_num = 0x9e3779b9;
        std::size_t hash = 512927377;
        hash ^= std::hash<int>()(job_id.grp_id) +
          magic_num + (hash << 6) + (hash >> 2);
        hash ^= std::hash<KeyType>()(job_id.range.begin()) +
          magic_num + (hash << 6) + (hash >> 2);
        hash ^= std::hash<KeyType>()(job_id.range.end()) +
          magic_num + (hash << 6) + (hash >> 2);
        return hash;
      }
    };

    struct JobInfo {
      JobStatus status;
      int ref_count;

      JobInfo() :
        status(JobStatus::PENDING),
        ref_count(0) {
        // do nothing
      }

      JobInfo(JobStatus in_status, int in_ref_count) :
        status(in_status),
        ref_count(in_ref_count) {
        // do nothing
      }

      JobInfo(const JobInfo& other) :
        status(other.status),
        ref_count(other.ref_count) {
        // do nothing
      }

      JobInfo& operator= (const JobInfo& rhs) {
        status = rhs.status;
        ref_count = rhs.ref_count;
        return *this;
      }

      void setStatus(const JobStatus new_status) {
        if (new_status > status && new_status < JobStatus::NUM) {
          status = new_status;
        }
        return;
      }

      void increaseRef() { ref_count++; }
      void decreaseRef() { ref_count--; }
    };

    struct LoadedData {
      SArray<KeyType> parameter_key;
      SArray<ValueType> parameter_value;
      SArray<ValueType> delta;
      SArray<KeyType> feature_key;
      SArray<OffsetType> feature_offset;
      SArray<ValueType> feature_value;

      LoadedData() {};

      LoadedData(const LoadedData& other) :
        parameter_key(other.parameter_key),
        parameter_value(other.parameter_value),
        delta(other.delta),
        feature_key(other.feature_key),
        feature_offset(other.feature_offset),
        feature_value(other.feature_value) {
        // do nothing
      }

      LoadedData& operator= (const LoadedData& rhs) {
        parameter_key = rhs.parameter_key;
        parameter_value = rhs.parameter_value;
        delta = rhs.delta;
        feature_key = rhs.feature_key;
        feature_offset = rhs.feature_offset;
        feature_value = rhs.feature_value;
        return *this;
      }
    };

    class JobInfoTable {
      public:
        JobInfoTable() {};
        ~JobInfoTable() {};

        // will insert new JobInfo if not exists
        void setStatus(const JobID job_id, const JobStatus new_status) {
          Lock l(table_mu_);
          table_[job_id].setStatus(new_status);
        }

        // return JobStatus::NUM if job_id does not exist
        // will not insert new JobInfo
        JobStatus getStatus(const JobID job_id) {
          Lock l(table_mu_);
          auto iter = table_.find(job_id);
          if (table_.end() != iter) {
            return iter->second.status;
          }
          return JobStatus::NUM;
        }

        // will insert new JobInfo if not exists
        void increaseRef(const JobID job_id) {
          Lock l(table_mu_);
          table_[job_id].increaseRef();
        }

        // will not insert new JobInfo
        void decreaseRef(const JobID job_id) {
          Lock l(table_mu_);
          auto iter = table_.find(job_id);
          if (table_.end() != iter) {
            iter->second.decreaseRef();
          }
        }

        // get reference count
        // return 0:
        //  if job_id not exists
        //  if the job is not used by anybody
        int getRef(const JobID job_id) {
          Lock l(table_mu_);
          auto iter = table_.find(job_id);
          if (table_.end() != iter) {
            return iter->second.ref_count;
          }
          return 0;
        };

        // remove job information
        void erase(const JobID job_id) {
          Lock l(table_mu_);
          table_.erase(job_id);
        }
      private:
        std::unordered_map<JobID, JobInfo, JobIDHash> table_;
        std::mutex table_mu_;
    };

  private: // methods
    Ocean();

    // add new directory
    // return false on:
    //  1. dir is not a regular directory
    //  2. I have no read & write permission
    bool addDirectory(const string& dir);

    // pick a random directory from directories_
    string pickDirRandomly();

    void prefetchThreadFunc();

    // dump an range of input into Ocean
    // true on success
    bool dumpSarraySegment(
      SArray<char>& input,
      const GrpID grp_id,
      const Range<KeyType>& global_range,
      const DataType type);

    // read data from disk
    LoadedData loadFromDiskSynchronously(const JobID job_id);

    // make sure that what I want resides in loaded_data_ already
    // true on Success
    void makeMemoryDataReady(const JobID job_id);

    string dataTypeToString(const Ocean::DataType type);

  private: // attributes
    string identity_;
    // all block partitions
    //  includes group ID and key range within that group
    std::unordered_map<GrpID, std::vector<Range<KeyType>>> partition_info_;
    // maintaining all dumped file paths
    std::array<
      ThreadsafeMap<JobID, string>,
      static_cast<size_t>(Ocean::DataType::NUM)> lakes_;
    // pending jobs
    threadsafe_queue<JobID> pending_jobs_;
    // Job status
    JobInfoTable job_info_table_;
    // memory pool for all loaded data
    ThreadsafeMap<JobID, LoadedData> loaded_data_;
    // JobID -> Column range
    // generate by w_::key
    ThreadsafeMap<JobID, SizeR> column_ranges_;
    // available directories
    //  you may take the advantage of multi-disks and multi-threaded prefetch
    std::vector<string> directories_;
    // running permission for prefetch threads
    std::atomic_bool go_on_prefetching_;
    std::vector<std::thread> thread_vec_;
    const string log_prefix_;
    std::default_random_engine rng_;
    std::mutex general_mu_;
    std::mutex prefetch_limit_mu_;
    std::condition_variable prefetch_limit_cond_;
}; // class Ocean
}; // namespace PS
