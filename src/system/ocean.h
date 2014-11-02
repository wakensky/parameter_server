#pragma once
#include <atomic>
#include <array>
#include "util/common.h"
#include "util/threadsafe_queue.h"
#include "util/threadsafe_map.h"
#include "util/threadsafe_limited_set.h"
#include "util/threadpool.h"
#include "proto/linear_method.pb.h"
#include "base/sparse_matrix.h"

namespace PS {

DECLARE_int32(prefetch_job_limit);
DECLARE_bool(less_memory);

class Ocean {
  public:
    typedef uint32 KeyType;
    typedef size_t OffsetType;
    typedef double ValueType;
    typedef int GrpID;
    typedef uint128_t JobID;
    enum class DataType: unsigned char {
      FEATURE_KEY = 0,
      FEATURE_OFFSET,
      FEATURE_VALUE,
      PARAMETER_KEY,
      PARAMETER_VALUE,
      DELTA,
      NUM
    };
    enum class JobStatus: unsigned char {
      PENDING = 0,
      LOADING = 1,
      FAILED,
      LOADED,
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
    bool dump(const SArray<char>& input, const GrpID grp_id, const Ocean::DataType type);

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
        JobInfo() {};
        ~JobInfo() {};

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
            return iter->status;
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
            iter->decreaseRef();
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
            return iter->ref_count;
          }
          return 0;
        };

        // remove job information
        void erase(const JobID job_id) {
          Lock l(table_mu_);
          table_.erase(job_id);
        }
      private:
        std::unordered_map<JobID, JobInfo> table_;
        std::mutex table_mu_;
    };

  private: // methods
    Ocean();

    // assemble JobsID
    // from significant bits to lower bits
    //  10 bits: grp_id
    //  54 bits: range size
    //  64 bits: range begin
    JobID makeJobID(GrpID grp_id, const Range<KeyType>& range);
    // restore group id and range from JobID
    void parseJobID(const JobID job_id, GrpID& grp_id, Range<KeyType>& range);

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
      const SArray<char>& input,
      const GrpID grp_id,
      const Range<KeyType>& global_range,
      const DataType type);

    // read data from disk
    LoadedData loadFromDiskSynchronously(const JobID job_id);

    // make sure that what I want resides in loaded_data_ already
    // true on Success
    void makeMemoryDataReady(const JobID job_id);

  private: // attributes
    string identity_;
    // all block partitions
    //  includes group ID and key range within that group
    std::unordered_map<GrpID, std::vector<Range<KeyType>>> partition_info_;
    // maintaining all dumped file paths
    std::array<ThreadsafeMap<JobID, string>, Ocean::DataType::NUM> lakes_;
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
