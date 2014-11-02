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
    bool dump(const SArray<char>& input, GrpID grp_id, Ocean::DataType type);

    // add prefetch job
    // prefetch may not start immediately if prefetch_job_limit reached
    void prefetch(GrpID grp_id, const Range<FeatureKeyType>& key_range);

  private: // internal types
    struct JobInfo {
      JobStatus status;
      int ref_count;
      std::mutex mu;

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
        Lock l(mu);

        status = rhs.status;
        ref_count = rhs.ref_count;
        return *this;
      }

      void setStatus(const JobStatus new_status) {
        if (new_status > status && new_status < JobStatus::NUM) {
          Lock l(mu);
          status = new_status;
        }
        return;
      }

      void increaseRef() {
        Lock l(mu);
        ref_count++;
      }

      void decreaseRef() {
        Lock l(mu);
        ref_count--;
      }
    };

    struct LoadedData {
      // LoadedData invalid if parameter_key is empty
      SArray<ParameterKeyType> parameter_key;
      SArray<ParameterValueType> parameter_value;
      SArray<DeltaType> delta;
      SparseMatrixPtr<FeatureKeyType, FeatureValueType> feature_matrix;

      LoadedData(const LoadedData& other) = delete;
      LoadedData& operator= (const LoadedData& rhs) = delete;
    };

  private: // methods
    Ocean();

    // assemble JobsID
    // from significant bits to lower bits
    //  10 bits: grp_id
    //  54 bits: range size
    //  64 bits: range begin
    JobID makeJobID(GrpID grp_id, const Range<FeatureKeyType>& range);
    // restore group id and range from JobID
    void parseJobID(const JobID job_id, GrpID& grp_id, Range<FeatureKeyType>& range);

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
      const Range<FeatureKeyType>& global_range,
      const DataType type);

  private: // attributes
    string identity_;
    // all block partitions
    //  includes group ID and key range within that group
    std::unordered_map<GrpID, std::vector<Range<FeatureKeyType>>> partition_info_;
    // maintaining all dumped file paths
    std::array<ThreadsafeMap<JobID, string>, Ocean::DataType::NUM> lakes_;
    // pending jobs
    threadsafe_queue<JobID> pending_jobs_;
    // Job Status
    ThreadsafeMap<JobID, JobInfo> job_status_;
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
}; // class Ocean
}; // namespace PS
