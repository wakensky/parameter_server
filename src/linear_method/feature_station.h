#pragma once
#include "util/common.h"
#include "base/matrix.h"
#include "system/message.h"

namespace PS {

class FeatureStation {
  public:
    FeatureStation(const std::vector<string>& directories);
    ~FeatureStation();
    FeatureStation(const FeatureStation& other) = delete;
    FeatureStation& operator= (const FeatureStation& rhs) = delete;

    // dump specific feature group to disk
    // return false on failure
    //      for example: cannot write to disk file
    //      for example: grp_id has existed
    bool addFeatureGrp(const int grp_id, const MatrixPtr<double> feature);

    // add prefetch job
    // prefetch may not start immediately if memory usage reaches limit
    void prefetch(const int task_id, const int grp_id, const SizeR range);

    // get feature used by a specific task
    // returned matrix resides in memory
    //   something wrong if returned shared_ptr is empty
    MatrixPtr<double> getFeature(const int task_id);

    // frees memory space corresponds to a specific task
    void dropFeature(const int task_id);

  private: // internal types
    struct PrefetchJob {
      int task_id;
      int grp_id;
      SizeR range;
      size_t mem_size;

      PrefetchJob(
        const int in_task_id, const in_grp_id,
        const SizeR in_range, const size_t in_mem_size) :
          task_id(in_task_id),
          grp_id(in_grp_id),
          range(in_range),
          mem_size(in_mem_size) {
        // do nothing
      }

      PrefetchJob(const PrefetchJob& other) :
        task_id(other.task_id),
        grp_id(other.grp_id),
        range(other.range),
        mem_size(other.mem_size) {
        // do nothing
      }

      PrefetchJob& operator= (const PrefetchJob& rhs) :
        task_id(rhs.task_id),
        grp_id(rhs.grp_id),
        range(rhs.range),
        mem_size(rhs.mem_size) {
        // do nothing
      }
    }; // struct PrefetchJob

    enum class DataSourceType : unsigned char {
      MMAP = 0,
      MEMORY,
      NUM
    };

    // first: pointer to data; second: size in bytes
    // The pointer may refer to real memory region or a mapped file
    typedef std::pair<const char*, size_t> DataSource;

    // represents a specific feature group
    struct DataSourceCollection {
      DataSource colidx;
      DataSource rowsiz;
      DataSource value;
      DataSourceType type;

      DataSourceCollection(const DataSourceType in_type) :
        colidx(std::make_pair(nullptr, 0)),
        rowsiz(std::make_pair(nullptr, 0)),
        value(std::make_pair(nullptr, 0)),
        type(in_type) {
        // do nothing
      }

      DataSourceCollection(const DataSourceCollection& other) :
        colidx(other.colidx),
        rowsiz(other.rowsiz),
        value(other.value),
        type(other.type) {
        // do nothing
      }

      DataSourceCollection& operator= (const DataSourceCollection& rhs) :
        colidx(rhs.colidx),
        rowsiz(rhs.rowsiz),
        value(rhs.value),
        type(rhs.type) {
        // do nothing
      }

      bool operator! () const {
        return !valid();
      }

      bool valid() const {
        return nullptr != colidx.first ||
          nullptr != rowsiz.first ||
          nullptr != value.first;
      }
    }; // struct MapCollection

  private: // methods
    // assemble a DataSource
    DataSource makeDataSource(
      const char* in_ptr = nullptr, const size_t in_size = 0);

    // dump feature group to disk
    // return the path of file just created
    //  empty string on failure
    string dumpFeature(const int grp_id, const MatrixPtr<double> feature);

    // map colidx/rowsiz/value files to DataSourceCollection
    // failure if returned DataSourceCollection is invalid
    DataSourceCollection mapFiles(const string& file_prefix);

    // map a specific file into memory
    DataSource mapOneFile(const string& full_file_path);

    // pick a random directory from directories_
    string pickDirRandomly();

    // current memory usage
    size_t memSize();

    void prefetchThreadFunc();

  private: // attributes
    std::mutex general_mu_;
    std::unordered_map<int, DataSourceCollection> grp_to_data_source_;
    threadsafe_queue<PrefetchJob> pending_jobs_;
    threadsafe_map<PrefetchJob> loading_jobs_;
    // trace the memory usage of prefetch threads
    // {task_id, memory capacity}
    threadsafeLimitedSet<int> prefetch_mem_record_;
    // available directories
    //  you may take the advantage of multi-disks and multi-threaded prefetch
    std::vector<string> directories_;
    // stores all prefetched matrixes
    threadsafe_map<int, MatrixPtr<double> > loaded_features_;

}; // class Featurestation

}; // namespace PS
