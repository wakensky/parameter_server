#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdexcept>
#include <malloc.h>
#include <gperftools/malloc_extension.h>
#include "linear_method/feature_station.h"
#include "base/shared_array_inl.h"

namespace PS {

DEFINE_int32(prefetch_mem_limit_kb, 1024 * 1024,
  "memory usage limit (in KiloBytes) while prefetching training data "
  "in the process of UPDATE_MODEL");
DEFINE_bool(mmap_training_data, false,
  "move training data to disk");
DEFINE_bool(prefetch_detail, false,
  "print detailed prefetch log. enabled with -verbose");
DECLARE_bool(verbose);
DECLARE_int32(num_threads);

FeatureStation::FeatureStation() :
  loaded_features_mem_size_(0),
  memory_features_mem_size_(0),
  go_on_prefetching_(true),
  log_prefix_("[FeatureStation] "),
  max_task_id_(0),
  rng_(time(0)) {
  prefetch_mem_record_.setMaxCapacity(FLAGS_prefetch_mem_limit_kb);

  // launch prefetch threads
  for (int i = 0; i < FLAGS_num_threads; ++i) {
    thread_vec_.push_back(std::move(
      std::thread(&FeatureStation::prefetchThreadFunc, this)));
  }

  if (!FLAGS_verbose) {
    FLAGS_prefetch_detail = false;
  }
}

FeatureStation::~FeatureStation() {
  // stop prefetching
  go_on_prefetching_ = false;
  for (auto& thread : thread_vec_) {
    for (int i = 0; i < FLAGS_num_threads + 1; ++i) {
      // pump in a lot of illegal prefetch jobs
      // let prefetch threads move on
      prefetch(i + 1, 0, SizeR());
    }
    thread.join();
  }
}

void FeatureStation::init(const string& identity, const LM::Config& conf) {
  identity_ = identity;

  for (int i = 0; i < conf.local_cache().file_size(); ++i) {
    addDirectory(conf.local_cache().file(i));
  }

  CHECK(!identity_.empty());
  CHECK(!directories_.empty());

  return;
}

bool FeatureStation::addFeatureGrp(
  const int grp_id, const MatrixPtr<ValType> feature) {
  if (!feature || feature->empty()) {
    return true;
  }
  CHECK(!identity_.empty());

  if (!FLAGS_mmap_training_data) {
    // simply store all training in memory
    if (memory_features_.addWithoutModify(grp_id, feature)) {
      memory_features_mem_size_ += feature->memSize();
    }
    grp_to_matrix_info_.addWithoutModify(grp_id, feature->info());
    return true;
  }

  // dump feature group to HDD
  string file_path = dumpFeature(grp_id, feature);
  if (file_path.empty()) {
    LL << log_prefix_ << "dumpFeature grp_id[" << grp_id << "] failed";
    return false;
  }
  if (FLAGS_verbose) {
    LI << log_prefix_ << "dumped grp[" << grp_id << "] to [" << file_path << "]";
  }

  // map files into DataSourceCollection
  DataSourceCollection dsc = mapFiles(
    file_path,
    std::static_pointer_cast<SparseMatrix<KeyType, ValType>>(feature)->binary());
  if (!dsc) {
    LL << log_prefix_ << "mapFiles failed. path [" << file_path << "]";
    return false;
  }

  grp_to_data_source_.addWithoutModify(grp_id, dsc);
  grp_to_matrix_info_.addWithoutModify(grp_id, feature->info());

  return true;
}

bool FeatureStation::addDirectory(const string& dir) {
  Lock l(general_mu_);

  // check permission
  struct stat st;
  if (-1 == stat(dir.c_str(), &st)) {
    LL << log_prefix_ << "dir [" << dir << "] cannot be added since error [" <<
      strerror(errno) << "]";
    return false;
  }
  if (!S_ISDIR(st.st_mode)) {
    LL << log_prefix_ << "dir [" << dir << "] is not a regular directory";
    return false;
  }
  if (0 != access(dir.c_str(), R_OK | W_OK)) {
    LL << log_prefix_ << "I donnot have read&write permission on dir [" << dir << "]";
    return false;
  }

  if (FLAGS_verbose) {
    LI << log_prefix_ << "dir [" << dir << "] has been added to featureStation";
  }

  // add
  directories_.push_back(dir);
  return false;
}

string FeatureStation::dumpFeature(
  const int grp_id, const MatrixPtr<ValType> feature) {
  if (!feature) {
    return "";
  }

  // target filename (directory + prefix)
  string prefix = pickDirRandomly() + "/" + identity_ +
    ".feature.slot_" + std::to_string(grp_id);

  // dump
  if (!feature->writeToBinFile(prefix)) {
    return "";
  }

  return prefix;
}

FeatureStation::DataSourceCollection FeatureStation::mapFiles(
  const string& file_prefix, const bool binary) {
  DataSourceCollection dsc(DataSourceType::MMAP);
  dsc.colidx = mapOneFile(file_prefix + ".index");
  dsc.rowsiz = mapOneFile(file_prefix + ".offset");
  if (!binary) {
    dsc.value = mapOneFile(file_prefix + ".value");
  }

  return dsc;
}

FeatureStation::DataSource FeatureStation::mapOneFile(
  const string& full_file_path) {
  int fd = ::open(full_file_path.c_str(), O_RDONLY);
  if (-1 == fd) {
    LL << log_prefix_ << "mapOnefile [" << full_file_path << "] failed. error [" <<
      strerror(errno) << "]";
    return makeDataSource();
  }

  try {
    // check file type, get file size
    struct stat st;
    if (-1 == fstat(fd, &st)) {
      throw std::runtime_error("fstat failed");
    }
    if (!S_ISREG(st.st_mode)) {
      throw std::runtime_error("not a regular file");
    }
    close(fd);
    return makeDataSource(full_file_path, st.st_size);
  } catch (std::exception& e) {
    LL << log_prefix_ << "file [" << full_file_path <<
      "] mapOneFile failed. error [" <<
      e.what() << "] description [" << strerror(errno) << "]";
    if (fd > 0) {
      close(fd);
    }
  }

  return makeDataSource();
}

string FeatureStation::pickDirRandomly() {
  Lock l(general_mu_);
  if (directories_.empty()) {
    return "";
  }

  std::uniform_int_distribution<size_t> distribution(0, directories_.size() - 1);
  const size_t random_idx = distribution(rng_);
  return directories_.at(random_idx);
}

void FeatureStation::prefetch(
  const int task_id, const int grp_id, const SizeR range) {
  if (!FLAGS_mmap_training_data) {
    return;
  }

  if (task_id > max_task_id_) {
    max_task_id_ = task_id;
  }

  PrefetchJob new_job(task_id, grp_id, range, estimateRangeMemSize(grp_id, range));
  pending_jobs_.addWithoutModify(task_id, new_job);

  if (FLAGS_prefetch_detail) {
    LI << log_prefix_ << "add PrefetchJob [" <<
      new_job.shortDebugString() << "]";
  }
  return;
}

void FeatureStation::prefetchThreadFunc() {
  while (go_on_prefetching_) {
    // take out a job
    int task_id = 0;
    PrefetchJob job;
    pending_jobs_.waitAndPop(task_id, job);

    // check memory usage
    // If memory exceeds limit, I will wait
    //   until dropFeature frees some memory
    prefetch_mem_record_.waitAndAdd(job.task_id, job.mem_size / 1024);

    if (!loading_jobs_.addWithoutModify(job.task_id, job) ||
        loaded_features_.test(job.task_id)) {
      // task_id is being loaded
      // or
      // task_id has been loaded before
      // simply drop the PrefetchJob
      prefetch_mem_record_.erase(job.task_id);
      continue;
    }

    if (FLAGS_prefetch_detail) {
      LI << log_prefix_ << "prefetching PrefetchJob [" <<
        job.shortDebugString() << "]";
    }
    // prefetch
    MatrixPtr<ValType> feature = assembleFeatureMatrix(job);
    if (feature) {
      // store in loaded_features_
      if (loaded_features_.addWithoutModify(job.task_id, feature)) {
        loaded_features_mem_size_ += feature->memSize();
        // wakensky
        LI << "wakensky; loaded_size:" << loaded_features_mem_size_ <<
          " prefetched_size:" << feature->memSize();
      }
    } else {
      LL << "assembleFeatureMatrix failed. [" << job.shortDebugString() << "]";
    }
    if (FLAGS_prefetch_detail) {
      LI << log_prefix_ << "prefetched PrefetchJob [" <<
        job.shortDebugString() << "]";
    }

    // remove from loading_jobs_
    loading_jobs_.erase(job.task_id);
  };
}

MatrixPtr<FeatureStation::ValType> FeatureStation::assembleFeatureMatrix(
  const PrefetchJob& job) {
  if (0 == job.mem_size) {
    LL << "empty job [" << job.shortDebugString() << "]";
    return MatrixPtr<ValType>();
  }

  DataSourceCollection dsc;
  if (!grp_to_data_source_.tryGet(job.grp_id, dsc)) {
    LL << "assembleFeatureMatrix cannot locate DataSourceCollection for grp_id [" <<
      job.grp_id << "]";
    return MatrixPtr<ValType>();
  }

  // load offset
  SArray<OffType> offset;
  SizeR range(job.range.begin(), job.range.end() + 1);
  CHECK(offset.readFromFile(range, dsc.rowsiz.first));

  // load index
  SArray<KeyType> index;
  range.set(offset.front(), offset.back());
  CHECK(index.readFromFile(range, dsc.colidx.first));

  // load value if necessary
  SArray<ValType> value;
  if (!dsc.value.first.empty()) {
    range.set(offset.front(), offset.back());
    CHECK(value.readFromFile(range, dsc.value.first));
  }

  // matrix info
  MatrixInfo info;
  CHECK(grp_to_matrix_info_.tryGet(job.grp_id, info));
  info.set_nnz(index.size());
  info.clear_ins_info();
  if (!info.row_major()) {
    SizeR range(0, job.range.end() - job.range.begin());
    range.to(info.mutable_col());
  }

  // assemble
  return MatrixPtr<ValType>(
    new SparseMatrix<KeyType, ValType>(info, offset, index, value));
}

FeatureStation::DataSource FeatureStation::makeDataSource(
  const string in_ptr, const size_t in_size) {
  return std::make_pair(in_ptr, in_size);
}

size_t FeatureStation::estimateRangeMemSize(const int grp_id, const SizeR range) {
  if (range.empty()) {
    return 0;
  }

  // locate DataSourceCollection
  DataSourceCollection dsc;
  if (!grp_to_data_source_.tryGet(grp_id, dsc)) {
    LL << "illegal grp_id [" << grp_id << "]";
    return 0;
  }

  // check range
  const size_t full_offset_length = dsc.rowsiz.second / sizeof(OffType);
  if (range.end() >= full_offset_length) {
    LL << "illegal range:[" << range.begin() << "," << range.end() <<
      ") offset_len:" << full_offset_length;
    return 0;
  }

  // NOTE:
  //    On MMAP mode, full_offset resides on disk.
  //    I hope two random reads here won't hurt performance too much.
  File* offset_file = File::openOrDie(dsc.rowsiz.first, "r");
  OffType offset_begin = 0;
  offset_file->seek(range.begin() * sizeof(OffType));
  offset_file->readOrDie(&offset_begin, sizeof(OffType));
  OffType offset_end = 0;
  offset_file->seek(range.end() * sizeof(OffType));
  offset_file->readOrDie(&offset_end, sizeof(OffType));
  offset_file->close();

  // estimate memory usage
  const size_t count = offset_end - offset_begin;
  size_t sum = sizeof(KeyType) * count;  // sizeof index
  sum += sizeof(OffType) * (range.end() - range.begin()); // sizeof offset
  if (!dsc.value.first.empty()) {
    sum += sizeof(ValType) * count; // sizeof value
  }
  return sum;
}

MatrixPtr<FeatureStation::ValType> FeatureStation::getFeature(
  const int task_id, const int grp_id, const SizeR range) {
  MatrixPtr<ValType> ret_ptr(nullptr);
  if (!FLAGS_mmap_training_data) {
    memory_features_.tryGet(grp_id, ret_ptr);
    if (ret_ptr && !range.empty()) {
      ret_ptr = ret_ptr->colBlock(range);
    }
    return ret_ptr;
  }

  if (task_id > max_task_id_) {
    max_task_id_ = task_id;
  }

  if (FLAGS_prefetch_detail) {
    LI << log_prefix_ << "getting task_id [" << task_id << "]";
  }

  if (loaded_features_.tryGet(task_id, ret_ptr)) {
    // What I want is ready, simply return it
  } else if (loading_jobs_.test(task_id)) {
    if (FLAGS_prefetch_detail) {
      LI << log_prefix_ << "waiting task_id [" << task_id << "]";
    }
    // What I want is being loaded, wait
    loaded_features_.waitAndGet(task_id, ret_ptr);
  } else {
    // Nobody cares about me
    // I must load all by myself
    size_t mem_size = estimateRangeMemSize(grp_id, range);
    PrefetchJob job(task_id, grp_id, range, mem_size);

    if (FLAGS_prefetch_detail) {
      LI << log_prefix_ << "loading job synchronously [" <<
        job.shortDebugString() << "]";
    }

    ret_ptr = assembleFeatureMatrix(job);
    if (ret_ptr) {
      if (loaded_features_.addWithoutModify(task_id, ret_ptr)) {
        loaded_features_mem_size_ += ret_ptr->memSize();
        // wakensky
        LI << "wakensky; loaded_size:" << loaded_features_mem_size_ <<
          " synchronously_size:" << ret_ptr->memSize();
      }
    } else {
      LL << "getFeature encounters an error while loading job[" <<
        job.shortDebugString() << "] synchronously";
    }
  }

  if (FLAGS_prefetch_detail) {
    LI << log_prefix_ << "got task_id [" << task_id << "]";
  }

  return ret_ptr;
}

void FeatureStation::dropFeature(const int task_id) {
  if (!FLAGS_mmap_training_data) {
    return;
  }

  MatrixPtr<ValType> feature_ptr(nullptr);
  if (loaded_features_.tryGet(task_id, feature_ptr)) {
    loaded_features_mem_size_ -= feature_ptr->memSize();
    // wakensky
    LI << "wakensky; dropFeature loaded_size:" << loaded_features_mem_size_ <<
      " dropped_size:" << feature_ptr->memSize();
  }
  loaded_features_.addAndModify(task_id, MatrixPtr<ValType>(nullptr));
  prefetch_mem_record_.erase(task_id);

  if (FLAGS_prefetch_detail) {
    LI << log_prefix_ << "dropped task_id [" << task_id << "]";
  }

#ifdef TCMALLOC
  // wakensky; tcmalloc force return memory to kernel
  MallocExtension::instance()->ReleaseFreeMemory();
#endif
#if 0
  int trim = malloc_trim(0);
  // wakensky
  if (trim) {
    LI << "malloc_trim returned";
  } else {
    LI << "malloc_trim not returned";
  }
#endif

  return;
}

size_t FeatureStation::pendingPrefetchJobCount() {
  return pending_jobs_.size();
}

bool FeatureStation::taskIDUsed(const int task_id) {
  return pending_jobs_.test(task_id) ||
    loading_jobs_.test(task_id) ||
    loaded_features_.test(task_id);
}

MatrixInfo FeatureStation::getMatrixInfo(const int grp_id) {
  MatrixInfo info;
  grp_to_matrix_info_.tryGet(grp_id, info);
  return info;
}

}; // namespace PS
