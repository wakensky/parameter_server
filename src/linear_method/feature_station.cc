#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdexcept>
#include "linear_method/feature_station.h"
#include "base/shared_array_inl.h"

namespace PS {

DEFINE_int32(prefetch_mem_limit_mb, 1024,
  "memory usage limit (in MBytes) while prefetching training data "
  "in the process of UPDATE_MODEL");
DEFINE_bool(mmap_training_data, false,
  "move training data to disk");
DECLARE_bool(verbose);
DECLARE_int32(num_threads);

FeatureStation::FeatureStation() :
  go_on_prefetching_(true) {
  prefetch_mem_record_.setMaxCapacity(FLAGS_prefetch_mem_limit_mb);

  // launch prefetch threads
  for (int i = 0; i < FLAGS_num_threads; ++i) {
    thread_vec_.push_back(std::move(
      std::thread(&FeatureStation::prefetchThreadFunc, this)));
  }
}

FeatureStation::~FeatureStation() {
  // stop prefetching
  go_on_prefetching_ = false;
  for (auto& thread : thread_vec_) {
    for (int i = 0; i < FLAGS_num_threads * 64; ++i) {
      // pump in a lot of illegal prefetch jobs
      // let prefetch threads move on
      prefetch(0, 0, SizeR());
    }
    thread.join();
  }

  // munmap
  int grp_id = 0;
  DataSourceCollection dsc;
  while (grp_to_data_source_.tryPop(grp_id, dsc)) {
    if (DataSourceType::MMAP == dsc.type) {
      if (nullptr != dsc.colidx.first) {
        munmap((void*)dsc.colidx.first, dsc.colidx.second);
      }
      if (nullptr != dsc.rowsiz.first) {
        munmap((void*)dsc.rowsiz.first, dsc.rowsiz.second);
      }
      if (nullptr != dsc.value.first) {
        munmap((void*)dsc.value.first, dsc.value.second);
      }
    }
  }
}

bool FeatureStation::addFeatureGrp(
  const int grp_id, const MatrixPtr<ValType> feature) {
  if (!feature) {
    return true;
  }

  if (!FLAGS_mmap_training_data) {
    // simply store all training in memory
    memory_features_.addWithoutModify(grp_id, feature);
    return true;
  }

  // dump feature group to HDD
  string file_path = dumpFeature(grp_id, feature);
  if (file_path.empty()) {
    LL << "dumpFeature grp_id[" << grp_id << "] failed";
    return false;
  }
  if (FLAGS_verbose) {
    LI << "dumped grp[" << grp_id << "] at [" << file_path << "]";
  }

  // map files into DataSourceCollection
  DataSourceCollection dsc = mapFiles(file_path);
  if (!dsc) {
    LL << "mapFiles failed. path [" << file_path << "]";
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
    LL << "dir [" << dir << "] cannot be added since error [" <<
      strerror(errno) << "]";
    return false;
  }
  if (!S_ISDIR(st.st_mode)) {
    LL << "dir [" << dir << "] is not a regular directory";
    return false;
  }
  if (0 != access(dir.c_str(), R_OK | W_OK)) {
    LL << "I donnot have read&write permission on dir [" << dir << "]";
    return false;
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
  string prefix = pickDirRandomly() + "/feature.slot_" + std::to_string(grp_id);

  // dump
  if (!feature->writeToBinFile(prefix)) {
    return "";
  }

  return prefix;
}

FeatureStation::DataSourceCollection FeatureStation::mapFiles(
  const string& file_prefix) {
  DataSourceCollection dsc(DataSourceType::MMAP);
  dsc.colidx = mapOneFile(file_prefix + ".colidx");
  dsc.rowsiz = mapOneFile(file_prefix + ".rowsiz");
  dsc.value = mapOneFile(file_prefix + ".value");

  return dsc;
}

FeatureStation::DataSource FeatureStation::mapOneFile(
  const string& full_file_path) {
  // open
  int fd = ::open(full_file_path.c_str(), O_RDONLY);
  if (-1 == fd) {
    LL << "mapOnefile [" << full_file_path << "] failed. error [" <<
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

    // mmap
    const char* mmap_ptr = (const char*)mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (MAP_FAILED == mmap_ptr) {
      throw std::runtime_error("mmap failed");
    }
    close(fd);

    return makeDataSource(mmap_ptr, st.st_size);
  } catch (std::exception& e) {
    LL << "file [" << full_file_path << "] mapOneFile failed. error [" <<
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

  std::default_random_engine rng(time(nullptr));
  std::uniform_int_distribution<size_t> distribution(0, directories_.size() - 1);
  const size_t random_idx = distribution(rng);
  return directories_.at(random_idx);
}

void FeatureStation::prefetch(
  const int task_id, const int grp_id, const SizeR range) {
  if (!FLAGS_mmap_training_data) {
    return;
  }

  PrefetchJob new_job(task_id, grp_id, range, estimateRangeMemSize(grp_id, range));
  pending_jobs_.addWithoutModify(task_id, new_job);
  return;
}

void FeatureStation::prefetchThreadFunc() {
  while (go_on_prefetching_) {
    // take out a job
    int task_id = 0;
    PrefetchJob job;
    pending_jobs_.waitAndPop(task_id, job);
    if (!loading_jobs_.addWithoutModify(job.task_id, job) ||
        loaded_features_.test(job.task_id)) {
      // task_id is being loaded
      // or
      // task_id has been loaded before
      // simply drop the PrefetchJob
      continue;
    }

    // check memory usage
    // If memory exceeds limit, I will wait
    //   until dropFeature frees some memory
    prefetch_mem_record_.waitAndAdd(job.task_id, job.mem_size / 1024 / 1024);

    // prefetch
    MatrixPtr<ValType> feature = assembleFeatureMatrix(job);
    if (!feature) {
      LL << "assembleFeatureMatrix failed. [" << job.shortDebugString() << "]";
    }

    // store in loaded_features_
    loaded_features_.addWithoutModify(job.task_id, feature);

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
  const OffType* full_offset = reinterpret_cast<const OffType*>(dsc.rowsiz.first);
  CHECK(nullptr != full_offset);
  SArray<OffType> offset(job.range.end() + 1 - job.range.begin());
  offset.copyFrom(
    full_offset + job.range.begin(),
    offset.size());

  // load index
  const KeyType* full_index = reinterpret_cast<const KeyType*>(dsc.colidx.first);
  CHECK(nullptr != full_index);
  SArray<KeyType> index(offset.back() - offset.front());
  index.copyFrom(
    full_index + offset[0],
    index.size());

  // load value if necessary
  const ValType* full_value = reinterpret_cast<const ValType*>(dsc.value.first);
  SArray<ValType> value;
  if (nullptr != full_value) {
    value.resize(index.size());
    value.copyFrom(
      full_value + offset[0],
      value.size());
  }

  // localize offset
  // TODO performance issue, maybe
  offset.eigenArray() -= offset[0];

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
  const char* in_ptr, const size_t in_size) {
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
  const OffType* full_offset = reinterpret_cast<const OffType*>(dsc.rowsiz.first);
  if (nullptr == dsc.rowsiz.first) {
    LL << "requesting an empty dsc.rowsiz. grp_id [" << grp_id << "]";
    return 0;
  }
  const size_t full_offset_length = dsc.rowsiz.second / sizeof(OffType);
  if (range.end() >= full_offset_length) {
    LL << "illegal range:[" << range.begin() << "," << range.end() <<
      ") offset_len:" << full_offset_length;
    return 0;
  }

  // estimate memory usage
  // NOTE:
  //    On MMAP mode, full_offset resides on disk.
  //    I hope two random reads here won't hurt performance too much.
  const size_t count = full_offset[range.end()] - full_offset[range.begin()];
  size_t sum = (sizeof(KeyType) + sizeof(OffType)) * count;
  if (nullptr != dsc.value.first) {
    sum += sizeof(ValType) * count;
  }
  return sum;
}

MatrixPtr<FeatureStation::ValType> FeatureStation::getFeature(
  const int task_id, const int grp_id, const SizeR range) {
  MatrixPtr<ValType> ret_ptr;
  if (!FLAGS_mmap_training_data) {
    memory_features_.tryGet(grp_id, ret_ptr);
    if (ret_ptr) {
      ret_ptr = ret_ptr->colBlock(range);
    }
    return ret_ptr;
  }

  if (loaded_features_.tryGet(task_id, ret_ptr)) {
    // What I want is ready, simply return it
  } else if (loading_jobs_.test(task_id)) {
    // What I want is being loaded, wait
    loaded_features_.waitAndGet(task_id, ret_ptr);
  } else {
    // Nobody cares about me
    // I must load all by myself
    size_t mem_size = estimateRangeMemSize(grp_id, range);
    PrefetchJob job(task_id, grp_id, range, mem_size);
    ret_ptr = assembleFeatureMatrix(job);
    if (!ret_ptr) {
      LL << "getFeature encounters an error while loading job[" <<
        job.shortDebugString() << "] synchronously";
    }
    loaded_features_.addWithoutModify(task_id, ret_ptr);
  }

  return ret_ptr;
}

void FeatureStation::dropFeature(const int task_id) {
  if (!FLAGS_mmap_training_data) {
    return;
  }

  loaded_features_.addAndModify(task_id, MatrixPtr<ValType>());
  prefetch_mem_record_.erase(task_id);
}

size_t FeatureStation::pendingPrefetchJobCount() {
  return pending_jobs_.size();
}

bool FeatureStation::taskIDUsed(const int task_id) {
  return pending_jobs_.test(task_id) ||
    loading_jobs_.test(task_id) ||
    loaded_features_.test(task_id);
}

}; // namespace PS
