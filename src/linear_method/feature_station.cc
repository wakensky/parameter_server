#include "linear_method/feature_station.h"

namespace PS {

DECLARE_bool(verbose);

FeatureStation::FeatureStation(
  const std::vector<string>& directories) :
  directories_(directories) {
  CHECK(!directories_.empty());
  prefetch_mem_record_.setMaxCapacity(FLAGS_prefetch_mem_limit_mb * 1024 * 1024);
}

FeatureStation::~FeatureStation() {
  // munmap
  for (auto& item : grp_to_data_source_) {
    auto& dsc = item.second;
    if (DataSourceType::MMAP != dsc.type) {
      continue;
    }

    if (nullptr != dsc.colidx.first) {
      munmap(dsc.first, dsc.second);
    }
    if (nullptr != dsc.rowsiz.first) {
      munmap(dsc.first, dsc.second);
    }
    if (nullptr != dsc.value.first) {
      munmap(dsc.first, dsc.second);
    }
  }
}

bool FeatureStation::addFeatureGrp(
  const int grp_id, const MatrixPtr<ValType> feature) {
  if (!feature) {
    return true;
  }
  {
    Lock l(general_mu_);

    auto iter = grp_to_file_path_.find(grp_id);
    if (grp_to_file_path_.end() != iter) {
      LL << "grp_id[" << grp_id << "] existed in FeatureStation";
      return false;
    }
  }

  // dump feature group
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

  // store the relationship: grp_id -> file_path
  {
    Lock l(general_mu_);

    grp_to_file_path_[grp_id] = file_path;
    grp_to_matrix_info_[grp_id] = feature.info();
  }

  return true;
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

DataSourceCollection FeatureStation::mapFiles(const string& file_prefix) {
  DataSourceCollection dsc(DataSourceType::MMAP);
  dsc.colidx = mapOneFile(file_prefix + ".colidx");
  dsc.rowsiz = mapOneFile(file_prefix + ".rowsiz");
  dsc.value = mapOneFile(file_prefix + ".value");

  return dsc;
}

DataSource FeatureStation::mapOneFile(const string& full_file_path) {
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
    const char* mmap_ptr = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (MAP_FAILED == mmap_ptr) {
      throw std::runtime_error("mmap failed");
    }
    close(fd);

    return makeDataSource(mmap_ptr, st.st_size);
  } catch (std::except& e) {
    LL << "file [" << full_file_path << "] mapOneFile failed. error [" <<
      e.what() << "] description [" << strerror(errno) << "]";
    if (fd > 0) {
      close(fd);
    }
  }

  return makeDataSource();
}

string FeatureStation::pickDirRandomly() {
  std::default_random_engine rng(time(nullptr));
  std::uniform_int_distribution<size_t> distribution(0, directories_.size() - 1);
  const size_t random_idx = distribution(rng);
  return directories_.at(random_idx);
}

size_t FeatureStation::memSize() {
  Lock l(general_mu_);

  size_t sum = 0;
  for (const auto& item : pool_) {
    sum += item.second->memSize();
  }

  return sum;
}

void FeatureStation::prefetch(
  const int task_id, const int grp_id, const SizeR range) {
  PrefetchJob new_job(task_id, grp_id, estimateRangeMemSize(grp_id, range));
  pending_jobs_.push(new_job);

  return;
}

void FeatureStation::prefetchThreadFunc() {
  while (1) {
    // take out a job
    PrefetchJob job;
    pending_jobs_.wait_and_pop(job);
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
    prefetch_mem_record_.push(job.task_id, job.mem_size);

    // prefetch
    MatrixPtr<ValType> feature = assembleFeatureMatrix(job);
    if (!feature) {
      LL << "assembleFeatureMatrix failed. [" << job.shortDebugString() << "]";
    }

    // store in loaded_features_
    loaded_features_[job.task_id] = feature;

    // remove from loading_jobs_
    loading_jobs_.erase(job.task_id);
  };
}

MatrixPtr<ValType> FeatureStation::assembleFeatureMatrix(const PrefetchJob& job) {
  if (0 == job.mem_size) {
    LL << "empty job [" << job.shortDebugString() << "]";
    return MatrixPtr<ValType>();
  }

  auto iter = grp_to_data_source_.find(job.task_id);
  if (grp_to_data_source_::end() == iter) {
    LL << "assembleFeatureMatrix cannot locate DataSourceCollection for grp_id [" <<
      job.grp_id << "]";
    return MatrixPtr<ValType>();
  }
  const DataSourceCollection& dsc = iter->second;

  // load offset
  const OffType* full_offset = reinterpret_cast<const OffType*>(dsc.offset.first);
  CHECK(nullptr != full_offset);
  SArray<OffType> offset(job.range.end() + 1 - job.range.begin());
  offset.copyFrom(
    full_offset + job.range.begin(),
    offset.size());

  // load index
  const KeyType* full_index = reinterpret_cast<const KeyType*>(dsc.index.first);
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
  auto info_iter = grp_to_matrix_info_.find(job.grp_id);
  CHECK(grp_to_matrix_info_.end() != info_iter);
  MatrixInfo info = info_iter->second;
  info.set_nnz(index.size());
  info.clear_ins_info();
  if (!info.row_major()) {
    info.mutable_col()->set(0, job.range.end() - job.range.begin());
  }

  // assemble
  return MatrixPtr<ValType>(
    new SparseMatrix<KeyType, ValType>(info, offset, index, value));
}

DataSource FeatureStation::makeDataSource(
  const char* in_ptr, const size_t in_size) {
  return std::make_pair(in_ptr, in_size);
}

size_t FeatureStation::estimateRangeMemSize(const int grp_id, const SizeR range) {
  if (range.empty()) {
    return 0;
  }

  // locate DataSourceCollection
  auto iter = grp_to_data_source_.find(grp_id);
  if (grp_to_data_source_::end() == iter) {
    LL << "illegal grp_id [" << grp_id << "]";
    return 0;
  }

  // check range
  const DataSourceCollection& dsc = iter->second;
  const OffType* full_offset = reinterpret_cast<const OffType*>(dsc.rowsiz.first);
  if (nullptr == dsc.rowsiz.first) {
    LL << "requesting an empty dsc.rowsiz. grp_id [" << grp_id << "]";
    return 0;
  }
  const full_offset_length = dsc.rowsiz.second / sizeof(OffType);
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

MatrixPtr<ValType> FeatureStation::getFeature(const int task_id) {
  MatrixPtr<ValType> ret_ptr;

  if (loaded_features_.tryGet(task_id, ret_ptr)) {
    // If what I want is ready, simply return it
  } else if (loading_jobs_.test(task_id)) {
    // If what I want is being loaded, wait
    loaded_features_.waitAndGet(task_id, ret_ptr);
  } else {
    // Nobody cares about me
    // I must load all by myself
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
  loaded_features_.addAndModify(task_id, MatrixPtr<ValType>());
  prefetch_mem_record_.tryErase(task_id);
}

}; // namespace PS
