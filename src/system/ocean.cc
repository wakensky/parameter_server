#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdexcept>
#include <malloc.h>
#include <gperftools/malloc_extension.h>
#include "base/shared_array_inl.h"
#include "system/ocean.h"

namespace PS {

DEFINE_int32(prefetch_job_limit, 64,
  "control memory usage via limit prefetch job count");
DEFINE_bool(less_memory, false,
  "use little memory as possible; disabled in default");

Ocean::Ocean() :
  go_on_prefetching_(true),
  log_prefix_("[Ocean] "),
  rng_(time(0)) {
  // launch prefetch threads
  for (int i = 0; i < FLAGS_num_threads; ++i) {
    thread_vec_.push_back(std::move(
      std::thread(&Ocean::prefetchThreadFunc, this)));
  }
}

Ocean::~Ocean() {
  // join prefetch threads
  go_on_prefetching_ = false;
  for (auto& thread : thread_vec_) {
    for (int i = 0; i < FLAGS_num_threads + 1; ++i) {
      // pump in some illegal prefetch jobs
      //  push threads moving on
      prefetch(0, Range<FullKeyType>());
    }
    thread.join();
  }
}

void Ocean::init(
  const string& identity,
  const LM::Config& conf,
  const Task& task) {
  identity_ = identity;

  for (int i = 0; i < conf.local_cache().file_size(); ++i) {
    addDirectory(conf.local_cache().file(i));
  }
  CHECK(!identity_.empty());
  CHECK(!directories_.empty());

  for (int i = 0; i < task.partition_info_size(); ++i) {
    Range<FullKeyType> range(task.partition_info(i).key());
    partition_info_[task.partition_info(i).fea_grp()].push_back(range);
  }
  CHECK(!partition_info_.empty());

  return;
}

bool Ocean::addDirectory(const string& dir) {
  Lock l(general_mu_);

  // check permission
  struct stat st;
  if (-1 == stat(dir.c_str(), &st)) {
    LL << log_prefix_ << "dir [" << dir << "] cannot be added " <<
      "since error [" << strerror(errno) << "]";
    return false;
  }
  if (!S_ISDIR(st.st_mode)) {
    LL << log_prefix_ << "dir [" << dir << "] is not a regular directory";
    return false;
  }
  if (0 != access(dir.c_str(), R_OK | W_OK)) {
    LL << log_prefix_ << "I donnot have read&write permission " <<
      "on directory [" << dir << "]";
    return false;
  }

  // add
  directories_.push_back(dir);

  if (FLAGS_verbose) {
    LI << log_prefix_ << "dir [" << dir << "] has been added to Ocean";
  }
  return true;
}

string Ocean::pickDirRandomly() {
  CHECK(!directories_.empty());
  Lock l(general_mu_);

  std::uniform_int_distribution<size_t> dist(0, directories_.size() - 1);
  const size_t random_idx = dist(rng_);
  return directories_.at(random_idx);
}

bool Ocean::dump(
  SArray<char> input,
  GrpID grp_id,
  Ocean::DataType type) {
  if (input.empty()) {
    return true;
  }
  CHECK(!identity_.empty());

  // traverse all blocks corresponding to grp_id
  if (partition_info_.end() != partition_info_.find(grp_id)) {
    for (const auto& partition_range : partition_info_[grp_id]) {
      if (!dumpSarraySegment(input, grp_id, partition_range, type)) {
        LL << "failed; group [" << grp_id << "] while dumping; " <<
          "data type [" << dataTypeToString(type) << "]";
        return false;
      }
    }
  } else {
    LL << "unknown; group [" << grp_id << "] while dumping; " <<
      "data type [" << dataTypeToString(type) << "]";
    return false;
  }

  return true;
}

bool Ocean::dumpSarraySegment(
  SArray<char>& input,
  const GrpID grp_id,
  const Range<FullKeyType>& global_range,
  const DataType type) {
  // make job id
  JobID job_id(grp_id, global_range);

  // get column range
  SizeR column_range;
  if (Ocean::DataType::PARAMETER_KEY == type) {
    // locate column range
    SArray<FullKeyType> keys(input);
    column_range = keys.findRange(global_range);
    column_ranges_.addWithoutModify(job_id, column_range);
    group_key_counts_[grp_id] += column_range.size();
  } else {
    // find column range
    if (!column_ranges_.tryGet(job_id, column_range)) {
      LL << log_prefix_ << __PRETTY_FUNCTION__ <<
        "could not find column range for grp [" << grp_id <<
        "], global_range " << global_range.toString();
      return false;
    }
  }

  if (!FLAGS_less_memory) {
    // on memory mode
    // store segmented SArray in loaded_data_ directly
    LoadedData memory_data;
    loaded_data_.tryGet(job_id, memory_data);

    switch (type) {
      case Ocean::DataType::PARAMETER_KEY:
        memory_data.parameter_key = SArray<FullKeyType>(input).segment(column_range);
        break;
      case Ocean::DataType::PARAMETER_VALUE:
        memory_data.parameter_value = SArray<ValueType>(input).segment(column_range);
        break;
      case Ocean::DataType::DELTA:
        memory_data.delta = SArray<ValueType>(input).segment(column_range);
        break;
      case Ocean::DataType::FEATURE_KEY:
        memory_data.feature_key = SArray<ShortKeyType>(input).segment(column_range);
        break;
      case Ocean::DataType::FEATURE_OFFSET:
        {
          SizeR offset_column_range(column_range.begin(), column_range.end() + 1);
          memory_data.feature_offset =
            SArray<OffsetType>(input).segment(offset_column_range);
        }
        break;
      case Ocean::DataType::FEATURE_VALUE:
        memory_data.feature_value = SArray<ValueType>(input).segment(column_range);
        break;
      default:
        LL << __PRETTY_FUNCTION__ << "; invalid datatype [" <<
          static_cast<size_t>(type) << "]";
        break;
    };
    loaded_data_.addAndModify(job_id, memory_data);

    return true;
  }

  // full path
  std::stringstream ss;
  ss << pickDirRandomly() << "/" << identity_ <<
    "." << dataTypeToString(type) << "." << grp_id <<
    "." << global_range.begin() << "-" << global_range.end();
  string full_path = ss.str();

  // dump
  try {
    if (Ocean::DataType::PARAMETER_KEY == type) {
      SArray<FullKeyType> array(input);
      if (!array.segment(column_range).writeToFile(full_path)) {
        throw std::runtime_error("FullKeyType");
      }
    } else if (Ocean::DataType::FEATURE_KEY == type) {
      SArray<ShortKeyType> array(input);
      if (!array.segment(column_range).writeToFile(full_path)) {
        throw std::runtime_error("ShortKeyType");
      }
    } else if (Ocean::DataType::FEATURE_VALUE == type ||
               Ocean::DataType::PARAMETER_VALUE == type ||
               Ocean::DataType::DELTA == type) {
      SArray<ValueType> array(input);
      if (!array.segment(column_range).writeToFile(full_path)) {
        throw std::runtime_error("ValueType");
      }
    } else if (Ocean::DataType::FEATURE_OFFSET == type) {
      SArray<OffsetType> array(input);
      SizeR offset_column_range(column_range.begin(), column_range.end() + 1);
      if (!array.segment(offset_column_range).writeToFile(full_path)) {
        throw std::runtime_error("OffsetType");
      }
    }
  } catch (std::exception& e) {
    LL << log_prefix_ << __PRETTY_FUNCTION__ <<
      "SArray writeToFile failed on path [" << full_path << "] column_range " <<
      column_range.toString() << " data type [" << dataTypeToString(type) << "]";
    return false;
  }

  // record
  lakes_[static_cast<size_t>(type)].addWithoutModify(job_id, full_path);

  return true;
}

void Ocean::prefetch(GrpID grp_id, const Range<FullKeyType>& key_range) {
  if (!FLAGS_less_memory) {
    return;
  }

  // enqueue
  JobID job_id(grp_id, key_range);
  pending_jobs_.push(job_id);

  // reference count
  job_info_table_.increaseRef(job_id);

  if (FLAGS_verbose) {
    LI << log_prefix_ << "add prefetch job; grp_id:" <<
      grp_id << ", range:" << key_range.toString();
  }
  return;
}

void Ocean::prefetchThreadFunc() {
  while (go_on_prefetching_) {
    // take out a job
    JobID job_id;
    pending_jobs_.wait_and_pop(job_id);

    // check job status
    JobStatus job_status = job_info_table_.getStatus(job_id);
    if (JobStatus::LOADING == job_status ||
        JobStatus::LOADED == job_status) {
      // already been prefetched
      continue;
    }

    // hang on if prefetch limit reached
    {
      std::unique_lock<std::mutex> l(prefetch_limit_mu_);
      prefetch_limit_cond_.wait(
        l, [this]{ return loaded_data_.size() <= FLAGS_prefetch_job_limit; });
    }

    // change job status
    job_info_table_.setStatus(job_id, JobStatus::LOADING);

    // load from disk
    LoadedData loaded = loadFromDiskSynchronously(job_id);
    loaded_data_.addWithoutModify(job_id, loaded);
    job_info_table_.setStatus(job_id, JobStatus::LOADED);
  };
}

Ocean::LoadedData Ocean::loadFromDiskSynchronously(const JobID job_id) {
  LoadedData memory_data;

  // load parameter_key
  string full_path;
  if (lakes_[static_cast<size_t>(DataType::PARAMETER_KEY)].tryGet(job_id, full_path)) {
    memory_data.parameter_key.readFromFile(full_path);
  }

  // load parameter_value
  if (lakes_[static_cast<size_t>(DataType::PARAMETER_VALUE)].tryGet(job_id, full_path)) {
    memory_data.parameter_value.readFromFile(full_path);
  }

  // load delta
  if (lakes_[static_cast<size_t>(DataType::DELTA)].tryGet(job_id, full_path)) {
    memory_data.delta.readFromFile(full_path);
  }

  // load feature key
  if (lakes_[static_cast<size_t>(DataType::FEATURE_KEY)].tryGet(job_id, full_path)) {
    memory_data.feature_key.readFromFile(full_path);
  }

  // load feature offset
  if (lakes_[static_cast<size_t>(DataType::FEATURE_OFFSET)].tryGet(job_id, full_path)) {
    memory_data.feature_offset.readFromFile(full_path);
  }

  // load feature value
  if (lakes_[static_cast<size_t>(DataType::FEATURE_VALUE)].tryGet(job_id, full_path)) {
    memory_data.feature_value.readFromFile(full_path);
  }

  return memory_data;
}

SArray<Ocean::FullKeyType> Ocean::getParameterKey(
  const GrpID grp_id,
  const Range<Ocean::FullKeyType>& range) {
  JobID job_id(grp_id, range);
  makeMemoryDataReady(job_id);

  LoadedData memory_data;
  if (loaded_data_.tryGet(job_id, memory_data)) {
    return memory_data.parameter_key;
  }
  return SArray<FullKeyType>();
}

SArray<Ocean::ValueType> Ocean::getParameterValue(
  const GrpID grp_id,
  const Range<Ocean::FullKeyType>& range) {
  JobID job_id(grp_id, range);
  makeMemoryDataReady(job_id);

  LoadedData memory_data;
  if (loaded_data_.tryGet(job_id, memory_data)) {
    return memory_data.parameter_value;
  }
  return SArray<ValueType>();
}

SArray<Ocean::ValueType> Ocean::getDelta(
  const GrpID grp_id,
  const Range<Ocean::FullKeyType>& range) {
  JobID job_id(grp_id, range);
  makeMemoryDataReady(job_id);

  LoadedData memory_data;
  if (loaded_data_.tryGet(job_id, memory_data)) {
    return memory_data.delta;
  }
  return SArray<ValueType>();
}

SArray<Ocean::ShortKeyType> Ocean::getFeatureKey(
  const GrpID grp_id,
  const Range<Ocean::FullKeyType>& range) {
  JobID job_id(grp_id, range);
  makeMemoryDataReady(job_id);

  LoadedData memory_data;
  if (loaded_data_.tryGet(job_id, memory_data)) {
    return memory_data.feature_key;
  }
  return SArray<ShortKeyType>();
}

SArray<Ocean::OffsetType> Ocean::getFeatureOffset(
  const GrpID grp_id,
  const Range<Ocean::FullKeyType>& range) {
  JobID job_id(grp_id, range);
  makeMemoryDataReady(job_id);

  LoadedData memory_data;
  if (loaded_data_.tryGet(job_id, memory_data)) {
    return memory_data.feature_offset;
  }
  return SArray<OffsetType>();
}

SArray<Ocean::ValueType> Ocean::getFeatureValue(
  const GrpID grp_id,
  const Range<Ocean::FullKeyType>& range) {
  JobID job_id(grp_id, range);
  makeMemoryDataReady(job_id);

  LoadedData memory_data;
  if (loaded_data_.tryGet(job_id, memory_data)) {
    return memory_data.feature_value;
  }
  return SArray<ValueType>();
}

void Ocean::drop(const GrpID grp_id, const Range<Ocean::FullKeyType>& range) {
  if (!FLAGS_less_memory) {
    return;
  }

  JobID job_id(grp_id, range);
  job_info_table_.decreaseRef(job_id);

  if (job_info_table_.getRef(job_id) <= 0) {
    // write mutable data back to disk
    LoadedData memory_data;
    if (loaded_data_.tryGet(job_id, memory_data)) {
      if (!memory_data.parameter_value.empty()) {
        writeToDisk(SArray<char>(memory_data.parameter_value), job_id, DataType::PARAMETER_VALUE);
      }
      if (!memory_data.delta.empty()) {
        writeToDisk(SArray<char>(memory_data.delta), job_id, DataType::DELTA);
      }
    }

    // release
    if (job_info_table_.getRef(job_id) <= 0) {
      // release LoadedData
      loaded_data_.erase(job_id);
      // remove from job_info_table_
      job_info_table_.erase(job_id);
    }
  }

  return;
}

void Ocean::makeMemoryDataReady(const JobID job_id) {
  if (!FLAGS_less_memory) {
    return;
  }

  if (loaded_data_.test(job_id)) {
  } else if (JobStatus::LOADING == job_info_table_.getStatus(job_id)) {
    LoadedData memory_data;
    loaded_data_.waitAndGet(job_id, memory_data);
  } else {
    job_info_table_.setStatus(job_id, JobStatus::LOADING);
    LoadedData memory_data = loadFromDiskSynchronously(job_id);
    loaded_data_.addWithoutModify(job_id, memory_data);
    job_info_table_.setStatus(job_id, JobStatus::LOADED);
  }
  return;
}

string Ocean::dataTypeToString(const Ocean::DataType type) {
  switch (type) {
    case Ocean::DataType::FEATURE_KEY:
      return "FEATURE_KEY";
    case Ocean::DataType::FEATURE_OFFSET:
      return "FEATURE_OFFSET";
    case Ocean::DataType::FEATURE_VALUE:
      return "FEATURE_VALUE";
    case Ocean::DataType::DELTA:
      return "DELTA";
    case Ocean::DataType::PARAMETER_KEY:
      return "PARAMETER_KEY";
    case Ocean::DataType::PARAMETER_VALUE:
      return "PARAMETER_VALUE";
    default:
      return "UNKNOWN_DATATYPE";
  };
}

SizeR Ocean::getBaseRange(const GrpID grp_id, const Range<FullKeyType>& range) {
  SizeR ret;
  if (!column_ranges_.tryGet(JobID(grp_id, range), ret)) {
    ret = SizeR();
  }
  return ret;
}

size_t Ocean::pendingPrefetchCount() {
  return pending_jobs_.size();
}

bool Ocean::writeToDisk(
  SArray<char> input,
  const JobID& job_id,
  const Ocean::DataType type) {
  if (input.empty()) {
    return true;
  }

  string full_path;
  if (lakes_[static_cast<size_t>(type)].tryGet(job_id, full_path)) {
    return input.writeToFile(full_path, true);
  } else {
    LL << "cannot not find full_path in lakes_; job_id: " << job_id.toString();
    return false;
  }
}

size_t Ocean::groupKeyCount(const GrpID grp_id) {
  return group_key_counts_[grp_id];
}
}; // namespace PS
