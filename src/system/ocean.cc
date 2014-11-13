#include <unistd.h>
#include <fcntl.h>
#include "util/split.h"
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
  cpu_profiler_started_(false) {
  // launch prefetch threads
  for (int i = 0; i < FLAGS_num_threads; ++i) {
    thread_vec_.push_back(std::move(
      std::thread(&Ocean::prefetchThreadFunc, this)));
  }
  // launch write threads
  for (int i = 0; i < FLAGS_num_threads * 4; ++i) {
    write_thread_vec_.push_back(std::move(
      std::thread(&Ocean::writeThreadFunc, this)));
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
  // join write threads
  for (auto& thread : write_thread_vec_) {
    for (int i = 0; i < FLAGS_num_threads + 1; ++i) {
      // pump in some illegal write jobs
      pending_writes_.push(JobID(0, Range<FullKeyType>()));
    }
    thread.join();
  }
}

void Ocean::init(
  const string& identity,
  const LM::Config& conf,
  const Task& task,
  PathPicker& path_picker) {
  identity_ = identity;
  path_picker_ = &path_picker;
  conf_ = conf;
  CHECK(!identity_.empty());

  for (int i = 0; i < task.partition_info_size(); ++i) {
    Range<FullKeyType> range(task.partition_info(i).key());
    partition_info_[task.partition_info(i).fea_grp()].push_back(range);
  }
  CHECK(!partition_info_.empty());

  return;
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
    for (const auto& partition_range : partition_info_.at(grp_id)) {
      SizeR column_range;
      if (!locateColumnRange(column_range, grp_id, partition_range, type, input)) {
        LL << log_prefix_ << __FUNCTION__ <<
          " could not find column range for grp [" << grp_id <<
          "], global_range " << partition_range.toString() <<
          " DataType [" << dataTypeToString(type) << "]";
        return false;
      }
      if (!dumpSArraySegment(
            input, JobID(grp_id, partition_range),
            column_range, type)) {
        LL << log_prefix_ << __FUNCTION__ << " failed; group [" << grp_id << "] while dumping; " <<
          "data type [" << dataTypeToString(type) << "]";
        return false;
      }
    }
  } else {
    LL << log_prefix_ << __FUNCTION__ << " unknown; group [" << grp_id << "] while dumping; " <<
      "data type [" << dataTypeToString(type) << "]";
    return false;
  }

  return true;
}

bool Ocean::dump(
  SparseMatrixPtr<ShortKeyType, ValueType> X,
  const GrpID grp_id) {
  if (!X) { return true; }
  CHECK(!identity_.empty());

  //traverse all blocks corresponding to grp_id
  if (partition_info_.end() != partition_info_.find(grp_id)) {
    for (const auto& partition_range : partition_info_.at(grp_id)) {
      // column range for offset
      SizeR column_range;
      if (!locateColumnRange(
        column_range, grp_id, partition_range,
        DataType::FEATURE_OFFSET, SArray<char>(X->offset()))) {
        LL << log_prefix_ << __FUNCTION__ <<
          " could not find column range for grp [" << grp_id <<
          "], global_range " << partition_range.toString() <<
          " DataType [" << dataTypeToString(DataType::FEATURE_OFFSET) << "]";
        return false;
      }

      // dump feature_offset
      SizeR offset_column_range(column_range.begin(), column_range.end() + 1);
      if (!dumpSArraySegment(
          SArray<char>(X->offset()), JobID(grp_id, partition_range),
          offset_column_range, DataType::FEATURE_OFFSET)) {
        LL << log_prefix_ << __FUNCTION__ << " failed; group [" <<
          grp_id << "] while dumping; data type [" <<
          dataTypeToString(DataType::FEATURE_OFFSET) << "]";
        return false;
      }

      SizeR feature_key_range(
        X->offset()[offset_column_range.begin()],
        X->offset()[offset_column_range.end() - 1]);
      // dump feature_key
      if (!dumpSArraySegment(
          SArray<char>(X->index()), JobID(grp_id, partition_range),
          feature_key_range, DataType::FEATURE_KEY)) {
        LL << log_prefix_ << __FUNCTION__ << " failed; group [" <<
          grp_id << "] while dumping; data type [" <<
          dataTypeToString(DataType::FEATURE_KEY) << "]";
        return false;
      }

      // dump feature_value
      if (!dumpSArraySegment(
          SArray<char>(X->value()), JobID(grp_id, partition_range),
          feature_key_range, DataType::FEATURE_VALUE)) {
        LL << log_prefix_ << __FUNCTION__ << " failed; group [" <<
          grp_id << "] while dumping; data type [" <<
          dataTypeToString(DataType::FEATURE_VALUE) << "]";
        return false;
      }

      CHECK_EQ(
        X->offset().segment(offset_column_range).back() -
          X->offset().segment(offset_column_range).front(),
        X->index().segment(feature_key_range).size()) <<
        " dumping job failed " << JobID(grp_id, partition_range).toString();
    }
  } else {
    LL << log_prefix_ << __FUNCTION__ << " unknown; group [" <<
      grp_id << "] while dumping SparseMatrix";
    return false;
  }

  // dump matrix info
  matrix_info_.addWithoutModify(grp_id, X->info());

  return true;
}

bool Ocean::dumpSArraySegment(
  SArray<char> input,
  const JobID& job_id,
  const SizeR& column_range,
  const DataType type) {
  if (input.empty()) { return true; }

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
        memory_data.feature_offset = SArray<OffsetType>(input).segment(column_range);
        break;
      case Ocean::DataType::FEATURE_VALUE:
        memory_data.feature_value = SArray<ValueType>(input).segment(column_range);
        break;
      default:
        LL << __FUNCTION__ << "; invalid datatype [" <<
          static_cast<size_t>(type) << "]";
        break;
    };
    loaded_data_.addAndModify(job_id, memory_data);
    return true;
  }

  // full path
  std::stringstream ss;
  ss << "blockcache." << identity_ <<
    "." << dataTypeToString(type) << "." << job_id.grp_id <<
    "." << job_id.range.begin() << "-" << job_id.range.end();
  string full_path = path_picker_->getPath(ss.str()).c_str();

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
      if (!array.segment(column_range).writeToFile(full_path)) {
        throw std::runtime_error("OffsetType");
      }
    }
  } catch (std::exception& e) {
    LL << log_prefix_ << __FUNCTION__ <<
      " SArray writeToFile failed on path [" << full_path << "] column_range " <<
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

    if (FLAGS_verbose) {
      LI << log_prefix_ << "prefetchThreadFunc ready to prefetch [" <<
        job_id.toString() << "]";
    }

    // hang on if prefetch limit reached
    {
      std::unique_lock<std::mutex> l(prefetch_limit_mu_);
      prefetch_limit_cond_.wait(
        l, [this]{ return loaded_data_.size() <= FLAGS_prefetch_job_limit; });
    }

    // check job status
    JobStatus job_status = job_info_table_.getStatus(job_id);
    if (JobStatus::LOADING == job_status ||
        JobStatus::LOADED == job_status) {
      if (FLAGS_verbose) {
        LI << log_prefix_ << "prefetchThreadFunc finish prefetching [" <<
          job_id.toString() << "] since already LOADING/LOADED";
      }
      // already been prefetched
      continue;
    }

    if (FLAGS_verbose) {
      LI << log_prefix_ << "prefetchThreadFunc start prefetching [" <<
        job_id.toString() << "]";
    }

    // change job status
    job_info_table_.setStatus(job_id, JobStatus::LOADING);

    // load from disk
    LoadedData loaded = loadFromDiskSynchronously(job_id);
    loaded_data_.addWithoutModify(job_id, loaded);
    job_info_table_.setStatus(job_id, JobStatus::LOADED);

    if (FLAGS_verbose) {
      LI << log_prefix_ << "prefetchThreadFunc finish prefetching [" <<
        job_id.toString() << "] since succ";
    }
  };
}

Ocean::LoadedData Ocean::loadFromDiskSynchronously(const JobID job_id) {
  LoadedData memory_data;

  // load parameter_key
  string full_path;
  if (lakes_[static_cast<size_t>(DataType::PARAMETER_KEY)].tryGet(job_id, full_path)) {
    SArray<char> stash;
    CHECK(stash.readFromFile(full_path));
    memory_data.parameter_key = stash;
  }

  // load parameter_value
  if (lakes_[static_cast<size_t>(DataType::PARAMETER_VALUE)].tryGet(job_id, full_path)) {
    SArray<char> stash;
    CHECK(stash.readFromFile(full_path));
    memory_data.parameter_value = stash;
  }

  // load delta
  if (lakes_[static_cast<size_t>(DataType::DELTA)].tryGet(job_id, full_path)) {
    SArray<char> stash;
    CHECK(stash.readFromFile(full_path));
    memory_data.delta = stash;
  }

  // load feature key
  if (lakes_[static_cast<size_t>(DataType::FEATURE_KEY)].tryGet(job_id, full_path)) {
    SArray<char> stash;
    CHECK(stash.readFromFile(full_path));
    memory_data.feature_key = stash;
  }

  // load feature offset
  if (lakes_[static_cast<size_t>(DataType::FEATURE_OFFSET)].tryGet(job_id, full_path)) {
    SArray<char> stash;
    CHECK(stash.readFromFile(full_path));
    memory_data.feature_offset = stash;
  }

  // load feature value
  if (lakes_[static_cast<size_t>(DataType::FEATURE_VALUE)].tryGet(job_id, full_path)) {
    SArray<char> stash;
    CHECK(stash.readFromFile(full_path));
    memory_data.feature_value = stash;
  }

  if (!memory_data.feature_offset.empty()) {
    CHECK_EQ(
      memory_data.feature_offset.back() - memory_data.feature_offset.front(),
      memory_data.feature_key.size());
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

  if (FLAGS_verbose) {
    LI << log_prefix_ << "dropped [" << job_id.toString() <<
      "]; remaining ref_count: " << job_info_table_.getRef(job_id);
  }

  pending_writes_.push(job_id);
}

void Ocean::makeMemoryDataReady(const JobID job_id) {
  if (!FLAGS_less_memory) {
    return;
  }

  if (loaded_data_.test(job_id)) {
  } else if (JobStatus::LOADING == job_info_table_.getStatus(job_id)) {
    if (FLAGS_verbose) {
      LI << log_prefix_ << "waiting synchronously [" << job_id.toString() << "]";
    }

    LoadedData memory_data;
    loaded_data_.waitAndGet(job_id, memory_data);

    if (FLAGS_verbose) {
      LI << log_prefix_ << "waited synchronously [" << job_id.toString() << "]";
    }
  } else {
    if (FLAGS_verbose) {
      LI << log_prefix_ << "loading synchronously [" << job_id.toString() << "]";
    }

    job_info_table_.setStatus(job_id, JobStatus::LOADING);
    LoadedData memory_data = loadFromDiskSynchronously(job_id);
    loaded_data_.addWithoutModify(job_id, memory_data);
    job_info_table_.setStatus(job_id, JobStatus::LOADED);

    if (FLAGS_verbose) {
      LI << log_prefix_ << "loaded synchronously [" << job_id.toString() << "]";
    }
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
    return input.writeToFile(full_path);
  } else {
    LL << "cannot not find full_path in lakes_; job_id: " << job_id.toString();
    return false;
  }
}

size_t Ocean::groupKeyCount(const GrpID grp_id) {
  return group_key_counts_[grp_id];
}

std::vector<Range<Ocean::FullKeyType>> Ocean::getPartitionInfo(const GrpID grp_id) {
  auto iter = partition_info_.find(grp_id);
  if (partition_info_.end() != iter) {
    return iter->second;
  }
  return std::vector<Range<FullKeyType>>();
}

bool Ocean::locateColumnRange(
  SizeR& output_range,
  const GrpID grp_id,
  const Range<FullKeyType>& partition_range,
  const Ocean::DataType type,
  SArray<char> input) {
  if (Ocean::DataType::PARAMETER_KEY == type) {
    // binary search
    SArray<FullKeyType> keys(input);
    output_range = keys.findRange(partition_range);
    column_ranges_.addWithoutModify(JobID(grp_id, partition_range), output_range);
    group_key_counts_[grp_id] += output_range.size();
  } else {
    // locate calculated column range
    if (!column_ranges_.tryGet(JobID(grp_id, partition_range), output_range)) {
      LL << log_prefix_ << __FUNCTION__ <<
        " could not find column range for grp [" << grp_id <<
        "], global_range " << partition_range.toString() <<
        " DataType [" << dataTypeToString(type) << "]";
      return false;
    }
  }
  return true;
}

std::vector<std::pair<Ocean::JobID, string>> Ocean::getAllDumpedPath(
  const Ocean::DataType type) {
  CHECK_LT(static_cast<size_t>(type), static_cast<size_t>(Ocean::DataType::NUM));
  return lakes_.at(static_cast<size_t>(type)).all();
}

std::vector<std::pair<Ocean::JobID, SArray<char>>> Ocean::getAllLoadedArray(
  const Ocean::DataType type) {
  CHECK_LT(static_cast<size_t>(type), static_cast<size_t>(Ocean::DataType::NUM));

  std::vector<std::pair<Ocean::JobID, SArray<char>>> vec;
  auto all_loaded = loaded_data_.all();
  for (const auto& item : all_loaded) {
    SArray<char> array;
    switch (type) {
      case Ocean::DataType::PARAMETER_KEY:
        array = item.second.parameter_key;
        break;
      case Ocean::DataType::PARAMETER_VALUE:
        array = item.second.parameter_value;
        break;
      case Ocean::DataType::DELTA:
        array = item.second.delta;
        break;
      case Ocean::DataType::FEATURE_KEY:
        array = item.second.feature_key;
        break;
      case Ocean::DataType::FEATURE_OFFSET:
        array = item.second.feature_offset;
        break;
      case Ocean::DataType::FEATURE_VALUE:
        array = item.second.feature_value;
        break;
      default:
        CHECK(false) << "UNKNOWN_DATATYPE " << static_cast<size_t>(type);
        return vec;
    };

    vec.push_back(std::make_pair(item.first, array));
  }
  return vec;
}

void Ocean::writeBlockCacheInfo() {
  const string output_file_path = path_picker_->getPath(
    string("blockcache.") + identity_ + ".info");
  File* f = File::openOrDie(output_file_path, "w");

  for (size_t data_type = 0; data_type < lakes_.size(); ++data_type) {
    auto blockcache_vec = getAllDumpedPath(static_cast<Ocean::DataType>(data_type));
    for (const auto& item : blockcache_vec) {
      std::stringstream ss;
      ss << data_type << "\t" << item.first.grp_id << "\t" <<
        item.first.range.begin() << "\t" << item.first.range.end() << "\t" <<
        item.second << "\t" << File::size(item.second) << "\n";
      f->writeString(ss.str());
    }
  }
  f->close();

  const string column_range_file_path = path_picker_->getPath(
    string("column_range.") + identity_ + ".info");
  f = File::openOrDie(column_range_file_path, "w");

  auto column_range_vec = column_ranges_.all();
  for (const auto& item : column_range_vec) {
    std::stringstream ss;
    ss << item.first.grp_id << "\t" <<
      item.first.range.begin() << "\t" << item.first.range.end() << "\t" <<
      item.second.begin() << "\t" << item.second.end() << "\n";
    f->writeString(ss.str());
  }
  f->close();

  const string matrix_info_file_path = path_picker_->getPath(
    string("matrix_info.") + identity_ + ".info");
  f = File::openOrDie(matrix_info_file_path, "w");

  auto matrix_info_vec = matrix_info_.all();
  for (const auto& item : matrix_info_vec) {
    const string dumped_matrix_info_path = path_picker_->getPath(
      string("matrix_info.") + identity_ + "." + std::to_string(item.first));
    writeProtoToASCIIFileOrDie(
      item.second,
      dumped_matrix_info_path);
    std::stringstream ss;
    ss << item.first << "\t" <<
      dumped_matrix_info_path << "\n";
    f->writeString(ss.str());
  }
  f->close();

  return;
}

bool Ocean::readBlockCacheInfo() {
  const string input_file_path = path_picker_->getPath(
    string("blockcache.") + identity_ + ".info");
  File* f = File::open(input_file_path, "r");
  if (nullptr == f) {
    // file not exists
    return false;
  }
  threadsafe_queue<std::tuple<string, DataType, size_t>> mutable_files;

  char buf[2048];
  while (nullptr != f->readLine(buf, sizeof(buf))) {
    string each_blockcache(buf);

    // remove tailing line-break
    if (!each_blockcache.empty() && '\n' == each_blockcache.back()) {
      each_blockcache.resize(each_blockcache.size() - 1);
    }

    try {
      auto vec = split(each_blockcache, '\t');
      if (6 != vec.size()) {
        throw std::runtime_error("column number wrong");
      }

      size_t data_type = std::stoul(vec[0]);
      if (data_type >= static_cast<size_t>(DataType::NUM)) {
        throw std::runtime_error("illegal data_type");
      }

      GrpID grp_id = std::stoul(vec[1]);
      FullKeyType range_begin = std::stoull(vec[2]);
      FullKeyType range_end = std::stoull(vec[3]);
      string path = vec[4];
      const size_t file_size = std::stoull(vec[5]);

      if (isMutableDataType(static_cast<DataType>(data_type))) {
        // mutable files will be zeroed later
        mutable_files.push(std::make_tuple(
          path, static_cast<DataType>(data_type), file_size));
      } else if (file_size != File::size(path)) {
        // make sure corresponding non-mutable file exists
        std::stringstream ss;
        ss << "incorrect file size. " << file_size << " vs " << File::size(path);
        throw std::runtime_error(ss.str());
      }

      // track
      lakes_[data_type].addWithoutModify(
        JobID(grp_id, Range<FullKeyType>(range_begin, range_end)),
        path);
    } catch (std::exception& e) {
      LL << log_prefix_ << __FUNCTION__ <<
        " encountered illegal blockcache info line [" <<
        each_blockcache << "] [" << e.what() << "]";
      f->close();
      return false;
    }
  }
  f->close();

  // zero mutable files
  auto resetMutable = [this, &mutable_files]() {
    std::tuple<string, DataType, size_t> mutable_file;
    while (mutable_files.try_pop(mutable_file)) {
      int init_value = 0;
      if (DataType::DELTA == std::get<1>(mutable_file)) {
        init_value = conf_.darling().delta_init_value();
      }

      SArray<ValueType> buf(std::get<2>(mutable_file) / sizeof(ValueType), init_value);
      buf.writeToFile(std::get<0>(mutable_file));
    };
  };
  {
    ThreadPool pool(FLAGS_num_threads);
    for (size_t i = 0; i < FLAGS_num_threads; ++i) {
      pool.add(resetMutable);
    }
    pool.startWorkers();
  }

  const string column_range_file_path = path_picker_->getPath(
    string("column_range.") + identity_ + ".info");
  f = File::open(column_range_file_path, "r");
  if (nullptr == f) {
    // file not exists
    return false;
  }

  while (nullptr != f->readLine(buf, sizeof(buf))) {
    string each_column_range(buf);

    // remove tailing line-break
    if (!each_column_range.empty() && '\n' == each_column_range.back()) {
      each_column_range.resize(each_column_range.size() - 1);
    }

    try {
      auto vec = split(each_column_range, '\t');
      if (5 != vec.size()) {
        throw std::runtime_error("column number wrong");
      }

      GrpID grp_id = std::stoul(vec[0]);
      FullKeyType range_begin = std::stoull(vec[1]);
      FullKeyType range_end = std::stoull(vec[2]);
      size_t column_begin = std::stoul(vec[3]);
      size_t column_end = std::stoul(vec[4]);

      SizeR current_range(column_begin, column_end);
      column_ranges_.addWithoutModify(
        JobID(grp_id, Range<FullKeyType>(range_begin, range_end)),
        current_range);
      group_key_counts_[grp_id] += current_range.size();
    } catch (std::exception& e) {
      LL << log_prefix_ << __FUNCTION__ <<
        " encountered illegal column_range info line [" <<
        each_column_range << "] [" << e.what() << "]";
      f->close();
      return false;
    }
  }
  f->close();

  const string matrix_info_file_path = path_picker_->getPath(
    string("matrix_info.") + identity_ + ".info");
  f = File::open(matrix_info_file_path, "r");
  if (nullptr == f) {
    // file not exists
    return false;
  }

  while (nullptr != f->readLine(buf, sizeof(buf))) {
    string each_matrix_info(buf);

    // remove tailing line-break
    if (!each_matrix_info.empty() && '\n' == each_matrix_info.back()) {
      each_matrix_info.resize(each_matrix_info.size() - 1);
    }

    try {
      auto vec = split(each_matrix_info, '\t');
      if (2 != vec.size()) {
        throw std::runtime_error("column number wrong");
      }

      GrpID grp_id = std::stoul(vec[0]);
      const string matrix_info_path = vec[1];
      MatrixInfo current_matrix_info;
      readFileToProtoOrDie(matrix_info_path, &current_matrix_info);

      matrix_info_.addWithoutModify(grp_id, current_matrix_info);
    } catch (std::exception& e) {
      LL << log_prefix_ << __FUNCTION__ <<
        " encountered illegal matrix_info info line [" <<
        each_matrix_info << "] [" << e.what() << "]";
      f->close();
      return false;
    }
  }
  f->close();

  return true;
}

MatrixInfo Ocean::getMatrixInfo(const GrpID grp_id) {
  MatrixInfo info;
  CHECK(matrix_info_.tryGet(grp_id, info));
  return info;
}

void Ocean::writeThreadFunc() {
  while (go_on_prefetching_) {
    // take out a write job
    JobID job_id;
    pending_writes_.wait_and_pop(job_id);

    if (FLAGS_verbose) {
      LI << log_prefix_ << "writeThreadFunc ready to write [" <<
        job_id.toString() << "]";
    }

    LoadedData memory_data;
    if (loaded_data_.tryGet(job_id, memory_data)) {
      if (!memory_data.parameter_value.empty()) {
        writeToDisk(SArray<char>(memory_data.parameter_value), job_id, DataType::PARAMETER_VALUE);
      }
      if (!memory_data.delta.empty()) {
        writeToDisk(SArray<char>(memory_data.delta), job_id, DataType::DELTA);
      }
    }

    if (FLAGS_verbose) {
      LI << log_prefix_ << "written to disk [" << job_id.toString() << "]";
    }

    // release
    if (job_info_table_.getRef(job_id) <= 0) {
      // remove from job_info_table_
      job_info_table_.erase(job_id);
      // release LoadedData
      loaded_data_.erase(job_id);
      prefetch_limit_cond_.notify_all();
#ifdef TCMALLOC
      // tcmalloc force return memory to kernel
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
      if (FLAGS_verbose) {
        LI << log_prefix_ << "released [" << job_id.toString() << "]";
      }
    }

  }
}
}; // namespace PS
