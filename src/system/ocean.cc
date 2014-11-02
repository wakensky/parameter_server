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
      prefetch(0, Range<KeyType>());
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
    Range<KeyType> range(task.partition_info(i).key());
    partition_info_[task.partition_info(i).fea_grp()].push_back(range);
  }
  CHECK(!partition_info.empty());

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

JobID Ocean::makeJobID(GrpID grp_id, const Range<KeyType>& range) {
  JobID job_id = 0;
  job_id |= (static_cast<JobID>(grp_id) << 118);
  job_id |= (static_cast<JobID>(range.size()) << 64);
  job_id |= range.begin();

  return job_id;
}

void Ocean::parseJobID(
  const JobID job_id,
  GrpID& grp_id,
  Range<KeyType>& range) {
  grp_id = static_cast<GrpID>(job_id >> 118);
  KeyType begin = static_cast<KeyType>((job_id << 64) >> 64);
  KeyType end = static_cast<KeyType>((job_id << 10) >> 74);
  range.set(begin, end);
  return;
}

bool Ocean::dump(
  const SArray<char>& input,
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
          "data type [" << type << "]";
        return false;
      }
    }
  } else {
    LL << "unknown; group [" << grp_id << "] while dumping; " <<
      "data type [" << type << "]";
    return false;
  }

  return true;
}

bool Ocean::dumpSarraySegment(
  const SArray<char>& input,
  const GrpID grp_id,
  const Range<KeyType>& global_range,
  const DataType type) {
  // make job id
  JobID job_id = makeJobID(grp_id, global_range);

  // get column range
  SizeR column_range;
  if (Ocean::DataType::PARAMETER_KEY == type) {
    // locate column range
    SArray<KeyType> keys = input;
    column_range = keys.findRange(global_range);
    column_ranges_.addWithoutModify(job_id, column_range);
  } else {
    // find column range
    if (!column_ranges_.tryGet(job_id, column_range)) {
      LL << log_prefix_ << __PRETTY_FUNCTION__ <<
        "could not find column range for grp [" << grp_id <<
        "], global_range " << global_range.toString();
      return false;
    }
  }

  // full path
  string full_path = pickDirRandomly() + "/" + identity_ +
    "." << dataTypeToString(type) << "." << grp_id <<
    "." << global_range.begin() << "-" << global_range.end();

  // dump
  try {
    if (Ocean::DataType::FEATURE_KEY == type ||
        Ocean::DataType::PARAMETER_KEY == type) {
      SArray<KeyType> array = input;
      if (!array.segment(column_range).writeToFile(full_path)) {
        throw std::runtime_error("KeyType");
      }
    } else if (Ocean::DataType::FEATURE_VALUE == type ||
               Ocean::DataType::PARAMETER_VALUE == type ||
               Ocean::DataType::DELTA == type) {
      SArray<ValueType> array = input;
      if (!array.segment(column_range).writeToFile(full_path)) {
        throw std::runtime_error("ValueType");
      }
    } else if (Ocean::DataType::FEATURE_OFFSET == type) {
      SArray<OffsetType> array = input;
      if (!array.segment(column_range).writeToFile(full_path)) {
        throw std::runtime_error("OffsetType");
      }
    }
  } catch (std::exception& e) {
    LL << log_prefix_ << __PRETTY_FUNCTION__ <<
      "sarray writeToFile failed on path [" << full_path << "] column_range " <<
      column_range.toString() << " data type [" << dataTypeToString(type) << "]";
    return false;
  }

  // record
  lakes_[static_cast<size_t>(type)].addWithoutModify(job_id, full_path);

  return true;
}

void Ocean::prefetch(GrpID grp_id, const Range<KeyType>& key_range) {
  // enqueue
  JobID job_id = makeJobID(grp_id, key_range);
  pending_jobs_.push(job_id);

  // reference count
  job_status_[job_id].increaseRef();

  if (FLAGS_verbose) {
    LI << log_prefix_ << "add prefetch job; grp_id:" <<
      grp_id << ", range:" << key_range.toString();
  }
  return;
}

void Ocean::prefetchThreadFunc() {
  while (go_on_prefetching_) {
    // take out a job
    JobID job_id = 0;
    pending_jobs_.wait_and_pop(job_id);

    // check job status
    JobInfo job_info;
    if (job_status_.tryGet(job_id, job_info) &&
        (JobStatus::LOADING == job_info.status ||
         JobStatus::LOADED == job_info.status) {
      // already been fetched
      continue;
    }

    // hang on if prefetch limit reached
    {
      std::unique_lock<std::mutex> l(prefetch_limit_mu_);
      prefetch_limit_cond_.wait(
        l, []{loaded_data_.size() <= FLAGS_prefetch_job_limit});
    }

    // change job status
    job_info.setStatus(JobStatus::LOADING);
    job_status_.addAndModify(job_id, job_info);

    // load from disk
    LoadedData loaded = loadFromDiskSynchronously(job_id);
    if (loaded.valid()) {
      loaded_data_.addWithoutModify(job_id, loaded);
      job_info.setStatus(JobStatus::LOADED);
    } else {
      LL << log_prefix_ << "prefetch job failed. [" <<
        jobIDToString(job_id) << "]";
      job_info.setStatus(JobStatus::FAILED);
    }
    job_status_.addAndModify(job_info);
  };
}

}; // namespace PS
