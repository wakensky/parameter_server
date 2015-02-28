#include "system/ocean.h"
#include <unistd.h>
#include <fcntl.h>
#include <iomanip>
#include "util/split.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdexcept>
#include <gperftools/malloc_extension.h>
#include "base/shared_array_inl.h"

namespace PS {

DEFINE_int32(in_memory_unit_limit, 64,
  "control memory usage via limit in-memory unit count");

Ocean::Ocean() :
  go_on_(true),
  in_memory_unit_count_(0) {
  // launch prefetch threads
  for (int i = 0; i < FLAGS_num_threads; ++i) {
    prefetch_threads_.push_back(std::move(
      std::thread(&Ocean::prefetchThreadFunc, this)));
  }
  // launch write threads
  for (int i = 0; i < 4 * FLAGS_num_threads; ++i) {
    write_threads_.push_back(std::move(
      std::thread(&Ocean::writeThreadFunc, this)));
  }
}

Ocean::~Ocean() {
  go_on_ = false;
  // join prefetch threads
  for (auto& thread : prefetch_threads_) {
    for (int i = 0; i < FLAGS_num_threads + 1; ++i) {
      // pump in some illegal prefetch jobs
      //  push threads moving on
      prefetch(0, Range<FullKey>(), kFakeTaskID);
    }
    thread.join();
  }
  // join write threads
  for (auto& thread : write_threads_) {
    // pump in some illegal write jobs
    //   push threads moving on
    for (int i = 0; i < FLAGS_num_threads + 1; ++i) {
      write_queue_.push(nullptr);
    }
    thread.join();
  }
}

void Ocean::init(
  const string& identity, const LM::Config& conf,
  const Task& task, PathPicker* path_picker) {
  CHECK(!identity.empty());
  identity_ = identity;
  path_picker_ = path_picker;
  conf_ = conf;

  for (int i = 0; i < task.partition_info_size(); ++i) {
    Range<FullKey> global_range(task.partition_info(i).key());
    group_partition_ranges_[task.partition_info(i).fea_grp()].push_back(global_range);
  }
}

bool Ocean::dump(
  const GroupID grp_id,
  SArray<FullKey> parameter_key,
  SArray<Value> parameter_value,
  SArray<Value> delta,
  SparseMatrixPtr<ShortKey, Value> matrix) {
  group_key_count_[grp_id] = parameter_key.size();
  if (parameter_key.empty()) {
    LL << "parameter_key empty in group [" << grp_id << "]";
    return false;
  }
  auto iterator = group_partition_ranges_.find(grp_id);
  if (group_partition_ranges_.end() == iterator) {
    LL << "unknown group [" << grp_id << "] while Ocean dumpping";
    return false;
  }

  for (const auto& global_range : iterator->second) {
    SizeR in_group_anchor = parameter_key.findRange(global_range);

    UnitBody unit_body;
    unit_body.in_group_anchor = in_group_anchor;
    anchors_[UnitID(grp_id, global_range)] = in_group_anchor;

    dumpColumnPartitionedSArray(
      SArray<char>(parameter_key), UnitID(grp_id, global_range),
      in_group_anchor, DataSource::PARAMETER_KEY, &unit_body);
    dumpColumnPartitionedSArray(
      SArray<char>(parameter_value), UnitID(grp_id, global_range),
      in_group_anchor, DataSource::PARAMETER_VALUE, &unit_body);
    dumpColumnPartitionedSArray(
      SArray<char>(delta), UnitID(grp_id, global_range),
      in_group_anchor, DataSource::DELTA, &unit_body);

    if (matrix) { matrix_info_[grp_id] = matrix->info(); };
    if (matrix && !matrix->empty()) {
      SizeR feature_in_group_anchor(
        matrix->offset()[in_group_anchor.begin()],
        matrix->offset()[in_group_anchor.end()]);
      dumpColumnPartitionedSArray(
        SArray<char>(matrix->offset()), UnitID(grp_id, global_range),
        SizeR(in_group_anchor.begin(), in_group_anchor.end() + 1),
        DataSource::FEATURE_OFFSET, &unit_body);
      dumpColumnPartitionedSArray(
        SArray<char>(matrix->index()), UnitID(grp_id, global_range),
        feature_in_group_anchor, DataSource::FEATURE_KEY, &unit_body);
      dumpColumnPartitionedSArray(
        SArray<char>(matrix->value()), UnitID(grp_id, global_range),
        feature_in_group_anchor, DataSource::FEATURE_VALUE, &unit_body);

      auto segmented_offset = matrix->offset().segment(
        SizeR(in_group_anchor.begin(), in_group_anchor.end() + 1));
      auto segmented_feature_key = matrix->index().segment(feature_in_group_anchor);
      CHECK_EQ(
        segmented_offset.back() - segmented_offset.front(),
        segmented_feature_key.size());
      // wakensky
      LI << "dumped unit [" << UnitID(grp_id, global_range).toString() <<
        "]; off.front: " << segmented_offset.front() <<
        "; off.back: " << segmented_offset.back() <<
        "; fea_key.size: " << segmented_feature_key.size();
    }

    // record
    UnitHashMap::accessor accessor;
    units_.insert(accessor, UnitID(grp_id, global_range));
    accessor->second = unit_body;
  }
  return true;
}

bool Ocean::dumpColumnPartitionedSArray(
  SArray<char> in, const UnitID unit_id,
  const SizeR& anchor, const DataSource data_source,
  UnitBody* unit_body) {
  CHECK(nullptr != unit_body);
  if (in.empty()) {
    return true;
  }

  // generate dump path
  std::stringstream file_name;
  file_name << "blockcache." << identity_ <<
    "." << printDataSource(data_source) << "." << unit_id.grp_id <<
    "." << unit_id.global_range.begin() << "-" << unit_id.global_range.end();
  string full_path = path_picker_->getPath(file_name.str());

  // dump
  try {
    if (Ocean::DataSource::PARAMETER_KEY == data_source) {
      SArray<FullKey> array(in);
      if (!array.segment(anchor).writeToFile(full_path)) {
        throw std::runtime_error("FullKey");
      }
    } else if (Ocean::DataSource::FEATURE_KEY == data_source) {
      SArray<ShortKey> array(in);
      if (!array.segment(anchor).writeToFile(full_path)) {
        throw std::runtime_error("ShortKey");
      }
    } else if (Ocean::DataSource::FEATURE_VALUE == data_source ||
               Ocean::DataSource::PARAMETER_VALUE == data_source ||
               Ocean::DataSource::DELTA == data_source) {
      SArray<Value> array(in);
      if (!array.segment(anchor).writeToFile(full_path)) {
        throw std::runtime_error("Value");
      }
    } else if (Ocean::DataSource::FEATURE_OFFSET == data_source) {
      SArray<Offset> array(in);
      if (!array.segment(anchor).writeToFile(full_path)) {
        throw std::runtime_error("Offset");
      }
    }
  } catch (std::exception& e) {
    LL << "SArray writeToFile failed on path [" << full_path <<
      "] anchor [" << anchor.toString() <<
      "] DataSource [" << printDataSource(data_source) << "]";
    return false;
  }

  // record
  unit_body->path_pack.set(data_source, full_path);
  return true;
}

void Ocean::prefetch(
  const GroupID grp_id,
  const Range<FullKey>& global_range,
  const TaskID task_id) {
  if (kFakeTaskID != task_id && prefetched_tasks_.count(task_id) > 0) {
    return;
  }
  prefetched_tasks_.insert(task_id);
  UnitID unit_id(grp_id, global_range);

  // enqueue
  prefetch_queue_.push(std::make_pair(unit_id, task_id));

  // wakensky
  LI << "prefetch added [" << unit_id.toString() <<
    "] [" << task_id << "]";
}

Ocean::DataPack Ocean::get(
  const GroupID grp_id,
  const Range<FullKey>& global_range,
  const TaskID task_id) {
  UnitID unit_id(grp_id, global_range);
  UnitHashMap::accessor accessor;
  if (!units_.find(accessor, unit_id)) {
    LL << "Ocean::get cannot find unit [" << unit_id.toString() << "]";
    return DataPack();
  }

  LI << "Ocean::get tries to read unit [" << unit_id.toString() <<
    "] [" << task_id << "]";
  if (UnitStatus::LOADED == accessor->second.status) {
    return accessor->second.data_pack;
  } else if (UnitStatus::LOADING == accessor->second.status) {
    // should not happen
    // since tbb::concurrent_hash_map::accessor would block,
    //   while prefetch thread is loading this UnitBody
    CHECK(false) << "Ocean::get got LOADING status wrongly";
  } else {
    // record
    InUseTaskHashMap::const_accessor task_const_accessor;
    accessor->second.in_use_tasks.insert(task_const_accessor, task_id);

    // load
    LI << "Ocean::get loading unit [" << unit_id.toString() <<
      "] [" << task_id << "] synchronously";
    accessor->second.setStatus(UnitStatus::LOADING);
    CHECK(loadFromDiskSynchronously(
      unit_id, task_id, accessor->second.path_pack, &accessor->second.data_pack));
    accessor->second.setStatus(UnitStatus::LOADED);
    LI << "Ocean::get loaded unit [" << unit_id.toString() <<
      "] [" << task_id << "] synchronously";
  }
  return accessor->second.data_pack;
}

void Ocean::drop(
  const GroupID grp_id,
  const Range<FullKey>& global_range,
  const TaskID task_id) {
  UnitID unit_id(grp_id, global_range);
  std::shared_ptr<UnitHashMap::accessor> accessor_ptr(new UnitHashMap::accessor());
  if (!units_.find(*accessor_ptr, unit_id)) {
    // LL << "Ocean::drop cannot find unit [" << unit_id.toString() << "]";
    return;
  }

  // mark as loaded
  tasks_do_not_prefetch_.insert(task_id);

  // remove finished task_id from in_use_tasks
  (*accessor_ptr)->second.in_use_tasks.erase(task_id);

  // dump back to disk, release memory
  if ((*accessor_ptr)->second.in_use_tasks.empty()) {
    (*accessor_ptr)->second.setStatus(UnitStatus::DROPPING);
    write_queue_.push(accessor_ptr);
    LI << "Ocean::drop added unit to write queue [" << unit_id.toString() <<
      "] [" << task_id << "]";
  }
}

SizeR Ocean::fetchAnchor(
  const GroupID grp_id, const Range<FullKey>& global_range) {
  auto iterator = anchors_.find(UnitID(grp_id, global_range));
  if (anchors_.end() != iterator) {
    return iterator->second;
  } else {
    return SizeR();
  }
}

bool Ocean::saveModel(const string& path) {
  // open file
  std::ofstream out(path);
  CHECK(out.good());
  LI << "Ocean::saveModel is dumping model... ";

  // traverse all possible UnitID
  for (const auto& gid_ranges : group_partition_ranges_) {
    const GroupID group_id = gid_ranges.first;
    for (const auto& global_range : gid_ranges.second) {
      UnitID unit_id(group_id, global_range);
      UnitHashMap::accessor accessor;
      if (!units_.find(accessor, unit_id)) {
        continue;
      }

      // read model
      SArray<FullKey> parameter_key;
      SArray<Value> parameter_value;
      if (UnitStatus::LOADED == accessor->second.status) {
        parameter_key = accessor->second.data_pack.arrays[
          static_cast<size_t>(DataSource::PARAMETER_KEY)];
        parameter_value = accessor->second.data_pack.arrays[
          static_cast<size_t>(DataSource::PARAMETER_VALUE)];
      } else {
        SArray<char> key_stash;
        CHECK(key_stash.readFromFile(accessor->second.path_pack.path[
          static_cast<size_t>(DataSource::PARAMETER_KEY)]));
        parameter_key = key_stash;

        SArray<char> value_stash;
        CHECK(value_stash.readFromFile(accessor->second.path_pack.path[
          static_cast<size_t>(DataSource::PARAMETER_VALUE)]));
        parameter_value = value_stash;
      }
      CHECK_EQ(parameter_key.size(), parameter_value.size());

      // save model
      for (size_t i = 0; i < parameter_key.size(); ++i) {
        double v = parameter_value[i];
        if (!(v != v || 0 == v)) {
          out << parameter_key[i] << "\t" <<
            std::setprecision(std::numeric_limits<double>::digits10+2) <<
            v << "\n";
        }
      }
    }
  }
  LI << "Ocean::saveModel dumped model";

  return true;
}

bool Ocean::snapshot() {
  // blockcache
  std::stringstream ss;
  ss << identity_ << ".blockcache.ocean.guide";
  const string blockcache_guide_path = path_picker_->getPath(ss.str());
  File* blockcache_file = File::openOrDie(blockcache_guide_path, "w");
  for (const auto& unit : units_) {
    size_t data_source = 0;
    for (const auto& path : unit.second.path_pack.path) {
      if (!path.empty()) {
        std::stringstream line;
        line << unit.first.grp_id << "\t" <<
          unit.first.global_range.begin() << "\t" <<
          unit.first.global_range.end() << "\t" <<
          data_source << "\t" <<
          path << "\t" <<
          File::size(path) << "\n";
        CHECK_EQ(blockcache_file->writeString(line.str()), line.str().size());
      }
      data_source++;
    }
  }
  CHECK(blockcache_file->close());

  // anchor
  ss.str("");
  ss << identity_ << ".anchor.ocean.guide";
  const string anchor_guide_path = path_picker_->getPath(ss.str());
  File* anchor_file = File::openOrDie(anchor_guide_path, "w");
  for (const auto& anchor : anchors_) {
    std::stringstream line;
    line << anchor.first.grp_id << "\t" <<
      anchor.first.global_range.begin() << "\t" <<
      anchor.first.global_range.end() << "\t" <<
      anchor.second.begin() << "\t" <<
      anchor.second.end() << "\n";
    CHECK_EQ(anchor_file->writeString(line.str()), line.str().size());
  }
  CHECK(anchor_file->close());

  // group key count
  ss.str("");
  ss << identity_ << ".group_key_count.ocean.guide";
  const string count_guide_path = path_picker_->getPath(ss.str());
  File* count_file = File::openOrDie(count_guide_path, "w");
  for (const auto& count : group_key_count_) {
    std::stringstream line;
    line << count.first << "\t" << count.second << "\n";
    CHECK_EQ(count_file->writeString(line.str()), line.str().size());
  }
  CHECK(count_file->close());

  // MatrixInfo
  // one summary file
  // some proto files
  ss.str("");
  ss << identity_ << ".matrix_info.ocean.guide.summary";
  const string matrix_summary_path = path_picker_->getPath(ss.str());
  File* matrix_summary_file = File::openOrDie(matrix_summary_path, "w");
  for (const auto& matrix_info : matrix_info_) {
    // matrix proto file
    ss.str("");
    ss << identity_ << ".matrix_info.ocean.guide." << matrix_info.first;
    const string proto_path = path_picker_->getPath(ss.str());
    writeProtoToASCIIFileOrDie(matrix_info.second, proto_path);

    // record in matrix summary file
    ss.str("");
    ss << matrix_info.first << "\t" << proto_path << "\n";
    CHECK_EQ(matrix_summary_file->writeString(ss.str()), ss.str().size());
  }
  CHECK(matrix_summary_file->close());

  return true;
}

bool Ocean::resume() {
  std::stringstream ss;
  ss << identity_ << ".blockcache.ocean.guide";
  const string blockcache_guide_path = path_picker_->getPath(ss.str());
  ss.str("");
  ss << identity_ << ".anchor.ocean.guide";
  const string anchor_guide_path = path_picker_->getPath(ss.str());
  ss.str("");
  ss << identity_ << ".group_key_count.ocean.guide";
  const string count_guide_path = path_picker_->getPath(ss.str());
  ss.str("");
  ss << identity_ << ".matrix_info.ocean.guide.summary";
  const string matrix_summary_path = path_picker_->getPath(ss.str());

  if (!File::exists(blockcache_guide_path.c_str()) ||
      !File::exists(anchor_guide_path.c_str()) ||
      !File::exists(count_guide_path.c_str()) ||
      !File::exists(matrix_summary_path.c_str())) {
    LL << "Ocean::resume: key guide files not found";
    return false;
  }

  const size_t kBufLen = 2048;
  char buf[kBufLen + 1];

  // blockcache
  File* blockcache_file = File::openOrDie(blockcache_guide_path, "r");
  while (nullptr != blockcache_file->readLine(buf, kBufLen)) {
    string line(buf);

    // remove tailing line-break
    if (!line.empty() && '\n' == line.back()) {
      line.resize(line.size() - 1);
    }

    try {
      const auto vec = split(line, '\t');
      if (6 != vec.size()) {
        throw std::runtime_error("wrong column number (blockcache)");
      }

      UnitID unit_id;
      unit_id.grp_id = std::stoul(vec.at(0));
      unit_id.global_range.set(std::stoull(vec.at(1)), std::stoull(vec.at(2)));
      const size_t data_source = std::stoul(vec.at(3));
      const string path = vec.at(4);
      const size_t size = std::stoull(vec.at(5));
      if (data_source >= static_cast<size_t>(DataSource::NUM)) {
        throw std::runtime_error("illegal data source");
      }
      if (DataSource::PARAMETER_VALUE != static_cast<DataSource>(data_source) &&
          DataSource::DELTA != static_cast<DataSource>(data_source)) {
        if (size != File::size(path)) {
          throw std::runtime_error(
            std::to_string(size) + " vs " + std::to_string(File::size(path)) +
            " file size does not match");
        }
      } else {
        // truncate mutable files to empty
        File *truncate_file = File::openOrDie(path, "w");
        truncate_file->close();
      }

      UnitHashMap::accessor accessor;
      if (units_.find(accessor, unit_id)) {
      } else {
        units_.insert(accessor, unit_id);
      }
      accessor->second.path_pack.path.at(data_source) = path;
    } catch (std::exception& e) {
      LL << "Ocean::resume encountered wrong blockcache info [" << line <<
        "] [" << e.what() << "]";
      blockcache_file->close();
      return false;
    }
  }
  blockcache_file->close();

  // anchor
  File* anchor_file = File::openOrDie(anchor_guide_path, "r");
  anchors_.clear();
  while (nullptr != anchor_file->readLine(buf, kBufLen)) {
    string line(buf);

    // remove tailing line-break
    if (!line.empty() && '\n' == line.back()) {
      line.resize(line.size() - 1);
    }

    try {
      const auto vec = split(line, '\t');
      if (5 !=vec.size()) {
        throw std::runtime_error("wrong column number (anchor)");
      }

      UnitID unit_id;
      unit_id.grp_id = std::stoul(vec.at(0));
      unit_id.global_range.set(std::stoul(vec.at(1)), std::stoul(vec.at(2)));
      SizeR anchor(std::stoull(vec.at(3)), std::stoull(vec.at(4)));

      anchors_[unit_id] = anchor;
    } catch (std::exception& e) {
      LL << "Ocean::resume encountered wrong anchor info [" << line <<
        "] [" << e.what() << "]";
      anchor_file->close();
      return false;
    }
  }
  anchor_file->close();

  // group key count
  File* count_file = File::openOrDie(count_guide_path, "r");
  group_key_count_.clear();
  while (nullptr != count_file->readLine(buf, kBufLen)) {
    string line(buf);

    // remove tailing line-break
    if (!line.empty() && '\n' == line.back()) {
      line.resize(line.size() - 1);
    }

    try {
      const auto vec = split(line, '\t');
      if (2 != vec.size()) {
        throw std::runtime_error("wrong column number (group key count)");
      }

      const GroupID grp_id = std::stoul(vec.at(0));
      const size_t count = std::stoull(vec.at(1));

      group_key_count_[grp_id] = count;
    } catch (std::exception& e) {
      LL << "Ocean::resume encountered wrong group key count info [" << line <<
        "] [" << e.what() << "]";
      count_file->close();
      return false;
    }
  }
  count_file->close();

  // matrix info files
  File* matrix_summary_file = File::openOrDie(matrix_summary_path, "r");
  matrix_info_.clear();
  while (nullptr != matrix_summary_file->readLine(buf, kBufLen)) {
    string line(buf);

    // remove tailing line-break
    if (!line.empty() && '\n' == line.back()) {
      line.resize(line.size() - 1);
    }

    try {
      const auto vec = split(line, '\t');
      if (2 != vec.size()) {
        throw std::runtime_error("wrong column number (matrix info summary)");
      }

      const GroupID grp_id = std::stoul(vec.at(0));
      const string proto_path = vec.at(1);

      MatrixInfo info;
      readFileToProtoOrDie(proto_path, &info);
      matrix_info_[grp_id] = info;
    } catch (std::exception& e) {
      LL << "Ocean::resume encountered wrong matrix summary info [" <<
        line << "] [" << e.what() << "]";
      matrix_summary_file->close();
      return false;
    }
  }
  matrix_summary_file->close();

  return true;
}

size_t Ocean::getGroupKeyCount(const GroupID grp_id) {
  auto iter = group_key_count_.find(grp_id);
  if (group_key_count_.end() != iter) {
    return iter->second;
  }
  return 0;
}

bool Ocean::matrix_binary(const GroupID grp_id) {
  return MatrixInfo::SPARSE_BINARY == matrix_info_[grp_id].type();
}

size_t Ocean::matrix_rows(const GroupID grp_id) {
  return matrix_info_[grp_id].row().end() - matrix_info_[grp_id].row().begin();
}

int Ocean::pendingPrefetchCount() {
  return prefetch_queue_.size();
}

void Ocean::prefetchThreadFunc() {
  while (go_on_) {
    // take out a UnitID which needs prefetch
    std::pair<UnitID, TaskID> prefetch_job;
    prefetch_queue_.pop(prefetch_job);
    UnitID unit_id = prefetch_job.first;
    TaskID task_id = prefetch_job.second;
    if (tasks_do_not_prefetch_.count(task_id) > 0) {
      if (FLAGS_verbose) {
        LI << "prefetch thread skipped unit since associated task has been loaded [" <<
          unit_id.toString() << "] [" << task_id << "]";
      }
      continue;
    }

    // hang on if in-memory unit count limit reached
    LI << "in_memory_unit_count_: " << in_memory_unit_count_;
    {
      std::unique_lock<std::mutex> l(in_memory_limit_mu_);
      in_memory_unit_not_full_cv_.wait(
        l, [this]{return in_memory_unit_count_ <= FLAGS_in_memory_unit_limit;});
    }

    if (FLAGS_verbose) {
      LI << "prefetch thread ready to prefetch unit [" <<
        unit_id.toString() << "] [" << task_id << "]";
    }

    // make sure {grp_id, global_range} has already resides in units_
    // otherwise, I do not know how to load it
    UnitHashMap::accessor accessor;
    if (!units_.find(accessor, unit_id)) {
#if 0
      LL << "prefetch thread cannot prefetch unit [" <<
        unit_id.toString() << "] [" << task_id << "]";
#endif
      continue;
    }

    // append TaskID to in_use_tasks
    InUseTaskHashMap::const_accessor task_const_accessor;
    accessor->second.in_use_tasks.insert(task_const_accessor, task_id);

    // check unit status
    // skip those units who are being loaded or has been loaded
    if (UnitStatus::LOADING == accessor->second.getStatus() ||
        UnitStatus::LOADED == accessor->second.getStatus()) {
      if (FLAGS_verbose) {
        LI << "prefetch thread finishes unit [" << unit_id.toString() <<
          "] [" << task_id << "] since already LOADING/LOADED";
      }
      continue;
    }

    if (FLAGS_verbose) {
      LI << "prefetch thread starts prefetching unit [" <<
        unit_id.toString() << "] [" << task_id << "]";
    }

    // load from disk
    accessor->second.setStatus(UnitStatus::LOADING);
    loadFromDiskSynchronously(
      unit_id, task_id, accessor->second.path_pack, &accessor->second.data_pack);
    accessor->second.setStatus(UnitStatus::LOADED);

    if (FLAGS_verbose) {
      LI << "prefetch thread finishes prefetching unit [" <<
        unit_id.toString() << "] [" << task_id << "]";
    }
  };
}

void Ocean::writeThreadFunc() {
  while (go_on_) {
    // take out an UnitHash::accessor which needs write mutable data back to disk
    std::shared_ptr<UnitHashMap::accessor> accessor_ptr;
    write_queue_.pop(accessor_ptr);
    if (!accessor_ptr) {
      continue;
    }

    UnitBody& unit_body = (*accessor_ptr)->second;
    // write back parameter_value and delta
    for (size_t data_source = 0;
         data_source < static_cast<size_t>(DataSource::NUM);
         ++data_source) {
      if (DataSource::PARAMETER_VALUE == static_cast<DataSource>(data_source) ||
          DataSource::DELTA == static_cast<DataSource>(data_source)) {
        SArray<char> array = unit_body.data_pack.arrays.at(data_source);
        if (!array.empty()) {
          CHECK(array.writeToFile(unit_body.path_pack.path.at(data_source), false));
        }
        CHECK_EQ(array.size(), File::size(unit_body.path_pack.path.at(data_source)));
      }
    }

    // memory release
    unit_body.data_pack.clear();
    in_memory_unit_count_--;
    in_memory_unit_not_full_cv_.notify_all();

    // mark as dropped
    unit_body.setStatus(UnitStatus::DROPPED);

    LI << "Ocean::writeThreadFunc dropped unit: [" <<
      (*accessor_ptr)->first.toString() << "]";
  };
}

bool Ocean::loadFromDiskSynchronously(
  const UnitID unit_id,
  const TaskID task_id,
  const PathPack& path_pack,
  Ocean::DataPack* data_pack) {
  CHECK(nullptr != data_pack);
  data_pack->clear();

  for (size_t data_source = 0;
       data_source < static_cast<size_t>(DataSource::NUM);
       ++data_source) {
    string path = path_pack.path[data_source];
    if (path.empty()) {
      continue;
    }
    CHECK(data_pack->arrays[data_source].readFromFile(path));
  }

  // adjust
  // If mutable data got empty dump file and associated parameter_key is NOT empty,
  // I should allocate mutable data on the fly
  SArray<FullKey> parameter_key(data_pack->arrays[static_cast<size_t>(DataSource::PARAMETER_KEY)]);
  SArray<Value> parameter_value(data_pack->arrays[static_cast<size_t>(DataSource::PARAMETER_VALUE)]);
  SArray<Value> delta(data_pack->arrays[static_cast<size_t>(DataSource::DELTA)]);
  if (!parameter_key.empty() && parameter_value.empty() &&
      !path_pack.path[static_cast<size_t>(DataSource::PARAMETER_VALUE)].empty()) {
    parameter_value.resize(parameter_key.size());
    parameter_value.setValue(0);
    data_pack->arrays[static_cast<size_t>(DataSource::PARAMETER_VALUE)] = parameter_value;
  }
  if (!parameter_key.empty() && delta.empty() &&
      !path_pack.path[static_cast<size_t>(DataSource::DELTA)].empty()) {
    delta.resize(parameter_key.size());
    delta.setValue(conf_.darling().delta_init_value());
    data_pack->arrays[static_cast<size_t>(DataSource::DELTA)] = delta;
  }

  // check
  SArray<Offset> offset(
    data_pack->arrays[static_cast<size_t>(DataSource::FEATURE_OFFSET)]);
  if (!offset.empty()) {
    SArray<ShortKey> feature_key(
      data_pack->arrays[static_cast<size_t>(DataSource::FEATURE_KEY)]);
    CHECK_EQ(offset.size(), parameter_key.size() + 1);
    CHECK_EQ(offset.back() - offset.front(), feature_key.size());
  }
  if (!parameter_value.empty()) {
    CHECK_EQ(parameter_key.size(), parameter_value.size());
  }
  if (!delta.empty()) {
    CHECK_EQ(parameter_key.size(), delta.size());
  }

  in_memory_unit_count_++;
  tasks_do_not_prefetch_.insert(task_id);
  return true;
}

string Ocean::printDataSource(const DataSource data_source) {
  switch (data_source) {
    case Ocean::DataSource::FEATURE_KEY:
      return "FEATURE_KEY";
    case Ocean::DataSource::FEATURE_OFFSET:
      return "FEATURE_OFFSET";
    case Ocean::DataSource::FEATURE_VALUE:
      return "FEATURE_VALUE";
    case Ocean::DataSource::DELTA:
      return "DELTA";
    case Ocean::DataSource::PARAMETER_KEY:
      return "PARAMETER_KEY";
    case Ocean::DataSource::PARAMETER_VALUE:
      return "PARAMETER_VALUE";
    default:
      return "UNKNOWN_DATASOURCE";
  };
}






};  // namespace PS
