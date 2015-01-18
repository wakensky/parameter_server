#include <unistd.h>
#include <fcntl.h>
#include "util/split.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdexcept>
#include <gperftools/malloc_extension.h>
#include "base/shared_array_inl.h"
#include "system/ocean.h"

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
}

Ocean::~Ocean() {
  go_on_ = false;
  // join prefetch threads
  for (auto& thread : prefetch_threads_) {
    for (int i = 0; i < FLAGS_num_threads + 1; ++i) {
      // pump in some illegal prefetch jobs
      //  push threads moving on
      prefetch(0, Range<FullKey>(), 0);
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
  UnitID unit_id(grp_id, global_range);

  // enqueue
  prefetch_queue_.push(std::make_pair(unit_id, task_id));
  prefetch_queue_not_empty_cv_.notify_all();
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

  if (UnitStatus::LOADED == accessor->second.status) {
    return accessor->second.data_pack;
  } else if (UnitStatus::LOADING == accessor->second.status) {
    // should not happen
    // since tbb::concurrent_hash_map::accessor would block
    //   if prefetch thread is loading this UnitBody
    CHECK(false) << "Ocean::get got LOADING status wrongly";
  } else {
    // record
    InUseTaskHashMap::const_accessor task_const_accessor;
    accessor->second.in_use_tasks.insert(task_const_accessor, task_id);

    // load
    LI << "Ocean::get loading unit [" << unit_id.toString() << "] synchronously";
    accessor->second.setStatus(UnitStatus::LOADING);
    CHECK(loadFromDiskSynchronously(
      unit_id, accessor->second.path_pack, &accessor->second.data_pack));
    accessor->second.setStatus(UnitStatus::LOADED);
    LI << "Ocean::get loaded unit [" << unit_id.toString() << "] synchronously";
  }
  return accessor->second.data_pack;
}

void Ocean::drop(
  const GroupID grp_id,
  const Range<FullKey>& global_range,
  const TaskID task_id) {
  UnitID unit_id(grp_id, global_range);
  UnitHashMap::accessor accessor;
  if (!units_.find(accessor, unit_id)) {
    LL << "Ocean::drop cannot find unit [" << unit_id.toString() << "]";
    return;
  }

  // remove finished task_id from in_use_tasks
  accessor->second.in_use_tasks.erase(task_id);

  // dump back to disk, release memory
  if (accessor->second.in_use_tasks.empty()) {
    accessor->second.setStatus(UnitStatus::DROPPING);
    accessor->second.setStatus(UnitStatus::DROPPED);
    in_memory_unit_count_--;
    // TODO write back to disk synchronously; write back to disk asynchronously
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

  // traverse all column partitioned units
  for (auto iterator = units_.begin();
       iterator != units_.end(); ++iterator) {
    SArray<FullKey> parameter_key;
    SArray<Value> parameter_value;

    if (UnitStatus::LOADED == iterator->second.status) {
      parameter_key = iterator->second.data_pack.arrays[
        static_cast<size_t>(DataSource::PARAMETER_KEY)];
      parameter_value = iterator->second.data_pack.arrays[
        static_cast<size_t>(DataSource::PARAMETER_VALUE)];
    } else {
      // load from disk
      SArray<char> key_stash;
      CHECK(key_stash.readFromFile(iterator->second.path_pack.path[
        static_cast<size_t>(DataSource::PARAMETER_KEY)]));
      parameter_key = key_stash;

      SArray<char> value_stash;
      CHECK(value_stash.readFromFile(iterator->second.path_pack.path[
        static_cast<size_t>(DataSource::PARAMETER_VALUE)]));
      parameter_value = value_stash;
    }
    CHECK_EQ(parameter_key.size(), parameter_value.size());

    for (size_t i = 0; i < parameter_key.size(); ++i) {
      double v = parameter_value[i];
      if (!(v != v || 0 == v)) {
        out << parameter_key[i] << "\t" << v << "\n";
      }
    }
  }
  return true;
}

void Ocean::prefetchThreadFunc() {
  while (go_on_) {
    // take out a UnitID which needs prefetch
    std::pair<UnitID, TaskID> prefetch_job;
    {
      std::unique_lock<std::mutex> l(prefetch_queue_mu_);
      prefetch_queue_not_empty_cv_.wait(
        l, [this, &prefetch_job]{return prefetch_queue_.try_pop(prefetch_job);});
    }
    UnitID unit_id = prefetch_job.first;
    TaskID task_id = prefetch_job.second;

    // hang on if in-memory unit count limit reached
    {
      std::unique_lock<std::mutex> l(in_memory_limit_mu_);
      in_memory_unit_not_full_cv_.wait(
        l, [this]{return in_memory_unit_count_ <= FLAGS_in_memory_unit_limit;});
    }

    if (FLAGS_verbose) {
      LI << "prefetch thread ready to prefetch unit [" <<
        unit_id.toString() << "]";
    }

    // make sure {grp_id, global_range} has already resides in units_
    // otherwise, I do not know how to load it
    UnitHashMap::accessor accessor;
    if (!units_.find(accessor, unit_id)) {
      LL << "prefetch thread cannot prefetch unit [" << unit_id.toString() << "]";
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
          "] since already LOADING/LOADED";
      }
      continue;
    }

    if (FLAGS_verbose) {
      LI << "prefetch thread starts prefetching unit [" <<
        unit_id.toString() << "]";
    }

    // load from disk
    accessor->second.setStatus(UnitStatus::LOADING);
    loadFromDiskSynchronously(
      unit_id, accessor->second.path_pack, &accessor->second.data_pack);
    accessor->second.setStatus(UnitStatus::LOADED);

    if (FLAGS_verbose) {
      LI << "prefetch thread finishes prefetching unit [" <<
        unit_id.toString() << "]";
    }
  };
}

bool Ocean::loadFromDiskSynchronously(
  const UnitID unit_id,
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

  // check
  SArray<Offset> offset(
    data_pack->arrays[static_cast<size_t>(DataSource::FEATURE_OFFSET)]);
  if (!offset.empty()) {
    SArray<ShortKey> feature_key(
      data_pack->arrays[static_cast<size_t>(DataSource::FEATURE_KEY)]);
    CHECK_EQ(offset.back() - offset.front(), feature_key.size());
  }

  in_memory_unit_count_++;
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
