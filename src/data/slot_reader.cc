#include <sys/stat.h>
#include <fcntl.h>
#include "data/slot_reader.h"
#include "data/example_parser.h"
#include "util/threadpool.h"
#include "util/filelinereader.h"
#include "util/split.h"
#include "util/parallel_sort.h"
namespace PS {

DEFINE_int32(load_data_max_mb_per_thread,
  2 * 1024,
  "maximum memory usage (MB) allowed for each worker thread "
  "while downloading data (LOAD_DATA)");
DECLARE_bool(verbose);

void SlotReader::init(const DataConfig& data, const DataConfig& cache) {
  CHECK(data.format() == DataConfig::TEXT);
  if (cache.file_size()) dump_to_disk_ = true;
  data_ = data;
  addDirectories(cache);
  CHECK(!directories_.empty());
  rng_.seed(time(0));
}

string SlotReader::cacheName(const DataConfig& data, int slot_id) const {
  CHECK_GT(data.file_size(), 0);
  return std::to_string(DJBHash32(data.file(0))) + "-" + \
    getFilename(data.file(0)) + "_slot_" + std::to_string(slot_id);
}

size_t SlotReader::nnzEle(int slot_id) const {
  size_t nnz = 0;
  for (int i = 0; i < info_.slot_size(); ++i) {
    if (info_.slot(i).id() == slot_id) nnz = info_.slot(i).nnz_ele();
  }
  return nnz;
}

int SlotReader::read(ExampleInfo* info) {
  CHECK_GT(FLAGS_num_threads, 0);
  {
    Lock l(mu_);
    loaded_file_count_ = 0;
  }
  {
    if (FLAGS_verbose) {
      for (size_t i = 0; i < data_.file_size(); ++i) {
        LI << "I will load data file [" << i + 1 << "/" <<
          data_.file_size() << "] [" << data_.file(i) << "]";
      }
    }

    ThreadPool pool(FLAGS_num_threads);
    for (int i = 0; i < data_.file_size(); ++i) {
      auto one_file = ithFile(data_, i);
      pool.add([this, one_file](){ readOneFile(one_file); });
    }
    pool.startWorkers();
  }
  if (info) *info = info_;
  for (int i = 0; i < info_.slot_size(); ++i) {
    slot_info_[info_.slot(i).id()] = info_.slot(i);
  }
  return 0;
}

bool SlotReader::readOneFile(const DataConfig& data) {
  if (FLAGS_verbose) {
    Lock l(mu_);
    LI << "loading data file [" << data.file(0) << "]; loaded [" <<
      loaded_file_count_ << "/" << data_.file_size() << "]";
  }

  string info_name = fullPath(
    std::to_string(DJBHash32(data.file(0))) +
    "-" + getFilename(data.file(0)) + ".info");
  ExampleInfo info;
  if (readFileToProto(info_name, &info)) {
    // the data is already in cache_dir
    Lock l(mu_);
    info_ = mergeExampleInfo(info_, info);
    if (FLAGS_verbose) {
      LI << "loaded data file [" << data.file(0) << "]; loaded [" <<
        loaded_file_count_ << "/" << data_.file_size() << "]";
    }
    return true;
  }

  ExampleParser parser;
  parser.init(data.text(), data.ignore_feature_group());
  struct VSlot {
    SArray<float> val;
    SArray<uint64> col_idx;
    SArray<uint16> row_siz;
    // how many elements in row_siz has been created since construction
    //  used for zero-padding
    size_t row_siz_total_size;
    bool writeToFile(const string& name) {
      return val.compressTo().writeToFile(name+".value")
          && col_idx.compressTo().writeToFile(name+".colidx")
          && row_siz.compressTo().writeToFile(name+".rowsiz");
    }
    VSlot() : row_siz_total_size(0) {
      // do nothing
    }
    void clear() {
      val.clear();
      col_idx.clear();
      row_siz.clear();
    }
    size_t memSize() {
      return val.memSize() + col_idx.memSize() + row_siz.memSize();
    }
    bool appendToFile(SlotReader* reader, const string& name) {
      CHECK(nullptr != reader);
      // lambda: append "start\tsize\n" to the file that contains partition info
      //   We named a compressed block "partition" here
      auto appendPartitionInfo = [] (
        const size_t start, const size_t size, const string& partition_file_path) {
        std::stringstream ss;
        ss << start << "\t" << size << "\n";
        File *file = File::open(partition_file_path, "a+");
        file->writeString(ss.str());
        file->close();
      };

      string file_name = reader->fullPath(name + ".value");
      size_t start = File::size(file_name);
      auto val_compressed = val.compressTo();
      CHECK(val_compressed.appendToFile(file_name));
      appendPartitionInfo(start, val_compressed.dataMemSize(), file_name + ".partition");

      file_name = reader->fullPath(name + ".colidx");
      start = File::size(file_name);
      auto col_compressed = col_idx.compressTo();
      CHECK(col_compressed.appendToFile(file_name));
      appendPartitionInfo(start, col_compressed.dataMemSize(), file_name + ".partition");

      file_name = reader->fullPath(name + ".rowsiz");
      start = File::size(file_name);
      auto row_compressed = row_siz.compressTo();
      CHECK(row_compressed.appendToFile(file_name));
      appendPartitionInfo(start, row_compressed.dataMemSize(), file_name + ".partition");

      // sort and unique
      parallelSort(&col_idx, FLAGS_num_threads, [](const uint64& a, const uint64& b) {
        return a < b; });
      uint64* new_end = std::unique(col_idx.begin(), col_idx.end());
      file_name = reader->fullPath(name + ".colidx_sorted_uniq");
      start = File::size(file_name);
      auto uniq_compressed =
        col_idx.segment(SizeR(0, new_end - col_idx.begin())).compressTo();
      CHECK(uniq_compressed.appendToFile(file_name));
      appendPartitionInfo(
        start, uniq_compressed.dataMemSize(), file_name + ".partition");

      return true;
    }
  };
  VSlot vslots[kSlotIDmax];
  uint32 num_ex = 0;
  Example ex;

  // first parse data into slots
  std::function<void(char*)> handle = [&] (char *line) {
    if (!parser.toProto(line, &ex)) return;
    // store them in slots
    for (int i = 0; i < ex.slot_size(); ++i) {
      const auto& slot = ex.slot(i);
      CHECK_LT(slot.id(), kSlotIDmax);
      auto& vslot = vslots[slot.id()];
      int key_size = slot.key_size();
      for (int j = 0; j < key_size; ++j) vslot.col_idx.pushBack(slot.key(j));
      int val_size = slot.val_size();
      for (int j = 0; j < val_size; ++j) vslot.val.pushBack(slot.val(j));
      while (vslot.row_siz_total_size < num_ex) {
        vslot.row_siz.pushBack(0);
        ++vslot.row_siz_total_size;
      }
      vslot.row_siz.pushBack(std::max(key_size, val_size));
      ++vslot.row_siz_total_size;
    }
    ++ num_ex;
  };

  // invoked by FileLineReader every N lines
  std::function<void(void*)> periodicity_check_handle = [&] (void*) {
    // check memory usage
    size_t mem_size = 0;
    for (auto& slot : vslots) {
      mem_size += slot.memSize();
    }
    if (mem_size / 1024 / 1024 < FLAGS_load_data_max_mb_per_thread ||
        0 == FLAGS_load_data_max_mb_per_thread) {
      return;
    }

    // log
    if (FLAGS_verbose) {
      LI << "dumping vslots ... ";
    }

    // dump memory image to file
    // release vslots
    for (int i = 0; i < kSlotIDmax; ++i) {
      auto& slot = vslots[i];
      if (slot.row_siz.empty() && slot.val.empty()) {
        continue;
      }

      CHECK(slot.appendToFile(this, cacheName(data, i)));
      slot.clear();
    }

    // log
    if (FLAGS_verbose) {
      LI << "vslots dumped";
    }
  };

  FileLineReader reader(data);
  reader.set_line_callback(handle);
  reader.set_periodicity_callback(periodicity_check_handle);
  reader.Reload();

  // save in cache
  info = parser.info();
  writeProtoToASCIIFileOrDie(info, info_name);
  for (int i = 0; i < kSlotIDmax; ++i) {
    auto& vslot = vslots[i];
    if (vslot.row_siz.empty() && vslot.val.empty()) continue;
    // zero-padding
    while (vslot.row_siz_total_size < num_ex) {
      vslot.row_siz.pushBack(0);
      ++vslot.row_siz_total_size;
    }
    CHECK(vslot.appendToFile(this, cacheName(data, i)));
  }
  {
    Lock l(mu_);
    info_ = mergeExampleInfo(info_, info);
    loaded_file_count_++;

    if (FLAGS_verbose) {
      LI << "loaded data file [" << data.file(0) << "]; loaded [" <<
        loaded_file_count_ << "/" << data_.file_size() << "]";
    }
  }
  return true;
}

SArray<uint64> SlotReader::index(int slot_id) {
  auto nnz = nnzEle(slot_id);
  if (nnz == 0) return SArray<uint64>();
  SArray<uint64> idx = index_cache_[slot_id];
  if (idx.size() == nnz) return idx;
  for (int i = 0; i < data_.file_size(); ++i) {
    string file = fullPath(cacheName(ithFile(data_, i), slot_id) + ".colidx");
    SArray<char> comp; CHECK(comp.readFromFile(file));
    SArray<uint64> uncomp;
    {
      SArray<char> buffer;
      CHECK(assemblePartitions(buffer, comp, file + ".partition"));
      uncomp = buffer;
    }
    idx.append(uncomp);
  }
  CHECK_EQ(idx.size(), nnz);
  index_cache_[slot_id] = idx;
  return idx;
}

SArray<size_t> SlotReader::offset(int slot_id) {
  if (offset_cache_[slot_id].size() == info_.num_ex()+1) {
    return offset_cache_[slot_id];
  }
  SArray<size_t> os(1); os[0] = 0;
  if (nnzEle(slot_id) == 0) return os;
  for (int i = 0; i < data_.file_size(); ++i) {
    string file = fullPath(cacheName(ithFile(data_, i), slot_id) + ".rowsiz");
    SArray<char> comp; CHECK(comp.readFromFile(file));
    SArray<uint16> uncomp;
    {
      SArray<char> buffer;
      CHECK(assemblePartitions(buffer, comp, file + ".partition"));
      uncomp = buffer;
    }
    size_t n = os.size();
    os.resize(n + uncomp.size());
    for (size_t i = 0; i < uncomp.size(); ++i) os[i+n] = os[i+n-1] + uncomp[i];
  }
  CHECK_EQ(os.size(), info_.num_ex()+1);
  offset_cache_[slot_id] = os;
  return os;
}

bool SlotReader::assemblePartitions(
  SArray<char>& out, SArray<char>& in, const string& partition_file_name) const {
  if (in.empty()) {
    out.clear();
    return true;
  }

  // if no corresponding partition file exists,
  //   we simply uncompress the whole SArray
  if (0 == File::size(partition_file_name)) {
    out.uncompressFrom(in);
    return true;
  }

  // store partitions' info: {start_byte, size}
  File* partition_file = File::openOrDie(partition_file_name, "r");
  std::vector<std::pair<size_t, size_t> > partitions;
  const size_t kLineMaxLen = 1024;
  char line_buffer[kLineMaxLen];
  while (nullptr != partition_file->readLine(line_buffer, kLineMaxLen)) {
    auto vec = split(line_buffer, '\t');
    CHECK_EQ(2, vec.size());

    partitions.push_back(std::make_pair(
      std::stoull(vec[0], nullptr),
      std::stoull(vec[1], nullptr)));
  }
  partition_file->close();
  CHECK(!partitions.empty());

  // decompress each partition, merge into output
  for (const auto& partition : partitions) {
    CHECK_LE(partition.first + partition.second, in.memSize());
    SArray<char> decompressed;
    decompressed.uncompressFrom(
      in.data() + partition.first, partition.second);
    out.append(decompressed);
  }

  return true;
}

void SlotReader::addDirectories(const DataConfig& cache) {
  CHECK(cache.file_size() > 0);

  for (size_t i = 0; i < cache.file_size(); ++i) {
    // check permission
    struct stat st;
    if (-1 == stat(cache.file(i).c_str(), &st)) {
      LL << "dir [" << cache.file(i) << "] cannot be added since error [" <<
        strerror(errno) << "]";
      continue;
    }
    if (!S_ISDIR(st.st_mode)) {
      LL << "dir [" << cache.file(i) << "] is not a regular directory";
      continue;
    }
    if (0 != access(cache.file(i).c_str(), R_OK | W_OK)) {
      LL << "I donnot have read&write permission on dir [" <<
        cache.file(i) << "]";
      continue;
    }

    // add
    directories_.push_back(cache.file(i));
  }
}

string SlotReader::fullPath(const string& file_name) {
  CHECK(!directories_.empty());

  struct stat st;
  string dir_path;
  for (const auto& dir : directories_) {
    if (-1 != stat((dir + "/" + file_name).c_str(), &st)) {
      dir_path = dir;
      break;
    }
  }

  if (!dir_path.empty()) {
    return dir_path + "/" + file_name;
  } else {
    std::uniform_int_distribution<size_t> dist(0, directories_.size() - 1);
    const size_t random_idx = dist(rng_);
    return directories_.at(random_idx) + "/" + file_name;
  }

}

SlotReader::DataPack SlotReader::nextPartition(
  const int slot_id, const LoadMode load_mode) {
  CHECK_LT(slot_id, kSlotIDmax);
  DataPack dp;

  auto& locator = partition_locator_[slot_id];
  locator.partition_idx++;

  if (locator.partition_idx >= locator.partition_count) {
    locator.file_idx++;
    if (locator.file_idx < data_.file_size()) {
      // load next data file's partitions
      loadPartitionRanges(locator.file_idx, slot_id);
      // initialize locator
      locator.partition_idx = 0;
      string file_name = fullPath(
        cacheName(ithFile(data_, locator.file_idx), slot_id) + ".colidx.partition");
      locator.partition_count = partition_ranges_[file_name].size();
    } else {
      // all partitions exhausted
      return dp;
    }
  }

  dp.is_ok = true;
  // load current partition
  if (load_mode & LoadMode::COLIDX) {
    string file = fullPath(
      cacheName(ithFile(data_, locator.file_idx), slot_id) + ".colidx");
    SArray<char> compressed;
    auto my_range = partition_ranges_[file + ".partition"].at(locator.partition_idx);
    compressed.readFromFile(my_range, file);
    dp.colidx.uncompressFrom(compressed);
  }
  if (load_mode & LoadMode::UNIQ_COLIDX) {
    string file = fullPath(
      cacheName(ithFile(data_, locator.file_idx), slot_id) + ".colidx_sorted_uniq");
    SArray<char> compressed;
    auto my_range = partition_ranges_[file + ".partition"].at(locator.partition_idx);
    compressed.readFromFile(my_range, file);
    dp.uniq_colidx.uncompressFrom(compressed);
  }
  if (load_mode & LoadMode::ROWSIZ) {
    string file = fullPath(
      cacheName(ithFile(data_, locator.file_idx), slot_id) + ".rowsiz");
    SArray<char> compressed;
    auto my_range = partition_ranges_[file + ".partition"].at(locator.partition_idx);
    compressed.readFromFile(my_range, file);
    dp.rowsiz.uncompressFrom(compressed);
  }
  if (load_mode & LoadMode::VALUE) {
    string file = fullPath(
      cacheName(ithFile(data_, locator.file_idx), slot_id) + ".val");
    SArray<char> compressed;
    auto my_range = partition_ranges_[file + ".partition"].at(locator.partition_idx);
    compressed.readFromFile(my_range, file);
    dp.val.uncompressFrom(compressed);
  }

  return dp;
}

void SlotReader::returnToFirstPartition(const int slot_id) {
  CHECK_LT(slot_id, kSlotIDmax);
  partition_locator_[slot_id].file_idx = -1;
  partition_locator_[slot_id].partition_idx = 0;
  partition_locator_[slot_id].partition_count = 0;
}

void SlotReader::loadPartitionRanges(
  const int file_idx, const int slot_id) {
  auto getRangeVector = [](const string& full_path) -> std::vector<SizeR> {
    std::vector<SizeR> ret_vec;
    File* partition_file = File::openOrDie(full_path, "r");
    const size_t kLineMaxLen = 1024;
    char line_buffer[kLineMaxLen];
    while (nullptr != partition_file->readLine(line_buffer, kLineMaxLen)) {
      auto tmp_vec = split(line_buffer, '\t');
      CHECK_EQ(2, tmp_vec.size());
      ret_vec.push_back(SizeR(
        std::stoull(tmp_vec[0], nullptr),
        std::stoull(tmp_vec[1], nullptr)));
    };
    return ret_vec;
  };

  const string prefix = cacheName(ithFile(data_, file_idx), slot_id);
  partition_ranges_[fullPath(prefix + ".colidx.partition")] =
    getRangeVector(fullPath(prefix + ".colidx.partition"));
  partition_ranges_[fullPath(prefix + ".colidx_sorted_uniq.partition")] =
    getRangeVector(fullPath(prefix + ".colidx_sorted_uniq.partition"));
  partition_ranges_[fullPath(prefix + ".rowsiz.partition")] =
    getRangeVector(fullPath(prefix + ".rowsiz.partition"));
  partition_ranges_[fullPath(prefix + ".value.partition")] =
    getRangeVector(fullPath(prefix + ".value.partition"));
}
} // namespace PS
