#include "data/slot_reader.h"
#include "data/example_parser.h"
#include "util/threadpool.h"
#include "util/filelinereader.h"
#include "util/split.h"
namespace PS {

DEFINE_int32(load_data_max_mb_per_thread,
  2 * 1024,
  "maximum memory usage (MB) allowed for each worker thread "
  "while downloading data (LOAD_DATA)");
DEFINE_int32(num_downloading_threads,
  8,
  "number of threads cooperating while downloading training data");
DECLARE_bool(verbose);

void SlotReader::init(const DataConfig& data, const DataConfig& cache,
  PathPicker* path_picker, KVVectorPtr<Key,double> w,
  const int start_time, const int finishing_time,
  const int count_min_k, const float count_min_n,
  const string& identity) {
  CHECK(data.format() == DataConfig::TEXT);
  if (cache.file_size()) dump_to_disk_ = true;
  data_ = data;
  path_picker_ = path_picker;
  w_ = w;
  time_ = start_time;
  finishing_time_ = finishing_time;
  count_min_k_ = count_min_k;
  count_min_n_ = count_min_n;
  identity_ = identity;
}

string SlotReader::cacheName(const DataConfig& data, int slot_id) const {
  CHECK_GT(data.file_size(), 0);
  return identity_ + "-" + std::to_string(DJBHash32(data.file(0))) + "-" + \
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

    ThreadPool pool(FLAGS_num_downloading_threads);
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

  if (w_) {
    // send the closing message to servers
    MessagePtr boundary(new Message(kServerGroup, finishing_time_));
    CHECK_EQ(finishing_time_, w_->push(boundary));
  }
  return 0;
}

bool SlotReader::readOneFile(const DataConfig& data) {
  if (FLAGS_verbose) {
    LI << "loading data file [" << data.file(0) << "]; loaded [" <<
      loaded_file_count_ << "/" << data_.file_size() << "]";
  }

  string info_name = path_picker_->getPath(
    std::to_string(DJBHash32(data.file(0))) +
    "-" + getFilename(data.file(0)) + ".info");
  ExampleInfo info;
  if (readFileToProto(info_name, &info)) {
    // the data is already in cache_dir
    Lock l(mu_);
    info_ = mergeExampleInfo(info_, info);
    if (FLAGS_verbose) {
      LI << "loaded data file [" << data.file(0) << "]; loaded [" <<
        loaded_file_count_++ << "/" << data_.file_size() << "]";
    }
    return true;
  }

  ExampleParser parser;
  parser.init(data.text(), data.ignore_feature_group());
  struct VSlot {
    SArray<float> val;
    SArray<uint64> col_idx;
    SArray<uint16> row_siz;
    size_t cumulative_row_siz_size;

    bool writeToFile(const string& name) {
      return val.compressTo().writeToFile(name+".value")
          && col_idx.compressTo().writeToFile(name+".colidx")
          && row_siz.compressTo().writeToFile(name+".rowsiz");
    }

    VSlot() :
      cumulative_row_siz_size(0) {
      // do nothing
    }

    void clear() {
      val.clear();
      col_idx.clear();
      row_siz.clear();
    }

    size_t memSize() const {
      return val.memSize() + col_idx.memSize() + row_siz.memSize();
    }

    bool appendToFile(
      PathPicker* path_picker, const int slot_id, const string& prefix,
      KVVectorPtr<Key,double> w, const int push_time,
      const int count_min_k, const float count_min_n) {
      const string kPartitionSuffix = ".partition";
      CHECK(nullptr != path_picker);
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

      // partitioned value
      string path = path_picker->getPath(prefix + ".value");
      size_t start = File::size(path);
      auto val_compressed = val.compressTo();
      CHECK(val_compressed.appendToFile(path));
      appendPartitionInfo(start, val_compressed.activeMemSize(), path + kPartitionSuffix);

      // partitioned colidx
      path = path_picker->getPath(prefix + ".colidx");
      start = File::size(path);
      auto col_compressed = col_idx.compressTo();
      CHECK(col_compressed.appendToFile(path));
      appendPartitionInfo(start, col_compressed.activeMemSize(), path + kPartitionSuffix);

      // partitioned rowsiz
      path = path_picker->getPath(prefix + ".rowsiz");
      start = File::size(path);
      auto row_compressed = row_siz.compressTo();
      CHECK(row_compressed.appendToFile(path));
      appendPartitionInfo(start, row_compressed.activeMemSize(), path + kPartitionSuffix);

      if (!col_idx.empty()) {
        // sort
        parallelSort(&col_idx, FLAGS_num_threads,
          [](const uint64& a, const uint64& b) { return a < b; });

        // unique and count
        SArray<uint64> unique_key;
        unique_key.reserve(col_idx.size() / 8 + 1);
        SArray<uint8> count_key;
        count_key.reserve(col_idx.size() / 8 + 1);

        auto current_key = col_idx.front();
        size_t current_cnt = 0;
        for (size_t i = 0; i < col_idx.size(); ++i) {
          if (current_key != col_idx[i]) {
            unique_key.pushBack(current_key);
            count_key.pushBack(current_cnt < kuint8max ? current_cnt : kuint8max);

            // reset
            current_key = col_idx[i];
            current_cnt = 0;
          }
          ++current_cnt;
        }
        // the last key
        unique_key.pushBack(current_key);
        count_key.pushBack(current_cnt < kuint8max ? current_cnt : kuint8max);

        if (w) {
          // push unique keys with their counts to servers
          MessagePtr count(new Message(kServerGroup, push_time));
          count->setKey(unique_key);
          count->addValue(count_key);
          count->task.set_key_channel(slot_id);
          Range<uint64>(unique_key.front(), unique_key.back() + 1).to(
            count->task.mutable_key_range());
          w->set(count)->set_insert_key_freq(true);
          w->set(count)->set_countmin_k(count_min_k);
          w->set(count)->set_countmin_n(count_min_n);
          CHECK_EQ(push_time, w->push(count));
        }

        // dump unique keys to disk
        path = path_picker->getPath(prefix + ".colidx_sorted_uniq");
        start = File::size(path);
        auto unique_key_compressed = unique_key.compressTo();
        CHECK(unique_key_compressed.appendToFile(path));
        appendPartitionInfo(start, unique_key_compressed.activeMemSize(), path + kPartitionSuffix);
      }

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
      // zero padding
      while (vslot.cumulative_row_siz_size < num_ex) {
        vslot.row_siz.pushBack(0);
        ++vslot.cumulative_row_siz_size;
      }
      vslot.row_siz.pushBack(std::max(key_size, val_size));
      ++vslot.cumulative_row_siz_size;
    }
    ++ num_ex;
  };

  // check memory usage every N lines
  std::function<void(int)> periodic_check_handle = [&] (int arg) {
    if (0 != arg) {
      // check memory usage
      size_t mem_size = 0;
      for (auto& slot : vslots) {
        mem_size += slot.memSize();
      }
      if (mem_size / 1024 / 1024 < FLAGS_load_data_max_mb_per_thread ||
          0 == FLAGS_load_data_max_mb_per_thread) {
        // memory threshold not touched yet
        return;
      }
    }

    if (FLAGS_verbose) {
      LI << "dumping vslots ...";
    }

    // dump memory image to file
    // release vslots
    for (int i = 0; i < kSlotIDmax; ++i) {
      auto& slot = vslots[i];
      if (slot.row_siz.empty() && slot.val.empty()) {
        // slot is empty
        continue;
      }
      // append to disk file
      CHECK(slot.appendToFile(
        this->path_picker_, i, cacheName(data, i),
        w_, time_++, count_min_k_, count_min_n_));
      slot.clear();
    }

    if (FLAGS_verbose) {
      LI << "vslots dumped";
    }
  };

  FileLineReader reader(data);
  reader.set_line_callback(handle);
  reader.set_periodic_callback(periodic_check_handle);
  reader.Reload();

  // dump the last partition remaining in VSlot
  for (int i = 0; i < kSlotIDmax; ++i) {
    auto& vslot = vslots[i];
    if (vslot.row_siz.empty() && vslot.val.empty()) continue;
    while (vslot.cumulative_row_siz_size < num_ex) {
      vslot.row_siz.pushBack(0);
      ++vslot.cumulative_row_siz_size;
    }
    CHECK(vslot.appendToFile(
      this->path_picker_, i, cacheName(data, i),
      w_, time_++, count_min_k_, count_min_n_));
  }

  // generate info
  info = parser.info();
  writeProtoToASCIIFileOrDie(info, info_name);

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
    string file = path_picker_->getPath(cacheName(ithFile(data_, i), slot_id) + ".colidx");
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
    string file = path_picker_->getPath(cacheName(ithFile(data_, i), slot_id) + ".rowsiz");
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
  out.clear();
  if (in.empty()) {
    return true;
  }

  // if no corresponding partition file exists,
  //   we simply uncompress the whole SArray
  if (0 == File::size(partition_file_name)) {
    out.uncompressFrom(in);
    return true;
  }

  // store partitions' info: {start_byte, size} into partitions
  File* partition_file = File::openOrDie(partition_file_name, "r");
  std::vector<std::pair<size_t, size_t>> partitions;
  const size_t kLineMaxLen = 1024;
  char line_buffer[kLineMaxLen + 1];
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

void SlotReader::getAllPartitions(
  const int slot_id,
  const string& type_str,
  std::vector<std::pair<string, SizeR>>& out_partitions) {
  out_partitions.clear();

  for (int file_idx = 0; file_idx < data_.file_size(); ++file_idx) {
    string data_path = path_picker_->getPath(cacheName(
      ithFile(data_, file_idx), slot_id) + "." + type_str);

    string partition_info_path = data_path + ".partition";
    File* partition_info_file = File::openOrDie(partition_info_path, "r");
    const size_t kLineMaxLen = 1024;
    char line[kLineMaxLen + 1];
    while (nullptr != partition_info_file->readLine(line, kLineMaxLen)) {
      auto vec = split(line, '\t');
      CHECK_EQ(2, vec.size());

      out_partitions.push_back(std::make_pair(
        data_path,
        SizeR(
          std::stoull(vec[0], nullptr),
          std::stoull(vec[0], nullptr) + std::stoull(vec[1], nullptr))));
    }
    partition_info_file->close();
  }
}
} // namespace PS
