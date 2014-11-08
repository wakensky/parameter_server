#pragma once
#include "base/shared_array_inl.h"
#include "proto/example.pb.h"
#include "data/common.h"

namespace PS {

// read all slots in *data* with multithreadd, save them into *cache*.
class SlotReader {
 public:
  SlotReader() { }
  SlotReader(const DataConfig& data, const DataConfig& cache) {
    init(data, cache);
  }

  void init(const DataConfig& data, const DataConfig& cache);

  // first read, then save
  int read(ExampleInfo* info = nullptr);

  template<typename V> MatrixInfo info(int slot_id) const {
    return readMatrixInfo(info_, slot_id, sizeof(uint64), sizeof(V));
  }

  // load a slot from cache
  SArray<size_t> offset(int slot_id);
  SArray<uint64> index(int slot_id);
  template<typename V> SArray<V> value(int slot_id);

  // return the full path according to file_name
  //    if file exists among directories_, return the corresponding full path
  //    if not found, randomly pick a new directory
  string fullPath(const string& file_name);

  struct DataPack {
    SArray<uint64> colidx;
    SArray<uint64> uniq_colidx;
    SArray<uint16> rowsiz;
    SArray<float> val;
    bool is_ok;

    DataPack() :
      is_ok(false) {
      // do nothing
    }
  };

  enum LoadMode {
    COLIDX = 1L << 0,
    UNIQ_COLIDX = 1L << 1,
    ROWSIZ = 1L << 2,
    VALUE = 1L << 3,
    END = 1L << 16
  };

  // which partition I am loading
  struct PartitionLocator {
    int file_idx;
    size_t partition_idx;
    size_t partition_count;

    PartitionLocator() :
      file_idx(-1),
      partition_idx(0),
      partition_count(0) {
      // do nothing
    }

    PartitionLocator(const PartitionLocator& other) :
      file_idx(other.file_idx),
      partition_idx(other.partition_idx),
      partition_count(other.partition_count) {
      // do nothing
    }

    PartitionLocator& operator= (const PartitionLocator& rhs) {
      file_idx = rhs.file_idx;
      partition_idx = rhs.partition_idx;
      partition_count = rhs.partition_count;
      return *this;
    }
  };

  // load partitions one by one
  //   return invalid DataPack on failure
  DataPack nextPartition(const int slot_id, const LoadMode load_mode);

  // return to the first partition
  //   nextPartition will bring us to the first partition again
  void returnToFirstPartition(const int slot_id);

  void clear(int slot_id) {
    offset_cache_.erase(slot_id);
    index_cache_.erase(slot_id);
  }

  void keepPartitionRange(const string& path, const SizeR& range);

 private:
  string cacheName(const DataConfig& data, int slot_id) const;
  void addDirectories(const DataConfig& cache);
  std::default_random_engine rng_;
  size_t nnzEle(int slot_id) const;
  bool readOneFile(const DataConfig& data);
  bool assemblePartitions(
    SArray<char>& out, SArray<char>& in, const string& partition_file_name) const;
  DataConfig data_;
  bool dump_to_disk_;
  ExampleInfo info_;
  std::unordered_map<int, SlotInfo> slot_info_;
  std::mutex mu_;
  size_t loaded_file_count_;
  std::unordered_map<int, SArray<size_t>> offset_cache_;
  std::unordered_map<int, SArray<uint64>> index_cache_;
  // available directories
  std::vector<string> directories_;
  // partition ranges for each file
  // file_name -> range
  std::unordered_map<string, std::vector<SizeR>> partition_ranges_;
  // where the current partition is
  //    std::array<int, 3> <=>
  //    {i-th file, current partition idx, partition numbers of i-th file}
  std::unordered_map<int, PartitionLocator> partition_locator_;
};

template<typename V> SArray<V> SlotReader::value(int slot_id) {
  // TODO support cache (but this is a template function...)
  SArray<V> val;
  if (nnzEle(slot_id) == 0) return val;
  for (int i = 0; i < data_.file_size(); ++i) {
    string file = fullPath(cacheName(ithFile(data_, i), slot_id) + ".value");
    SArray<char> comp; CHECK(comp.readFromFile(file));
    SArray<float> uncomp;
    {
      SArray<char> buffer;
      CHECK(assemblePartitions(buffer, comp, file + ".partition"));
      uncomp = buffer;
    }
    size_t n = val.size();
    val.resize(n+uncomp.size());
    for (size_t i = 0; i < uncomp.size(); ++i) val[n+i] = uncomp[i];
  }
  CHECK_EQ(val.size(), nnzEle(slot_id)) << slot_id;
  return val;
}

} // namespace PS
