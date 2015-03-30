#pragma once
#include "base/shared_array_inl.h"
#include "proto/example.pb.h"
#include "data/common.h"
#include "system/path_picker.h"

namespace PS {

template <typename K, typename V>
class KVVector;

// read all slots in *data* with multithreadd, save them into *cache*.
class SlotReader {
 public:
  SlotReader() { }
  SlotReader(const DataConfig& data, const DataConfig& cache,
    PathPicker* path_picker,
    const int start_time, const int finishing_time,
    const int count_min_k, const float count_min_n,
    const string& identity) {
    init(data, cache, path_picker, start_time, finishing_time,
         count_min_k, count_min_n, identity);
  }

  void init(const DataConfig& data, const DataConfig& cache,
    PathPicker* path_picker,
    const int start_time, const int finishing_time,
    const int count_min_k, const float count_min_n,
    const string& identity);

  // first read, then save
  int read(
    std::shared_ptr<KVVector<Key,double>> w,
    ExampleInfo* info = nullptr);

  template<typename V> MatrixInfo info(int slot_id) const {
    return readMatrixInfo(info_, slot_id, sizeof(uint64), sizeof(V));
  }

  // load a slot from cache
  SArray<size_t> offset(int slot_id);
  SArray<uint64> index(int slot_id);
  template<typename V> SArray<V> value(int slot_id) const;

  void clear(int slot_id) {
    offset_cache_.erase(slot_id);
    index_cache_.erase(slot_id);
  }

  // get all column partitions belong to slot_id
  // type_str: colidx, value, rowsiz, colidx_sorted_uniq
  void getAllPartitions(
    const int slot_id,
    const string& type_str,
    std::vector<std::pair<string, SizeR>>& out_partitions);

 private:
  // Produce regularized file name
  string cacheName(const DataConfig& data, int slot_id) const;
  size_t nnzEle(int slot_id) const;
  bool readOneFile(
    const DataConfig& data,
    std::shared_ptr<KVVector<Key,double>> w);
  bool assemblePartitions(
    SArray<char>& out, SArray<char>& in, const string& partition_file_name) const;
  DataConfig data_;
  bool dump_to_disk_;
  ExampleInfo info_;
  std::unordered_map<int, SlotInfo> slot_info_;
  std::mutex mu_;
  std::atomic_size_t loaded_file_count_;
  std::unordered_map<int, SArray<size_t>> offset_cache_;
  std::unordered_map<int, SArray<uint64>> index_cache_;
  PathPicker* path_picker_;
  std::atomic_int time_;
  int finishing_time_;
  int count_min_k_;
  float count_min_n_;
  string identity_;
};

template<typename V> SArray<V> SlotReader::value(int slot_id) const {
  // TODO support cache (but this is a template function...)
  SArray<V> val;
  if (nnzEle(slot_id) == 0) return val;
  for (int i = 0; i < data_.file_size(); ++i) {
    string file = path_picker_->getPath(cacheName(ithFile(data_, i), slot_id) + ".value");
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
