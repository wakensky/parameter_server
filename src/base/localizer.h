#pragma once
#include <tbb/concurrent_vector.h>
#include "base/shared_array_inl.h"
#include "base/sparse_matrix.h"
#include "util/parallel_sort.h"
#include "data/slot_reader.h"

namespace PS {

template<typename I, typename V>
class Localizer {
 public:
  Localizer() { }
  // find the unique indeces with their number of occrus in *idx*
  void countUniqIndex(
      const SArray<I>& idx, SArray<I>* uniq_idx, SArray<uint32>* idx_frq = nullptr);

  void countUniqIndex(
      const MatrixPtr<V>& mat, SArray<I>* uniq_idx, SArray<uint32>* idx_frq = nullptr);

  // return a matrix with index mapped: idx_dict[i] -> i. Any index does not exists
  // in *idx_dict* is dropped. Assume *idx_dict* is ordered
  MatrixPtr<V> remapIndex(int grp_id, const SArray<I>& idx_dict, SlotReader* reader) const;

  MatrixPtr<V> remapIndex(const SArray<I>& idx_dict);

  MatrixPtr<V> remapIndex(
      const MatrixInfo& info, const SArray<size_t>& offset,
      const SArray<I>& index, const SArray<V>& value,
      const SArray<I>& idx_dict) const;
  MatrixPtr<V> remapIndex(
    int grp_id, const SArray<I>& idx_dict, SlotReader* slot_reader,
    PathPicker* path_picker, const string& node_id) const;
  void clear() { pair_.clear(); }
 private:
#pragma pack(4)
  struct Pair {
    I k; uint32 i;
  };
  SArray<Pair> pair_;
  SparseMatrixPtr<I,V> mat_;
};

template<typename I, typename V>
void Localizer<I,V>::countUniqIndex(
     const MatrixPtr<V>& mat, SArray<I>* uniq_idx, SArray<uint32>* idx_frq) {
  mat_ = std::static_pointer_cast<SparseMatrix<I,V>>(mat);
  countUniqIndex(mat_->index(), uniq_idx, idx_frq);
}



template<typename I, typename V>
void Localizer<I,V>::countUniqIndex(
    const SArray<I>& idx, SArray<I>* uniq_idx, SArray<uint32>* idx_frq) {
  if (idx.empty()) return;
  CHECK(uniq_idx);
  CHECK_LT(idx.size(), kuint32max)
      << "well, you need to change Pair.i from uint32 to uint64";
  CHECK_GT(FLAGS_num_threads, 0);

  pair_.resize(idx.size());
  for (size_t i = 0; i < idx.size(); ++i) {
    pair_[i].k = idx[i];
    pair_[i].i = i;
  }
  parallelSort(&pair_, FLAGS_num_threads, [](const Pair& a, const Pair& b) {
      return a.k < b.k; });

  uniq_idx->clear();
  if (idx_frq) idx_frq->clear();

  I curr = pair_[0].k;
  uint32 cnt = 0;
  for (const Pair& v : pair_) {
    if (v.k != curr) {
      uniq_idx->pushBack(curr);
      curr = v.k;
      if (idx_frq) idx_frq->pushBack(cnt);
      cnt = 0;
    }
    ++ cnt;
  }
  uniq_idx->pushBack(curr);
  if (idx_frq) idx_frq->pushBack(cnt);

  // for debug
  // index_.writeToFile("index");
  // uniq_idx->writeToFile("uniq");
  // idx_frq->writeToFile("cnt");
}

template<typename I, typename V>
MatrixPtr<V> Localizer<I,V>::remapIndex(
  int grp_id, const SArray<I>& idx_dict, SlotReader* slot_reader,
  PathPicker* path_picker, const string& node_id) const {
  if (idx_dict.empty()) {
    return MatrixPtr<V>();
  }
  CHECK(nullptr != slot_reader);
  CHECK(nullptr != path_picker);

  auto info = slot_reader->info<V>(grp_id);
  CHECK_NE(info.type(), MatrixInfo::DENSE)
      << "dense matrix already have compact indeces\n" << info.DebugString();

  // all global key partitions associated with the group
  std::vector<std::pair<string, SizeR>> global_key_partitions;
  slot_reader->getAllPartitions(grp_id, "colidx", global_key_partitions);

  // sub thread function, generate local keys partition by partition
  std::atomic_size_t partition_idx(0);
  std::atomic_size_t global_matched(0);
  std::mutex partition_mu;
  std::vector<string> remapped_idx_path(global_key_partitions.size());
  auto thr_generate_remapped_idx =
    [this, &global_key_partitions, &partition_idx, &remapped_idx_path,
     &idx_dict, &node_id, path_picker, &global_matched, &partition_mu]() {
    while (true) {
      size_t my_partition_idx = 0;
      {
        Lock l(partition_mu);
        my_partition_idx = partition_idx++;
      }
      if (my_partition_idx >= global_key_partitions.size()) {
        // all partitions done
        break;
      }
      size_t in_partition_matched = 0;

      // fetch next partition
      std::pair<string, SizeR> partition;
      partition = global_key_partitions.at(my_partition_idx);

      // load from disk
      SArray<char> compressed;
      compressed.readFromFile(partition.second, partition.first);
      SArray<Key> global_keys;
      global_keys.uncompressFrom(compressed);
      compressed.clear();

      // sorted pair: {global key, its index}
      SArray<Pair> pair;
      pair.resize(global_keys.size());
      for (size_t i = 0; i < global_keys.size(); ++i) {
        pair[i].k = global_keys[i];
        pair[i].i = i;
      }
      global_keys.clear();
      parallelSort(&pair, FLAGS_num_threads, [](const Pair& a, const Pair& b) {
        return a.k < b.k;});

      // generate local keys
      SArray<uint32> remapped_idx(pair.size(), 0);
      const I* cur_dict = idx_dict.begin();
      const Pair* cur_pair = pair.begin();
      while (cur_dict != idx_dict.end() && cur_pair != pair.end()) {
        if (*cur_dict < cur_pair->k) {
          ++cur_dict;
        } else {
          if (*cur_dict == cur_pair->k) {
            remapped_idx[cur_pair->i] = (uint32)(cur_dict - idx_dict.begin()) + 1;
            ++in_partition_matched;
          }
          ++cur_pair;
        }
      };

      // dump local keys
      string path = path_picker->getPath(
        node_id + ".partitioned_remapped_idx." + std::to_string(my_partition_idx));
      CHECK(remapped_idx.writeToFile(path)) <<
        node_id << " at " << hostname() << " writeToFile failed on path [" <<
        path << "] error [" << strerror(errno) << "]";
      remapped_idx_path.at(my_partition_idx) = path;
      global_matched += in_partition_matched;
    };
  };

  {
    ThreadPool pool(FLAGS_num_threads);
    for (size_t i = 0; i < FLAGS_num_threads; ++i) {
      pool.add(thr_generate_remapped_idx);
    }
    pool.startWorkers();
  }

  // construct new Sparsematrix with single thread
  SArray<uint32> new_index(global_matched);
  SArray<size_t> new_offset(
    SizeR(slot_reader->info<V>(grp_id).row()).size() + 1);
  new_offset[0] = 0;
  size_t k = 0; // number of survival keys
  size_t row_count = 0; // examples number across partitions
  std::vector<std::pair<string, SizeR>> rowsiz_partitions;
  slot_reader->getAllPartitions(grp_id, "rowsiz", rowsiz_partitions);
  CHECK_EQ(remapped_idx_path.size(), rowsiz_partitions.size());
  for (size_t partition_idx = 0;
       partition_idx < rowsiz_partitions.size(); ++partition_idx) {
    // load remapped_idx from disk
    SArray<char> remapped_idx_stash;
    CHECK(remapped_idx_stash.readFromFile(remapped_idx_path.at(partition_idx)));
    SArray<uint32> remapped_idx(remapped_idx_stash);

    // load partitioned rowsiz from disk
    SArray<char> compressed;
    CHECK(compressed.readFromFile(
      rowsiz_partitions[partition_idx].second,
      rowsiz_partitions[partition_idx].first));
    SArray<uint16> rowsiz;
    rowsiz.uncompressFrom(compressed);

    // skip empty partition
    if (rowsiz.empty()) {
      continue;
    }

    // fill new_index and new_offset
    size_t col_start = 0;
    for (size_t i = 0; i < rowsiz.size(); ++i) {
      size_t n = 0;
      for (size_t j = 0; j < rowsiz[i]; ++j) {
        if (0 == remapped_idx[j + col_start]) {
          continue;
        }
        ++n;
        new_index[k++] = remapped_idx[j + col_start] - 1;
      }
      new_offset[row_count + 1] = new_offset[row_count] + n;
      col_start += rowsiz[i];
      row_count++;
    }
  }
  CHECK_EQ(new_offset.size(), row_count + 1);
  CHECK_EQ(k, global_matched);

  LI << "global_matched: " << global_matched <<
    " len(new_index): " << new_index.size() <<
    " len(new_offset): " << new_offset.size();

  // establish matrix info
  auto new_info = info;
  new_info.set_sizeof_index(sizeof(uint32));
  new_info.set_nnz(new_index.size());
  new_info.clear_ins_info();
  SizeR local(0, idx_dict.size());
  if (new_info.row_major()) {
    local.to(new_info.mutable_col());
  } else {
    local.to(new_info.mutable_row());
  }
  return MatrixPtr<V>(new SparseMatrix<uint32, V>(
    new_info, new_offset, new_index, SArray<V>()));
}

template<typename I, typename V>
MatrixPtr<V> Localizer<I,V>::remapIndex(const SArray<I>& idx_dict) {
  CHECK(mat_);
  return remapIndex(mat_->info(), mat_->offset(), mat_->index(), mat_->value(), idx_dict);
}

template<typename I, typename V>
MatrixPtr<V> Localizer<I, V>::remapIndex(
    int grp_id, const SArray<I>& idx_dict, SlotReader* reader) const {
  SArray<V> val;
  auto info = reader->info<V>(grp_id);
  CHECK_NE(info.type(), MatrixInfo::DENSE)
      << "dense matrix already have compact indeces\n" << info.DebugString();
  if (info.type() == MatrixInfo::SPARSE) val = reader->value<V>(grp_id);
  return remapIndex(info, reader->offset(grp_id), reader->index(grp_id), val, idx_dict);
}

template<typename I, typename V>
MatrixPtr<V> Localizer<I, V>::remapIndex(
    const MatrixInfo& info, const SArray<size_t>& offset,
    const SArray<I>& index, const SArray<V>& value,
    const SArray<I>& idx_dict) const {
  if (index.empty() || idx_dict.empty()) return MatrixPtr<V>();
  CHECK_LT(idx_dict.size(), kuint32max);
  CHECK_EQ(offset.back(), index.size());
  CHECK_EQ(index.size(), pair_.size());
  bool bin = value.empty();
  if (!bin) CHECK_EQ(value.size(), index.size());

  // TODO multi-thread
  uint32 matched = 0;
  SArray<uint32> remapped_idx(pair_.size(), 0);
  const I* cur_dict = idx_dict.begin();
  const Pair* cur_pair = pair_.begin();
  while (cur_dict != idx_dict.end() && cur_pair != pair_.end()) {
    if (*cur_dict < cur_pair->k) {
      ++ cur_dict;
    } else {
      if (*cur_dict == cur_pair->k) {
        remapped_idx[cur_pair->i] = (uint32)(cur_dict-idx_dict.begin()) + 1;
        ++ matched;
      }
      ++ cur_pair;
    }
  }

  // construct the new matrix
  SArray<uint32> new_index(matched);
  SArray<size_t> new_offset(offset.size()); new_offset[0] = 0;
  SArray<V> new_value(std::min(value.size(), (size_t)matched));

  size_t k = 0;
  for (size_t i = 0; i < offset.size() - 1; ++i) {
    size_t n = 0;
    for (size_t j = offset[i]; j < offset[i+1]; ++j) {
      if (remapped_idx[j] == 0) continue;
      ++ n;
      if (!bin) new_value[k] = value[j];
      new_index[k++] = remapped_idx[j] - 1;
    }
    new_offset[i+1] = new_offset[i] + n;
  }
  CHECK_EQ(k, matched);

  auto new_info = info;
  new_info.set_sizeof_index(sizeof(uint32));
  new_info.set_nnz(new_index.size());
  new_info.clear_ins_info();
  SizeR local(0, idx_dict.size());
  if (new_info.row_major())  {
    local.to(new_info.mutable_col());
  } else {
    local.to(new_info.mutable_row());
  }
  // LL << curr_o << " " << local.end() << " " << curr_j;
  return MatrixPtr<V>(new SparseMatrix<uint32, V>(new_info, new_offset, new_index, new_value));
}

} // namespace PS
