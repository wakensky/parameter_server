#pragma once
#include "base/shared_array_inl.h"
#include "base/sparse_matrix.h"
#include "util/parallel_sort.h"
#include "util/split.h"
#include "data/slot_reader.h"

namespace PS {

DECLARE_int32(in_group_parallel);

template<typename I, typename V>
class Localizer {
 public:
  Localizer(const NodeID& node_id, const int grp_id = 0) :
    node_id_(node_id),
    grp_id_(grp_id) {
    // do nothing
  }
  // find the unique indeces with their number of occrus in *idx*
  void countUniqIndex(
      const SArray<I>& idx, SArray<I>* uniq_idx, SArray<uint32>* idx_frq = nullptr);

  void countUniqIndex(
      const MatrixPtr<V>& mat, SArray<I>* uniq_idx, SArray<uint32>* idx_frq = nullptr);

  // return a matrix with index mapped: idx_dict[i] -> i. Any index does not exists
  // in *idx_dict* is dropped. Assume *idx_dict* is ordered
  MatrixPtr<V> remapIndex(
    int grp_id, const SArray<I>& idx_dict,
    SlotReader* reader, PathPicker* path_picker) const;

  MatrixPtr<V> remapIndex(const SArray<I>& idx_dict);

  MatrixPtr<V> remapIndex(
      const MatrixInfo& info, const SArray<size_t>& offset,
      const SArray<I>& index, const SArray<V>& value,
      const SArray<I>& idx_dict) const;
  void clear() { pair_.clear(); }
 private:
#pragma pack(4)
  struct Pair {
    I k; uint32 i;
  };
  SArray<Pair> pair_;
  SparseMatrixPtr<I,V> mat_;
  const NodeID node_id_;
  const int grp_id_;
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
MatrixPtr<V> Localizer<I,V>::remapIndex(const SArray<I>& idx_dict) {
  CHECK(mat_);
  return remapIndex(mat_->info(), mat_->offset(), mat_->index(), mat_->value(), idx_dict);
}

template<typename I, typename V>
MatrixPtr<V> Localizer<I,V>::remapIndex(
  int grp_id, const SArray<I>& idx_dict, SlotReader* reader,
  PathPicker* path_picker) const {
  if (idx_dict.empty()) {
    return MatrixPtr<V>();
  }
  CHECK(nullptr != reader);

  auto info = reader->info<V>(grp_id);
  CHECK_NE(info.type(), MatrixInfo::DENSE)
      << "dense matrix already have compact indeces\n" << info.DebugString();

  struct remapped_path_less {
    bool operator() (const string& a, const string& b) {
      auto a_vec = split(a, '.');
      auto b_vec = split(b, '.');
      CHECK_EQ(a_vec.size(), b_vec.size());
      CHECK_EQ(a_vec.size(), 3);
      return std::stod(a_vec.back()) > std::stod(b_vec.back());
    }
  };
  std::priority_queue<
    string,
    std::vector<string>,
    remapped_path_less> remapped_idx_path_queue;
  size_t global_matched = 0;
  std::mutex remapped_idx_lock;

  // traverse all partitions
  // generate segmented remapped_idx
  auto generate_remapped_idx = [&]() {
    while (true) {
      // fetch next partition
      SlotReader::DataPack dp;
      {
        Lock l(remapped_idx_lock);
        if (!(dp = reader->nextPartition(grp_id_, SlotReader::COLIDX, false)).is_ok) {
          break;
        }
      }
      size_t local_matched = 0;

      // load dp.colidx
      SArray<char> compressed;
      compressed.readFromFile(
        SizeR(
          dp.colidx_info.second.begin(),
          dp.colidx_info.second.begin() + dp.colidx_info.second.end()),
        dp.colidx_info.first);
      dp.colidx.uncompressFrom(compressed);

      // sorted pair
      SArray<Pair> pair;
      pair.resize(dp.colidx.size());
      for (size_t i = 0; i < dp.colidx.size(); ++i) {
        pair[i].k = dp.colidx[i];
        pair[i].i = i;
      }
      parallelSort(&pair, FLAGS_num_threads, [](const Pair& a, const Pair& b) {
        return a.k < b.k; });

      // generate remapped_idx
      SArray<uint32> remapped_idx(pair.size(), 0);
      const I* cur_dict = idx_dict.begin();
      const Pair* cur_pair = pair.begin();
      while (cur_dict != idx_dict.end() && cur_pair != pair.end()) {
        if (*cur_dict < cur_pair->k) {
          ++cur_dict;
        } else {
          if (*cur_dict == cur_pair->k) {
            remapped_idx[cur_pair->i] = (uint32)(cur_dict - idx_dict.begin()) + 1;
            ++local_matched;
          }
          ++cur_pair;
        }
      };

      // dump remapped_idx
      string path = path_picker->getPath(
        node_id_ + ".segmented_remapped_idx." + std::to_string(dp.sn));
      CHECK(remapped_idx.writeToFile(path));
      {
        Lock l(remapped_idx_lock);

        remapped_idx_path_queue.push(path);
        // increase matched
        global_matched += local_matched;
      }

    };
  };

  reader->returnToFirstPartition(grp_id_);
  {
    size_t thread_num = FLAGS_num_threads;
    ThreadPool remapped_idx_pool(thread_num);
    for (size_t i = 0; i < thread_num; ++i) {
      remapped_idx_pool.add(generate_remapped_idx);
    }
    remapped_idx_pool.startWorkers();
  }

  // construct new SparseMatrix
  //   containing localized feature ids
  SArray<uint32> new_index(global_matched);
  SArray<size_t> new_offset(
    SizeR(reader->info<V>(grp_id_).row()).size() + 1);
  new_offset[0] = 0;
  size_t k = 0;
  size_t row_count = 0;
  SlotReader::DataPack dp;
  reader->returnToFirstPartition(grp_id);
  while ((dp = reader->nextPartition(grp_id_, SlotReader::ROWSIZ)).is_ok) {
    // load remapped_idx from disk
    CHECK(!remapped_idx_path_queue.empty());
    string remapped_idx_path = remapped_idx_path_queue.top();
    remapped_idx_path_queue.pop();

    // wakensky
    LI << "[remapIndex] 2-nd loop: before load remapped_idx: " << remapped_idx_path;

    SArray<char> stash;
    CHECK(stash.readFromFile(remapped_idx_path));
    SArray<uint32> remapped_idx(stash);

    // wakensky
    LI << "[remapIndex] 2-nd loop: loaded remapped_idx";

    // skip empty partition
    if (dp.rowsiz.empty()) { continue; }

    // fill new_index
    size_t col_start = 0;
    for (size_t i = 0; i < dp.rowsiz.size(); ++i) {
      size_t n = 0;
      for (size_t j = 0; j < dp.rowsiz[i]; ++j) {
        if (0 == remapped_idx[j + col_start]) { continue; }
        ++n;
        new_index[k++] = remapped_idx[j + col_start] - 1;
      }
      new_offset[row_count + 1] = new_offset[row_count] + n;
      col_start += dp.rowsiz[i];
      row_count++;
    }

    // wakensky
    LI << "[remapIndex] 2-nd loop: filled new_index";
  }
  CHECK_EQ(k, global_matched);

  // establish new info
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
  return MatrixPtr<V>(new SparseMatrix<uint32, V>(
    new_info, new_offset, new_index, SArray<V>()));
}

#if 0
template<typename I, typename V>
MatrixPtr<V> Localizer<I, V>::remapIndex(
    int grp_id, const SArray<I>& idx_dict, SlotReader* reader) const {
  if (idx_dict.empty()) {
    return MatrixPtr<V>(new SparseMatrix<uint32, V>());
  }
  SArray<V> val;
  auto info = reader->info<V>(grp_id);
  CHECK_NE(info.type(), MatrixInfo::DENSE)
      << "dense matrix already have compact indeces\n" << info.DebugString();
  if (info.type() == MatrixInfo::SPARSE) val = reader->value<V>(grp_id);
  return remapIndex(info, reader->offset(grp_id), reader->index(grp_id), val, idx_dict);
}
#endif

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
