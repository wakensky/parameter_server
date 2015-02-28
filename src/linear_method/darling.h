#pragma once
#include <float.h>
#include "tbb/concurrent_hash_map.h"
#include "linear_method/batch_solver.h"
#include "base/bitmap.h"
#include "filter/sparse_filter.h"

namespace PS {
namespace LM {

// batch algorithm for sparse logistic regression
class Darling : public BatchSolver {
 public:
  virtual void runIteration();
  virtual void preprocessData(const MessageCPtr& msg);
  virtual void updateModel(const MessagePtr& msg);

 protected:
  SArrayList<double> computeGradients(int grp, SizeR global_range, int task_id);
  void updateDual(
    int grp, SizeR global_range, SArray<double> new_w, const int task_id);
  void updateWeight(
    int grp, SizeR global_range,
    SArray<double> G, SArray<double> U,
    const int task_id, const bool is_priority);

  Progress evaluateProgress();
  void showProgress(int iter);
  void showKKTFilter(int iter);

  void computeGradients(
    int grp, SizeR thr_anchor, const size_t group_anchor_begin,
    SArray<double> G, SArray<double> U,
    SArray<uint32> feature_key,
    SArray<size_t> feature_offset,
    SArray<double> feature_value,
    SArray<double> delta);
  void updateDual(
    int grp, SizeR th_row_range, SizeR anchor, SArray<double> w_delta,
    SArray<uint32> feature_index,
    SArray<size_t> feature_offset,
    SArray<double> feature_value,
    SArray<double> delta);

  double newDelta(double delta_w) {
    return std::min(conf_.darling().delta_max_value(), 2 * fabs(delta_w) + .1);
  }

  // {nnz_w, objv} for each column partitioned unit on servers
  std::unordered_map<
    Ocean::UnitID,
    std::pair<size_t, double>,
    Ocean::UnitIDHash> progress_stat_;

  SparseFilter kkt_filter_;

  double kkt_filter_threshold_;
  double violation_;

  DarlingConfig darling_conf_;
};

} // namespace LM
} // namespace PS
