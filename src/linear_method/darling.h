#pragma once
#include <float.h>
#include "linear_method/batch_solver.h"
#include "base/bitmap.h"

namespace PS {
namespace LM {

// batch algorithm for sparse logistic regression
class Darling : public BatchSolver {
 public:
  virtual void init();
 protected:
  virtual void runIteration();
  virtual void preprocessData(const MessageCPtr& msg);
  virtual void updateModel(const MessagePtr& msg);

  SArrayList<double> computeGradients(int task_id, int grp, Range<Key> g_key_range);
  void updateDual(int grp, Range<Key> g_key_range, SArray<double> new_weight);
  // serial
  void parallelUpdateWeight(
    int grp, Range<Key> g_key_range, SizeR base_range,
    SArray<double> G, SArray<double> U);
  // parallel
  void updateWeight(
    int grp, Range<Key> g_key_range, SizeR base_range,
    SArray<double> G, SArray<double> U);
  void threadUpdateWeight(
          int grp, size_t begin, size_t size, size_t base_range_begin,
          SArray<double> G, SArray<double> U,
          size_t* nnz_w, double* objv, double* violation,
          std::vector<size_t>* clear_idx,
          SArray<double>& value, SArray<double>& delta, Bitmap& active_set);

  Progress evaluateProgress();
  void showProgress(int iter);
  void showKKTFilter(int iter);

  void computeGradients(
    const SArray<uint32>& feature_index,
    const SArray<size_t>& feature_offset,
    const SArray<double>& feature_value,
    const SArray<double>& delta,
    int grp, SizeR col_range, const size_t base_range_begin,
    SArray<double> G, SArray<double> U);
  void updateDual(
    const SArray<uint32>& feature_index,
    const SArray<size_t>& feature_offset,
    const SArray<double>& feature_value,
    const SArray<double>& delta,
    int grp, SizeR row_range, SizeR col_range,
    const size_t base_range_begin,
    SArray<double> w_delta);

  double newDelta(double delta_w) {
    return std::min(conf_.darling().delta_max_value(), 2 * fabs(delta_w) + .1);
  }

  bool inactive(double val) { return val != val; }
  double kInactiveValue_;

  std::unordered_map<int, Bitmap> active_set_;
  std::unordered_map<int, SArray<double>> delta_;

  double KKT_filter_threshold_;
  double violation_;

  DarlingConfig darling_conf_;

  // all tasks been prefetched
  // identify with msg->task.time()
  std::unordered_set<int> prefetched_task_;

  // statistic for weight (on server)
  //   JobID -> {nnz_w, objv}
  std::unordered_map<
    Ocean::JobID,
    std::pair<size_t, double>,
    Ocean::JobIDHash> weight_stat_;
};

} // namespace LM
} // namespace PS
