#pragma once
#include <cstdint>
#include <cinttypes>
#include <atomic>
#include <array>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_unordered_set.h>
#include "proto/task.pb.h"
#include "util/common.h"
#include "util/threadpool.h"
#include "base/sparse_matrix.h"
#include "base/auc.h"
#include "base/localizer.h"
#include "data/slot_reader.h"
#include "system/path_picker.h"
#include "system/ocean.h"

namespace PS {

DECLARE_bool(verbose);

class Validation {
  public:
    using WeightLookupTable =
      tbb::concurrent_hash_map<Ocean::FullKey, Ocean::Value>;

  public:
    Validation(
      const string& identity,
      const LM::Config& conf,
      PathPicker* path_picker);
    ~Validation();
    Validation(const Validation& other) = delete;
    Validation& operator= (const Validation& rhs) = delete;

  // download validation data
  bool download();

  // preprocess validation data
  bool preprocess(const Task& task);

  // submit an AUC calculation task asynchronizedly
  void submit(
    const Ocean::GroupID grp_id,
    const Range<Ocean::FullKey> global_range,
    const Ocean::TaskID task_id,
    SArray<Ocean::FullKey> model_keys,
    SArray<Ocean::Value> model_weights);

  // get validation result
  // need wait all prediction tasks finished
  // prediction result will be cleared afterwards
  AUCData waitAndGetResult();

  // internal types
  private:
    struct PredictionRequest {
      Ocean::UnitID unit_id;
      Ocean::TaskID task_id;
      std::shared_ptr<WeightLookupTable> lookup;
      PredictionRequest():
        unit_id(),
        task_id(0),
        lookup() {
        // do nothing
      }

      PredictionRequest(
        const Ocean::UnitID& in_unit_id,
        const Ocean::TaskID in_task_id,
        std::shared_ptr<WeightLookupTable> in_lookup):
        unit_id(in_unit_id),
        task_id(in_task_id),
        lookup(in_lookup) {
        // do nothing
      }

      PredictionRequest(const PredictionRequest& other):
        unit_id(other.unit_id),
        task_id(other.task_id),
        lookup(other.lookup) {
        // do nothing
      }

      PredictionRequest& operator= (const PredictionRequest& rhs) {
        unit_id = rhs.unit_id;
        task_id = rhs.task_id;
        lookup = rhs.lookup;
        return *this;
      }
    };

  // methods
  private:
    void predictThreadFunc();

    // add partial prediction to global result
    void prophet(
      const SizeR& th_row_range,
      const SizeR& anchor,
      SArray<Ocean::FullKey> parameter_key,
      SArray<Ocean::ShortKey> feature_key,
      SArray<Ocean::Offset> feature_offset,
      std::shared_ptr<WeightLookupTable> weight_lookup);

  // attributes
  private:
    string identity_;
    // download validation data
    SlotReader slot_reader_;
    // select a directory for a file name
    PathPicker* path_picker_;
    // storage column partitioned validation data
    Ocean ocean_;
    LM::Config conf_;
    // switch for asynchronized threads
    std::atomic_bool go_on_;
    // labels
    MatrixPtr<double> y_;
    // predictions
    MatrixPtr<double> prediction_;

    using PredictionPendingQueue = tbb::concurrent_bounded_queue<PredictionRequest>;
    PredictionPendingQueue prediction_pending_queue_;
    // An element has been pop from prediction_pending_queue_
    std::condition_variable queue_pop_cv_;
    std::mutex queue_pop_mu_;

    // how many validation examples to be predicted
    size_t num_examples_;

    // average of clicks for validation data
    double click_average_;

    // statistic for AUC
    AUC auc_;
};

};