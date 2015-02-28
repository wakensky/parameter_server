#pragma once
#include "base/auc.h"
#include "system/ocean.h"

namespace PS {

DECLARE_bool(verbose);

class SlotReader;
class Validation {
  public:
    Validation();
    ~Validation();
    Validation(const Validation& other) = delete;
    Validation& operator= (const Validation& rhs) = delete;

    void init(
      const string& identity,
      const LM::Config& conf,
      PathPicker* path_picker);

    // download validation data
    bool download();

    // preprocess validation data
    bool preprocess(const Task& task);

    // submit an AUC calculation task asynchronizedly
    void submit(
      const Ocean::GroupID grp_id,
      const Range<Ocean::FullKey> global_range,
      const Ocean::TaskID task_id,
      SArray<Ocean::Value> validation_weights);

    // get validation result
    // need wait all prediction tasks finished
    // prediction result will be cleared afterwards
    AUCData waitAndGetResult();

    // add prefetch job to internal Ocean object
    void prefetch(
      const Ocean::GroupID grp_id,
      const Range<Ocean::FullKey>& global_range,
      const Ocean::TaskID task_id);

    SArray<Ocean::FullKey> getKey(
      const Ocean::GroupID grp_id,
      const Range<Ocean::FullKey>& global_range,
      const Ocean::TaskID task_id);

    SizeR fetchAnchor(
      const Ocean::GroupID grp_id, const Range<Ocean::FullKey>& global_range);

    // whether the Validation is enabled
    bool isEnabled() { return enable_; }

  private:
    using WeightLookupTable =
      tbb::concurrent_hash_map<Ocean::FullKey, Ocean::Value>;

    static const Ocean::TaskID kFakeTaskID = -1;

    struct PredictionRequest {
      Ocean::UnitID unit_id;
      Ocean::TaskID task_id;
      std::shared_ptr<WeightLookupTable> lookup;
      PredictionRequest():
        unit_id(),
        task_id(kFakeTaskID),
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

  public:
    string identity_;
    std::shared_ptr<SlotReader> slot_reader_ptr_;
    PathPicker* path_picker_;
    Ocean ocean_;
    LM::Config conf_;

    // switch for asynchronized threads
    std::atomic_bool go_on_;

    SArray<double> label_;
    SArray<double> prediction_;

    using PredictionPendingQueue =
      tbb::concurrent_bounded_queue<PredictionRequest>;
    PredictionPendingQueue prediction_pending_queue_;
    // An element has been pop from prediction_pending_queue_
    std::condition_variable queue_pop_cv_;
    std::mutex queue_pop_mu_;

    // how many validation examples to be predicted
    size_t num_examples_;

    // sum of clicks for validation data
    double click_sum_;

    // statistic for AUC
    AUC auc_;

    // predict sub-thread function
    std::shared_ptr<std::thread> predict_thread_ptr_;

    // whether conf_ has validation data information
    bool enable_;
};  // class Validation
};  // namespace PS