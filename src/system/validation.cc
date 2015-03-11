#include "system/validation.h"
#include <unistd.h>
#include <fcntl.h>
#include <iomanip>
#include "util/split.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdexcept>
#include <gperftools/malloc_extension.h>
#include "base/shared_array_inl.h"
#include "base/localizer.h"
#include "data/slot_reader.h"

namespace PS {

Validation::Validation():
  go_on_(true),
  num_examples_(0u),
  click_sum_(0.0),
  enable_(false) {
}

Validation::~Validation() {
  go_on_ = false;

  // join prediction thread
  if (predict_thread_ptr_) {
    submit(0, Range<Ocean::FullKey>(), kFakeTaskID,
      SArray<Ocean::Value>());
    predict_thread_ptr_->join();
  }
}

void Validation::init(
  const string& identity,
  const LM::Config& conf,
  PathPicker* path_picker) {
  CHECK(!identity.empty());
  identity_ = identity;
  conf_ = conf;
  path_picker_ = path_picker;

  enable_ = conf_.has_validation_data() && conf_.validation_data().file_size() > 0;
  predict_thread_ptr_.reset(
    new std::thread(&Validation::predictThreadFunc, this));
  slot_reader_ptr_.reset(new SlotReader());
}

bool Validation::download() {
  if (!enable_) { return true; }

  slot_reader_ptr_->init(
    conf_.validation_data(), conf_.local_cache(),
    path_picker_, 0, 0, 0, 0, identity_);

  ExampleInfo example_info;
  return 0 == slot_reader_ptr_->read(nullptr, &example_info);
}

bool Validation::preprocess(const Task& task) {
  ocean_.init(identity_, conf_, task, path_picker_);
  if (!enable_) { return true; }

  const int grp_size = task.linear_method().fea_grp_size();
  std::vector<Ocean::GroupID> fea_grp;
  for (int i = 0; i < grp_size; ++i) {
    fea_grp.push_back(task.linear_method().fea_grp(i));
  }

  // preprocess groups one by one
  for (int group_order = 0; group_order < fea_grp.size(); ++group_order) {
    Ocean::GroupID group_id = fea_grp.at(group_order);

    // unique key partitions
    std::vector<std::pair<string, SizeR>> unique_key_partitions;
    slot_reader_ptr_->getAllPartitions(
      group_id, "colidx_sorted_uniq", unique_key_partitions);

    // merge all unique keys
    SArray<Ocean::FullKey> merged_unique_keys;
    for (const auto& partition : unique_key_partitions) {
      // read from disk (compressed)
      SArray<char> compressed;
      compressed.readFromFile(partition.second, partition.first);

      // uncompress
      SArray<Ocean::FullKey> uncompressed;
      uncompressed.uncompressFrom(compressed);

      // merge and unique (already sorted)
      if (!uncompressed.empty()) {
        merged_unique_keys = merged_unique_keys.setUnion(uncompressed);
      }
    }

    // localize
    Localizer<Ocean::FullKey, Ocean::Value>* localizer =
      new Localizer<Ocean::FullKey, Ocean::Value>();
    if (FLAGS_verbose) {
      LI << "Validation started remapIndex [" << group_order + 1 << "/" <<
        grp_size << "]; grp: " << group_id;
    }
    auto X = localizer->remapIndex(
      group_id, merged_unique_keys,
      slot_reader_ptr_.get(), path_picker_, identity_);
    if (FLAGS_verbose) {
      LI << "finished remapIndex [" << group_order + 1 << "/" << grp_size <<
        "]; grp: " << group_id;
    }
    delete localizer;
    slot_reader_ptr_->clear(group_id); // clear SlotReader's internal cache

    // transformation
    if (FLAGS_verbose) {
      LI << "started toColMajor [" << group_order + 1 << "/" <<
        grp_size << "]; grp: " << group_id;
    }
    if (X) {
      // TODO toColMajor is necessary since I have assumed that
      //   training data could be partition column-wise
      if (conf_.solver().has_feature_block_ratio()) {
        X = X->toColMajor();
      }
    }
    if (FLAGS_verbose) {
      LI << "finished toColMajor [" << group_order + 1 << "/" << grp_size <<
        "]; grp: " << group_id;
    }

    // dump to Ocean
    CHECK(ocean_.dump(group_id, merged_unique_keys, SArray<Ocean::Value>(),
      SArray<Ocean::Value>(), SArray<Ocean::Value>(),
      std::static_pointer_cast<SparseMatrix<uint32, double>>(X)));
  }

  // load labels
  label_ = slot_reader_ptr_->value<double>(0);

  // initialize predictions
  prediction_ = slot_reader_ptr_->value<double>(0);
  prediction_.setZero();

  // statistic for clicks
  num_examples_ = label_.size();
  for (size_t i = 0; i < num_examples_; ++i) {
    if (label_[i] > 0) {
      click_sum_ += label_[i];
    }
  }

  return true;
}

void Validation::submit(
  const Ocean::GroupID grp_id,
  const Range<Ocean::FullKey> global_range,
  const Ocean::TaskID task_id,
  SArray<Ocean::Value> validation_weights) {
  if (!enable_) { return; }

  // get keys
  SArray<Ocean::FullKey> validation_keys = getKey(
    grp_id, global_range, task_id);
  CHECK_EQ(validation_keys.size(), validation_weights.size());

  // generate key->weight lookup table
  std::shared_ptr<WeightLookupTable> lookup_ptr(new WeightLookupTable());
  CHECK(lookup_ptr);
  {
    ThreadPool pool(FLAGS_num_threads);
    SizeR keys_position_range(0, validation_keys.size());

    for (int i = 0; i < FLAGS_num_threads; ++i) {
      auto thr_keys_position_range =
        keys_position_range.evenDivide(FLAGS_num_threads, i);
      pool.add([this, thr_keys_position_range,
                &validation_keys, &validation_weights,
                lookup_ptr]() {
                  for (size_t i = thr_keys_position_range.begin();
                       i < thr_keys_position_range.end(); ++i) {
                    WeightLookupTable::accessor accessor;
                    lookup_ptr->insert(accessor, validation_keys[i]);
                    accessor->second = validation_weights[i];
                  }
                });
    }

    pool.startWorkers();
  }

  // in-queue
  prediction_pending_queue_.push(PredictionRequest(
    Ocean::UnitID(grp_id, global_range),
    task_id, lookup_ptr));

  LI << "Validation::submit has enqueued [" <<
    Ocean::UnitID(grp_id, global_range).toString() << "] [" <<
    task_id << "]; queue_size: " << prediction_pending_queue_.size();
}

void Validation::predictThreadFunc() {
  while (go_on_) {
    // take out a PredictionRequest from pending queue
    PredictionRequest prediction_request;
    prediction_pending_queue_.pop(prediction_request);
    if (kFakeTaskID ==  prediction_request.task_id) {
      continue;
    }

    LI << "Validation::predictThreadFunc has popped [" <<
      prediction_request.unit_id.toString() << "] [" <<
      prediction_request.task_id << "]; queue_size: " <<
      prediction_pending_queue_.size();

    // load data
    Ocean::DataPack data_pack = ocean_.get(
      prediction_request.unit_id.grp_id,
      prediction_request.unit_id.global_range,
      prediction_request.task_id);
    // on-demand usage
    SArray<Ocean::FullKey> parameter_key(
      data_pack.arrays[static_cast<size_t>(Ocean::DataSource::PARAMETER_KEY)]);
    SArray<Ocean::ShortKey> feature_key(
      data_pack.arrays[static_cast<size_t>(Ocean::DataSource::FEATURE_KEY)]);
    SArray<Ocean::Offset> feature_offset(
      data_pack.arrays[static_cast<size_t>(Ocean::DataSource::FEATURE_OFFSET)]);
    SizeR anchor = ocean_.fetchAnchor(
      prediction_request.unit_id.grp_id,
      prediction_request.unit_id.global_range);
    // check
    CHECK_EQ(parameter_key.size(), anchor.size());
    CHECK_EQ(feature_offset.size(), anchor.size() + 1);
    CHECK_EQ(
      feature_offset.back() - feature_offset.front(),
      feature_key.size());

    // multi-threaded prediction
    // tasks are partitioned by row
    SizeR row_range(0, ocean_.matrix_rows(prediction_request.unit_id.grp_id));
    {
      ThreadPool pool(FLAGS_num_threads);
      for (int i = 0; i < FLAGS_num_threads; ++i) {
        auto thr_row_range = row_range.evenDivide(FLAGS_num_threads, i);
        if (thr_row_range.empty()) { continue; }

        pool.add([this, thr_row_range, anchor,
                  &parameter_key, &feature_key, &feature_offset,
                  &prediction_request]() {
                    prophet(thr_row_range, anchor, parameter_key,
                            feature_key, feature_offset, prediction_request.lookup);
                  });
      }
      pool.startWorkers();
    }

    // drop
    ocean_.drop(
      prediction_request.unit_id.grp_id,
      prediction_request.unit_id.global_range,
      prediction_request.task_id);

    // notify
    queue_pop_cv_.notify_all();
  };
}

AUCData Validation::waitAndGetResult() {
  AUCData auc_data;
  if (!enable_) { return auc_data; }

  // wait until prediction_pending_queue_ is empty
  if (prediction_pending_queue_.size() > 0) {
    std::unique_lock<std::mutex> l(queue_pop_mu_);
    queue_pop_cv_.wait(
      l, [this]() { return prediction_pending_queue_.empty(); });
  }

  // prediction average
  double prediction_sum = 0;
  for (size_t i = 0; i < prediction_.size(); ++i) {
    // logit
    prediction_[i] = 1 / (1 + exp(0 - prediction_[i]));
    prediction_sum += prediction_[i];
  }

  // fill AUCData
  auc_.compute<Ocean::Value>(label_, prediction_, &auc_data);
  auc_data.set_num_examples(num_examples_);
  auc_data.set_click_sum(click_sum_);
  auc_data.set_prediction_sum(prediction_sum);

  // clear prediction_ for the next iteration
  prediction_.setZero();
  return auc_data;
}

SArray<Ocean::FullKey> Validation::getKey(
  const Ocean::GroupID grp_id,
  const Range<Ocean::FullKey>& global_range,
  const Ocean::TaskID task_id) {
  if (!enable_) { return SArray<Ocean::FullKey>(); }

  Ocean::DataPack data_pack = ocean_.get(grp_id, global_range, task_id);
  return SArray<Ocean::FullKey>(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::PARAMETER_KEY)]);
}

SizeR Validation::fetchAnchor(
  const Ocean::GroupID grp_id,
  const Range<Ocean::FullKey>& global_range) {
  if (!enable_) { return SizeR(); }

  return ocean_.fetchAnchor(grp_id, global_range);
}

void Validation::prefetch(
  const Ocean::GroupID grp_id,
  const Range<Ocean::FullKey>& global_range,
  const Ocean::TaskID task_id) {
  if (!enable_) { return; }

  ocean_.prefetch(grp_id, global_range, task_id);
}

void Validation::prophet(
  const SizeR& th_row_range,
  const SizeR& anchor,
  SArray<Ocean::FullKey> parameter_key,
  SArray<Ocean::ShortKey> feature_key,
  SArray<Ocean::Offset> feature_offset,
  std::shared_ptr<WeightLookupTable> weight_lookup) {
  Ocean::ShortKey* index = feature_key.data();

  // j: column id, i: row id
  for (size_t j = 0; j < anchor.size(); ++j) {
    // fetch weight
    Ocean::FullKey key = parameter_key[j];
    double weight = 0;
    {
      WeightLookupTable::const_accessor const_accessor;
      if (weight_lookup->find(const_accessor, key)) {
        weight = const_accessor->second;
      }
    }

    // skip unused weights
    if (weight != weight || 0 == weight) {
      index += feature_offset[j + 1] - feature_offset[j];
      continue;
    }

    // accumulate the weight to corresponding examples
    for (size_t o = feature_offset[j]; o < feature_offset[j + 1]; ++o) {
      auto i = *(index++);
      if (!th_row_range.contains(i)) { continue; }
      prediction_[i] += weight;
    }
  }
}

};  // namespace PS
