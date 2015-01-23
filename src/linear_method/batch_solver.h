#pragma once
#include "base/bitmap.h"
#include "linear_method/linear_method.h"
#include "data/slot_reader.h"

namespace PS {

DECLARE_int32(preprocess_mem_limit);

namespace LM {

class BatchSolver : public LinearMethod {
 public:
  virtual void init();
  virtual void run();

 protected:
  static const int kPace = 10;

  virtual int loadData(const MessageCPtr& msg, ExampleInfo* info);
  virtual void preprocessData(const MessageCPtr& msg);
  virtual void updateModel(const MessagePtr& msg);
  virtual void runIteration();

  virtual Progress evaluateProgress();
  virtual void showProgress(int iter);

  void computeEvaluationAUC(AUCData *data);
  void saveModel(const MessageCPtr& msg);

  bool loadCache(const string& name) { return dataCache(name, true); }
  bool saveCache(const string& name) { return dataCache(name, false); }
  bool dataCache(const string& name, bool load);

  typedef shared_ptr<KVVector<Key, double>> KVVectorPtr;
  KVVectorPtr w_;

  // feature block info, only available at the scheduler, format: pair<fea_grp_id, fea_range>
  typedef std::vector<std::pair<int, Range<Key>>> FeatureBlocks;
  FeatureBlocks fea_blk_;
  std::vector<int> blk_order_;
  std::vector<int> prior_blk_order_;
  std::vector<int> fea_grp_;

  // global data information, only available at the scheduler
  ExampleInfo g_train_info_;

  // training data, available at the workers
  std::map<int, MatrixPtr<double>> X_;
  MatrixPtr<double> y_;
  SlotReader slot_reader_;
  // dual_ = X * w
  SArray<double> dual_;
  std::array<MatrixInfo, 2048> matrix_info_;
  std::unordered_map<int, Bitmap> active_set_;
  std::unordered_map<int, SArray<double>> delta_;


  std::mutex mu_;

 private:
  class PreprocessHelper {
    public:
      enum class Status : unsigned char {
        PENDING = 0,
        GOING,
        FINISHED,
        NUM
      };

      PreprocessHelper(
        KVVectorPtr w, SlotReader* slot_reader,
        const int tail_frequency, const int grp_id) :
          w_(w),
          slot_reader_(slot_reader),
          tail_frequency_(tail_frequency),
          grp_id_(grp_id),
          status_(Status::PENDING),
          all_filters_sent_(false),
          partition_idx_(0) {
        slot_reader_->getAllPartitions(grp_id, "colidx_sorted_uniq", partitions_);

        // wakensky
        for (const auto& item : partitions_) {
          LI << "PreprocessHelper got partitions: " <<
            item.first << "; " <<
            item.second.toString();
        }
      }
      PreprocessHelper(const PreprocessHelper& other) = delete;
      PreprocessHelper& operator= (const PreprocessHelper& rhs) = delete;

      Status getStatus() {
        return status_;
      }
      void setStatus(const Status status) {
        status_ = status;
      }

      // Send next keys batch to servers
      // If something been sent indeed, return time+1;
      //   else, return time
      int pullNextFilter(int time) {
        if (all_filters_sent_) {
          return time;
        }

        // in-group concurrency control for memory saving
        // If last batch has not been replied, next batch could not be launched
        if (!pulled_filters_.empty() &&
            !w_->tryWaitOutMsg(kServerGroup, pulled_filters_.back())) {
          return time;
        }

        // merge several unique_key partitions until memory limit reaches
        SArray<Key> merged_unique_keys;
        while (partition_idx_ < partitions_.size()) {
          auto info = partitions_[partition_idx_++];
          SArray<char> compressed;
          compressed.readFromFile(info.second, info.first);
          SArray<Key> unique_keys;
          unique_keys.uncompressFrom(compressed);
          if (!unique_keys.empty()) {
            merged_unique_keys = merged_unique_keys.setUnion(unique_keys);
          }

          // memory limit check
          if (merged_unique_keys.memSize() >= FLAGS_preprocess_mem_limit) {
            break;
          }
        }

        // send
        if (!merged_unique_keys.empty()) {
          MessagePtr filter(new Message(kServerGroup, time));
          filter->setKey(merged_unique_keys);
          filter->task.set_key_channel(grp_id_);
          Range<Key>(merged_unique_keys.front(), merged_unique_keys.back() + 1).to(
            filter->task.mutable_key_range());
          // filter->addFilter(FilterConfig::KEY_CACHING)->set_clear_cache_if_done(true);
          w_->set(filter)->set_query_key_freq(tail_frequency_);
          CHECK_EQ(time, w_->pull(filter));
          pulled_filters_.push_back(time);

          // wakensky
          LI << "sent filter msg: " << filter->shortDebugString();
        }

        if (partition_idx_ >= partitions_.size()) {
          all_filters_sent_ = true;
        }
        return time + 1;
      }

      bool allFiltersFinished() {
        if (!all_filters_sent_) {
          return false;
        }

        for (auto& time : pulled_filters_) {
          // 0 indicates an finished job
          if (0 != time && !w_->tryWaitOutMsg(kServerGroup, time)) {
            return false;
          } else {
            time = 0;
          }
        }
        return true;
      }

    private:
      KVVectorPtr w_;
      SlotReader* slot_reader_;
      const int tail_frequency_;
      const int grp_id_;
      Status status_;
      bool all_filters_sent_;
      size_t partition_idx_;
      std::vector<std::pair<string, SizeR>> partitions_;
      std::vector<int> pulled_filters_;
  }; // class PreprocessHelper
};


} // namespace LM
} // namespace PS

  // void saveAsDenseData(const Message& msg);
