#pragma once
#include "system/executor.h"
#include "linear_method/linear_method.h"
#include "data/slot_reader.h"

namespace PS {

DECLARE_int32(preprocess_memory_limit_each_group);

namespace LM {

class BatchSolver : public LinearMethod {
 public:
  virtual void init();
  virtual void run();

 protected:
  static const int kPace = 10;
  static const int kFilterPace = 1000000;

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

  // whether training data for grp_id is in binary format
  bool binary(const int grp_id) const;

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

  MatrixPtr<double> y_;
  SlotReader slot_reader_;
  // dual_ = X * w
  SArray<double> dual_;

  std::mutex mu_;

 private:
  class PreprocessStatus {
    public:
      enum class Progress : unsigned char {
        PENDING = 0,
        GOING,
        FINISHED,
        NUM
      };

    public:
      PreprocessStatus(
        KVVectorPtr w, SlotReader* slot_reader,
        const Config& conf, const int grp_id,
        const int time_filter_start) :
        w_(w),
        slot_reader_(slot_reader),
        conf_(conf),
        grp_id_(grp_id),
        time_filter_(time_filter_start),
        prog_(Progress::PENDING),
        filter_finished_(false),
        all_filter_sent_(false) {
      }
      PreprocessStatus(const PreprocessStatus& other) = delete;
      PreprocessStatus& operator= (const PreprocessStatus& rhs) = delete;

    public:
      Progress getProgress() {
        return prog_;
      }
      void setProgress(const Progress progress) {
        prog_ = progress;
      }

      void pullNextFilter() {
        if (all_filter_sent_) { return; }

        // in-group parallel control for memory saving
        if (!pulled_filters_.empty() &&
            !w_->tryWaitOutMsg(kServerGroup, pulled_filters_.back())) {
          return;
        }

        SlotReader::DataPack dp;
        SArray<uint64> unique_keys;
        while (true) {
          dp = slot_reader_->nextPartition(
            grp_id_, SlotReader::UNIQ_COLIDX);
          if (dp.is_ok) {
            if (!dp.uniq_colidx.empty()) {
              unique_keys = unique_keys.setUnion(dp.uniq_colidx);
            }
            // memory limit check
            if (unique_keys.memSize() >= FLAGS_preprocess_memory_limit_each_group) {
              break;
            }
          } else {
            all_filter_sent_ = true;
            break;
          }
        }

        if (!unique_keys.empty()) {
          MessagePtr filter(new Message(kServerGroup, time_filter_));
          filter->key = unique_keys;
          filter->task.set_key_channel(grp_id_);
          Range<uint64>(unique_keys.front(), unique_keys.back() + 1).to(
            filter->task.mutable_key_range());
          filter->task.set_erase_key_cache(true);
          w_->set(filter)->set_query_key_freq(conf_.solver().tail_feature_freq());
          CHECK_EQ(time_filter_, w_->pull(filter));
          pulled_filters_.push_back(time_filter_++);
        }

        return;
      }

      bool allFilterFinished() {
        if (!all_filter_sent_) { return false; }
        if (filter_finished_) { return true; }

        for (const auto msg_id : pulled_filters_) {
          if (!w_->tryWaitOutMsg(kServerGroup, msg_id)) {
            return false;
          }
        }
        return (filter_finished_ = true);
      }

      void finish() { prog_ = Progress::FINISHED; }
      void start() { prog_ = Progress::GOING; }

    private:
      KVVectorPtr w_;
      SlotReader* slot_reader_;
      Config conf_;
      int grp_id_;
      int time_filter_;
      Progress prog_;
      bool filter_finished_;
      // all filter messages been sent to servers
      bool all_filter_sent_;
      std::vector<int> pulled_filters_;
  }; // end of PreprocessStatus

  std::vector<std::shared_ptr<PreprocessStatus>> preprocess_status_;
};


} // namespace LM
} // namespace PS

  // void saveAsDenseData(const Message& msg);
