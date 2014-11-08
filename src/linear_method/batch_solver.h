#pragma once
#include "linear_method/linear_method.h"
#include "data/slot_reader.h"

namespace PS {

DECLARE_int32(in_group_parallel);

namespace LM {

class BatchSolver : public LinearMethod {
 public:
  virtual void init();
  virtual void run();

 protected:
  static const int kPace = 10;
  static const int kFilterPace = 10000;

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

  // matrix for training data, available at the workers
  std::map<int, MatrixInfo> matrix_info_;
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
        const int grp_id,
        const int time_count_start, const int time_boundary,
        const int time_filter_start) :
        prog_(Progress::PENDING),
        count_finished_(false),
        filter_finished_(false),
        grp_id_(grp_id),
        slot_reader_(slot_reader),
        time_count_(time_count_start),
        time_filter_(time_filter_start),
        time_boundary_(time_boundary) {
        // do nothing
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
      bool pushNextCount() {
        if (all_count_sent_) { return false; }

        // in-group parallel control
        for (const auto msg_id : pushed_counts_) {
          if (msg_id > time_count_ - FLAGS_in_group_parallel) {
            continue;
          }
          if (!w_->tryWaitOutMsg(kServerGroup, msg_id)) {
            return false;
          }
        }

        DataPack dp = slot_reader_.nextPartition(
          grp_id_, SlotReader::UNIQ_COLIDX);
        if (dp.is_ok) {
          MessagePtr count(new Message(kServerGroup, time_count_));
          count->key = dp.uniq_colidx;
          count->task.set_key_channel(grp_id_);
          auto arg = w_->set(count);
          arg->set_insert_key_freq(true);
          arg->set_countmin_k(conf_.solver().countmin_k());
          arg->set_countmin_n(static_cast<int>( // wakensky; 1000 was chosen casually
            dp.uniq_colidx.size() * 1000 * conf_.solver().countmin_n_ratio()));
          CHECK_EQ(time_count_, w_->push(count));

          waiting_counts_.push_back(time_count_++);
          return true;
        }
        all_count_sent_ = true;
        return false;
      }

      bool pullNextFilter() {
        if (all_filter_sent_) { return false; }

        // in-group parallel control
        for (const auto msg_id : pulled_filters_) {
          if (msg_id > time_filter_ - FLAGS_in_group_parallel) {
            continue;
          }
          if (!w_->tryWaitOutMsg(kServerGroup, msg_id)) {
            return false;
          }
        }

        DataPack dp = slot_reader_.nextPartition(
          grp_id_, SlotReader::UNIQ_COLIDX);
        if (dp.is_ok) {
          MessagePtr filter(
            new Message(KServerGroup, time_filter++, time_boundary_ + 1));
          filter->key = dp.uniq_colidx;
          filter->task.set_key_channel(grp_id_);
          filter->task.set_erase_key_cache(true);
          w_->set(filter)->set_query_key_freq(conf_.solver().tail_feature_freq());
          CHECK_EQ(time_filter_, w_->pull(filter));

          waiting_filters_.push_back(time_filter_++);
          return true;
        }
        all_filter_sent_ = true;
        return false;
      }

      // whether all count PUSHes acknowledged by servers?
      bool allCountFinished() {
        if (!all_count_sent_) { return false; }
        if (count_finished_) { return true; }

        for (const auto msg_id : pushed_counts_) {
          if (!w_->tryWaitOutMsg(kServerGroup, msg_id)) {
            return false;
          }
        }

        // send boundary message
        MessagePtr boundary(new Message(kServerGroup, time_boundary_));
        boundary->task.set_key_channel(grp_id_);
        CHECK_EQ(time_boundary_, w_->push(boundary));

        return (count_finished_ = true);
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
      Progress prog_;
      bool count_finished_;
      bool filter_finished_;
      std::vector<int> pushed_counts_;
      std::vector<int> pulled_filters_;
      SlotReader* slot_reader_;
      int grp_id_;
      int time_count_;
      int time_filter_;
      int time_boundary_;
  };

  std::vector<PreprocessStatus> preprocess_status_;
};


} // namespace LM
} // namespace PS

  // void saveAsDenseData(const Message& msg);
