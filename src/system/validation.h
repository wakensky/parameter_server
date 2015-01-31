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
#include "system/path_picker.h"
#include "system/ocean.h"

namespace PS {

DECLARE_bool(verbose);

/*
 * download validation data
 * AUC calculation
 */
class Validation {
  public:
    using FullKey = uint64;
    using Value = double;
    using GroupID = int;
    using TaskID = int;
    using BatchID = Ocean::UnitID;
    using PackageID = uint32;
    using ExampleID = uint32;

  public:
    SINGLETON(Validation);
    ~Validation();
    Validation(const Validation& other) = delete;
    Validation& operator= (const Validation& rhs) = delete;

    // initialization
    void init(
      const string& identity,
      const LM::DataConfig& file_list,
      PathPicker* path_picker);

    // download validation data to disk
    bool download(const Task& preprocess_task);

    SArray<FullKey> getUniqueKeys(const BatchID& batch_id);

    // predict examples under the batch
    void predictBatch(
      const BatchID& batch_id,
      SArray<FullKey> keys,
      SArray<Value> weight);

  private:
    class LineReader {
      public:
        LineReader(const DataConfig& files, const size_t line_limit):
          files_(files),
          file_being_read_(nullptr),
          readed_line_count_(0),
          line_limit_(line_limit),
          buf_(new char[kBufLen + 1]) {
        }
        LineReader(const LineReader& other) = delete;
        LineReader& operator= (const LineReader& rhs) = delete;
        // read files line by line
        // return: empty string means all files exhausted
        string readLine() {
          if (0 != readed_line_count_ && readed_line_count_ >= line_limit_) {
            if (file_being_read_ != nullptr) {
              file_being_read_->close();
              file_being_read_ = nullptr;
            }
            return string();
          }

          while (1) {
            // open the next file
            if (nullptr == file_being_read_) {
              if (file_idx_ >= files_.file_size()) {
                return string();
              }

              while (file_idx_ < files_.file_size()) {
                file_being_read_ = File::open(ithFile(files_, file_idx_++));
                if (nullptr != file_being_read_) {
                  break;
                } else {
                  LL << "Validation::LineReader open failed [" <<
                    files_.file(file_idx_ - 1) << "]";
                }
              }
              if (nullptr == file_being_read_) {
                // all files exhausted
                return string();
              }
            }

            // read one line
            const char* ret = file_being_read_->readLine(buf_.get(), kBufLen);
            if (nullptr == ret) {
              // current file exhausted
              file_being_read_->close();
              file_being_read_ = nullptr;
            } else {
              readed_line_count_++;
              return string(buf_.get());
            }
          }
        }
      private:
        const size_t kBufLen = 1024 * 1024;
        const DataConfig files_;
        // which file is being read
        size_t file_idx_;
        File* file_being_read_;
        // how many lines have been read
        size_t readed_line_count_;
        // maximum lines I could read
        // 0 means no limit
        const size_t line_limit_;
        // line buffer
        std::unique_ptr<char[]> buf_;
    };

    struct ExampleText {
      ExampleID id;
      string text;
    };

    struct DumpedPackage {
      string proto_path;
      string uniq_keys_path;
    };

    struct LivePackage {
      ValidationData proto;
      std::vector<FullKey> keys;
    };

  private:
    Validation();

    // dump live_packages_ onto disk
    // dump packages associated with the same examples to the disk
    // We separated FLAGS_VALIDATION_BATCH_VOLUMN examples into several packages
    // Each package contains training features within a specified BatchID ({group, key_range})
    bool dumpPackages();

    // sub-threads process examples
    void exampleThreadFunc();

  private:
    string identity_;
    PathPicker* path_picker_;
    // which validation files I should download
    DataConfig file_list_;
    // grp_id -> (N+1) Fullkeys
    //   N: the number of column partitioned blocks
    //   the last key is the ending guard
    std::unordered_map<GroupID, std::vector<FullKey>> group_ranges_;
    // examples need to be processed
    tbb::concurrent_bounded_queue<std::<ExampleID, string>> pending_examples_;

    // BatchID -> dumped proto path
    using DumpedBatchHashMap = tbb::concurrent_hash_map<BatchID, std::vector<DumpedPackage>>;
    DumpedBatchHashMap dumped_batches_;

    // BatchID -> one package
    using LivePackageHashMap = tbb::concurrent_hash_map::<BatchID, LivePackage>;
    LivePackageHashMap live_packages_;

    // sub threads who process the original examples (in text format)
    std::vector<std::thread> example_threads_;

    // ExampleID -> {actual label, model's prediction}
    tbb::concurrent_hash_map<ExampleID, std::pair<double, double>> label_prediction_;
};

}; // namespace PS
