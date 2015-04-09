#pragma once
#include <cstdint>
#include <cinttypes>
#include "proto/task.pb.h"
#include "util/common.h"

namespace PS {

DECLARE_bool(verbose);
DECLARE_int32(prefetch_job_limit);
DECLARE_bool(less_memory);

class PathPicker {
  public:
    enum class PathType: unsigned char {
      TRAINING = 0,
      DUMPED_MODEL,
      NEO_MODEL,
      NUM
    };

  public:
    SINGLETON(PathPicker);
    ~PathPicker() {};
    PathPicker(const PathPicker& other) = delete;
    PathPicker& operator= (const PathPicker& rhs) = delete;

    void init(const LM::Config& conf);

    // return the full path for file_name
    //   if file exists already, return the existing path
    //   otherwise, return a randomly picked new path
    string getPath(
      const string& file_name,
      const PathType type = PathType::TRAINING);

    // Enumerate all candidate directories under certain PathType
    std::vector<string> allPath(const PathType type);

  private:
    PathPicker() {
      for (auto& item : cursors_) item = 0;
    };
    // return false if
    //   dir not exists
    //   W/R permission not allowed
    bool addDirectory(
      const string& dir,
      const PathType type);

  private:
    std::array<std::vector<string>, static_cast<size_t>(PathType::NUM)> directories_;
    std::array<size_t, static_cast<size_t>(PathType::NUM)> cursors_;
};
};
