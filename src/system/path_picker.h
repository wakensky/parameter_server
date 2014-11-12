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
    SINGLETON(PathPicker);
    ~PathPicker();
    PathPicker(const PathPicker& other) = delete;
    PathPicker& operator= (const PathPicker& rhs) = delete;

    void init(const LM::Config& conf);
    // return the full path for file_name
    //   if file exists already, return the existing path
    //   otherwise, return a randomly picked new path
    string getPath(const string& file_name);

  private:
    PathPicker();
    // return false if
    //   dir not exists
    //   W/R permission not allowed
    bool addDirectory(const string& dir);

  private:
    std::vector<string> directories_;
    size_t cursor_;
};
};
