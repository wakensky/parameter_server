#include "system/path_picker.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace PS {
void PathPicker::init(const LM::Config& conf) {
  for (int i = 0; i < conf.local_cache().file_size(); ++i) {
    addDirectory(conf.local_cache().file(i) + "/training_data/", PathType::TRAINING);
    addDirectory(conf.local_cache().file(i) + "/models_dumped/", PathType::DUMPED_MODEL);
    addDirectory(conf.local_cache().file(i) + "/models_to_neo/", PathType::NEO_MODEL);
  }

  for (size_t i = 0; i < directories_.size(); ++i) {
    CHECK(!directories_[i].empty()) << "PathType [" << i << "] found no directory";
  }
}

string PathPicker::getPath(
  const string& file_name,
  const PathPicker::PathType type) {
  CHECK_LT(static_cast<size_t>(type), static_cast<size_t>(PathType::NUM)) <<
    "Illegal PathType " << __PRETTY_FUNCTION__ << " got; " <<
    static_cast<size_t>(type) << " vs " << static_cast<size_t>(PathType::NUM);
  CHECK(!directories_[static_cast<size_t>(type)].empty());

  struct stat st;
  string settlement;
  for (const auto& dir : directories_[static_cast<size_t>(type)]) {
    if (-1 != stat((dir + "/" + file_name).c_str(), &st)) {
      // file found
      settlement = dir;
      break;
    }
  }

  if (!settlement.empty()) {
    return settlement + "/" + file_name;
  } else {
    // pick a new directory
    auto& candidate_dir_vec = directories_[static_cast<size_t>(type)];
    auto& cursor = cursors_[static_cast<size_t>(type)];
    return candidate_dir_vec[cursor++ % candidate_dir_vec.size()] + "/" + file_name;
  }
}

bool PathPicker::addDirectory(const string& dir, const PathPicker::PathType type) {
  struct stat st;
  if (-1 == stat(dir.c_str(), &st)) {
    LL << "dir [" << dir << "] cannot be added since error [" <<
      strerror(errno) << "]";
    return false;
  }
  if (!S_ISDIR(st.st_mode)) {
    LL << "dir [" << dir << "] is not a regular directory";
    return false;
  }
  if (0 != access(dir.c_str(), R_OK | W_OK)) {
    LL << "I donnot have read&write permission on dir [" << dir << "]";
    return false;
  }

  CHECK_LT(static_cast<size_t>(type), static_cast<size_t>(PathType::NUM)) <<
    "Illegal PathType for [" << __PRETTY_FUNCTION__ << "]; " <<
    static_cast<size_t>(type) << " vs " << static_cast<size_t>(PathType::NUM);
  directories_[static_cast<size_t>(type)].push_back(dir);
  return true;
}

std::vector<string> PathPicker::allPath(const PathType type) {
  CHECK_LT(static_cast<size_t>(type), static_cast<size_t>(PathType::NUM)) <<
    "Illegal PathType " << __PRETTY_FUNCTION__ << " got; " <<
    static_cast<size_t>(type) << " vs " << static_cast<size_t>(PathType::NUM);
  return directories_[static_cast<size_t>(type)];
}
};
