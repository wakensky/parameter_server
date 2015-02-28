#include "system/path_picker.h"
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

namespace PS {
void PathPicker::init(const LM::Config& conf) {
  cursor_ = 0;

  for (int i = 0; i < conf.local_cache().file_size(); ++i) {
    addDirectory(conf.local_cache().file(i));
  }
  CHECK(!directories_.empty());
}

string PathPicker::getPath(const string& file_name) {
  CHECK(!directories_.empty());

  struct stat st;
  string settlement;
  for (const auto& dir : directories_) {
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
    return directories_[cursor_++ % directories_.size()] + "/" + file_name;
  }
}

bool PathPicker::addDirectory(const string& dir) {
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

  directories_.push_back(dir);
  return true;
}
};
