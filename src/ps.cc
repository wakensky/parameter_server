#include "system/postoffice.h"

DEFINE_bool(log_to_file, false, "redirect INFO log to file; eg. log_w1_datetime");
DEFINE_bool(log_instant, false, "disable buffer of glog");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  if (FLAGS_log_instant) {
    FLAGS_logbuflevel = -1;
  }

  PS::Postoffice::instance().run();

  return 0;
}
