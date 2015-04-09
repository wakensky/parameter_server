#include <gperftools/malloc_extension.h>
#include "system/postoffice.h"

DEFINE_bool(log_instant, false, "disable buffer of glog");
DEFINE_int32(memory_release_rate, 9,
  "By default, tcmalloc will release no-longer-used memory back to the kernel gradually, over time. "
  "The FLAGS_memory_release_rate flag controls how quickly this happens. "
  "Reasonable rates are in the range [0,10], 9 as default. "
  "Increase this flag to return memory faster. 0 means never give back. "
  "Any given value who violates this range will fall on 9. ");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  if (FLAGS_log_instant) FLAGS_logbuflevel = -1;

#ifdef TCMALLOC
  if (FLAGS_memory_release_rate >= 0 && FLAGS_memory_release_rate <= 10) {
    MallocExtension::instance()->SetMemoryReleaseRate(FLAGS_memory_release_rate);
  } else {
    MallocExtension::instance()->SetMemoryReleaseRate(9);
  }
#endif

  PS::Postoffice::instance().run();
  LL << "exists main function";

  return 0;
}
