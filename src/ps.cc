#include <execinfo.h>
#include "system/postoffice.h"

DEFINE_bool(log_instant, false, "disable buffer of glog");

int main(int argc, char *argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;

  if (FLAGS_log_instant) {
    FLAGS_logbuflevel = -1;
  }

  auto terminateHandler = []() {
    std::exception_ptr exp_ptr = std::current_exception();
    try {
      std::rethrow_exception(exp_ptr);
    } catch (std::exception& e) {
        std::cerr << "Terminated due to exception:" << e.what() << std::endl;
    }

    void *array[20];
    size_t size = backtrace(array, sizeof(array) / sizeof(array[0]));
    backtrace_symbols_fd(array, size, STDERR_FILENO);
  };
  std::set_terminate(terminateHandler);

  PS::Postoffice::instance().run();

  return 0;
}
