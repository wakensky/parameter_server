#include "system/postoffice.h"

int main(int argc, char *argv[]) {
  FLAGS_logtostderr = 1;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  PS::Postoffice::instance().run("");

  LL << "done";
  return 0;
}
