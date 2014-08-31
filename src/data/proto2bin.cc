#include "base/matrix_io.h"
#include "base/io.h"

DEFINE_string(input, "../config/data.config", "");
DEFINE_string(output, "../data/bin/xx", "");

int main(int argc, char *argv[]) {
  FLAGS_logtostderr = 1;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  using namespace PS;

  DataConfig cf;
  ReadFileToProtoOrDie(FLAGS_input, &cf);

  auto cf2 = searchFiles(cf);
  InstanceInfo info;
  auto data = readMatrices<double>(cf2, info, "proto2bin", -1, false);

  SArray<Key> key;
  auto X = data[1]->localize(&key);

  auto out = FLAGS_output;
  data[0]->writeToBinFile(out + ".y");
  X->writeToBinFile(out + ".X");
  key.writeToFile(SizeR::all(), out+".X.key");

  return 0;
}
