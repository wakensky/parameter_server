/***************************************************************************
 *
 * Copyright (c) 2015 xxxx.com, Inc. All Rights Reserved
 * $Id$
 *
 **************************************************************************/

 /**
 * @file model_merger.cc
 * @author di.wu(di.wu@xxxx.com)
 * @date 2015/03/18 22:50:20
 * @version $Revision$
 * @brief
 *
 **/

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <memory>
#include "util/common.h"
#include "proto/neo_model.pb.h"

DEFINE_string(
  in_dir, "",
  "Input directory that contains partitioned model files. "
  "Models files are partitioned by Morpheus servers");
DEFINE_string(
  out_dir, "",
  "Output directory that contains merged models, "
  "which could be loaded by NEO");
DEFINE_string(
  in_file_pattern, "",
  "Regex pattern that indicates file name pattern for input model files");

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_in_dir.empty() || FLAGS_out_dir.empty() ||
      FLAGS_in_file_pattern.empty()) {
    std::cerr << "Illegal input" << std::endl;
    return -1;
  }

  // check permision on work directory
  struct stat st;
  if (-1 == stat(FLAGS_in_dir.c_str(), &st)) {
    std::cerr << "cannot stat work dir [" << FLAGS_in_dir << "] since [" <<
      strerror(errno) << "]" << std::endl;
    return -1;
  }
  if (0 != access(FLAGS_in_dir.c_str(), R_OK | W_OK)) {
    std::cerr << "permission not complete on work dir [" <<
      FLAGS_in_dir << "]" << std::endl;
    return -1;
  }

  using ModelClusterHashTable =
    std::unordered_map<std::string, std::vector<std::string>>;
  ModelClusterHashTable cluster_table;

  // scan work directory
  // select those files whose names contain FLAGS_in_file_pattern
  struct dirent** eps;
  int n = scandir(FLAGS_in_dir.c_str(), &eps, nullptr, alphasort);
  if (n < 0) {
    std::cerr << "scandir failed on [" << FLAGS_in_dir << "] [" <<
      strerror(errno) << "]" << std::endl;
    return -1;
  }
  for (int i = 0; i < n; ++i) {
    std::string file_name(eps[i]->d_name);
    if (std::string::npos != file_name.find(FLAGS_in_file_pattern)) {
      std::string cluster_name = file_name.substr(file_name.find(".") + 1);
      if (cluster_name.empty()) {
        std::cerr << "work directory [" << FLAGS_in_dir <<
          "] contains illegal file name [" << file_name << "]" << std::endl;
        return -1;
      }
      cluster_table[cluster_name].push_back(file_name);
    }
  }

  if (cluster_table.empty()) {
    std::cerr << "work directory [" << FLAGS_in_dir <<
      "] contains no file with component [" << FLAGS_in_file_pattern << "]" << std::endl;
    return -1;
  }

  // make sure the output directory exists and is clean
  std::stringstream shell_cmd;
  shell_cmd << "mkdir -p " << FLAGS_out_dir << "; find " << FLAGS_out_dir <<
    " -type f | xargs rm -f";
  if (0 != system(shell_cmd.str().c_str())) {
    std::cerr << "shell cmd [" << shell_cmd.str() << "] failed" << std::endl;
    return -1;
  }

  // merge all model files
  std::stringstream ss_err;
  size_t kBufLen = 1024 * 1024; // 1MB
  std::unique_ptr<char[]> buf_ptr(new char[kBufLen + 1]);
  for (const auto& item : cluster_table) {
    try {
      // open files, accumulate model_size
      size_t total_model_size = 0;
      std::vector<std::shared_ptr<std::ifstream>> partitions;
      for (const auto& file_name : item.second) {
        // open
        std::shared_ptr<std::ifstream> in_ptr(
          new std::ifstream(FLAGS_in_dir + "/" + file_name));
        if (!in_ptr->is_open()) {
          ss_err << "open partitioned model file failed [" << file_name <<
            "] under [" << FLAGS_in_dir << "]";
          throw std::runtime_error(ss_err.str());
        }

        // parse meta
        size_t meta_size = 0;
        if (!in_ptr->read(reinterpret_cast<char*>(&meta_size), sizeof(meta_size))) {
          ss_err << "read meta size failed [" << file_name <<
            "] under [" << FLAGS_in_dir << "]";
          throw std::runtime_error(ss_err.str());
        }
        if (0 == meta_size || meta_size > kBufLen ||
            !in_ptr->read(buf_ptr.get(), meta_size)) {
          ss_err << "read meta body failed [" << file_name <<
            "] meta_size [" << meta_size << "] under [" << FLAGS_in_dir << "]";
          throw std::runtime_error(ss_err.str());
        }
        neo::proto::ModelMeta meta_proto;
        if (!meta_proto.ParseFromArray(buf_ptr.get(), meta_size)) {
          ss_err << "parse meta body failed [" << file_name <<
            "] under [" << FLAGS_in_dir << "]";
          throw std::runtime_error(ss_err.str());
        }

        // accumulate
        total_model_size += meta_proto.model_size();

        // save ifstream
        partitions.push_back(in_ptr);
      }

      // put head into output model
      std::ofstream out(FLAGS_out_dir + "/" + item.first);
      if (!out) {
        ss_err << "open output file failed [" << item.first <<
          "] under [" << FLAGS_in_dir << "]";
        throw std::runtime_error(ss_err.str());
      }
      neo::proto::ModelMeta out_meta;
      out_meta.set_model_size(total_model_size);
      std::string serialized;
      if (!out_meta.SerializeToString(&serialized)) {
        ss_err << "serialize output meta failed";
        throw std::runtime_error(ss_err.str());
      }
      const size_t meta_size = serialized.size();
      if (!out.write(reinterpret_cast<const char*>(&meta_size), sizeof(meta_size))) {
        ss_err << "write new meta size failed [" << item.first <<
          "] under [" << FLAGS_out_dir << "]";
        throw std::runtime_error(ss_err.str());
      }
      if (!out.write(serialized.data(), meta_size)) {
        ss_err << "write new meta body failed [" << item.first <<
          "] under [" << FLAGS_out_dir << "]";
        throw std::runtime_error(ss_err.str());
      }

      // append ModelFeatures protos into output model
      for (auto& in_ptr : partitions) {
        out << in_ptr->rdbuf();
      }
    } catch (std::exception& e) {
      std::cerr << e.what() << "[" << strerror(errno) << "]" << std::endl;
      return -1;
    }
  }

  return 0;
}

/* vim: set ts=4 sw=4 sts=4 tw=100 */
