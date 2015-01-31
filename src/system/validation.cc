#include <unistd.h>
#include <fcntl.h>
#include "util/split.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdexcept>
#include <gperftools/malloc_extension.h>
#include "base/shared_array_inl.h"
#include "system/validation.h"

namespace PS {

Validation::Validation() {
  ;
}

Validation::~Validation() {
  ;
}

void Validation::init(
  const string& identity,
  LM::DataConfig& file_list,
  PathPicker* path_picker) {
  CHECK(!identity.empty());
  identity_ = identity;
  path_picker_ = path_picker;
  file_list_ = file_list;
}

bool Validation::download(const Task& preprocess_task) {
  // fill group_ranges_
  for (int i = 0; i < preprocess_task.partition_info_size(); ++i) {
    const GroupID grp_id = preprocess_task.partition_info(i).fea_grp();
    Range<FullKey> global_range(preprocess_task.partition_info(i).key());
    group_ranges_[grp_id].push_back(global_range.begin());
    group_ranges_[grp_id].push_back(global_range.end());
  }

  // sort and unique
  size_t number_of_blocks = 0;
  for (auto& ranges : group_ranges_) {
    std::sort(ranges.begin(), ranges.end());
    auto last = std::unique(ranges.begin(), ranges.end());
    ranges.erase(last, ranges.end());

    CHECK_GE(ranges.size(), 2);
    if (!ranges.empty()) {
      number_of_blocks += ranges.size() - 1;
    }
  }
  CHECK_EQ(number_of_blocks, preprocess_task.partition_info_size());

  // read training files line by line
  LineReader line_reader(file_list, FLAGS_validation_line_limit);
  string line;
  // I restrict the number of lines in each package,
  //   making memory usage controllable
  size_t package_line_count = 0;
  ExampleID example_id = 0;
  ExampleText example_text;
  while (!(example_text.text = line_reader.readLine()).empty()) {
    example_text.id = example_id++;
    pending_examples_.push(example_text);
    package_line_count++;

    if (package_line_count >= FLAGS_validation_package_volumn) {
      // wait
      LI << "Validation::download is waiting package [" << package_id << "]";
      while (1) {
        if (pending_lines_.size() <= 0 - FLAGS_validation_thread_num) {
          break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      LI << "Validation::download finishes waiting package [" << package_id << "]";
    }

    // dump package
    CHECK(dumpPackages());
    package_line_count = 0;
  }
  CHECK(dumpPackages());
  package_line_count = 0;
}

void Validation::exampleThreadFunc() {
  ExampleParser parser;
  parser.init(file_list_.text(), file_list_.ignore_feature_group());

  auto group_ranges = group_ranges_;

  // Separate one example to different batches
  LivePackageHashMap separated_one_example;

  auto separate = [](
    const ExampleID example_id;
    const Example& full_proto,
    const std::unordered_map<GroupID, std::vector<FullKey>>& group_ranges,
    LivePackageHashMap& separated_one_example) {
    separated_one_example.clear();

    ValidationPartialExample partial_proto;
    std::vector<FullKey> partial_keys;

    for (int i = 0; i < full_proto.slot_size(); ++i) {
      const GroupID group_id = full_proto.slot(i).id();
      auto& range_vec = group_ranges[group_id];

      if (!range_vec.empty()) {
        size_t last_range_idx = 0;
        for (int j = 0; j < full_proto.slot(i).key_size(); ++j) {
          FullKey key = full_proto.slot(i).key(j);

          if (range_vec[last_range_idx] <= key) {
            // store
            if (!partial_keys.empty()) {
              Range<FullKey> global_range(range_vec[last_range_idx - 1], range_vec[last_range_idx]);

              separated_one_example[BatchID(group_id, global_range)].proto = partial_proto;
              separated_one_example[BatchID(group_id, global_range)].keys = partial_keys;
            }

            // clear
            partial_proto.Clear();
            partial_keys.clear();

            // set example id
            partial_proto.set_id(example_id);
          }

          // locate
          while (range_vec[last_range_idx] <= key &&
                 last_range_idx < range_vec.size()) {
            ++last_range_idx;
          }

          if (last_range_idx > 0 && last_range_idx < range_vec.size()) {
            partial_proto.add_keys(key);
            partial_keys.push_back(key);
          } else {
            // TODO
            LL << "cannot locate key under group x";
            break;
          }
        }
      }
    }
  };

  while (go_on_) {
    // take out an example (plain text)
    ExampleText example_text;
    pending_examples_.pop(example_text);

    // remove tailing line-break
    if (!example_text.text.empty() && '\n' == example_text.text.back()) {
      example_text.text.resize(example_text.text.size() - 1);
    }

    // parse to Example proto, which contains all features for an example
    Example full_proto;
    if (!parser.toProto(example_text.text, &full_proto)) {
      LL << "Validation::exampleThreadFunc parses to full proto failed. [" <<
        example_text.text << "]";
      continue;
    }

    // separate keys into sub proto buffers
    LivePackageHashMap separated_one_example;
    separate(full_proto, group_ranges, separated_one_example);

    // merge fragments to global live_packages_
    for (const auto fragment : separated_one_example) {
      LiveBatchHashMap::accessor accessor;
      if (!live_packages_.find(accessor, fragment.first)) {
        LL << ;
        continue;
      }

      // merge partial proto
      accessor->second.proto.add_partitial_examples(fragment.second.proto);

      // merge partial keys
      accessor->second.keys.insert(
        accessor->second.keys.end(),
        fragment.second.keys.begin(),
        fragment.second.keys.end());
    }
  };
}

bool Validation::dumpPackages() {
  // traverse all live packages and dump to disk
  for (const auto& gid_ranges : group_ranges_) {
    const GroupID group_id = gid_ranges.first;
    if (gid_ranges.second.size() < 2) {
      // TODO
      LL << "ranges for group x not illegal";
    }
    for (size_t i = 0; i < gid_ranges.second.size() - 1; ++i) {
      // produce batch_id
      Range<FullKey> global_range(gid_ranges.second[i], gid_ranges.second[i + 1]);
      BatchID batch_id(group_id, global_range);

      LiveBatchHashMap::accessor live_accessor;
      if (live_packages_.find(live_accessor, batch_id)) {
        // proto file name
        std::stringstream ss_proto;
        ss_proto << "validation.proto." << identity_ << "." << group_id <<
          "." << global_range.begin() << "-" << global_range.end();
        const string proto_path = path_picker_->getPath(ss_proto.str());

        // dump proto file
        std::ofstream out(proto_path, ios::out | ios::trunc | ios::binary);
        CHECK(out.good());
        CHECK(live_accessor->second.proto.SerializeToOstream(&out));
        out.close();

        // release proto
        live_accessor->second.proto.Clear();

        // sort and unique
        std::sort(live_accessor->second.keys.begin(), live_accessor->second.keys.end());
        auto last = std::unique(merged.begin(), merged.end());
        merged.erase(last, merged.end());

        // dump unique keys
        std::stringstream ss_uniq;
        ss_uniq << "validation.uniq_key." << identity_ << "." << group_id <<
          "." << global_range.begin() << "-" << global_range.end();
        SArray<FullKey> uniq_keys({live_accessor->second.keys});
        CHECK(uniq_keys.writeToFile(ss_uniq.str()));

        // release keys
        live_accessor->second.keys.clear();

        // insert path into dumped_batches_
        {
          DumpedBatchHashMap::accessor dumped_accessor;
          if (dumped_batches_.find(dumped_accessor, batch_id)) {
            DumpedPackage new_package;
            new_package.proto_path = proto_path;
            new_package.uniq_keys_path = ss_uniq.str();

            dumped_accessor->second.push_back(new_package);
          } else {
            LL << "critical lost";
          }
        }
      }
    }
  }
}

SArray<FullKey> Validation::getUniqueKeys(const BatchID& batch_id) {
  SArray<FullKey> ret;

  // merge all packages together under the specified batch
  DumpedBatchHashMap::accessor dumped_batch_accessor;
  if (!dumped_batches_.find(dumped_batch_accessor, batch_id)) {
    // TODO
    LL << "cannot find batch_id when merging unique keys";
    return ret;
  }

  for (const auto& dumped_package : dumped_batch_accessor->second) {
    SArray<char> stash;
    CHECK(stash.readFromFile(dumped_package.uniq_keys_path));
    SArray<FullKey> sub_unique_keys(stash);

    ret.setUnion(sub_unique_keys);
  }

  return ret;
}

};  // namespace PS
