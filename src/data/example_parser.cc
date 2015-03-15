#include "data/example_parser.h"
#include <functional>
#include "util/strtonum.h"
#include "util/MurmurHash3.h"
// #include "base/matrix_io_inl.h"

namespace PS {

DEFINE_bool(shuffle_fea_id, false,
  "shuffle fea id of Terafea (lowest 54bits) with MurmurHash3 "
  "on downloading");
DEFINE_bool(add_beta_feature, false,
  "whether add feature id 0 into a special group 1023. "
  "help eliminate bias. false in default");

// NOTICE: Do not use strtok, it is not thread-safe, use strtok_r instead
void ExampleParser::init(TextFormat format, bool ignore_fea_slot) {
  ignore_fea_slot_ = ignore_fea_slot;
  format_ = format;
  using namespace std::placeholders;
  if (format == DataConfig::LIBSVM) {
    parser_ = std::bind(&ExampleParser::parseLibsvm, this, _1, _2);
  } else if (format == DataConfig::ADFEA) {
    parser_ = std::bind(&ExampleParser::parseAdfea, this, _1, _2);
  } else if (format == DataConfig::TERAFEA) {
    parser_ = std::bind(&ExampleParser::parseTerafea, this, _1, _2);
  }else {
    CHECK(false) << "unknown text format " << format;
  }
}

void ExampleParser::clear() {
  for (int i = 0; i < kSlotIDmax; ++i) slot_info_[i].Clear();
  info_.Clear();
  num_ex_ = 0;
}

bool ExampleParser::toProto(char* line, Example* ex) {
  // convert to protobuf format
  ex->Clear(); if (!parser_(line, ex)) return false;

  // update info
  for (int i = 0; i < ex->slot_size(); ++i) {
    const auto& slot = ex->slot(i);
    if (slot.id() >= kSlotIDmax) return false;
    auto& sinfo = slot_info_[slot.id()];
    for (int j = 0; j < slot.key_size(); ++j) {
      uint64 key = slot.key(j);
      sinfo.set_min_key(std::min((uint64)sinfo.min_key(), key));
      sinfo.set_max_key(std::max((uint64)sinfo.max_key(), key + 1));
    }
    sinfo.set_nnz_ex(sinfo.nnz_ex() + 1);
    sinfo.set_nnz_ele(sinfo.nnz_ele() + std::max(slot.key_size(), slot.val_size()));
  }
  ++ num_ex_;
  return true;
}

ExampleInfo ExampleParser::info() {
  info_.set_num_ex(num_ex_);
  info_.clear_slot();
  for (int i = 0; i < kSlotIDmax; ++i) {
    auto &sinfo = slot_info_[i];
    if (!sinfo.nnz_ele()) continue;
    sinfo.set_id(i);
    if (i == 0) {  // the label
      sinfo.set_format(SlotInfo::DENSE);
      sinfo.set_min_key(0);
      sinfo.set_max_key(1);
    } else {
      if (format_ == DataConfig::LIBSVM) {
        sinfo.set_format(SlotInfo::SPARSE);
      } else if (format_ == DataConfig::ADFEA) {
        sinfo.set_format(SlotInfo::SPARSE_BINARY);
      } else if (format_ == DataConfig::TERAFEA) {
        sinfo.set_format(SlotInfo::SPARSE_BINARY);
      }
    }
    *info_.add_slot() = sinfo;
  }
  return info_;
}

// libsvm:
//
//   label feature_id:weight feature_id:weight feature_id:weight ...
//
// assume feature_ids are ordered
bool ExampleParser::parseLibsvm(char* buff, Example* ex) {
  char *saveptr;
  // label
  char * pch = strtok_r(buff, " \t\r\n", &saveptr);
  float label;
  if (!strtofloat(pch, &label)) return false;
  auto lbl_slot = ex->add_slot();
  lbl_slot->set_id(0);
  lbl_slot->add_val(label);

  // feature and weights
  pch = strtok_r(NULL, " \t\r\n", &saveptr);
  auto fea_slot = ex->add_slot();
  fea_slot->set_id(1);
  uint64 idx = 0, last_idx=0;
  while (pch != NULL) {
    char *it;
    for (it = pch; *it != ':' && *it != 0; it ++);
    if (*it == 0) return false;
    *it = 0;

    if (!strtou64(pch, &idx)) return false;
    float val;
    if (!strtofloat(it+1, &val)) return false;
    if (last_idx > idx) return false;
    last_idx = idx;

    fea_slot->add_key(idx);
    fea_slot->add_val(val);
    pch = strtok_r(NULL, " \t\r\n", &saveptr);
  }
  return true;
}

// adfea format:
//
//   line_id 1 clicked_or_not key:grp_id key:grp_id ...
//
// same group_ids should appear together, but not necesary be ordered
bool ExampleParser::parseAdfea(char* line, Example* ex) {
  uint64 key = -1;
  int pre_slot_id = 0;
  Slot* slot = ex->add_slot();
  slot->set_id(0);

  char* saveptr;
  char* tk = strtok_r(line, " :", &saveptr);
  for (int i = 0; tk != NULL; tk = strtok_r(NULL, " :", &saveptr), ++i) {
    if (i == 0) {
      // skip the line id
    } else if (i == 1) {
      // skip, it is always 1
    } else if (i == 2) {
      int32 label;
      if (!strtoi32(tk, &label)) return false;
      slot->add_val(label > 0 ? 1.0 : -1.0);
    } else if (i % 2 == 1) {
      if (!strtou64(tk, &key)) return false;
    } else {
      int slot_id = 1;
      if (!ignore_fea_slot_ && !strtoi32(tk, &slot_id)) return false;
      if (slot_id != pre_slot_id) {
        slot = ex->add_slot();
        slot->set_id(slot_id);
        pre_slot_id = slot_id;
      }
      slot->add_key(key);
    }
  }
  // LL << ex->ShortDebugString();
  return true;
}

// terafea format:
//
//   clicked_or_not line_id | uint64 uint64 ...
//   uint64:
//      the most significant 10 bits    - group id
//      lower 54 bits                   - feature id
//
//  no guarantee that the same group ids stay contiguously
//
bool ExampleParser::parseTerafea(char* line, Example* ex) {
  std::vector<uint64> group_feature_vec;
  group_feature_vec.reserve(1024);

  // add the very first slot
  uint64 pre_grp_id = 0;
  Slot* slot = ex->add_slot();
  slot->set_id(0);

  char* saveptr;
  char* tk = strtok_r(line, " |", &saveptr);
  for (int i = 0; tk != NULL; tk = strtok_r(NULL, " |", &saveptr), ++i) {
    if (i == 0) {
      // label
      int32 label;
      if (!strtoi32(tk, &label)) return false;

#if 0
      // patch for abtest
      // It seems that some large group may produce segment fault
      {
        if (!(label > 0) && (rand() % 10 < 5)) {
          return false;
        }
      }
#endif

      slot->add_val(label > 0 ? 1.0 : -1.0);
    } else if (i == 1) {
      // skip, line_id
    } else {
      uint64 key = -1;
      if (!strtou64(tk, &key)) return false;

      // patch for abtest
      {
        const uint64 grp_id = key >> 54;
        if (20 == grp_id || 25 == grp_id ||
            26 == grp_id || 29 == grp_id) {
          continue;
        }
      }

      group_feature_vec.push_back(key);
    }
  }
  if (group_feature_vec.empty()) { return false; }

  // sort; make same group_ids appear together
  std::sort(group_feature_vec.begin(), group_feature_vec.end());

  for (const auto& key : group_feature_vec) {
    uint64 grp_id = key >> 54;
    uint64 fea_id = key;
    // uint64 fea_id = key & 0x3FFFFFFFFFFFFF;

    if (FLAGS_shuffle_fea_id) {
      uint64 murmur_out[2];
      MurmurHash3_x64_128(&fea_id, 8, 512927377, murmur_out);
      fea_id = (murmur_out[0] ^ murmur_out[1]);
    }

    if (grp_id != pre_grp_id) {
      slot = ex->add_slot();
      slot->set_id(grp_id);
      pre_grp_id = grp_id;
    }
    slot->add_key(fea_id);
  }

  // add feature id 0 to special group 1023
  if (FLAGS_add_beta_feature) {
    const uint64 grp_id = 1023;
    const uint64 fea_id = 0;

    CHECK_LE(slot->id(), grp_id);
    if (grp_id != slot->id()) {
      slot = ex->add_slot();
      slot->set_id(grp_id);
    }
    slot->add_key(fea_id);
  }

  // LL << ex->ShortDebugString();
  return true;
}


// ps format: TODO
//
// label; group_id feature[:weight] feature[:weight] ...; groud_id ...; ...
//
// - label: the label of the extance. integer for classification, float for
//   regression, and emtpy for unsupervised learning
//
// - group_id: the integer identity of a feature group, each extance should
//   contaex at least one feature group.
//
// - feature: an 64-bit integer presenting the feature id for sparse training
//   data, an float feature value for dense training data.
//
// - weight: only valid for non-bianry sparse training data, a float number
//   presenting the feature value.
//
//
// vw: TODO
//
}
