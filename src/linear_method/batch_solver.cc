#include <limits>
#include <gperftools/malloc_extension.h>
#include <gperftools/profiler.h>
#include "linear_method/batch_solver.h"
#include "util/split.h"
#include "base/matrix_io_inl.h"
#include "base/localizer.h"
#include "base/sparse_matrix.h"

namespace PS {

DEFINE_int32(preprocess_memory_limit_each_group, 1024 * 1024 * 1024,
  "approximate memory usage while preprocessing in Bytes for each group; "
  "1GB in default");
DECLARE_bool(verbose);

namespace LM {

void BatchSolver::init() {
  LinearMethod::init();
  w_ = KVVectorPtr(new KVVector<Key, double>());
  w_->name() = app_cf_.parameter_name(0);
  sys_.yp().add(std::static_pointer_cast<Customer>(w_));
}

void BatchSolver::run() {
  // start the system
  LinearMethod::startSystem();

  // load data
  auto active_nodes = taskpool(kActiveGroup);
  auto load_time = tic();
  Task load = newTask(Call::LOAD_DATA);
  int hit_cache = 0;
  active_nodes->submitAndWait(load, [this, &hit_cache](){
      DataInfo info; CHECK(info.ParseFromString(exec_.lastRecvReply()));
      // LL << info.DebugString();
      g_train_info_ = mergeExampleInfo(g_train_info_, info.example_info());
      hit_cache += info.hit_cache();
    });
  if (hit_cache > 0) {
    CHECK_EQ(hit_cache, FLAGS_num_workers) << "clear the local caches";
    LI << "Hit local caches for the training data";
  }
  LI << "Loaded " << g_train_info_.num_ex() << " examples in "
     << toc(load_time) << " sec";

  // partition feature blocks
  CHECK(conf_.has_solver());
  auto sol_cf = conf_.solver();
  for (int i = 0; i < g_train_info_.slot_size(); ++i) {
    auto info = g_train_info_.slot(i);
    CHECK(info.has_id());
    if (info.id() == 0) continue;  // it's the label
    CHECK(info.has_nnz_ele());
    CHECK(info.has_nnz_ex());
    fea_grp_.push_back(info.id());
    double nnz_per_row = (double)info.nnz_ele() / (double)info.nnz_ex();
    int n = 1;  // number of blocks for a feature group
    if (nnz_per_row > 1 + 1e-6) {
      // only parititon feature group whose features are correlated
      n = std::max((int)std::ceil(nnz_per_row * sol_cf.feature_block_ratio()), 1);
    }
    for (int i = 0; i < n; ++i) {
      auto block = Range<Key>(info.min_key(), info.max_key()).evenDivide(n, i);
      if (block.empty()) continue;
      fea_blk_.push_back(std::make_pair(info.id(), block));
    }
  }

  // preprocess the training data
  auto preprocess_time = tic();
  Task preprocess = newTask(Call::PREPROCESS_DATA);
  for (auto grp : fea_grp_) set(&preprocess)->add_fea_grp(grp);
  set(&preprocess)->set_hit_cache(hit_cache > 0);
  // add all block partitions into preprocess task
  for (const auto& block_info : fea_blk_) {
    // add
    PartitionInfo* partition = preprocess.add_partition_info();
    // set
    partition->set_fea_grp(block_info.first);
    block_info.second.to(partition->mutable_key());
  }
  active_nodes->submitAndWait(preprocess);
  if (sol_cf.tail_feature_freq()) {
    LI << "Features with frequency <= " << sol_cf.tail_feature_freq() << " are filtered";
  }
  LI << "Preprocessing is finished in " << toc(preprocess_time) << " sec";
  LI << "Features are partitioned into " << fea_blk_.size() << " blocks";

  // a simple block order
  for (int i = 0; i < fea_blk_.size(); ++i) blk_order_.push_back(i);

  // blocks for important feature groups
  std::vector<string> hit_blk;
  for (int i = 0; i < sol_cf.prior_fea_group_size(); ++i) {
    int grp_id = sol_cf.prior_fea_group(i);
    std::vector<int> tmp;
    for (int k = 0; k < fea_blk_.size(); ++k) {
      if (fea_blk_[k].first == grp_id) tmp.push_back(k);
    }
    if (tmp.empty()) continue;
    hit_blk.push_back(std::to_string(grp_id));
    for (int j = 0; j < sol_cf.num_iter_for_prior_fea_group(); ++j) {
      if (sol_cf.random_feature_block_order()) {
        std::random_shuffle(tmp.begin(), tmp.end());
      }
      prior_blk_order_.insert(prior_blk_order_.end(), tmp.begin(), tmp.end());
    }
  }
  if (!hit_blk.empty()) LI << "Prior feature groups: " + join(hit_blk, ", ");


  total_timer_.restart();
  runIteration();

  if (conf_.has_validation_data()) {
    // LI << "\tEvaluate with " << g_validation_info_[0].row().end()
    //    << " validation examples\n";
    Task test = newTask(Call::COMPUTE_VALIDATION_AUC);
    AUC validation_auc;
    active_nodes->submitAndWait(test, [this, &validation_auc](){
        mergeAUC(&validation_auc); });
    LI << "\tEvaluation accuracy: " << validation_auc.accuracy(0)
       << ", auc: " << validation_auc.evaluate();
  }

  Task save_model = newTask(Call::SAVE_MODEL);
  active_nodes->submitAndWait(save_model);
}

void BatchSolver::runIteration() {
  auto sol_cf = conf_.solver();
  auto pool = taskpool(kActiveGroup);
  int time = pool->time();
  int tau = sol_cf.max_block_delay();
  for (int iter = 0; iter < sol_cf.max_pass_of_data(); ++iter) {
    if (sol_cf.random_feature_block_order())
      std::random_shuffle(blk_order_.begin(), blk_order_.end());

    for (int b : blk_order_)  {
      Task update = newTask(Call::UPDATE_MODEL);
      update.set_wait_time(time - tau);
      // set the feature key range will be updated in this block
      fea_blk_[b].second.to(set(&update)->mutable_key());
      time = pool->submit(update);
    }

    Task eval = newTask(Call::EVALUATE_PROGRESS);
    eval.set_wait_time(time - tau);
    time = pool->submitAndWait(
        eval, [this, iter](){ LinearMethod::mergeProgress(iter); });

    showProgress(iter);

    double rel = g_progress_[iter].relative_objv();
    if (rel > 0 && rel <= sol_cf.epsilon()) {
      LI << "\tStopped: relative objective <= " << sol_cf.epsilon();
      break;
    }
  }
}

bool BatchSolver::dataCache(const string& name, bool load) {
  if (!conf_.has_local_cache()) return false;
  return false;  // FIXME
#if 0
  // load / save label
  auto cache = conf_.local_cache();
  auto y_conf = ithFile(cache, 0, name + "_label_" + myNodeID());
  if (load) {
    MatrixPtrList<double> y_list;
    if (!readMatrices<double>(y_conf, &y_list) || !(y_list.size()==1)) return false;
    y_ = y_list[0];
  } else {
    if (!y_->writeToBinFile(y_conf.file(0))) return false;
  }
  // load / save feature groups
  string info_file = cache.file(0) + name + "_" + myNodeID() + ".info";
  InstanceInfo info = y_->info().ins_info();
  if (load && !readFileToProto(info_file, &info)) return false;
  InstanceInfo new_info;
  for (int i = 0; i < info.fea_grp_size(); ++i) {
    int id = info.fea_grp(i).grp_id();
    string x_name = name + "_fea_grp_" + std::to_string(id) + "_" + myNodeID();
    auto x_conf = ithFile(cache, 0, x_name);
    string key_name = name + "_key_" + std::to_string(id) + "_" + myNodeID();
    auto key_conf = ithFile(cache, 0, key_name);
    if (load) {
      MatrixPtrList<double> x_list;
      if (!readMatrices<double>(x_conf, &x_list) || !(x_list.size()==1)) return false;
      X_[id] = x_list[0];
      if (!w_->key(id).readFromFile(SizeR(0, X_[id]->cols()), key_conf)) return false;
    } else {
      if (!X_[id]) continue;
      if (w_->key(id).empty()) LL << id << " " << X_[id]->debugString();
      if (!(X_[id]->writeToBinFile(x_conf.file(0))
            && w_->key(id).writeToFile(key_conf.file(0)))) return false;
      *new_info.add_fea_grp() = info.fea_grp(i);
    }
  }
  if (!load && !writeProtoToASCIIFile(new_info, info_file)) return false;
  return true;
#endif
}

int BatchSolver::loadData(const MessageCPtr& msg, ExampleInfo* info) {
  const int starting_count_time = msg->task.time() + 1;
  const int finishing_count_time = starting_count_time + 10000000;

  if (IamWorker()) {
    CHECK(conf_.has_local_cache());
    slot_reader_.init(
      conf_.training_data(), conf_.local_cache(), conf_.solver(),
      w_, starting_count_time, finishing_count_time, pathPicker());
    slot_reader_.read(info);
  } else {
    w_->waitInMsg(kWorkerGroup, finishing_count_time);
  }

  return false; // hit_cache disabled
}

void BatchSolver::preprocessData(const MessageCPtr& msg) {
  int time = msg->task.time() * kFilterPace;
  bool hit_cache = get(msg).hit_cache();
  int grp_size = get(msg).fea_grp_size();

  if (IamWorker()) {
    std::vector<std::promise<void>> wait_dual(grp_size);

    // initialize preprocess status
    for (int i = 0; i < grp_size; ++i) {
      preprocess_status_.push_back(
        std::shared_ptr<PreprocessStatus>(new PreprocessStatus(
          w_, &slot_reader_,
          conf_, fea_grp_.at(i),
          time + i * kFilterPace)));
    }

    const int max_parallel = std::max(
      1, conf_.solver().max_num_parallel_groups_in_preprocessing());
    auto groupCouldGo =
      [this, &max_parallel](const int grp_order) -> bool {
      std::shared_ptr<PreprocessStatus> preprocess = preprocess_status_.at(grp_order);
      if (PreprocessStatus::Progress::GOING == preprocess->getProgress()) {
        return true;
      } else if (PreprocessStatus::Progress::FINISHED == preprocess->getProgress()) {
        return false;
      } else {
        if (grp_order >= max_parallel) {
          // concurrency control
          for (int i = 0; i <= grp_order - max_parallel; ++i) {
            if (PreprocessStatus::Progress::FINISHED !=
                preprocess_status_.at(i)->getProgress()) {
              return false;
            }
          }
        }
        preprocess->start();
        return true;
      }
    };
    auto allGroupFinished = [this]() -> bool {
      for (auto& preprocess : preprocess_status_) {
        if (PreprocessStatus::Progress::FINISHED != preprocess->getProgress()) {
          return false;
        }
      }
      return true;
    };

    while (true) {
      for (int i = 0; i < grp_size; ++i) {
        int grp = fea_grp_.at(i);

        std::shared_ptr<PreprocessStatus> preprocess = preprocess_status_.at(i);
        if (!groupCouldGo(i)) {
          continue;
        }

        preprocess->pullNextFilter();
        if (preprocess->allFilterFinished()) {
          // remapIndex
          Localizer<Key, double> *localizer = new Localizer<Key, double>(myNodeID(), grp);
          if (FLAGS_verbose) {
            LI << "started remapIndex [" << i + 1 << "/" << grp_size << "]";
          }
          this->sys_.hb().startTimer(HeartbeatInfo::TimerType::BUSY);
          auto X = localizer->remapIndex(grp, w_->key(grp), &slot_reader_, &pathPicker());
          delete localizer;
          slot_reader_.clear(grp);
          this->sys_.hb().stopTimer(HeartbeatInfo::TimerType::BUSY);
          if (FLAGS_verbose) {
            LI << "finished remapIndex [" << i + 1 << "/" << grp_size << "]";
          }

          // toColMajor
          if (FLAGS_verbose) {
            LI << "started toColMajor [" << i + 1 << "/" << grp_size << "]";
          }
          if (X) {
            this->sys_.hb().startTimer(HeartbeatInfo::TimerType::BUSY);
            if (conf_.solver().has_feature_block_ratio() && !X->empty()) {
              X = X->toColMajor();
            }
            this->sys_.hb().stopTimer(HeartbeatInfo::TimerType::BUSY);
          }
          if (FLAGS_verbose) {
            LI << "finished toColMajor [" << i + 1 << "/" << grp_size << "]";
          }

          // push initial keys to servers
          const int time_push_initial_key =
            time + i * kFilterPace + kFilterPace - 1;
          MessagePtr push_initial_key(
            new Message(kServerGroup, time_push_initial_key));
          push_initial_key->key = w_->key(grp);
          push_initial_key->task.set_key_channel(grp);
          push_initial_key->task.set_erase_key_cache(true);
          push_initial_key->fin_handle =
            [this, i, grp, X, &wait_dual]() {
            if (ocean_.getCPUProfilerStarted()) {
              // ProfilerFlush();
            }

            // set parameter value
            w_->value(grp).resize(w_->key(grp).size());
            w_->value(grp).setValue(0);

            // dump to Ocean
            CHECK(ocean_.dump(SArray<char>(w_->key(grp)), grp, Ocean::DataType::PARAMETER_KEY));
            CHECK(ocean_.dump(SArray<char>(w_->value(grp)), grp, Ocean::DataType::PARAMETER_VALUE));
            auto sparse_x = std::static_pointer_cast<SparseMatrix<uint32, double>>(X);
            CHECK(ocean_.dump(sparse_x, grp));

            // release original parameter memory
            w_->key(grp).clear();
            w_->value(grp).clear();

            wait_dual[i].set_value();
#ifdef TCMALLOC
            // tcmalloc force return memory to kernel
            MallocExtension::instance()->ReleaseFreeMemory();
#endif
          };
          CHECK_EQ(time_push_initial_key, w_->push(push_initial_key));

          // set group finished
          // but push_initial_key may not finish yet,
          //   wait_dual could help
          preprocess->finish();
        } // end of: if (preprocess->allFilterFinished())
      } // end of group loop

      if (allGroupFinished()) {
        break;
      } else {
        // wakensky
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // std::this_thread::yield();
      }
    } // end of the whole while;

    // wait until all push_initial_key finishes
    for (int i = 0; i < grp_size; ++i) {
      wait_dual[i].get_future().wait();
    }
    saveCache("train");
  } else {
    for (int i = 0; i < grp_size; ++i) {
      if (ocean_.getCPUProfilerStarted()) {
        // ProfilerFlush();
      }

      if (hit_cache) continue;

      // Task IDs
      int time_push_initial_key = time + i * kFilterPace + kFilterPace - 1;

      // init keys
      w_->waitInMsg(kWorkerGroup, time_push_initial_key);
      int chl = fea_grp_[i];
      w_->keyFilter(chl).clear();
      w_->value(chl).resize(w_->key(chl).size());
      w_->value(chl).setValue(0);

      // dump to Ocean
      CHECK(ocean_.dump(SArray<char>(w_->key(chl)), chl, Ocean::DataType::PARAMETER_KEY));
      CHECK(ocean_.dump(SArray<char>(w_->value(chl)), chl, Ocean::DataType::PARAMETER_VALUE));
      // reset
      w_->key(chl).clear();
      w_->value(chl).clear();

#ifdef TCMALLOC
      // tcmalloc force return memory to kernel
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
    }
  }
}

Progress BatchSolver::evaluateProgress() {
  Progress prog;
  // if (IamWorker()) {
  //   mu_.lock();
  //   busy_timer_.start();
  //   prog.set_objv(loss_->evaluate({y_, dual_.matrix()}));
  //   prog.add_busy_time(busy_timer_.get());
  //   busy_timer_.reset();
  //   mu_.unlock();
  // } else if (IamServer()) {
  //   if (penalty_) prog.set_objv(penalty_->evaluate(w_->value().matrix()));
  //   prog.set_nnz_w(w_->nnz());
  // }
  // // LL << myNodeID() << ": objv " << prog.objv();
  return prog;
}

void BatchSolver::saveModel(const MessageCPtr& msg) {
  if (!IamServer()) return;
  if (!conf_.has_model_output()) return;

  auto output = conf_.model_output();
  if (output.format() == DataConfig::TEXT) {
    CHECK(output.file_size());
    std::string file = output.file(0) + "_" + myNodeID();
    std::ofstream out(file); CHECK(out.good());

    if (!FLAGS_less_memory) {
      typedef std::pair<Ocean::JobID, SArray<char>> BlockedArray;
      std::vector<BlockedArray> parameter_key_vec =
        ocean_.getAllLoadedArray(Ocean::DataType::PARAMETER_KEY);
      CHECK(!parameter_key_vec.empty());
      std::vector<BlockedArray> parameter_val_vec =
        ocean_.getAllLoadedArray(Ocean::DataType::PARAMETER_VALUE);
      CHECK(!parameter_val_vec.empty());
      CHECK_EQ(parameter_key_vec.size(), parameter_val_vec.size());

      // sort with JobID
      std::sort(parameter_key_vec.begin(), parameter_key_vec.end(),
                [](const BlockedArray& a, const BlockedArray& b) -> bool {
                  return a.first < b.first; });
      std::sort(parameter_val_vec.begin(), parameter_val_vec.end(),
                [](const BlockedArray& a, const BlockedArray& b) -> bool {
                  return a.first < b.first; });

      SArray<Key> key_array;
      SArray<double> val_array;
      for (size_t i = 0; i < parameter_key_vec.size(); ++i) {
        CHECK(parameter_key_vec.at(i).first == parameter_val_vec.at(i).first);
        CHECK_EQ(
          parameter_key_vec.at(i).second.size(),
          parameter_val_vec.at(i).second.size());

        key_array = parameter_key_vec.at(i).second;
        val_array = parameter_val_vec.at(i).second;
        for (size_t j = 0; j < key_array.size(); ++j) {
          auto key = key_array[j];
          auto val = val_array[j];
          if (val != 0 && !(val != val)) {
            out << key << "\t" << val << "\n";
          }
        }
      }
    } else {
      typedef std::pair<Ocean::JobID, string> BlockedPath;
      std::vector<BlockedPath> parameter_key_vec =
        ocean_.getAllDumpedPath(Ocean::DataType::PARAMETER_KEY);
      CHECK(!parameter_key_vec.empty());
      std::vector<BlockedPath> parameter_val_vec =
        ocean_.getAllDumpedPath(Ocean::DataType::PARAMETER_VALUE);
      CHECK(!parameter_val_vec.empty());
      CHECK_EQ(parameter_key_vec.size(), parameter_val_vec.size());

      // sort with JobID
      std::sort(parameter_key_vec.begin(), parameter_key_vec.end(),
                [](const BlockedPath& a, const BlockedPath& b) -> bool {
                  return a.first < b.first; });
      std::sort(parameter_val_vec.begin(), parameter_val_vec.end(),
                [](const BlockedPath& a, const BlockedPath& b) -> bool {
                  return a.first < b.first; });

      for (size_t i = 0; i < parameter_key_vec.size(); ++i) {
        CHECK(parameter_key_vec.at(i).first == parameter_val_vec.at(i).first);

        SArray<char> key_stash;
        CHECK(key_stash.readFromFile(parameter_key_vec.at(i).second));
        SArray<Key> key_array(key_stash);
        SArray<char> val_stash;
        CHECK(val_stash.readFromFile(parameter_val_vec.at(i).second));
        SArray<double> val_array(val_stash);
        CHECK_EQ(key_array.size(), key_array.size());

        for (size_t j = 0; j < key_array.size(); ++j) {
          auto key = key_array[j];
          auto val = val_array[j];
          if (val != 0 && !(val != val)) {
            out << key << "\t" << val << "\n";
          }
        }
      }
    }
    LI << myNodeID() << " writes model to " << file;
  } else {
    LL << "didn't implement yet";
  }
}

void BatchSolver::showProgress(int iter) {
  int s = iter == 0 ? -3 : iter;
  for (int i = s; i <= iter; ++i) {
    showObjective(i);
    showNNZ(i);
    showTime(i);
  }
}

bool BatchSolver::binary(const int grp_id) const {
  return MatrixInfo::SPARSE_BINARY == ocean_.getMatrixInfo(grp_id).type();
}

void BatchSolver::computeEvaluationAUC(AUCData *data) {
  if (!IamWorker()) return;

  // TODO
  // load data
  // CHECK(XXXX.has_validation_data());
  // if (!loadCache("valid")) {
  //   auto list = readMatricesOrDie<double>(XXXX.training_data());
  //   CHECK_EQ(list.size(), 2);
  //   y_ = list[0];
  //   auto X = std::static_pointer_cast<SparseMatrix<Key, double>>(list[1]);
  //   X->countUniqIndex(&w_->key());
  //   X_ = X->remapIndex(w_->key());
  //   saveCache("valid");
  // }

  // // fetch the model
  // MessagePtr pull_msg(new Message(kServerGroup, Message::kInvalidTime));
  // pull_msg->key = w_->key();
  // pull_msg->wait = true;
  // int time = w_->pull(pull_msg);
  // w_->value() = w_->received(time)[0].second;
  // CHECK_EQ(w_->key().size(), w_->value().size());

  // // w_->fetchValueFromServers();

  // // compute auc
  // AUC auc; auc.setGoodness(XXXX.block_solver().auc_goodness());
  // SArray<double> Xw(X_->rows());
  // for (auto& v : w_->value()) if (v != v) v = 0;
  // Xw.eigenVector() = *X_ * w_->value().eigenVector();
  // auc.compute(y_->value(), Xw, data);

  // debug
  // w.writeToFile("w");
  // double correct = 0;
  // for (int i = 0; i < Xw.size(); ++i)
  //   if (y_->value()[i] * Xw[i] >= 0) correct += 1;
  // LL << correct / Xw.size();

  // Xw.writeToFile("Xw_"+myNodeID());
  // y_->value().writeToFile("y_"+myNodeID());
  // LL << auc.evaluate();
}

// void BatchSolver::saveAsDenseData(const Message& msg) {
//   if (!exec_.isWorker()) return;
//   auto call = RiskMinimization::getCall(msg);
//   int n = call.reduce_range_size();
//   if (n == 0) return;
//   if (X_->rowMajor()) {
//     X_ = X_->toColMajor();
//   }
//   DenseMatrix<double> Xw(X_->rows(), n, false);
//   for (int i = 0; i < n; ++i) {
//     auto lr = w_->localRange(Range<Key>(call.reduce_range(i)));
//     Xw.colBlock(SizeR(i, i+1))->eigenArray() =
//         *(X_->colBlock(lr)) * w_->segment(lr).eigenVector();
//   }

//   Xw.writeToBinFile(call.name()+"_Xw");
//   y_->writeToBinFile(call.name()+"_y");
// }


void BatchSolver::updateModel(const MessagePtr& msg) {
  // FIXME several tiny bugs here...
  // int time = msg->task.time() * 10;
  // Range<Key> global_range(msg->task.risk().key());
  // auto local_range = w_->localRange(global_range);

  // if (exec_.isWorker()) {
  //   auto X = X_->colBlock(local_range);

  //   SArrayList<double> local_grads(2);
  //   local_grads[0].resize(local_range.size());
  //   local_grads[1].resize(local_range.size());
  //   AggGradLearnerArg arg;
  //   {
  //     Lock l(mu_);
  //     busy_timer_.start();
  //     learner_->compute({y_, X, dual_.matrix()}, arg, local_grads);
  //     busy_timer_.stop();
  //   }

  //   msg->finished = false;
  //   auto sender = msg->sender;
  //   auto task = msg->task;
  //   w_->roundTripForWorker(time, global_range, local_grads,
  //                          [this, X, local_range, sender, task] (int time) {
  //       Lock l(mu_);
  //       busy_timer_.start();

  //       if (!local_range.empty()) {
  //         auto data = w_->received(time);

  //         CHECK_EQ(data.size(), 1);
  //         CHECK_EQ(local_range, data[0].first);
  //         auto new_w = data[0].second;

  //         auto delta = new_w.eigenVector() - w_->segment(local_range).eigenVector();
  //         dual_.eigenVector() += *X * delta;
  //         w_->segment(local_range).eigenVector() = new_w.eigenVector();
  //       }

  //       busy_timer_.stop();

  //       taskpool(sender)->finishIncomingTask(task.time());
  //       sys_.reply(sender, task);
  //       // LL << myNodeID() << " done " << d.task.time();
  //     });
  // } else {
  //   // aggregate local gradients, then update model
  //   w_->roundTripForServer(time, global_range, [this, local_range] (int time) {
  //       SArrayList<double> aggregated_gradient;
  //       for (auto& d : w_->received(time)) {
  //         CHECK_EQ(local_range, d.first);
  //         aggregated_gradient.push_back(d.second);
  //       }
  //       AggGradLearnerArg arg;
  //       arg.set_learning_rate(XXXX.learning_rate().eta());
  //       learner_->update(aggregated_gradient, arg, w_->segment(local_range));
  //     });
  // }
}

} // namespace LM
} // namespace PS
