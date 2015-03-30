#include "linear_method/batch_solver.h"
#include <sys/stat.h>
#include <gperftools/malloc_extension.h>
#include "util/split.h"
#include "base/localizer.h"
#include "base/sparse_matrix.h"
#include "data/common.h"

namespace PS {

DEFINE_int32(preprocess_mem_limit, 1024 * 1024 * 1024,
  "approximate memory usage while preprocessing in Bytes for each group; "
  "1GB in default");
DECLARE_bool(verbose);
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

    int num_iter_for_prior = sol_cf.num_iter_for_prior_fea_group();
    if (1023 == grp_id && sol_cf.has_beta_feature_prior_num_iter()) {
      num_iter_for_prior = sol_cf.beta_feature_prior_num_iter();
    }

    for (int j = 0; j < num_iter_for_prior; ++j) {
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

#if 0
  Task save_model = newTask(Call::SAVE_MODEL);
  active_nodes->submitAndWait(save_model);
#endif
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

int BatchSolver::loadData(const MessageCPtr& msg, ExampleInfo* info) {
  const int starting_time = msg->task.time() + 1;
  const int finishing_time = starting_time + 1000000;

  validation_.init(myNodeID() + "-validation", conf_, &path_picker_);
  if (IamWorker()) {
    CHECK(conf_.has_local_cache());

    // download validation data
    ThreadPool load_validation_pool(1);
    load_validation_pool.add([this]() {
                                CHECK(validation_.download());
                             });
    load_validation_pool.startWorkers();

    // download training data
    slot_reader_.init(
      conf_.training_data(), conf_.local_cache(), &pathPicker(),
      starting_time, finishing_time,
      conf_.solver().countmin_k(),
      conf_.solver().countmin_n_ratio(),
      myNodeID());
    slot_reader_.read(w_, info);
  } else {
    w_->waitInMsg(kWorkerGroup, finishing_time);
  }
  return false;
}

void BatchSolver::preprocessData(const MessageCPtr& msg) {
  int pull_time = msg->task.time() * 1000000;
  int push_initial_key_time = pull_time + 1000000;
  const int grp_size = fea_grp_.size();

  if (IamWorker()) {
    std::vector<std::promise<void>> wait_dual(grp_size);
    std::vector<std::shared_ptr<PreprocessHelper>> preprocess_helpers;
    std::vector<int> pushed_initial_time(grp_size);

    // initialize preprocess helper
    for (int i = 0; i < grp_size; ++i) {
      preprocess_helpers.push_back(std::shared_ptr<PreprocessHelper>(
        new PreprocessHelper(
          w_, &slot_reader_,
          conf_.solver().tail_feature_freq(), fea_grp_[i])));
    }

    // how many groups could run concurrently
    const int max_parallel = std::max(
      1, conf_.solver().max_num_parallel_groups_in_preprocessing());

    // whether the associated group could run
    auto groupCouldPull =
      [this, max_parallel](
      std::vector<std::shared_ptr<PreprocessHelper>>& helpers,
      const int grp_order) -> bool {
      if (PreprocessHelper::Status::GOING == helpers.at(grp_order)->getStatus()) {
        return true;
      } else if (PreprocessHelper::Status::FINISHED == helpers.at(grp_order)->getStatus()) {
        return false;
      } else {
        if (grp_order >= max_parallel) {
          // concurrency control
          for (int i = 0; i <= grp_order - max_parallel; ++i) {
            if (PreprocessHelper::Status::FINISHED !=
                helpers.at(i)->getStatus()) {
              // Not all former groups finished
              return false;
            }
          }
        }
        helpers.at(grp_order)->setStatus(PreprocessHelper::Status::GOING);
        return true;
      }
    };

    auto allPullFinished = [](std::vector<std::shared_ptr<PreprocessHelper>>& preprocess_helpers) -> bool {
      for (auto helper : preprocess_helpers) {
        if (PreprocessHelper::Status::FINISHED != helper->getStatus()) {
          return false;
        }
      }
      return true;
    };

    auto compute =
      [this, grp_size](
      const int grp_order, SArray<Key> keys) {
      const int grp_id = fea_grp_.at(grp_order);

      // global -> local
      MilliTimer compute_milli_timer; compute_milli_timer.start();
      Localizer<Key, double> *localizer = new Localizer<Key, double>();
      if (true) {
        LI << "started remapIndex [" << grp_order + 1 << "/" << grp_size << "]; grp: " << grp_id;
      }
      auto X = localizer->remapIndex(
        grp_id, keys, &slot_reader_, &path_picker_, myNodeID());
      delete localizer;
      slot_reader_.clear(grp_id);
      if (true) {
        LI << "finished remapIndex [" << grp_order + 1 << "/" << grp_size << "]; grp: " << grp_id;
      }

      // matrix transforms to column major
      if (true) {
        LI << "started toColMajor [" << grp_order + 1 << "/" << grp_size << "]; grp: " << grp_id;
      }
      if (X) {
        // TODO toColMajor is necessary since I have assumed that
        //   training data could be partition column-wise
        if (conf_.solver().has_feature_block_ratio()) {
          X = X->toColMajor();
        }
      }
      if (true) {
        LI << "finished toColMajor [" << grp_order + 1 << "/" << grp_size << "]; grp: " << grp_id;
      }
      compute_milli_timer.stop();
      this->sys_.hb_collector().increaseTime(compute_milli_timer.get());

      // record MatrixInfo
      // matrix_info_[grp_id] = X->info();

      // reset parameter_value
      SArray<double> values;
      values.resize(keys.size());
      values.setValue(0);

      // reset delta
      delta_[grp_id].resize(keys.size());
      delta_[grp_id].setValue(conf_.darling().delta_init_value());

      // dump to Ocean
      CHECK(ocean_.dump(grp_id, keys, values,
        delta_[grp_id], SArray<double>(),
        std::static_pointer_cast<SparseMatrix<uint32, double>>(X)));

      // release memory resource
      delta_[grp_id].clear();
    };

    // pull loop
    while (true) {
      for (int grp_order = 0; grp_order < grp_size; ++grp_order) {
        auto helper = preprocess_helpers.at(grp_order);
        if (!groupCouldPull(preprocess_helpers, grp_order)) {
          continue;
        }

        pull_time = helper->pullNextFilter(pull_time);
        if (helper->allFiltersFinished()) {
          // stash w_->key(grp_id) on disk
          std::stringstream ss;
          ss << myNodeID() << ".key_stash." << fea_grp_.at(grp_order);
          const string key_path = path_picker_.getPath(ss.str());
          CHECK(w_->key(fea_grp_.at(grp_order)).writeToFile(key_path));

          // release w_->key
          w_->key(fea_grp_.at(grp_order)).clear();

          // finish tag
          helper->setStatus(PreprocessHelper::Status::FINISHED);

#ifdef TCMALLOC
          MallocExtension::instance()->ReleaseFreeMemory();
#endif
        }
      }

      if (allPullFinished(preprocess_helpers)) {
        break;
      } else {
        // not all groups finished, sleep for a while
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    } // pull loop for worker

    // computation loop
    for (int grp_order = 0; grp_order < grp_size; ++grp_order) {
      // load keys from disk
      std::stringstream ss;
      ss << myNodeID() << ".key_stash." << fea_grp_.at(grp_order);
      const string key_path = path_picker_.getPath(ss.str());
      SArray<char> stash;
      CHECK(stash.readFromFile(key_path));
      SArray<Key> keys(stash);

      // compute and dump
      compute(grp_order, keys);

#ifdef TCMALLOC
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
    }

    // push initial keys
    for (int grp_order = 0; grp_order < grp_size; ++grp_order) {
      if (grp_order >= max_parallel) {
        w_->waitOutMsg(kServerGroup, pushed_initial_time.at(grp_order - max_parallel));
      }

      // load keys from disk
      std::stringstream ss;
      ss << myNodeID() << ".key_stash." << fea_grp_.at(grp_order);
      const string key_path = path_picker_.getPath(ss.str());
      SArray<char> stash;
      CHECK(stash.readFromFile(key_path));
      SArray<Key> keys(stash);

      MessagePtr push_initial_key(
        new Message(kServerGroup, push_initial_key_time));
      push_initial_key->setKey(keys);
      push_initial_key->task.set_key_channel(fea_grp_.at(grp_order));
      push_initial_key->fin_handle = [this, &wait_dual, grp_order] {
        wait_dual.at(grp_order).set_value();
      };
      CHECK_EQ(push_initial_key_time, w_->push(push_initial_key));
      pushed_initial_time.at(grp_order) = push_initial_key_time++;

      LI << "Has pushed initial keys for group [" << fea_grp_.at(grp_order) <<
        "] [" << grp_order + 1 << "/" << grp_size << "]";

#ifdef TCMALLOC
      MallocExtension::instance()->ReleaseFreeMemory();
#endif
    }

    // wait until all initial keys push finished
    for (int grp_order = 0; grp_order < grp_size; ++grp_order) {
      wait_dual.at(grp_order).get_future().wait();
    }
  } else {
    for (int grp_order = 0; grp_order < grp_size; ++grp_order) {
      const int grp_id = fea_grp_.at(grp_order);

      // wait initial keys from workers
      w_->waitInMsg(kWorkerGroup, push_initial_key_time++);

      // reset parameter_value
      w_->value(grp_id).resize(w_->key(grp_id).size());
      w_->value(grp_id).setValue(0);

      // reset second-order gradient
      SArray<double> second_order_gradient(w_->key(grp_id).size(), 0);

      // reset delta
      delta_[grp_id].resize(w_->key(grp_id).size());
      delta_[grp_id].setValue(conf_.darling().delta_init_value());

      // dump to Ocean
      ocean_.dump(grp_id, w_->key(grp_id), w_->value(grp_id),
        delta_[grp_id], second_order_gradient,
        SparseMatrixPtr<uint32, double>());

      // release memory resource
      w_->clear(grp_id);
      delta_[grp_id].clear();
      w_->keyFilter(grp_id).clear();
#ifdef TCMALLOC
      MallocExtension::instance()->ReleaseFreeMemory();
#endif

      LI << "got init keys on group " << grp_id <<
        "[" << grp_order + 1 << "/" << grp_size << "]";
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
    CHECK(get(msg).has_iteration());

    // make sure the corresponding directory exists
    std::stringstream ss;
    ss << output.file(0) << "/iter_" << get(msg).iteration();
    system((std::string("mkdir -p ") + ss.str()).c_str());

    ss << "/ctr_" << myNodeID();
    CHECK(ocean_.saveModel(ss.str()));
    LI << myNodeID() << " writes model to " << ss.str();
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
