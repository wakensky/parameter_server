#include <gperftools/malloc_extension.h>
#include "linear_method/darling.h"
#include "base/matrix_io.h"
#include "base/sparse_matrix.h"

namespace PS {

DECLARE_bool(verbose);

namespace LM {

void Darling::runIteration() {
  CHECK(conf_.has_darling());
  CHECK_EQ(conf_.loss().type(), LossConfig::LOGIT);
  CHECK_EQ(conf_.penalty().type(), PenaltyConfig::L1);
  auto sol_cf = conf_.solver();
  int tau = sol_cf.max_block_delay();
  kkt_filter_threshold_ = 1e20;
  bool reset_kkt_filter = false;
  bool random_blk_order = sol_cf.random_feature_block_order();
  if (!random_blk_order) {
    LI << "Warning: Randomized block order often acclerates the convergence.";
  }
  LI << "Train l_1 logistic regression by " << tau << "-delayed block coordinate descent";

  // iterating
  int max_iter = sol_cf.max_pass_of_data();
  auto pool = taskpool(kActiveGroup);
  int time = pool->time() + fea_grp_.size() * 2;
  const int first_update_model_task_id = time + 1;
  // UpdateModel tasks that has been sent in the current iteration
  std::queue<int> update_model_tasks;
  for (int iter = 0; iter < max_iter; ++iter) {
    // pick up a updating order
    auto order = blk_order_;
    if (random_blk_order) std::random_shuffle(order.begin(), order.end());
    if (iter == 0) order.insert(
            order.begin(), prior_blk_order_.begin(), prior_blk_order_.end());

    // iterating on feature blocks
    for (int i = 0; i < order.size(); ++i) {
      Task update = newTask(Call::UPDATE_MODEL);
      update.set_time(time+1);
      if (iter == 0 && i < prior_blk_order_.size()) {
        // force zero delay for important feature blocks
        update.set_wait_time(time);
        update.set_is_priority(true);
      } else {
        update.set_wait_time(time - tau);
      }

      // make sure leading UPDATE_MODEL tasks could be picked up by workers
      if (update.wait_time() < first_update_model_task_id) {
        update.set_wait_time(-1);
      }

      auto cmd = set(&update);
      if (i == 0) {
        cmd->set_kkt_filter_threshold(kkt_filter_threshold_);
        if (reset_kkt_filter) cmd->set_kkt_filter_reset(true);
      }
      auto blk = fea_blk_[order[i]];
      blk.second.to(cmd->mutable_key());
      cmd->add_fea_grp(blk.first);
      time = pool->submit(update);

      update_model_tasks.push(time);
    }

    // wait all UpdateModel tasks finished
    while (!update_model_tasks.empty()) {
      pool->waitOutgoingTask(update_model_tasks.front());
      update_model_tasks.pop();
    }

    // evaluate the progress
    Task eval = newTask(Call::EVALUATE_PROGRESS);
    eval.set_wait_time(time);
    time = pool->submitAndWait(
        eval, [this, iter](){ LinearMethod::mergeProgress(iter); });
    showProgress(iter);

    // update the kkt filter strategy
    double vio = g_progress_[iter].violation();
    double ratio = conf_.darling().kkt_filter_threshold_ratio();
    kkt_filter_threshold_ = vio / (double)g_train_info_.num_ex() * ratio;

    // save model each iteration
    Task save_model = newTask(Call::SAVE_MODEL);
    save_model.set_wait_time(time);
    save_model.mutable_linear_method()->set_iteration(iter);
    time = pool->submitAndWait(save_model);
    LL << "H has dumped model for iteration " << iter;

    // check if finished
    double rel = g_progress_[iter].relative_objv();
    if (rel > 0 && rel <= sol_cf.epsilon()) {
      if (reset_kkt_filter) {
        LL << "Expected Stopped: relative objective <= " << sol_cf.epsilon();
        break;
      } else {
        reset_kkt_filter = true;
      }
    } else {
      reset_kkt_filter = false;
    }
    if (iter == max_iter - 1) {
      LL << "Expected Stopped: Reached maximal " << max_iter << " data passes";
    }
  }
}

void Darling::preprocessData(const MessageCPtr& msg) {
  const int grp_size = get(msg).fea_grp_size();
  fea_grp_.clear();
  for (int i = 0; i < grp_size; ++i) fea_grp_.push_back(get(msg).fea_grp(i));

  std::shared_ptr<std::thread> validation_thread_ptr;
  ocean_.init(myNodeID(), conf_, msg->task, &path_picker_);
  if (IamWorker()) {
    // validation preprocess
    validation_thread_ptr.reset(new std::thread([this, msg]() {
      CHECK(validation_.preprocess(msg->task));
    }));

    // load labels
    y_ = MatrixPtr<double>(new DenseMatrix<double>(
      slot_reader_.info<double>(0), slot_reader_.value<double>(0)));

    // dual_ = exp(y.*(X_*w_))
    dual_.resize(y_->rows());
    dual_.setZero();
    dual_.eigenArray() = exp(y_->value().eigenArray() * dual_.eigenArray());
  }

  if (!ocean_.resume()) {
    BatchSolver::preprocessData(msg);
    ocean_.snapshot();
  }

  // reset active_set_
  for (int grp_id : fea_grp_) {
    active_set_[grp_id].resize(ocean_.getGroupKeyCount(grp_id), true);
  }

  if (validation_thread_ptr) {
    validation_thread_ptr->join();
  }
}

void Darling::updateModel(const MessagePtr& msg) {
  if (FLAGS_verbose) {
    LI << "updateModel; msg [" << msg->shortDebugString() << "]";
  }

  CHECK_GT(FLAGS_num_threads, 0);
  // auto time = msg->task.time() * kPace;
  auto time = std::numeric_limits<int>::max() / 2 + msg->task.time() * kPace;
  auto call = get(msg);
  if (call.has_kkt_filter_threshold()) {
    kkt_filter_threshold_ = call.kkt_filter_threshold();
    violation_ = 0;
  }
  if (call.has_kkt_filter_reset() && call.kkt_filter_reset()) {
    for (int grp : fea_grp_) active_set_[grp].fill(true);
  }
  CHECK_EQ(call.fea_grp_size(), 1);
  int grp = call.fea_grp(0);
  Range<Key> g_key_range(call.key());
  auto anchor = ocean_.fetchAnchor(grp, g_key_range);

  if (IamWorker()) {
    // compute local gradients
    mu_.lock();

    MilliTimer grad_milli_timer; grad_milli_timer.start();
    busy_timer_.start();
    auto local_gradients = computeGradients(grp, g_key_range, msg->task.time());
    busy_timer_.stop();
    grad_milli_timer.stop();
    this->sys_.hb_collector().increaseTime(grad_milli_timer.get());

    mu_.unlock();

    // load data
    Ocean::DataPack data_pack = ocean_.get(grp, g_key_range, msg->task.time());
    SArray<Key> parameter_key(
      data_pack.arrays[static_cast<size_t>(Ocean::DataSource::PARAMETER_KEY)]);
    CHECK_EQ(anchor.size(), parameter_key.size());

    // time 0: push local gradients
    MessagePtr push_msg(new Message(kServerGroup, time));
    push_msg->setKey(parameter_key);
    push_msg->addValue(local_gradients);
    g_key_range.to(push_msg->task.mutable_key_range());
    push_msg->task.set_key_channel(grp);
    push_msg->task.set_owner_time(msg->task.time());
    // push_msg->addFilter(FilterConfig::KEY_CACHING);
    CHECK_EQ(time, w_->push(push_msg));

    // wakensky
    if (1023 == grp) {
      double sum_dual = 0.0;
      for (size_t i = 0; i < dual_.size(); ++i) {
        sum_dual += dual_[i];
      }
      LL << myNodeID() << " [after computeGradients] sum_dual: " << sum_dual;
    }

    // time 1: servers do update, none of my business
    // time 2: pull the updated model for validation
    // Whether I need send a pull request for validation

    // wakensky
    LI << "validation summary: " << msg->task.is_priority() <<
      " " << validation_.isEnabled() <<
      ": " << validation_.identity_;;

    msg->finished = false; // not finished until model updates are pulled
    bool validation_pull_sent = false;
    if (!msg->task.is_priority() && validation_.isEnabled()) {
      validation_pull_sent = true;

      LI << "ready to send validation pull: " <<
        grp << " " << grp << " " << g_key_range.toString() <<
        " " << msg->task.time() <<
        " " << validation_.isEnabled();

      MessagePtr validation_pull_msg(
        new Message(kServerGroup, time+2, time+1));
      validation_pull_msg->task.mutable_shared_para()->set_is_validation(true);
      g_key_range.to(validation_pull_msg->task.mutable_key_range());
      validation_pull_msg->setKey(
        validation_.getKey(grp, g_key_range, msg->task.time()));
      validation_pull_msg->task.set_key_channel(grp);
      validation_pull_msg->task.set_owner_time(msg->task.time());
      validation_pull_msg->fin_handle = [this, grp, time, msg,
                                         g_key_range]() {
        if (!validation_.fetchAnchor(grp, g_key_range).empty()) {
          validation_.submit(
            grp, g_key_range, msg->task.time(),
            w_->received(time+2).second[0]);
        } else {
          validation_.submit(
            grp, g_key_range, msg->task.time(), SArray<double>());
        }

        // Check whether training-pull finished.
        // If finished already, I will make servers release
        //   column-partitioned parameter value
        if (w_->tryWaitOutMsg(kServerGroup, time+3)) {
          MessagePtr task_over_msg(
            new Message(kServerGroup, time+4));
          task_over_msg->task.mutable_shared_para()->set_task_over(true);
          g_key_range.to(task_over_msg->task.mutable_key_range());
          task_over_msg->task.set_key_channel(grp);
          task_over_msg->task.set_owner_time(msg->task.time());
          CHECK_EQ(time+4, w_->push(task_over_msg));

          // now finished, reply the scheduler
          taskpool(msg->sender)->finishIncomingTask(msg->task.time());
          sys_.reply(msg->sender, msg->task);
        }
      };
      CHECK_EQ(time+2, w_->pull(validation_pull_msg));
    }

    // time 3: pull the updated model from Servers
    MessagePtr pull_msg(new Message(kServerGroup, time+3, time+1));
    pull_msg->setKey(parameter_key);
    g_key_range.to(pull_msg->task.mutable_key_range());
    pull_msg->task.set_key_channel(grp);
    pull_msg->task.set_owner_time(msg->task.time());
    // pull_msg->addFilter(FilterConfig::KEY_CACHING);
    // the callback for updating the local dual variable
    pull_msg->fin_handle =
      [this, grp, anchor, time, msg, g_key_range, validation_pull_sent] () {
      if (!anchor.empty()) {
        auto data = w_->received(time+3);
        CHECK_EQ(anchor, data.first); CHECK_EQ(data.second.size(), 1);
        mu_.lock();

        // wakensky
        if (1023 == grp) {
            double sum_dual = 0.0;
            for (size_t i = 0; i < dual_.size(); ++i) {
                sum_dual += dual_[i];
            }
            LL << myNodeID() << " [before updateDual] sum_dual: " << sum_dual <<
              "; new_wei: " << SArray<double>(data.second[0])[0];
        }

        MilliTimer dual_milli_timer; dual_milli_timer.start();
        busy_timer_.start();
        updateDual(grp, g_key_range, data.second[0], msg->task.time());
        busy_timer_.stop();
        dual_milli_timer.stop();
        this->sys_.hb_collector().increaseTime(dual_milli_timer.get());

        // wakensky
        if (1023 == grp) {
            double sum_dual = 0.0;
            for (size_t i = 0; i < dual_.size(); ++i) {
                sum_dual += dual_[i];
            }
            LL << myNodeID() << " [after updateDual] sum_dual: " << sum_dual <<
              "; new_wei: " << SArray<double>(data.second[0])[0];
        }

        mu_.unlock();
      }

      // Check whether validation-pull finished.
      // If finished already, I will make servers release
      //   column-partitioned parameter value
      if (!validation_pull_sent || w_->tryWaitOutMsg(kServerGroup, time+2)) {
        MessagePtr task_over_msg(
          new Message(kServerGroup, time+4));
        task_over_msg->task.mutable_shared_para()->set_task_over(true);
        g_key_range.to(task_over_msg->task.mutable_key_range());
        task_over_msg->task.set_key_channel(grp);
        task_over_msg->task.set_owner_time(msg->task.time());
        CHECK_EQ(time+4, w_->push(task_over_msg));

        // now finished, reply the scheduler
        taskpool(msg->sender)->finishIncomingTask(msg->task.time());
        sys_.reply(msg->sender, msg->task);
      }

      // ocean drop
      ocean_.drop(grp, g_key_range, msg->task.time());
    };
    CHECK_EQ(time+3, w_->pull(pull_msg));
  } else if (IamServer()) {
    // none of my bussiness
    if (w_->myKeyRange().setIntersection(g_key_range).empty()) {
      ocean_.drop(grp, g_key_range, msg->task.time());
      return;
    }

    // time 0: aggregate all workers' local gradients
    w_->waitInMsg(kWorkerGroup, time);

    // time 1: update model
    if (!anchor.empty()) {
      auto data = w_->received(time);
      CHECK_EQ(anchor, data.first);
      CHECK_EQ(data.second.size(), 2);

      MilliTimer weight_milli_timer; weight_milli_timer.start();
      updateWeight(
        grp, g_key_range, data.second[0], data.second[1],
        msg->task.time(), msg->task.is_priority());
      weight_milli_timer.stop();
      this->sys_.hb_collector().increaseTime(weight_milli_timer.get());
    }
    w_->finish(kWorkerGroup, time+1);

    // time 2: let the workers pull from me
  }

#ifdef TCMALLOC
  MallocExtension::instance()->ReleaseFreeMemory();
#endif
}

SArrayList<double> Darling::computeGradients(int grp, SizeR global_range, int task_id) {
  SArrayList<double> grads(2);
  // load data
  Ocean::DataPack data_pack = ocean_.get(grp, global_range, task_id);
  // on-demand usage
  SArray<uint32> feature_key(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::FEATURE_KEY)]);
  SArray<size_t> feature_offset(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::FEATURE_OFFSET)]);
  SArray<double> feature_value(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::FEATURE_VALUE)]);
  SArray<double> delta(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::DELTA)]);
  SizeR anchor = ocean_.fetchAnchor(grp, global_range);
  if (feature_key.empty()) {
    return grads;
  }

  // check
  if (!feature_value.empty()) {
    CHECK_EQ(feature_key.size(), feature_value.size());
  }
  CHECK_EQ(feature_offset.size(), anchor.size() + 1);
  CHECK_EQ(feature_offset.size(), delta.size() + 1);
  CHECK_EQ(
    feature_offset.back() - feature_offset.front(),
    feature_key.size());

  // allocate grads
  for (int i : {0, 1} ) {
    grads[i].resize(anchor.size());
    grads[i].setZero();
  }

  // TODO partition by rows for small col_range size
  int num_threads = anchor.size() < 64 ? 1 : FLAGS_num_threads;
  ThreadPool pool(num_threads);
  int npart = num_threads * 1;  // could use a larger partition number
  for (int i = 0; i < npart; ++i) {
    auto thr_anchor = anchor.evenDivide(npart, i);
    if (thr_anchor.empty()) continue;
    auto gr = thr_anchor - anchor.begin();
    pool.add([this, grp, thr_anchor, anchor, gr, &grads,
              &feature_key, &feature_offset, &feature_value, &delta]() {
        computeGradients(grp, thr_anchor, anchor.begin(),
          grads[0].segment(gr), grads[1].segment(gr),
          feature_key, feature_offset, feature_value, delta);
      });
  }
  pool.startWorkers();
  return grads;
}

void Darling::computeGradients(
  int grp, SizeR thr_anchor, const size_t group_anchor_begin,
  SArray<double> G, SArray<double> U,
  SArray<uint32> feature_key,
  SArray<size_t> feature_offset,
  SArray<double> feature_value,
  SArray<double> delta) {
  CHECK_EQ(G.size(), thr_anchor.size());
  CHECK_EQ(U.size(), thr_anchor.size());

  const auto& active_set = active_set_[grp]; // KKT utility
  const double* y = y_->value().data();  // labels

  // offset's head and delta's head are aligned,
  //   although offset is longer by 1
  const size_t* offset = feature_offset.data() + (thr_anchor.begin() - group_anchor_begin);
  double* delta_ptr = delta.data() + (thr_anchor.begin() - group_anchor_begin);
  // Positions reside in offset are group-wise.
  // Since a group has been seperated into several segments,
  //   we should have all positions minus their head,
  //   which transforms group-wise positions to segment-wise positions.
  uint32* index = feature_key.data() + (offset[0] - feature_offset.front());
  double* value = feature_value.data() + (offset[0] - feature_offset.front());

  bool is_binary = ocean_.matrix_binary(grp);
  // j: column id, i: row id
  for (size_t j = 0; j < thr_anchor.size(); ++j) {
    size_t k = j + thr_anchor.begin();
    size_t n = offset[j+1] - offset[j];
    if (!active_set.test(k)) {
      index += n;
      if (!is_binary) value += n;
      kkt_filter_.mark(&G[j]);
      kkt_filter_.mark(&U[j]);
      continue;
    }
    double g = 0, u = 0;
    double d = is_binary ? exp(delta_ptr[j]) : delta_ptr[j];
    // TODO unroll loop
    for (size_t o = 0; o < n; ++o) {
      auto i = *(index ++);
      double tau = 1 / ( 1 + dual_[i] );
      if (is_binary) {
        g -= y[i] * tau;
        u += std::min(tau*(1-tau)*d, .25);
        // u += tau * (1-tau);
      } else {
        double v = *(value++);
        g -= y[i] * tau * v;
        u += std::min(tau*(1-tau)*exp(fabs(v)*d), .25) * v * v;
        // u += tau * (1-tau) * v * v;
      }
    }
    G[j] = g; U[j] = u;
  }
}

void Darling::updateDual(
  int grp, SizeR global_range, SArray<double> new_w, const int task_id) {
  SArray<double> delta_w(new_w.size());
  auto& active_set = active_set_[grp];
  // load data
  Ocean::DataPack data_pack = ocean_.get(grp, global_range, task_id);
  // on-demand usage
  SArray<double> parameter_value(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::PARAMETER_VALUE)]);
  SArray<uint32> feature_key(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::FEATURE_KEY)]);
  SArray<size_t> feature_offset(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::FEATURE_OFFSET)]);
  SArray<double> feature_value(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::FEATURE_VALUE)]);
  SArray<double> delta(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::DELTA)]);
  SizeR anchor = ocean_.fetchAnchor(grp, global_range);

  // check
  CHECK_EQ(anchor.size(), parameter_value.size());
  if (!feature_value.empty()) {
    CHECK_EQ(feature_key.size(), feature_value.size());
  }
  CHECK_EQ(feature_offset.size(), anchor.size() + 1);
  CHECK_EQ(feature_offset.size(), delta.size() + 1);
  CHECK_EQ(
    feature_offset.back() - feature_offset.front(),
    feature_key.size());

  // update weight
  auto& cur_w = parameter_value;
  for (size_t i = 0; i < new_w.size(); ++i) {
    size_t j = anchor.begin() + i;
    double& cw = cur_w[i];
    double& nw = new_w[i];
    if (kkt_filter_.marked(nw)) {
      active_set.clear(j);
      cw = 0;
      delta_w[i] = 0;
      continue;
    }
    delta_w[i] = nw - cw;
    delta[i] = newDelta(delta_w[i]);
    cw = nw;
  }

  SizeR row_range(0, ocean_.matrix_rows(grp));
  ThreadPool pool(FLAGS_num_threads);
  int npart = FLAGS_num_threads;
  for (int i = 0; i < npart; ++i) {
    auto thr_row_range = row_range.evenDivide(npart, i);
    if (thr_row_range.empty()) continue;
    pool.add([this, grp, thr_row_range, anchor, delta_w,
              &feature_key, &feature_offset, &feature_value, &delta]() {
      updateDual(
        grp, thr_row_range, anchor, delta_w,
        feature_key, feature_offset, feature_value, delta);
    });
  }
  pool.startWorkers();
}

void Darling::updateDual(
  int grp, SizeR th_row_range, SizeR anchor, SArray<double> w_delta,
  SArray<uint32> feature_index,
  SArray<size_t> feature_offset,
  SArray<double> feature_value,
  SArray<double> delta) {
  CHECK_EQ(w_delta.size(), anchor.size());

  const auto& active_set = active_set_[grp];
  double* y = y_->value().data();

  size_t* offset = feature_offset.data();
  uint32* index = feature_index.data();
  double* value = feature_value.data();

  bool is_binary = ocean_.matrix_binary(grp);
  // j: column id, i: row id
  for (size_t j = 0; j < anchor.size(); ++j) {
    size_t k = j + anchor.begin();
    size_t n = offset[j+1] - offset[j];
    double wd = w_delta[j];
    if (wd == 0 || !active_set.test(k)) {
      index += n;
      continue;
    }
    // TODO unroll the loop
    for (size_t o = offset[j]; o < offset[j+1]; ++o) {
      auto i = *(index++);
      if (!th_row_range.contains(i)) continue;
      dual_[i] *= is_binary ? exp(y[i] * wd) : exp(y[i] * wd * value[o - offset[0]]);
    }
  }
}

void Darling::updateWeight(
  int grp, SizeR global_range,
  SArray<double> G, SArray<double> U,
  const int task_id, const bool is_priority) {
  // load data
  Ocean::DataPack data_pack = ocean_.get(grp, global_range, task_id);
  // on-demand usage
  SArray<double> value(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::PARAMETER_VALUE)]);
  SArray<double> delta(
    data_pack.arrays[static_cast<size_t>(Ocean::DataSource::DELTA)]);
  SizeR anchor = ocean_.fetchAnchor(grp, global_range);

  // check
  CHECK_EQ(G.size(), anchor.size());
  CHECK_EQ(U.size(), anchor.size());
  CHECK_EQ(value.size(), delta.size());

  // progress statistic
  size_t nnz_w = 0;
  double objv = 0.0;

  double eta = conf_.learning_rate().eta();
  if (1023 == grp && is_priority && conf_.solver().has_beta_feature_learning_rate()) {
    eta = conf_.solver().beta_feature_learning_rate();
  }

  double lambda = conf_.penalty().lambda(0);
  auto& active_set = active_set_[grp];
  for (size_t i = 0; i < anchor.size(); ++i) {
    size_t k = i + anchor.begin();
    if (!active_set.test(k)) continue;
    double g = G[i], u = U[i] / eta + 1e-10;
    double g_pos = g + lambda, g_neg = g - lambda;
    double& w = value[i];
    double d = - w, vio = 0;

    if (w == 0) {
      if (g_pos < 0) {
        vio = - g_pos;
      } else if (g_neg > 0) {
        vio = g_neg;
      } else if (g_pos > kkt_filter_threshold_ && g_neg < - kkt_filter_threshold_) {
        active_set.clear(k);
        kkt_filter_.mark(&w);
        continue;
      }
    }
    violation_ = std::max(violation_, vio);

    if (g_pos <= u * w) {
      d = - g_pos / u;
    } else if (g_neg >= u * w) {
      d = - g_neg / u;
    }
    d = std::min(delta[i], std::max(-delta[i], d));
    delta[i] = newDelta(d);
    w += d;

    if (!(kkt_filter_.marked(w) || 0 == w)) {
      nnz_w++;
      objv += fabs(w);
    }
  }
  progress_stat_[Ocean::UnitID(grp, global_range)] = std::make_pair(nnz_w, objv);
}


void Darling::showKKTFilter(int iter) {
  if (iter == -3) {
    fprintf(stderr, "|      KKT filter     ");
  } else if (iter == -2) {
    fprintf(stderr, "| threshold  #activet ");
  } else if (iter == -1) {
    fprintf(stderr, "+---------------------");
  } else {
    auto prog = g_progress_[iter];
    fprintf(stderr, "| %.1e %11llu ", kkt_filter_threshold_, (uint64)prog.nnz_active_set());
  }
}

void Darling::showProgress(int iter) {
  int s = iter == 0 ? -3 : iter;
  for (int i = s; i <= iter; ++i) {
    showObjective(i);
    showNNZ(i);
    showKKTFilter(i);
    showAUC(i);
    showTime(i);
  }
}

Progress Darling::evaluateProgress() {
  Progress prog;
  if (IamWorker()) {
    prog.set_objv(log(1+1/dual_.eigenArray()).sum());
    prog.add_busy_time(busy_timer_.stop());
    busy_timer_.restart();

    // wait and get AUC statistic
    *prog.mutable_validation_auc_data() = validation_.waitAndGetResult();

    // label statistics
    if (FLAGS_verbose) {
      size_t positive_label_count = 0;
      size_t negative_label_count = 0;
      size_t bad_label_count = 0;

      for (size_t i = 0; i < y_->value().size(); ++i) {
        int label = y_->value()[i];

        if (1 == label) {
          positive_label_count++;
        } else if (-1 == label) {
          negative_label_count++;
        } else {
          bad_label_count++;
        }
      }

      LI << "dual_sum[" << dual_.eigenArray().sum() << "] " <<
        "dual_.rows[" << dual_.eigenArray().rows() << "] " <<
        "dual_.avg[" << dual_.eigenArray().sum() / static_cast<double>(
          dual_.eigenArray().rows()) << "] " <<
        "y_.positive[" << positive_label_count << "] " <<
        "y_.negative[" << negative_label_count << "] " <<
        "y_.bad[" << bad_label_count << "] " <<
        "y_.positive_ratio[" << positive_label_count / static_cast<double>(
          positive_label_count + negative_label_count + bad_label_count) << "] ";
    }
  } else {
    size_t nnz_w = 0;
    size_t nnz_as = 0;
    double objv = 0;
    for (int grp : fea_grp_) {
      nnz_as += active_set_[grp].nnz();
    }
    for (const auto& column_partition : progress_stat_) {
      nnz_w += column_partition.second.first;
      objv += column_partition.second.second;
    }
    prog.set_objv(objv * conf_.penalty().lambda(0));
    prog.set_nnz_w(nnz_w);
    prog.set_violation(violation_);
    prog.set_nnz_active_set(nnz_as);
  }
  return prog;
}

} // namespace LM
} // namespace PS
