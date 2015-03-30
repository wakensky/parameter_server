#include "linear_method/linear_method.h"
#include "base/range.h"
#include "util/eigen3.h"
#include "proto/instance.pb.h"
#include "linear_method/darling.h"
#include "linear_method/batch_solver.h"
namespace PS {
namespace LM {

AppPtr LinearMethod::create(const Config& conf) {
  if (!conf.has_solver()) {
    CHECK(false) << "Solver should be provided in configuration";
  } else if (conf.solver().minibatch_size() <= 0) {
    // batch solver
    if (conf.has_darling()) {
      return AppPtr(new Darling());
    } else {
      return AppPtr(new BatchSolver());
    }
  }
  return AppPtr(nullptr);
}

void LinearMethod::init() {
  CHECK(app_cf_.has_linear_method());
  conf_ = app_cf_.linear_method();

  click_sum_ = 0;
  prediction_sum_ = 0;
  num_validation_examples_ = 0;

  // bool has_learner = app_cf_.has_learner();
  // if (has_learner) {
  //   learner_ = std::static_pointer_cast<AggGradLearner<double>>(
  //       LearnerFactory<double>::create(app_cf_.learner()));
  // }
  // if (has_learner) learner_->setLoss(loss_);
  // if (has_learner) learner_->setPenalty(penalty_);
}

void LinearMethod::process(const MessagePtr& msg) {
  switch (get(msg).cmd()) {
    case Call::EVALUATE_PROGRESS: {
      auto prog = evaluateProgress();
      // LL << myNodeID() << prog.DebugString();
      sys_.replyProtocalMessage(msg, prog);
      break;
    }
    case Call::REPORT_PROGRESS: {
      Progress prog; CHECK(prog.ParseFromString(msg->task.msg()));
      Lock l(progress_mu_);
      recent_progress_[msg->sender] = prog;
    }
    case Call::LOAD_DATA: {
      DataInfo info;
      info.set_hit_cache(loadData(msg, info.mutable_example_info()));
      sys_.replyProtocalMessage(msg, info);
      break;
    }
    case Call::PREPROCESS_DATA:
      preprocessData(msg);
      break;
    case Call::UPDATE_MODEL:
      updateModel(msg);
      break;
    case Call::SAVE_MODEL:
      saveModel(msg);
      break;
    case Call::RECOVER:
      // FIXME
      // W_.recover();
      break;
    case Call::COMPUTE_VALIDATION_AUC: {
      AUCData data;
      computeEvaluationAUC(&data);
      sys_.replyProtocalMessage(msg, data);
      break;
    }
    default:
      CHECK(false) << "unknown cmd: " << get(msg).cmd();
  }
}


void LinearMethod::mergeProgress(int iter) {
  Progress recv;
  CHECK(recv.ParseFromString(exec_.lastRecvReply()));
  auto& p = g_progress_[iter];

  p.set_objv(p.objv() + recv.objv());
  p.set_nnz_w(p.nnz_w() + recv.nnz_w());

  if (recv.busy_time_size() > 0) p.add_busy_time(recv.busy_time(0));
  p.set_total_time(total_timer_.stop());
  total_timer_.start();
  p.set_relative_objv(iter==0 ? 1 : g_progress_[iter-1].objv()/p.objv() - 1);
  p.set_violation(std::max(p.violation(), recv.violation()));
  p.set_nnz_active_set(p.nnz_active_set() + recv.nnz_active_set());

  if (recv.has_validation_auc_data()) {
    validation_auc_.merge(recv.validation_auc_data());
    click_sum_ += recv.validation_auc_data().click_sum();
    prediction_sum_ += recv.validation_auc_data().prediction_sum();
    num_validation_examples_ += recv.validation_auc_data().num_examples();
  }
}

void LinearMethod::mergeAUC(AUC* auc) {
  if (exec_.lastRecvReply().empty()) return;
  AUCData recv;
  CHECK(recv.ParseFromString(exec_.lastRecvReply()));
  auc->merge(recv);
}


void LinearMethod::startSystem() {
  Task start;
  start.set_request(true);
  start.set_customer(name());
  start.set_type(Task::MANAGE);
  start.mutable_mng_node()->set_cmd(ManageNode::INIT);

  // Range<Key> key_range = Range<Key>::all();
  Range<Key> key_range(0, std::numeric_limits<Key>::max() & kLower54Bits);
  int s = 0;
  for (auto& it : sys_.yp().nodes()) {
    auto node = it.second;
    auto key = node.role() != Node::SERVER ? key_range :
               key_range.evenDivide(FLAGS_num_servers, s++);
    key.to(node.mutable_key());
    *start.mutable_mng_node()->add_node() = node;

    LI << node.id() << " got key range: " << key.toString();
  }
  // let the scheduler connect all other nodes
  sys_.manageNode(start);

  // find all training (validation) files, and assign them to workers
  CHECK(conf_.has_training_data());
  std::vector<DataConfig> tr_parts, va_parts;
  auto tr_cf = searchFiles(conf_.training_data());
  LI << "Found " << tr_cf.file_size() << " training files";
  tr_parts = divideFiles(tr_cf, FLAGS_num_workers);
  int n = 0; for (const auto& p : tr_parts) n += p.file_size();
  LI << "Assigned " << n << " files to " << FLAGS_num_workers << " workers";

  if (conf_.has_validation_data()) {
    auto va_cf = searchFiles(conf_.validation_data());
    va_parts = divideFiles(va_cf, FLAGS_num_workers);
    LI << "Found " << va_cf.file_size() << " validation files";
  }

  // create the app on all other machines
  int time = 1;  // time 0 is used for
  int k = 0;
  start.mutable_mng_app()->set_cmd(ManageApp::ADD);
  start.set_time(time);
  for (auto& w : exec_.group(kActiveGroup)) {
    auto cf = app_cf_;
    auto lm = cf.mutable_linear_method();
    lm->clear_training_data();
    lm->clear_validation_data();
    if (w->role() == Node::WORKER) {
      if (conf_.has_validation_data()) {
        *lm->mutable_validation_data() = va_parts[k];
      }
      *lm->mutable_training_data() = tr_parts[k++];
    }
    *start.mutable_mng_app()->mutable_app_config() = cf;
    CHECK_EQ(time, w->submit(start));
  }
  taskpool(kActiveGroup)->waitOutgoingTask(time);
}

void LinearMethod::showTime(int iter) {
  if (iter == -3) {
    fprintf(stderr, "|    time (sec.)\n");
  } else if (iter == -2) {
    fprintf(stderr, "|(app:min max) total\n");
  } else if (iter == -1) {
    fprintf(stderr, "+-----------------\n");
  } else {
    auto prog = g_progress_[iter];
    double ttl_t = prog.total_time() - (
        iter > 0 ? g_progress_[iter-1].total_time() : 0);

    int n = prog.busy_time_size();
    Eigen::ArrayXd busy_t(n);
    for (int i = 0; i < n; ++i) {
      busy_t[i] = prog.busy_time(i);
      // if (iter > 0) busy_t[i] -= global_progress_[iter-1].busy_time(i);
    }
    // double mean = busy_t.sum() / n;
    // double var = (busy_t - mean).matrix().norm() / std::sqrt((double)n);
    fprintf(stderr, "|%6.1f%6.1f%6.1f\n", busy_t.minCoeff(), busy_t.maxCoeff(), ttl_t);
  }
}


void LinearMethod::showObjective(int iter) {
  if (iter == -3) {
    fprintf(stderr, "     |        training        ");
  } else if (iter == -2) {
    fprintf(stderr, "iter |  objective    relative ");
  } else if (iter == -1) {
    fprintf(stderr, " ----+------------------------");
  } else {
    auto prog = g_progress_[iter];
    fprintf(stderr, "%4d | %.5e  %.3e ",
            iter, prog.objv(), prog.relative_objv()); //o, prog.training_auc());
  }
}

void LinearMethod::showNNZ(int iter) {
  if (iter == -3) {
    fprintf(stderr, "|  sparsity ");
  } else if (iter == -2) {
    fprintf(stderr, "|     |w|_0 ");
  } else if (iter == -1) {
    fprintf(stderr, "+-----------");
  } else {
    auto prog = g_progress_[iter];
    fprintf(stderr, "|%10lu ", (size_t)prog.nnz_w());
  }
}

void LinearMethod::showAUC(int iter) {
  if (iter == -3) {
    fprintf(stderr, "| AUC ");
  } else if (iter == -2) {
    fprintf(stderr, "|    AUC    pa    ca    ");
  } else if (iter == -1) {
    fprintf(stderr, "+----------------------");
  } else {
    double click_average = 0.0;
    double prediction_average = 0.0;
    if (num_validation_examples_ > 0) {
      click_average = click_sum_ / num_validation_examples_;
      prediction_average = prediction_sum_ / num_validation_examples_;
    }
    fprintf(stderr, "|%.6e  %.6e  %.6e  ",
      validation_auc_.evaluate(), prediction_average, click_average);

    // clean up
    validation_auc_.clear();
    num_validation_examples_ = 0;
    click_sum_ = 0.0;
    prediction_sum_ = 0.0;
  }
}

} // namespace LM
} // namespace PS
