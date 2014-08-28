#include "risk_minimization/linear_method/linear_method.h"
#include "risk_minimization/learner/learner_factory.h"
#include "base/range.h"
#include "util/eigen3.h"
#include "base/matrix_io.h"
#include "proto/instance.pb.h"
#include "base/io.h"

namespace PS {

namespace LM {

void LinearMethod::init() {
  bool has_learner = app_cf_.has_learner();
  if (has_learner) {
    learner_ = std::static_pointer_cast<AggGradLearner<double>>(
        LearnerFactory<double>::create(app_cf_.learner()));
  }

  if (app_cf_.has_loss()) {
    loss_ = Loss<double>::create(app_cf_.loss());
    if (has_learner) learner_->setLoss(loss_);
  }

  if (app_cf_.has_penalty()) {
    penalty_ = Penalty<double>::create(app_cf_.penalty());
    if (has_learner) learner_->setPenalty(penalty_);
  }
}

void LinearMethod::startSystem() {
    App::requestNodes();

    // initialize other nodes
    Task start;
    start.set_request(true);
    start.set_customer(name());
    start.set_type(Task::MANAGE);
    start.mutable_mng_node()->set_cmd(ManageNode::INIT);

    g_fea_range_.set(0, std::numeric_limits<Key>::max());
    for (const auto &it : nodes_) {
        auto node = it.second;
        g_fea_range_.to(node.mutable_key());
        *start.mutable_mng_node()->add_nodes() = node;
    }

    // let the scheduler connect all other nodes
    sys_.manageNode(start);

    // create the app on other nodes
    int time = 0;
    start.mutable_mng_app()->set_cmd(ManageApp::ADD);
    *(start.mutable_mng_app()->mutable_app_config()) = app_cf_;
    for (auto &w : exec_.group(kActiveGroup)) {
        CHECK_EQ(time, w->submit(start));
    }

    // wait
    taskpool(kActiveGroup)->waitOutgoingTask(time);
}

} // namespace LM
} // namespace PS
