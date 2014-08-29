#include <chrono>
#include "system/running_status.h"

namespace PS {

RunningStatus::RunningStatus() {
    // CPU core number
    char buffer[1024];
    FILE *fp_pipe = popen("grep 'processor' /proc/cpuinfo | wc -l", "r");
    CHECK(nullptr != fp_pipe);
    CHECK(nullptr != fgets(buffer, sizeof(buffer), fp_pipe));
    string core_str(buffer);
    core_str.resize(core_str.size() - 1);
    core_number_ = std::stoul(core_str);
    pclose(fp_pipe);

    reset();
}

RunningStatus::~RunningStatus() {
    // do nothing
}

void RunningStatus::reset() {
    Lock l(mu_);

    // reset all timers
    if (timers_.empty()) {
        for (size_t i = 0; i < static_cast<size_t>(TimerType::NUM); ++i) {
            timers_.push_back(Timer());
        }
        // timers_.assign(static_cast<size_t>(TimerType::NUM), Timer());
    }
    for (auto &timer : timers_) {
        timer.reset();
    }
    total_timer_.reset();

    // initialize cpu snapshot
    last_cpu_snapshot_ = dumpCPUSnapshot();

    // clear report
    report_.Clear();
}

RunningStatusReport RunningStatus::get() {
    {
        Lock l(mu_);
        total_timer_.stop();

        // timestamp
        report_.set_seconds_since_epoch(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());

        report_.set_total_time_micro(
                static_cast<uint32>(total_timer_.getMicro()));
        report_.set_busy_time_micro(
                static_cast<uint32>(timers_[static_cast<size_t>(TimerType::BUSY)].getMicro()));
        report_.set_network_time_micro(
                static_cast<uint32>(timers_[static_cast<size_t>(TimerType::NETWORK)].getMicro()));

        // CPU
        RunningStatus::CPUSnapshot cpu_snapshot= dumpCPUSnapshot();
        report_.set_my_cpu_usage_user(
                core_number_ * 100 * (
                    static_cast<float>(cpu_snapshot.my_user - last_cpu_snapshot_.my_user) /
                    (cpu_snapshot.host_total - last_cpu_snapshot_.host_total)));
        report_.set_my_cpu_usage_sys(
                core_number_ * 100 * (
                    static_cast<float>(cpu_snapshot.my_sys - last_cpu_snapshot_.my_sys) /
                    (cpu_snapshot.host_total - last_cpu_snapshot_.host_total)));
        report_.set_host_cpu_usage_user(
                core_number_ * 100 * (
                    static_cast<float>(cpu_snapshot.host_user - last_cpu_snapshot_.host_user) /
                    (cpu_snapshot.host_total - last_cpu_snapshot_.host_total)));
        report_.set_host_cpu_usage_sys(
                core_number_ * 100 * (
                    static_cast<float>(cpu_snapshot.host_sys - last_cpu_snapshot_.host_sys) /
                    (cpu_snapshot.host_total - last_cpu_snapshot_.host_total)));

        // memory
        report_.set_my_virtual(ResUsage::myVirMem());
        report_.set_my_rss(ResUsage::myPhyMem());
        report_.set_host_free_memory(ResUsage::hostFreePhyMem());
    }

    // reset and return
    RunningStatusReport ret = report_;
    reset();
    return ret;
}

void RunningStatus::startTimer(const TimerType type) {
    Lock l(mu_);

    timers_[static_cast<size_t>(type)].start();
}

void RunningStatus::stopTimer(const TimerType type) {
    Lock l(mu_);

    timers_[static_cast<size_t>(type)].stop();
}

RunningStatus::CPUSnapshot RunningStatus::dumpCPUSnapshot() {
    RunningStatus::CPUSnapshot snapshot;

    std::ifstream my_cpu_stat("/proc/self/stat", std::ifstream::in);
    CHECK(my_cpu_stat) << "open /proc/self/stat failed";
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    my_cpu_stat >> pid >> comm >> state >> ppid >> pgrp >> session >>
        tty_nr >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >>
        utime >> stime >> cutime >> cstime >> priority >> nice;
    my_cpu_stat.close();

    std::ifstream host_cpu_stat("/proc/stat", std::ifstream::in);
    CHECK(host_cpu_stat) << "open /proc/stat failed";
    string label, host_user, host_nice, host_sys, host_idle, host_iowait;
    host_cpu_stat >> label >> host_user >> host_nice >> host_sys >> host_idle >>
        host_iowait;
    host_cpu_stat.close();

    snapshot.my_user = std::stoull(utime);
    snapshot.my_sys = std::stoull(stime);
    snapshot.host_user = std::stoull(host_user);
    snapshot.host_sys = std::stoull(host_sys);
    snapshot.host_total = std::stoull(host_user) + std::stoull(host_nice) +
        std::stoull(host_sys) + std::stoull(host_idle) + std::stoull(host_iowait);

    return snapshot;
}

}; // namespace PS

