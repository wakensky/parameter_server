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

    // reset in/out bytes
    in_bytes_ = 0;
    out_bytes_ = 0;

    // initialize cpu snapshot
    last_snapshot_ = dumpSnapshot();

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
        report_.set_netin_time_micro(
            static_cast<uint32>(timers_[static_cast<size_t>(TimerType::NETIN)].getMicro()));
        report_.set_netout_time_micro(
            static_cast<uint32>(timers_[static_cast<size_t>(TimerType::NETOUT)].getMicro()));

        // in/out bytes within the process
        report_.set_in_bytes(in_bytes_);
        report_.set_out_bytes(out_bytes_);

        // CPU
        RunningStatus::Snapshot snapshot= dumpSnapshot();
        report_.set_my_cpu_usage_user(
                core_number_ * 100 * (
                    static_cast<float>(snapshot.my_user - last_snapshot_.my_user) /
                    (snapshot.host_total - last_snapshot_.host_total)));
        report_.set_my_cpu_usage_sys(
                core_number_ * 100 * (
                    static_cast<float>(snapshot.my_sys - last_snapshot_.my_sys) /
                    (snapshot.host_total - last_snapshot_.host_total)));
        report_.set_host_cpu_usage_user(
                core_number_ * 100 * (
                    static_cast<float>(snapshot.host_user - last_snapshot_.host_user) /
                    (snapshot.host_total - last_snapshot_.host_total)));
        report_.set_host_cpu_usage_sys(
                core_number_ * 100 * (
                    static_cast<float>(snapshot.host_sys - last_snapshot_.host_sys) /
                    (snapshot.host_total - last_snapshot_.host_total)));

        // memory
        report_.set_my_virtual(ResUsage::myVirMem());
        report_.set_my_rss(ResUsage::myPhyMem());
        report_.set_host_free_memory(ResUsage::hostFreePhyMem());

        // host net bandwidth usage (average)
        report_.set_host_net_in_bw_usage(static_cast<uint32>(
            (snapshot.host_in_bytes - last_snapshot_.host_in_bytes) /
            (total_timer_.getMicro() / 1e3) / 1e3));
        report_.set_host_net_out_bw_usage(static_cast<uint32>(
            (snapshot.host_out_bytes - last_snapshot_.host_out_bytes) /
            (total_timer_.getMicro() / 1e3) / 1e3));
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

void RunningStatus::increaseInBytes(const size_t delta) {
    in_bytes_ += delta;
}

void RunningStatus::increaseOutBytes(const size_t delta) {
    out_bytes_ += delta;
}

void RunningStatus::setInterface(const string &name) {
    Lock l(mu_);
    net_interface_ = name;
}

RunningStatus::Snapshot RunningStatus::dumpSnapshot() {
    RunningStatus::Snapshot snapshot;

    std::ifstream my_cpu_stat("/proc/self/stat", std::ifstream::in);
    CHECK(my_cpu_stat) << "open /proc/self/stat failed [" << strerror(errno) << "]";
    string pid, comm, state, ppid, pgrp, session, tty_nr;
    string tpgid, flags, minflt, cminflt, majflt, cmajflt;
    string utime, stime, cutime, cstime, priority, nice;
    my_cpu_stat >> pid >> comm >> state >> ppid >> pgrp >> session >>
        tty_nr >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >>
        utime >> stime >> cutime >> cstime >> priority >> nice;
    my_cpu_stat.close();

    std::ifstream host_cpu_stat("/proc/stat", std::ifstream::in);
    CHECK(host_cpu_stat) << "open /proc/stat failed [" << strerror(errno) << "]";
    string label, host_user, host_nice, host_sys, host_idle, host_iowait;
    host_cpu_stat >> label >> host_user >> host_nice >> host_sys >> host_idle >>
        host_iowait;
    host_cpu_stat.close();

    if (!net_interface_.empty()) {
        std::ifstream host_net_dev_stat("/proc/net/dev", std::ifstream::in);
        CHECK(host_net_dev_stat) << "open /proc/net/dev failed [" << strerror(errno) << "]";
        string line;
        bool interface_found = false;
        while (std::getline(host_net_dev_stat, line)) {
            if (std::string::npos != line.find(net_interface_)) {
                interface_found = true;
                break;
            }
        }
        CHECK(interface_found) << "I cannot find interface[" << net_interface_ <<
            "] in /proc/net/dev";
        string face, r_bytes, r_packets, r_errs, r_drop, r_fifo, r_frame;
        string r_compressed, r_multicast, t_bytes, t_packets;
        std::stringstream ss(line);
        ss >> face >> r_bytes >> r_packets >> r_errs >> r_drop >> r_fifo >>
            r_frame >> r_compressed >> r_multicast >> t_bytes >> t_packets;
        host_net_dev_stat.close();

        snapshot.host_in_bytes = std::stoull(r_bytes);
        snapshot.host_out_bytes = std::stoull(t_bytes);
    }

    snapshot.my_user = std::stoull(utime);
    snapshot.my_sys = std::stoull(stime);
    snapshot.host_user = std::stoull(host_user);
    snapshot.host_sys = std::stoull(host_sys);
    snapshot.host_total = std::stoull(host_user) + std::stoull(host_nice) +
        std::stoull(host_sys) + std::stoull(host_idle) + std::stoull(host_iowait);

    return snapshot;
}

}; // namespace PS

