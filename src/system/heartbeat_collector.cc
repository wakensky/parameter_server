#include <chrono>
#include "system/heartbeat_collector.h"

namespace PS {
HeartbeatCollector::HeartbeatCollector() :
  milli_seconds_(0),
  in_bytes_(0),
  out_bytes_(0) {
  // get cpu core number
  const size_t kBufLen = 1024;
  char buffer[kBufLen + 1];
  FILE *fp_pipe = popen("grep 'processor' /proc/cpuinfo | wc -l", "r");
  CHECK(nullptr != fp_pipe);
  CHECK(nullptr != fgets(buffer, kBufLen, fp_pipe));
  pclose(fp_pipe);

  string core_str(buffer);
  core_str.resize(core_str.size() - 1);
  cpu_core_number_ = std::stoul(core_str);

  // initialize internal status
  produceReport();
}

HeartbeatCollector::~HeartbeatCollector() {
  // do nothing
}

HeartbeatCollector::Snapshot HeartbeatCollector::snapshot() {
  HeartbeatCollector::Snapshot ret;

  // time stamp
  ret.time_point = std::chrono::system_clock::now();

  // cpu usage under current process
  std::ifstream my_cpu_stat("/proc/self/stat", std::ifstream::in);
  CHECK(my_cpu_stat) << "open /proc/self/stat failed [" << strerror(errno) << "]";
  string pid, comm, state, ppid, pgrp, session, tty_nr;
  string tpgid, flags, minflt, cminflt, majflt, cmajflt;
  string utime, stime, cutime, cstime, priority, nice;
  my_cpu_stat >> pid >> comm >> state >> ppid >> pgrp >> session >>
    tty_nr >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt >>
    utime >> stime >> cutime >> cstime >> priority >> nice;
  my_cpu_stat.close();
  ret.process_user = std::stoull(utime);
  ret.process_sys = std::stoull(stime);

  // cpu usage under current host
  std::ifstream host_cpu_stat("/proc/stat", std::ifstream::in);
  CHECK(host_cpu_stat) << "open /proc/stat failed [" << strerror(errno) << "]";
  string label, host_user, host_nice, host_sys, host_idle, host_iowait;
  host_cpu_stat >> label >> host_user >> host_nice >> host_sys >> host_idle >>
    host_iowait;
  host_cpu_stat.close();
  ret.host_user = std::stoull(host_user);
  ret.host_sys = std::stoull(host_sys);
  ret.host_cpu_base = std::stoull(host_user) + std::stoull(host_nice) +
    std::stoull(host_sys) + std::stoull(host_idle) + std::stoull(host_iowait);

  // host network bandwidth usage
  if (!interface_.empty()) {
    std::ifstream host_net_dev_stat("/proc/net/dev", std::ifstream::in);
    CHECK(host_net_dev_stat) << "open /proc/net/dev failed [" << strerror(errno) << "]";

    // find interface
    string line;
    bool interface_found = false;
    while (std::getline(host_net_dev_stat, line)) {
      if (std::string::npos != line.find(interface_)) {
        interface_found = true;
        break;
      }
    }
    CHECK(interface_found) << "I cannot find interface[" << interface_ <<
      "] in /proc/net/dev";

    // read counters
    string face, r_bytes, r_packets, r_errs, r_drop, r_fifo, r_frame;
      string r_compressed, r_multicast, t_bytes, t_packets;
    std::stringstream ss(line);
    ss >> face >> r_bytes >> r_packets >> r_errs >> r_drop >> r_fifo >>
      r_frame >> r_compressed >> r_multicast >> t_bytes >> t_packets;
    host_net_dev_stat.close();

    ret.host_in_bytes = std::stoull(r_bytes);
    ret.host_out_bytes = std::stoull(t_bytes);
  }

  return ret;
}

HeartbeatReport HeartbeatCollector::produceReport() {
  Lock l(mu_);
  HeartbeatReport report;
  Snapshot snapshot_now = snapshot();

  // hostname
  report.set_hostname(hostname_);

  // interval between invocations
  uint32 total_milli = std::chrono::duration_cast<std::chrono::milliseconds>(
    snapshot_now.time_point - snapshot_last_.time_point).count();
  if (0 == total_milli) {
    total_milli = 1;
  }
  report.set_total_time_milli(total_milli);

  // busy time
  report.set_busy_time_milli(milli_seconds_);

  // transfered bytes under current process
  report.set_process_in_mb(in_bytes_ / 1024 / 1024);
  report.set_process_out_mb(out_bytes_ / 1024 / 1024);

  // network bandwidth used on the host
  report.set_host_net_in_bw(static_cast<uint32>(
    (snapshot_now.host_in_bytes - snapshot_last_.host_in_bytes) /
    (total_milli / 1e3) / 1024 / 1024));
  report.set_host_net_out_bw(static_cast<uint32>(
    (snapshot_now.host_out_bytes - snapshot_last_.host_out_bytes) /
    (total_milli / 1e3) / 1024 / 1024));

  // cpu under current process
  uint32 process_usage_now =
    snapshot_now.process_user +
    snapshot_now.process_sys;
  uint32 process_usage_last =
    snapshot_last_.process_user +
    snapshot_last_.process_sys;
  report.set_process_cpu_usage(cpu_core_number_ *
    100 * static_cast<float>(process_usage_now - process_usage_last) /
    (snapshot_now.host_cpu_base - snapshot_last_.host_cpu_base));

  // cpu under current host
  uint32 host_usage_now =
    snapshot_now.host_user +
    snapshot_now.host_sys;
  uint32 host_usage_last = snapshot_last_.host_user + snapshot_last_.host_sys;
  report.set_host_cpu_usage(cpu_core_number_ *
    100 * static_cast<float>(host_usage_now - host_usage_last) /
    (snapshot_now.host_cpu_base - snapshot_last_.host_cpu_base));

  // memory statistic
  report.set_process_rss_mb(ResUsage::myPhyMem());
  report.set_process_virt_mb(ResUsage::myVirMem());
  report.set_host_in_use_gb(ResUsage::hostInUseMem() / 1024);
  report.set_host_in_use_percentage(
    100 * ResUsage::hostInUseMem() / ResUsage::hostTotalMem());

  milli_seconds_ = 0;
  in_bytes_ = 0;
  out_bytes_ = 0;
  snapshot_last_ = snapshot_now;

  return report;
}

void HeartbeatCollector::init(const string& interface, const string& hostname) {
  interface_ = interface;
  hostname_ = hostname;
}
}; // namespace PS
