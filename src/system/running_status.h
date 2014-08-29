#pragma once

#include "proto/task.pb.h"
#include "util/resource_usage.h"
#include "util/common.h"

namespace PS {

class RunningStatus {
public:
    RunningStatus();
    ~RunningStatus();
    RunningStatus(const RunningStatus &other) = delete;
    RunningStatus& operator= (const RunningStatus &rhs) = delete;

    // get report as serialized protobuf
    RunningStatusReport get();

    // start timer
    void startTimer(const TimerType type);
    // stop timer; add delta to internal timer
    void stopTimer(const TimerType type);


private:
    RunningStatusReport report_;
    std::vector<Timer> timers_;

    // 两次计时的间隔
    Timer total_timer_;

    struct CPUSnapshot {
        uint64 my_user;
        uint64 my_sys;
        uint64 host_user;
        uint64 host_sys;
        uint64 host_total;
    };

    CPUSnapshot last_cpu_snapshot_;
    CPUSnapshot dumpCPUSnapshot();

    // reset all internal status
    void reset();

    std::mutex mu_;

    size_t core_number_;
}; // class Runningstatus
}; // namespace PS
