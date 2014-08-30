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

    // increase in/out bytes
    void increaseInBytes(const size_t delta);
    void increaseOutBytes(const size_t delta);

    void setInterface(const string &name);


private:
    RunningStatusReport report_;
    std::vector<Timer> timers_;
    size_t in_bytes_, out_bytes_;

    // the name of network interface such as "eth0"
    string net_interface_;

    // 两次计时的间隔
    Timer total_timer_;

    struct Snapshot {
        uint64 my_user;
        uint64 my_sys;
        uint64 host_user;
        uint64 host_sys;
        uint64 host_total; // host total cpu ticks
        uint64 host_in_bytes;
        uint64 host_out_bytes;

        Snapshot() :
            my_user(0),
            my_sys(0),
            host_user(0),
            host_sys(0),
            host_total(0),
            host_in_bytes(0),
            host_out_bytes(0) {
            // do nothing
        }
    };

    Snapshot last_snapshot_;
    Snapshot dumpSnapshot();

    // reset all internal status
    void reset();

    std::mutex mu_;

    size_t core_number_;
}; // class Runningstatus
}; // namespace PS
