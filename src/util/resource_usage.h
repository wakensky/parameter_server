#pragma once

#include "stdlib.h"
#include "stdio.h"
#include "string.h"

#include <ctime>
#include <ratio>
#include <chrono>

namespace PS {

using std::chrono::system_clock;
static system_clock::time_point tic() {
  return system_clock::now();
}

// return the time since tic, in microsecnods
static double toc(system_clock::time_point start) {
  size_t ct = std::chrono::duration_cast<std::chrono::microseconds>(
      system_clock::now() - start).count();
  return (double) ct;
}

class ScopedTimer {
 public:
  explicit ScopedTimer(double* aggregate_time) :
      aggregate_time_(aggregate_time) {
    timer_ = tic();
  }
  ~ScopedTimer() {
    *aggregate_time_ += toc(timer_) / 1e6;
  }

 private:
  double* aggregate_time_;
  system_clock::time_point timer_;
};

class Timer {
 public:
  void start() { tp_ = tic(); }
  void stop() { time_ += toc(tp_); }
  void reset() { time_ = 0; tp_ = tic(); }
  // get time interval in seconds
  double get() {
    return time_ / 1e6;
  }
  // get time interval in microsecnods
  double getMicro() {
    return time_;
  }
  // get time interval from construction to now in seconds
  double getToNow() {
    return toc(construction_tp_) / 1e6;
  }
  // get time interval from construction to now in microsecnods
  double getToNowMicro() {
    return toc(construction_tp_);
  }
 private:
  system_clock::time_point tp_ = tic();
  const system_clock::time_point construction_tp_ = tic();
  // in microsecnods
  double time_ = 0;
};

// http://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
// print memeory usage of the current process in Mb
// TODO CPU usage
class ResUsage {
 public:
  // in Mb
  static double myVirMem() {
    return getLine("/proc/self/status", "VmSize:") / 1e3;
  }
  static double myPhyMem() {
    return getLine("/proc/self/status", "VmRSS:") / 1e3;
  }
  // in Mb
  static double hostFreePhyMem() {
    return getLine("/proc/meminfo", "MemFree:") / 1e3;
  }
 private:
  static double getLine(const char *target, const char *name) {
    FILE* file = fopen(target, "r");
    char line[128];
    int result = -1;
    while (fgets(line, 128, file) != NULL){
      if (strncmp(line, name, strlen(name)) == 0){
        result = parseLine(line);
        break;
      }
    }
    fclose(file);
    return result;
  }

  static int parseLine(char* line){
    int i = strlen(line);
    while (*line < '0' || *line > '9') line++;
    line[i-3] = '\0';
    i = atoi(line);
    return i;
  }
};

} // namespace PS
