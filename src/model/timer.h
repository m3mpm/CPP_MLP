#pragma once

#include <time.h>
namespace cpp_mlp {
class Timer {
 public:
  Timer() : start_(clock()) {}
  double PastTime() { return (double)(clock() - start_) / CLOCKS_PER_SEC; }

 private:
  clock_t start_;
};
}  // namespace cpp_mlp
