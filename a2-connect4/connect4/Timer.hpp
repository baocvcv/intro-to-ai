#ifndef TIMER_HPP
#define TIMER_HPP

#include <ctime>

class Timer {
public:
  Timer() {
    start = clock();
  }

  double getElapsedMicroseconds() {
    end = clock();
    return 1e6 * (end - start) / CLOCKS_PER_SEC;
  }

protected:
  clock_t start, end;
};

#endif
