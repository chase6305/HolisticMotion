#pragma once

#include <chrono>

namespace holistic_motion::robotics::planning::detail {

inline std::chrono::steady_clock::time_point
DeadlineAfter(std::chrono::steady_clock::time_point start, double seconds) {
  using Clock = std::chrono::steady_clock;
  const auto remaining = Clock::time_point::max() - start;
  if (std::chrono::duration<double>(seconds) >= remaining)
    return Clock::time_point::max();
  return start + std::chrono::duration_cast<Clock::duration>(
                     std::chrono::duration<double>(seconds));
}

} // namespace holistic_motion::robotics::planning::detail
