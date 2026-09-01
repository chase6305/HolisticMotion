#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <Eigen/Core>

namespace holistic_motion::robotics::planning::detail {

constexpr double kPi = 3.14159265358979323846;

inline double WrappedDifference(double difference) {
  return std::remainder(difference, 2.0 * kPi);
}

inline Eigen::VectorXd Difference(const Eigen::VectorXd &from,
                                  const Eigen::VectorXd &to,
                                  const std::vector<bool> &continuous) {
  Eigen::VectorXd difference = to - from;
  for (Eigen::Index index = 0; index < difference.size(); ++index) {
    if (continuous[static_cast<std::size_t>(index)])
      difference[index] = WrappedDifference(difference[index]);
  }
  return difference;
}

inline double NormalizeCoordinate(double value, Eigen::Index index,
                                  const Eigen::VectorXd &lower,
                                  const Eigen::VectorXd &upper,
                                  const std::vector<bool> &continuous) {
  if (continuous[static_cast<std::size_t>(index)]) {
    return lower[index] +
           std::fmod(std::fmod(value - lower[index], 2.0 * kPi) + 2.0 * kPi,
                     2.0 * kPi);
  }
  return std::clamp(value, lower[index], upper[index]);
}

inline Eigen::VectorXd Normalize(Eigen::VectorXd state,
                                 const Eigen::VectorXd &lower,
                                 const Eigen::VectorXd &upper,
                                 const std::vector<bool> &continuous,
                                 bool clamp_bounded = true) {
  for (Eigen::Index index = 0; index < state.size(); ++index) {
    if (clamp_bounded || continuous[static_cast<std::size_t>(index)]) {
      state[index] =
          NormalizeCoordinate(state[index], index, lower, upper, continuous);
    }
  }
  return state;
}

inline std::size_t SegmentCount(const Eigen::VectorXd &difference,
                                double resolution) {
  const double requested =
      std::ceil(difference.cwiseAbs().maxCoeff() / resolution);
  constexpr std::size_t maximum = std::numeric_limits<std::size_t>::max() - 1;
  if (!std::isfinite(requested) || requested >= static_cast<double>(maximum))
    return maximum;
  return std::max<std::size_t>(1, static_cast<std::size_t>(requested));
}

} // namespace holistic_motion::robotics::planning::detail
