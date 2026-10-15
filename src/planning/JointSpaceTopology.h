#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <Eigen/Core>

namespace holistic_motion::robotics::planning::detail {

constexpr double kPi = 3.14159265358979323846;

inline double WrappedDifference(double difference) {
  return std::remainder(difference, 2.0 * kPi);
}

// Canonical periodic states occupy [lower, lower + 2*pi). At a coarse
// floating-point origin this interval can contain only one coordinate.
inline bool HasRepresentablePeriod(double lower, double upper) {
    return std::abs(lower) < 2.0 * kPi ||
           std::nextafter(lower, upper) - lower < 2.0 * kPi;
}

// Ordinary sums retain their original evaluation order. Below this bound,
// rounding an unweighted square to zero/subnormal can affect the weighted
// answer, so scale before taking a stable norm. The bound accounts for both
// unweighted-square and weighted-product underflow without squaring weights.
class WeightedGeometry {
  public:
    explicit WeightedGeometry(const Eigen::VectorXd &weights)
        : weights_(weights),
          minimum_squared_(
              (weights.array() * std::numeric_limits<double>::min()).sum() +
              weights.size() * std::numeric_limits<double>::min()) {}

    template <class Derived>
    double SquaredNorm(const Eigen::MatrixBase<Derived> &value) const {
        const double squared = RawSquaredNorm(value);
        if (UseOrdinary(squared) ||
            (squared == 0.0 && (value.array() == 0.0).all()))
            return squared;
        const double norm = ScaledNorm(value);
        return norm * norm;
    }

    template <class Derived>
    double Norm(const Eigen::MatrixBase<Derived> &value) const {
        const double squared = RawSquaredNorm(value);
        if (UseOrdinary(squared) ||
            (squared == 0.0 && (value.array() == 0.0).all()))
            return std::sqrt(squared);
        return ScaledNorm(value);
    }

    bool UseOrdinary(double squared) const {
        return std::isfinite(squared) && squared > minimum_squared_;
    }

  private:
    template <class Derived>
    double RawSquaredNorm(const Eigen::MatrixBase<Derived> &value) const {
        return (weights_.array() * value.array().square()).sum();
    }

    template <class Derived>
    double ScaledNorm(const Eigen::MatrixBase<Derived> &value) const {
        double norm = 0.0;
        for (Eigen::Index i = 0; i < value.size(); ++i)
            norm = std::hypot(norm, std::sqrt(weights_[i]) * value[i]);
        return norm;
    }

    const Eigen::VectorXd &weights_;
    const double minimum_squared_;
};

inline Eigen::VectorXd DefaultJointWeights(const Eigen::VectorXd &ranges) {
    Eigen::VectorXd weights = ranges.array().square().inverse().matrix();
    for (Eigen::Index i = 0; i < ranges.size(); ++i) {
        if (!(weights[i] > 0.0) || !std::isfinite(weights[i]))
            weights[i] = (1.0 / ranges[i]) / ranges[i];
    }
    return weights;
}

// Cache the topology once per planning request. Distance-only queries must not
// allocate a dynamic Eigen vector for each node in the search tree.
class JointSpaceMetric {
  public:
    JointSpaceMetric(const Eigen::VectorXd &weights,
                     const std::vector<bool> &continuous)
        : weights_(weights), geometry_(weights),
          continuous_(continuous.begin(), continuous.end()),
          has_continuous_(std::any_of(continuous.begin(), continuous.end(),
                                      [](bool value) { return value; })) {}

    Eigen::Index Dimension() const { return weights_.size(); }
    bool HasContinuous() const { return has_continuous_; }
    bool IsContinuous(Eigen::Index index) const {
        return continuous_[static_cast<std::size_t>(index)];
    }
    double SquaredAxisDistance(double delta, Eigen::Index index) const {
        const double squared = delta * delta;
        if (std::isnormal(squared) || delta == 0.0)
            return weights_[index] * squared;
        const double scaled = std::sqrt(weights_[index]) * delta;
        return scaled * scaled;
    }

    double SquaredDistance(const Eigen::VectorXd &from,
                           const Eigen::VectorXd &to) const {
        if (!has_continuous_)
            return geometry_.SquaredNorm(to - from);

        double squared_distance = 0.0;
        for (Eigen::Index index = 0; index < from.size(); ++index) {
            double delta = to[index] - from[index];
            // Nearby states already use the shortest arc; only distance-only
            // periodic queries need this fast path. Keep interpolation's
            // general difference helper small for the common bounded case.
            if (continuous_[static_cast<std::size_t>(index)] &&
                (delta < -kPi || delta > kPi))
                delta = WrappedDifference(delta);
            squared_distance += SquaredAxisDistance(delta, index);
        }
        if (geometry_.UseOrdinary(squared_distance))
            return squared_distance;
        const double norm = ScaledDistance(from, to);
        return norm * norm;
    }

    double Distance(const Eigen::VectorXd &from,
                    const Eigen::VectorXd &to) const {
        if (!has_continuous_)
            return geometry_.Norm(to - from);
        const double squared = SquaredDistance(from, to);
        if (std::isnormal(squared))
            return std::sqrt(squared);
        return ScaledDistance(from, to);
    }

    bool IsCloser(const Eigen::VectorXd &candidate,
                  const Eigen::VectorXd &selected, const Eigen::VectorXd &query,
                  double candidate_squared, double selected_squared,
                  bool prefer_equal = false) const {
        if (candidate_squared < selected_squared)
            return true;
        if (candidate_squared != selected_squared)
            return false;
        if (!std::isnormal(candidate_squared)) {
            const double candidate_distance = Distance(candidate, query);
            const double selected_distance = Distance(selected, query);
            if (candidate_distance != selected_distance)
                return candidate_distance < selected_distance;
        }
        return prefer_equal;
    }

    bool WithinRadius(const Eigen::VectorXd &state,
                      const Eigen::VectorXd &query, double squared_distance,
                      double radius, double squared_radius) const {
        // A normal squared radius is finite; comparison already rejects
        // infinite or NaN distances. Exceptional radii use actual norms.
        if (std::isnormal(squared_radius))
            return squared_distance <= squared_radius && radius >= 0.0;
        return Distance(state, query) <= radius;
    }

  private:
    double ScaledDistance(const Eigen::VectorXd &from,
                          const Eigen::VectorXd &to) const {
        double norm = 0.0;
        for (Eigen::Index i = 0; i < from.size(); ++i) {
            double delta = to[i] - from[i];
            if (continuous_[static_cast<std::size_t>(i)])
                delta = WrappedDifference(delta);
            norm = std::hypot(norm, std::sqrt(weights_[i]) * delta);
        }
        return norm;
    }

    const Eigen::VectorXd &weights_;
    const WeightedGeometry geometry_;
    // Distance queries visit every joint for every candidate. Cache bytes to
    // avoid unpacking vector<bool> bits in that inner loop.
    const std::vector<std::uint8_t> continuous_;
    const bool has_continuous_;
};

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
