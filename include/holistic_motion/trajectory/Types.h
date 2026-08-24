#pragma once

#include <cmath>
#include <list>
#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <fmt/ranges.h>

#include "holistic_motion/manif/LieGroup.h"
#include "holistic_motion/utility/Logging.h"
#include "holistic_motion/utility/Types.h"

namespace holistic_motion::robotics {

class Joint;

#define HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(group) \
    template class group<R1d>;                                  \
    template class group<R2d>;                                  \
    template class group<R3d>;                                  \
    template class group<R4d>;                                  \
    template class group<R5d>;                                  \
    template class group<R6d>;                                  \
    template class group<R7d>;                                  \
    template class group<R8d>;                                  \
    template class group<R9d>;                                  \
    template class group<R10d>;                                 \
    template class group<R11d>;                                 \
    template class group<R12d>;                                 \
    template class group<R13d>;                                 \
    template class group<R14d>;                                 \
    template class group<R15d>;                                 \
    template class group<R16d>;                                 \
    template class group<R17d>;                                 \
    template class group<R18d>;                                 \
    template class group<R19d>;                                 \
    template class group<R20d>;                                 \
    template class group<R21d>;                                 \
    template class group<R22d>;                                 \
    template class group<R23d>;                                 \
    template class group<R24d>;                                 \
    template class group<R25d>;                                 \
    template class group<R26d>;                                 \
    template class group<R27d>;                                 \
    template class group<R28d>;                                 \
    template class group<R29d>;                                 \
    template class group<R30d>;                                 \
    template class group<R31d>;                                 \
    template class group<R32d>;                                 \
    template class group<SE3d>;

inline constexpr double Epsilon = 1e-5;
inline constexpr double Small = 1e-8;

enum class PathType {
    NoBlend = 0,
    Bezier2ndJointSpace,
    Bezier5thJointSpace,
    Bezier2ndCartesianSpace,
    Bezier5thCartesianSpace,
};
enum class TrajType {
    DoubleS = 0,
    Trapezoidal = 1,
    Trapezium = Trapezoidal,  // Compatibility alias.
};
enum class PathSegType { LinearSeg = 0, Bezier2ndSeg, Bezier5thSeg };

struct TrajectorySeg {
    int seg_no{0};
    double timestamp{0.0};
    double pos{0.0};
    double vel{0.0};
    double acc{0.0};
    double jerk{0.0};

    TrajectorySeg() = default;
    TrajectorySeg(int segment,
                  double time,
                  double position,
                  double velocity,
                  double acceleration,
                  double segment_jerk)
        : seg_no(segment),
          timestamp(time),
          pos(position),
          vel(velocity),
          acc(acceleration),
          jerk(segment_jerk) {}
};

inline Eigen::VectorXd GetWeights(int dof, bool cartesian) {
    // Rotation must contribute to the path parameter so pure-orientation
    // Cartesian trajectories remain valid. Per-axis physical limits still
    // determine the final time scale.
    (void)cartesian;
    return Eigen::VectorXd::Ones(dof);
}

inline double WeightedNorm(const Eigen::VectorXd& target,
                           const Eigen::VectorXd& weights) {
    return std::sqrt((target.array().square() * weights.array()).sum());
}

template <typename Derived>
double WeightedNorm(const Eigen::MatrixBase<Derived>& target,
                    const Eigen::VectorXd& weights) {
    return std::sqrt((target.array().square() * weights.array()).sum());
}

template <typename Group>
double WeightedNorm(const Group& target, const Eigen::VectorXd& weights) {
    return std::sqrt(
            (target.Coeffs().array().square() * weights.array()).sum());
}

class TrajectoryConstraints {
public:
    TrajectoryConstraints() = default;
    explicit TrajectoryConstraints(const Eigen::VectorXd& max_velocity)
        : max_velocity_(max_velocity),
          max_acceleration_(Eigen::VectorXd::Ones(max_velocity.size())),
          max_jerk_(Eigen::VectorXd::Ones(max_velocity.size())) {}
    TrajectoryConstraints(const Eigen::VectorXd& max_velocity,
                          const Eigen::VectorXd& max_acceleration,
                          const Eigen::VectorXd& max_jerk)
        : max_velocity_(max_velocity),
          max_acceleration_(max_acceleration),
          max_jerk_(max_jerk) {}
    explicit TrajectoryConstraints(
            const std::vector<std::shared_ptr<Joint>>& joints);

    const Eigen::VectorXd& GetMaxVelocityConstraints() const {
        return max_velocity_;
    }
    const Eigen::VectorXd& GetMaxAccelerationConstraints() const {
        return max_acceleration_;
    }
    const Eigen::VectorXd& GetMaxJerkConstraints() const { return max_jerk_; }
    bool IsValid() const {
        return max_velocity_.size() > 0 &&
               max_velocity_.size() == max_acceleration_.size() &&
               max_velocity_.size() == max_jerk_.size() &&
               max_velocity_.allFinite() && max_acceleration_.allFinite() &&
               max_jerk_.allFinite() &&
               (max_velocity_.array() > 0.0).all() &&
               (max_acceleration_.array() > 0.0).all() &&
               (max_jerk_.array() > 0.0).all();
    }

private:
    Eigen::VectorXd max_velocity_;
    Eigen::VectorXd max_acceleration_;
    Eigen::VectorXd max_jerk_;
};

}  // namespace holistic_motion::robotics
