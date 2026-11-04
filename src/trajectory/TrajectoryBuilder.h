#pragma once

#include <type_traits>
#include <utility>

#include "PathSegmentEvaluation.h"
#include "TrajectorySampling.h"
#include "holistic_motion/trajectory/TrajectoryBase.h"

namespace holistic_motion::robotics {

template <typename LieGroup> class TrajectoryDoubleS;
template <typename LieGroup> class TrajectoryTrapezium;

namespace detail {

// Own the path-to-clock construction sequence once. The concrete profiles keep
// their scalar algorithms and Double-S backtracking; no runtime policy or state
// is added to a trajectory.
template <typename LieGroup> class TrajectoryBuilder {
  public:
    template <typename Profile>
    static void Build(Profile &trajectory,
                      const std::shared_ptr<PathBase<LieGroup>> &path,
                      const std::shared_ptr<TrajectoryConstraints> &constraints,
                      double vel_init, double vel_end) {
        constexpr bool double_s =
            std::is_same_v<Profile, TrajectoryDoubleS<LieGroup>>;
        static_assert(double_s ||
                      std::is_same_v<Profile, TrajectoryTrapezium<LieGroup>>);

        holistic_motion::utility::LogDebug("Constructing trajectory...");
        if (!path) {
            holistic_motion::utility::LogWarning("The input path is null!");
            return;
        }
        if (!std::isfinite(vel_init) || !std::isfinite(vel_end) ||
            vel_init < 0.0 || vel_end < 0.0) {
            holistic_motion::utility::LogWarning(
                "Initial and final path velocities must be finite and "
                "non-negative");
            return;
        }
        trajectory.path_ = path;
        if constexpr (double_s) {
            const auto path_type = path->GetType();
            if (path_type != PathType::Bezier5thJointSpace &&
                path_type != PathType::Bezier5thCartesianSpace) {
                holistic_motion::utility::LogWarning(
                    "Trajectory DoubleS profile is used for "
                    "Bezier5thJointSpace "
                    "or Bezier5thCartesianSpace, but preset path type is {}",
                    to_underlying_type(path_type));
                return;
            }
        }
        if (!path->IsValid()) {
            holistic_motion::utility::LogWarning(
                "The input path is not valid!");
            return;
        }
        const double path_length = path->GetLength();
        if (path_length == 0.0) {
            holistic_motion::utility::LogDebug(
                "The input path length: {}, skip trajectory build!",
                path_length);
            return;
        }
        if (path_length < 0.0) {
            holistic_motion::utility::LogWarning(
                "The input path is not valid, path length: {}!", path_length);
            return;
        }

        Eigen::VectorXd velocity_limits, acceleration_limits, jerk_limits;
        if (!trajectory.GetLimitFromConstraintProfile(
                constraints, velocity_limits, acceleration_limits, jerk_limits))
            return;

        std::list<TrajectorySeg> segments, linear_profile;
        segments.emplace_back(0, 0.0, 0.0, vel_init, 0.0, 0.0);
        const int segment_count = path->GetNumOfPathSegments();
        int index = 0;
        while (index < segment_count) {
            holistic_motion::utility::LogDebug("[{:4.2f}%] computing.....",
                                               double(index) / segment_count *
                                                   100);
            TrajectorySeg previous = segments.back();
            const auto line = path->GetPathSegmentByIndex(index);
            const bool has_line =
                line->GetPathSegType() == PathSegType::LinearSeg;
            const int line_index = index;
            double end_position = 0.0, end_velocity = 0.0;
            double max_velocity = std::numeric_limits<double>::max();
            double max_acceleration = max_velocity, max_jerk = max_velocity;
            if (has_line) {
                const auto tangent =
                    line->GetTangent(line->GetStartParameter());
                for (std::size_t joint = 0; joint < trajectory.dof_; ++joint) {
                    const double direction = std::abs(tangent[joint]);
                    max_velocity = std::min(max_velocity,
                                            velocity_limits[joint] / direction);
                    max_acceleration =
                        std::min(max_acceleration,
                                 acceleration_limits[joint] / direction);
                    if constexpr (double_s)
                        max_jerk =
                            std::min(max_jerk, jerk_limits[joint] / direction);
                }
                end_position = previous.pos + line->GetLength();
                ++index;
                end_velocity = index == segment_count ? vel_end : 0.0;
            }

            double blend_length = 0.0;
            double blend_velocity = std::numeric_limits<double>::max();
            bool has_blend = false;
            while (index < segment_count &&
                   path->GetPathSegmentByIndex(index)->GetPathSegType() !=
                       PathSegType::LinearSeg) {
                const auto blend = path->GetPathSegmentByIndex(index);
                blend_length += blend->GetLength();
                blend_velocity =
                    std::min(blend_velocity,
                             ComputeSegmentMaxSVel(
                                 blend, velocity_limits, acceleration_limits,
                                 jerk_limits, trajectory.dof_));
                end_velocity = blend_velocity;
                has_blend = true;
                ++index;
            }

            if (has_blend) {
                // Rounded endpoint tangents can give adjacent geometry slightly
                // different caps. Use one shared speed before building either.
                if (has_line)
                    end_velocity = std::min(end_velocity, max_velocity);
                if (index < segment_count) {
                    const auto next_line = path->GetPathSegmentByIndex(index);
                    const auto tangent =
                        next_line->GetTangent(next_line->GetStartParameter());
                    for (std::size_t joint = 0; joint < trajectory.dof_;
                         ++joint)
                        if (tangent[joint] != 0.0)
                            end_velocity = std::min(
                                end_velocity, velocity_limits[joint] /
                                                  std::abs(tangent[joint]));
                }
            }

            if (has_line) {
                bool computed;
                if constexpr (double_s) {
                    computed = trajectory._ComputeDoubleSProfile(
                        previous.pos, end_position, previous.vel, end_velocity,
                        max_velocity, max_acceleration, max_jerk,
                        previous.timestamp, linear_profile, line_index,
                        line_index == 0 || index == segment_count);
                } else {
                    computed = trajectory._ComputeTrapeziumProfile(
                        previous.pos, end_position, previous.vel, end_velocity,
                        max_velocity, max_acceleration, previous.timestamp,
                        linear_profile, line_index);
                    if (!computed) {
                        holistic_motion::utility::LogWarning(
                            "Construct trapezium profile failed!");
                        return;
                    }
                }
                if (linear_profile.empty()) {
                    holistic_motion::utility::LogWarning(
                        "Profile produced no trajectory segments");
                    return;
                }
                if (linear_profile.back().seg_no == 0 &&
                    std::abs(previous.vel - vel_init) > Epsilon) {
                    holistic_motion::utility::LogWarning(
                        "Desired initial velocity {} is not in limits [0.0, "
                        "{}]",
                        vel_init, previous.vel);
                    return;
                }

                segments.pop_back();
                if constexpr (double_s) {
                    if (!computed) {
                        if (segments.empty()) {
                            holistic_motion::utility::LogWarning(
                                "Initial velocity cannot be reached without an "
                                "earlier segment");
                            return;
                        }
                        segments.back().vel = previous.vel;
                        if (!trajectory._ReverseWithMaxJerk(segments) ||
                            !trajectory._AlignProfileAfter(segments.back(),
                                                           linear_profile))
                            return;
                    }
                }
                segments.splice(segments.end(), linear_profile);
            }

            if (has_blend) {
                if (!std::isfinite(end_velocity) || end_velocity <= 0.0) {
                    holistic_motion::utility::LogWarning(
                        "Bezier segment has no finite positive path velocity");
                    return;
                }
                previous = segments.back();
                segments.emplace_back(
                    index - 1, previous.timestamp + blend_length / end_velocity,
                    previous.pos + blend_length, end_velocity, 0.0, 0.0);
            }
        }

        trajectory.trajectory_segments_ = std::move(segments);
        trajectory.trajectory_pspline_ =
            trajectory.InterpolateToPSpline(trajectory.trajectory_segments_);
        if (trajectory.trajectory_pspline_->GetKnots().size() < 2 ||
            trajectory.trajectory_pspline_->GetLastTimeStamp() <= 0.0) {
            holistic_motion::utility::LogWarning(
                "Trajectory interpolation produced no positive-duration "
                "segments");
            return;
        }
        if (!trajectory.InitializePhasePathSegments())
            return;
        trajectory.valid_ = true;
        if (!trajectory.EnforceJointLimits(velocity_limits, acceleration_limits,
                                           jerk_limits)) {
            holistic_motion::utility::LogWarning(
                "Trajectory contains non-finite derivatives");
            return;
        }
        holistic_motion::utility::LogDebug("Constructing succeed!\n");
    }

    static double ComputeSegmentMaxSVel(
        const std::shared_ptr<PathSegmentBase<LieGroup>> &segment,
        const Eigen::VectorXd &velocity_limits,
        const Eigen::VectorXd &acceleration_limits,
        const Eigen::VectorXd &jerk_limits, std::size_t dof) {
        if (!segment)
            return 0.0;
        PathSpeedLimit speed_limit;
        double s = segment->GetStartParameter();
        const double length = segment->GetLength();
        const double ep = s + length;
        if (!std::isfinite(s) || !std::isfinite(length) || !std::isfinite(ep) ||
            length < 0.0 || (length > 0.0 && ep <= s))
            return 0.0;
        const double step = std::min(0.01, length);
        // Keep the ordinary 0.01 grid; bound long intervals to 4096 steps
        // before the separate composed-trajectory limit check.
        constexpr int maximum_intervals = 4096;
        const bool normalized_grid = length > 0.01 * maximum_intervals;
        const double start = s;
        int sample = 0;
        SegmentEvaluationSampler<LieGroup> derivatives(*segment);
        while (true) {
            typename LieGroup::Tangent tangent, curvature, torsion;
            derivatives.Compute(s, tangent, curvature, torsion);
            for (std::size_t joint = 0; joint < dof; ++joint) {
                if (!std::isfinite(tangent[joint]) ||
                    !std::isfinite(curvature[joint]) ||
                    !std::isfinite(torsion[joint]))
                    return 0.0;
                speed_limit.AddVelocity(velocity_limits[joint], tangent[joint]);
                speed_limit.AddAcceleration(acceleration_limits[joint],
                                            curvature[joint]);
                speed_limit.AddJerk(jerk_limits[joint], torsion[joint]);
            }
            if (s == ep)
                break;
            ++sample;
            const double next =
                normalized_grid
                    ? (sample == maximum_intervals
                           ? ep
                           : std::min(ep,
                                      start +
                                          length * (sample /
                                                    double(maximum_intervals))))
                    : std::min(ep, s + step);
            // Reject a fixed step that rounds back to the same large parameter.
            if (next <= s)
                return 0.0;
            s = next;
        }
        return speed_limit.Get();
    }
};

} // namespace detail
} // namespace holistic_motion::robotics
