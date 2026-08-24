#pragma once
#include "holistic_motion/trajectory/PathBase.h"
#include "holistic_motion/trajectory/PathBezierCurve.h"
#include "holistic_motion/trajectory/Polynomial.h"

namespace holistic_motion {
namespace robotics {

// class TrajectoryBase
template <typename LieGroup>
class TrajectoryBase
    : public std::enable_shared_from_this<TrajectoryBase<LieGroup>> {
private:
    using Tangent = typename LieGroup::Tangent;

public:
    struct State {
        LieGroup position;
        Tangent velocity;
        Tangent acceleration;
        Tangent jerk;
    };

    struct ConstraintReport {
        Eigen::VectorXd peak_velocity;
        Eigen::VectorXd peak_acceleration;
        Eigen::VectorXd peak_jerk;
        Eigen::VectorXd velocity_utilization;
        Eigen::VectorXd acceleration_utilization;
        Eigen::VectorXd jerk_utilization;
        Eigen::VectorXd maximum_velocity_jump;
        Eigen::VectorXd maximum_acceleration_jump;
        double maximum_utilization{0.0};
        bool within_limits{false};
        bool velocity_continuous{false};
        bool acceleration_continuous{false};
    };

    virtual ~TrajectoryBase() {
        holistic_motion::utility::LogDebug("Destructing TrajectoryBase.");
    };

    /// \brief judge whether the path is valid
    bool IsValid() const { return this->valid_; };

    /// \brief Manually set whether this part of the trajectory is valid (use
    /// with caution externally)
    bool SetValid(const bool is_valid) {
        this->valid_ = is_valid;
        return true;
    };

    /// \brief get the duration of the trajectory
    double GetDuration() const {
        return trajectory_pspline_
                       ? trajectory_pspline_->GetLastTimeStamp() * time_scale_
                       : 0.0;
    };

    double GetTimeScale() const { return time_scale_; }

    std::vector<double> GetBreakpoints() const;

    const Eigen::VectorXd& GetMaxVelocity() const { return max_velocity_; }
    const Eigen::VectorXd& GetMaxAcceleration() const {
        return max_acceleration_;
    }
    const Eigen::VectorXd& GetMaxJerk() const { return max_jerk_; }

    /// Slow the trajectory to at least the requested duration. The path is
    /// unchanged and derivative limits remain satisfied.
    bool SetMinimumDuration(double duration);

    /// \brief get the position of the trajectory
    LieGroup GetPosition(double t) const;

    /// \brief get the velocity of the trajectory
    Tangent GetVelocity(double t) const;

    /// \brief get the acceleration of the trajectory
    Tangent GetAcceleration(double t) const;

    /// \brief get the jerk of the trajectory
    Tangent GetJerk(double t) const;

    /// Evaluate position and the first three time derivatives together.
    State GetState(double t) const;

    /// Sample the complete trajectory, including both sides of internal
    /// breakpoints, and summarize derivative-limit utilization.
    ConstraintReport GetConstraintReport(std::size_t samples = 2001) const;

protected:
    /// \brief Interpolates a given trajectory segment list into a PSpline.
    ///
    /// \param traj_segs A list of trajectory segments to be interpolated.
    /// \return A shared pointer to the resulting PSpline after interpolation.
    std::shared_ptr<PSpline> InterpolateToPSpline(
            const std::list<TrajectorySeg>& traj_segs) const;

    /// \brief Retrieves the velocity, acceleration, and jerk limits from the
    /// given trajectory constraints.
    ///
    /// \param velocity_limits A reference to a vector where the velocity limits
    /// will be stored.
    /// \param acceleration_limits A reference to a vector where the
    /// acceleration limits will be stored.
    /// \param jerk_limits A reference to a vector where the jerk limits will be
    /// stored.
    ///  \return Returns true if the
    /// limits were successfully retrieved, otherwise false.
    bool GetLimitFromConstraintProfile(
            const std::shared_ptr<TrajectoryConstraints>& constraints,
            Eigen::VectorXd& velocity_limits,
            Eigen::VectorXd& acceleration_limits,
            Eigen::VectorXd& jerk_limits);

    /// Apply one conservative global time scale after path/time-law
    /// composition. This accounts for curvature terms that are not bounded by
    /// the scalar path profile alone.
    bool EnforceJointLimits(const Eigen::VectorXd& velocity_limits,
                            const Eigen::VectorXd& acceleration_limits,
                            const Eigen::VectorXd& jerk_limits);

protected:
    TrajectoryBase() = default;

    size_t dof_{0};

    std::shared_ptr<PathBase<LieGroup>> path_;
    PathType path_type_;
    std::list<TrajectorySeg> trajectory_segments_;
    std::shared_ptr<PSpline> trajectory_pspline_;
    Eigen::VectorXd max_velocity_;
    Eigen::VectorXd max_acceleration_;
    Eigen::VectorXd max_jerk_;
    double time_scale_{1.0};
    bool valid_{false};  ///< if path is valid
};

}  // namespace robotics
}  // namespace holistic_motion
