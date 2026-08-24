#pragma once

#include "holistic_motion/trajectory/TrajectoryBase.h"

namespace holistic_motion {
namespace robotics {

template <typename LieGroup>
class TrajectoryTrapezium : public TrajectoryBase<LieGroup> {
private:
    using Tangent = typename LieGroup::Tangent;

public:
    virtual ~TrajectoryTrapezium() {
        holistic_motion::utility::LogDebug("Destructing TrajectoryTrapezium.");
    };

    /// \brief Trapezium profile
    ///
    /// \param path path with init position between the end position
    /// \param velocity_limits
    /// \param acceleration_limits
    /// \param jerk_limits
    /// \param vel_init init velocity
    /// \param vel_end end velocity
    /// \param min_duration time control
    TrajectoryTrapezium(
            const std::shared_ptr<PathBase<LieGroup>>& path,
            const std::shared_ptr<TrajectoryConstraints>& constraints,
            const double& vel_init = .0,
            const double& vel_end = .0);

    /// \brief get the path
    ///
    /// \param s path parameter
    /// \return std::shared_ptr<PathSegmentBase>
    std::shared_ptr<PathBase<LieGroup>> GetPath() const { return this->path_; }

    /// \brief get the length of path of the trajectory
    ///
    /// \return double the path length
    double GetPathLength() const { return GetPath()->GetLength(); }

    // /// \brief get the waypoints of the path
    std::list<TrajectorySeg> GetWaypointSegments() const {
        return this->trajectory_segments_;
    };

protected:
    /// \brief compute max velocity of the segment
    ///
    /// \param segment
    /// \param velocity_limits
    /// \param acceleration_limits
    /// \param jerk_limits
    /// \return double
    double _ComputeSegmentMaxSVel(
            const std::shared_ptr<PathSegmentBase<LieGroup>>& segment,
            const Eigen::VectorXd& velocity_limits,
            const Eigen::VectorXd& acceleration_limits,
            const Eigen::VectorXd& jerk_limits);

    /// \brief compute trapezium trajectory
    ///
    /// \param q0 init position
    /// \param q1 end position
    /// \param v0 init velocity
    /// \param v1 end velocity
    /// \param max_velocity
    /// \param max_accelaration
    /// \param t0
    /// \param traj_seg
    /// \param seg_no the number of the segment
    /// \return bool
    bool _ComputeTrapeziumProfile(const double& q0,
                                  const double& q1,
                                  double v0,
                                  double& v1,
                                  const double& max_velocity,
                                  double max_acceleration,
                                  const double& t0,
                                  std::list<TrajectorySeg>& traj_seg,
                                  const int& seg_no);

    /// \brief compute trapezium trajectory
    /// According to the current location, timestamp, jerk
    /// to calculate the acceleration, speed, position of the next point
    ///
    /// \param segment trajectory segment
    /// \param t
    /// \param next_jerk next jerk
    /// \param next_seg_no the number of the next segment
    /// \return TrajectorySeg
    TrajectorySeg _ComputeNextTrajStep(const TrajectorySeg& traj_seg,
                                       const double& t,
                                       const double& next_jerk,
                                       const int& next_seg_no);

};

}  // namespace robotics
}  // namespace holistic_motion
