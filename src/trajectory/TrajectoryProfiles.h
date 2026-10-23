#pragma once

#include "TrajectoryIntegration.h"
#include "holistic_motion/trajectory/Types.h"

namespace holistic_motion::robotics::detail {

// Scalar clocks are independent of the path's Lie-group dimension. Keep one
// implementation while the protected class methods remain compatible wrappers.
bool ComputeDoubleSProfile(const double &q0, const double &q1,
                           double &start_velocity, double &end_velocity,
                           const double &requested_max_velocity,
                           double max_acceleration, const double &max_jerk,
                           const double &t0, std::list<TrajectorySeg> &traj_seg,
                           const int &seg_no, const bool &allow_concave);
bool AlignDoubleSProfileAfter(const TrajectorySeg &previous,
                              std::list<TrajectorySeg> &profile);
bool ComputeTrapeziumProfile(const double &q0, const double &q1, double v0,
                             double &v1, const double &requested_max_velocity,
                             double max_acceleration, const double &t0,
                             std::list<TrajectorySeg> &traj_segs,
                             const int &seg_no);

inline TrajectorySeg AdvanceTrajectoryStep(const TrajectorySeg &segment,
                                           const double &t,
                                           const double &next_jerk,
                                           const int &next_seg_no) {
    double acc = segment.acc + segment.jerk * t;
    double vel =
        segment.vel + segment.acc * t + QuadraticContribution(segment.jerk, t);
    double pos =
        segment.pos + segment.vel * t + QuadraticContribution(segment.acc, t);
    if (segment.jerk != 0.0)
        pos += CubicJerkDisplacement(segment.jerk, t);
    return TrajectorySeg(next_seg_no, (segment.timestamp + t), pos, vel, acc,
                         next_jerk);
}

} // namespace holistic_motion::robotics::detail
