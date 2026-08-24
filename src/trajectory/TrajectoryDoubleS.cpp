#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

namespace holistic_motion {
namespace robotics {

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(TrajectoryDoubleS)

template <typename LieGroup>
TrajectoryDoubleS<LieGroup>::TrajectoryDoubleS(
        const std::shared_ptr<PathBase<LieGroup>> &path,
        const std::shared_ptr<TrajectoryConstraints> &constraints,
        const double &vel_init,
        const double &vel_end) {
    holistic_motion::utility::LogDebug("Constructing TrajectoryDoubleS...");
    if (!path) {
        holistic_motion::utility::LogWarning("The input path is null!");
        return;
    }
    if (!std::isfinite(vel_init) || !std::isfinite(vel_end) ||
        vel_init < 0.0 || vel_end < 0.0) {
        holistic_motion::utility::LogWarning(
                "Initial and final path velocities must be finite and non-negative");
        return;
    }
    this->path_ = path;
    auto path_type = this->path_->GetType();
    this->path_type_ = path_type;
    if (!(path_type == PathType::Bezier5thJointSpace ||
          path_type == PathType::Bezier5thCartesianSpace)) {
        holistic_motion::utility::LogWarning(
                "Trajectory DoubleS profile is used for "
                "Bezier5thJointSpace "
                "or Bezier5thCartesianSpace, but preset path type is {}",
                to_underlying_type(path_type));
        return;
    }

    if (!path->IsValid()) {
        holistic_motion::utility::LogWarning("The input path is not valid!");
        return;
    }
    auto path_length = path->GetLength();
    if (0 == path_length) {
        holistic_motion::utility::LogDebug(
                "The input path length: {}, skip trajectory build!",
                path_length);
        return;
    } else if (path_length < 0) {
        holistic_motion::utility::LogWarning(
                "The input path is not valid, path length: {}!", path_length);
        return;
    }

    double bezier_velocity{0.0};  ///< the velocity of bezier segment
    double bezier_length{0.0};    ///< the length of bezier segment

    TrajectorySeg pre_seg;
    double &pre_time = pre_seg.timestamp;  ///< start time
    double &pre_pos = pre_seg.pos;         ///< start position
    double &pre_vel = pre_seg.vel;         ///< start velocity
    double end_pos{0.0};                   // end position
    double end_vel{0.0};                   // end velocity

    double max_vel{0.0};   ///< max velocity
    double max_acc{0.0};   ///< max acceleration
    double max_jerk{0.0};  ///< max jerk

    std::shared_ptr<PathSegmentBase<LieGroup>> linear_seg;
    std::shared_ptr<PathSegmentBase<LieGroup>> blend_seg;
    std::list<TrajectorySeg> traj_segs, linear_segs;

    Eigen::VectorXd velocity_limits, acceleration_limits, jerk_limits;
    if (!this->GetLimitFromConstraintProfile(
                constraints, velocity_limits, acceleration_limits,
                jerk_limits)) {
        return;
    }

    // update the limits
    auto max_vel_list = velocity_limits;
    auto max_acc_list = acceleration_limits;
    auto max_jerk_list = jerk_limits;

    traj_segs.push_back(TrajectorySeg(0, .0, .0, vel_init, .0, .0));

    int i = 0;
    int num_of_segments = this->path_->GetNumOfPathSegments();
    holistic_motion::utility::LogDebug("Num of path segments: {}", num_of_segments);
    while (i < num_of_segments) {
        holistic_motion::utility::LogDebug("[{:4.2f}%] computing.....",
                                (double)i / num_of_segments * 100);
        bool is_cur_linear_segment = false;
        bool is_next_bezier_segment = false;

        linear_seg = this->path_->GetPathSegmentByIndex(i);

        double sp_of_linear_seg = linear_seg->GetStartParameter();
        pre_seg = traj_segs.back();

        int bezier_segs_number = 0;
        if ((is_cur_linear_segment = (linear_seg->GetPathSegType() ==
                                      PathSegType::LinearSeg))) {
            auto tangent = linear_seg->GetTangent(sp_of_linear_seg);
            max_vel = (std::numeric_limits<double>::max)();
            max_acc = max_vel;
            max_jerk = max_vel;
            // save the max limits
            for (size_t i = 0; i < this->dof_; ++i) {
                max_vel = std::min(max_vel,
                                   velocity_limits[i] / std::abs(tangent[i]));
                max_acc = std::min(
                        max_acc, acceleration_limits[i] / std::abs(tangent[i]));
                max_jerk = std::min(max_jerk,
                                    jerk_limits[i] / std::abs(tangent[i]));
            }

            end_pos = pre_pos + linear_seg->GetLength();
            i++;
            end_vel = (i == num_of_segments) ? vel_end : 0.0;
        }

        // need to find the next Bezier5th svel if possible
        bezier_velocity = std::numeric_limits<double>::max();
        bezier_length = 0.0;

        // to find the bezier curve segment after the linear segment
        while (i < num_of_segments &&
               this->path_->GetPathSegmentByIndex(i)->GetPathSegType() !=
                       PathSegType::LinearSeg) {
            blend_seg = this->path_->GetPathSegmentByIndex(i);
            bezier_length += blend_seg->GetLength();
            bezier_velocity = std::min(
                    bezier_velocity,
                    _ComputeSegmentMaxSVel(blend_seg, max_vel_list,
                                           max_acc_list, max_jerk_list));
            end_vel = bezier_velocity;
            is_next_bezier_segment = true;
            bezier_segs_number++;
            i++;
        }

        if (is_cur_linear_segment) {
            holistic_motion::utility::LogDebug(
                    "DoubleS profile:[ pre_pos:{}, end_pos:{}, "
                    "pre_vel:{}, end_vel:{}, max_vel:{} max_acc:{}, "
                    "max_jerk:{}, pre_time:{} ]",
                    pre_pos, end_pos, pre_vel, end_vel, max_vel, max_acc,
                    max_jerk, pre_time);
            bool allow_concave = false;
            int seg_no = i - 1 - bezier_segs_number;
            if (seg_no == 0 || i == num_of_segments) {  // initial/final segment
                allow_concave = true;
            }

            // compute doubleS profile
            bool res = _ComputeDoubleSProfile(
                    pre_pos, end_pos, pre_vel, end_vel, max_vel, max_acc,
                    max_jerk, pre_time, linear_segs, seg_no, allow_concave);

            if (linear_segs.empty()) {
                holistic_motion::utility::LogWarning(
                        "Double-S profile produced no trajectory segments");
                return;
            }

            if (linear_segs.back().seg_no == 0 &&
                std::abs(pre_vel - vel_init) > Epsilon) {
                holistic_motion::utility::LogWarning(
                        "Desired initial velocity {} is "
                        "not in limits [0.0, {}]",
                        vel_init, pre_vel);
                return;
            }

            // compute doubleS profile failed for pre_vel is too large
            traj_segs.pop_back();

            if (!res) {
                traj_segs.back().vel = pre_vel;
                holistic_motion::utility::LogDebug(
                        "Compute doubleS profile with reverse max jerk");
                if (!_ReverseWithMaxJerk(traj_segs)) {
                    holistic_motion::utility::LogWarning(
                            "Compute doubleS profile failed!");
                    return;
                }
                double dt = (linear_segs.front().pos - traj_segs.back().pos) /
                                    traj_segs.back().vel -
                            (linear_segs.front().timestamp -
                             traj_segs.back().timestamp);
                for (auto &step : linear_segs) {
                    step.timestamp += dt;
                }
            }

            traj_segs.splice(traj_segs.end(), linear_segs);
        }

        if (is_next_bezier_segment) {
            if (!std::isfinite(end_vel) || end_vel <= 0.0) {
                holistic_motion::utility::LogWarning(
                        "Bezier segment has no finite positive path velocity");
                return;
            }
            pre_seg = traj_segs.back();
            traj_segs.push_back(
                    TrajectorySeg(i - 1, pre_time + bezier_length / end_vel,
                                  pre_pos + bezier_length, end_vel, 0.0, 0.0));
        }
    }
    holistic_motion::utility::LogDebug("[100%] computed complete! ");

    this->trajectory_segments_ = traj_segs;
    this->trajectory_pspline_ = this->InterpolateToPSpline(traj_segs);
    if (this->trajectory_pspline_->GetKnots().size() < 2 ||
        this->trajectory_pspline_->GetLastTimeStamp() <= 0.0) {
        holistic_motion::utility::LogWarning(
                "Trajectory interpolation produced no positive-duration segments");
        return;
    }
    this->valid_ = true;
    if (!this->EnforceJointLimits(
                velocity_limits, acceleration_limits, jerk_limits)) {
        holistic_motion::utility::LogWarning(
                "Trajectory contains non-finite derivatives");
        return;
    }
    holistic_motion::utility::LogDebug("Constructing succeed!\n");
}

template <typename LieGroup>
double TrajectoryDoubleS<LieGroup>::_ComputeSegmentMaxSVel(
        const std::shared_ptr<PathSegmentBase<LieGroup>> &segment,
        const Eigen::VectorXd &velocity_limits,
        const Eigen::VectorXd &acceleration_limits,
        const Eigen::VectorXd &jerk_limits) {
    // for a general segment segment, especially Bezier5th (whose max
    // tangent/curvature is hard to calculated analytically) use s(t) = kt, k is
    // selected to comply with dq, ddq limits
    double m = std::numeric_limits<double>::max();
    double s = segment->GetStartParameter();
    double ep = segment->GetLength() + s;
    double step = 0.01;
    if (segment->GetLength() <= step) {
        step = segment->GetLength();
    }

    bool quit_loop = false;
    while (!quit_loop) {
        if (s >= ep) {
            s = ep;
            quit_loop = true;
        }
        auto tangent = segment->GetTangent(s);
        auto curvature = segment->GetCurvature(s);
        auto torsion = segment->GetTorsion(s);
        for (size_t i = 0; i < this->dof_; i++) {
            if (std::abs(tangent[i]) > Epsilon) {
                m = std::min(m, velocity_limits[i] / std::abs(tangent[i]));
            }
            if (std::abs(curvature[i]) > Epsilon) {
                m = std::min(m, std::sqrt(acceleration_limits[i] /
                                          std::abs(curvature[i])));
            }
            if (std::abs(torsion[i]) > Epsilon) {
                m = std::min(m, std::pow(jerk_limits[i] / std::abs(torsion[i]),
                                         1.0 / 3.0));
            }
        }
        s += step;
    }

    return m;
}

template <typename LieGroup>
bool TrajectoryDoubleS<LieGroup>::_ComputeDoubleSProfile(
        const double &q0,
        const double &q1,
        double &v0,
        double &v1,
        const double &max_velocity,
        double max_acceleration,
        const double &max_jerk,
        const double &t0,
        std::list<TrajectorySeg> &traj_seg,
        const int &seg_no,
        const bool &allow_concave) {
    holistic_motion::utility::LogDebug(
            "Compute q0:{}, q1:{}, v0:{}, v1:{}, max_velocity:{} "
            "max_acceleration:{}, max_jerk:{}, seg_no:{}, allow_concave:{}",
            q0, q1, v0, v1, max_velocity, max_acceleration, max_jerk, seg_no,
            allow_concave);

    traj_seg.clear();
    if (!std::isfinite(q0) || !std::isfinite(q1) || !std::isfinite(v0) ||
        !std::isfinite(v1) || !std::isfinite(max_velocity) ||
        !std::isfinite(max_acceleration) || !std::isfinite(max_jerk) ||
        !std::isfinite(t0) || q1 < q0 || v0 < -Epsilon || v1 < -Epsilon ||
        max_velocity <= 0.0 || max_acceleration <= 0.0 || max_jerk <= 0.0) {
        holistic_motion::utility::LogWarning(
                "Invalid Double-S inputs: q0={}, q1={}, v0={}, v1={}, "
                "vmax={}, amax={}, jmax={}, t0={}",
                q0, q1, v0, v1, max_velocity, max_acceleration, max_jerk,
                t0);
        return false;
    }
    v0 = std::max(0.0, v0);
    v1 = std::max(0.0, v1);
    double acc_init = .0;

    // duration
    ///< jerk_accel_time constant duration of the jerk in the acceleration phase
    ///< ta acceleration period
    ///< tv constant speed period
    ///< jerk_decel_time constant duration of the jerk in the deceleration phase
    ///< td deceleration period
    ///< delta to compute
    double jerk_accel_time{0.0}, ta{0.0}, tv{0.0};
    double jerk_decel_time{0.0}, td{0.0}, delta{0.0}, jerk_time{0.0};
    bool feasible = false;
    bool need_reduce_v0 = v0 > max_velocity ? true : false;
    // const double v0_init = v0;

    // const double max_acceleration_const = max_acceleration;
    const double lower_scale = 0.9;

    bool v0_gt_max_velocity = false;
    bool v1_gt_max_velocity = false;

    if (!allow_concave) {
        v0 = v0 > max_velocity ? max_velocity : v0;
        v1 = v1 > max_velocity ? max_velocity : v1;
    } else {
        if (v0 > max_velocity) v0_gt_max_velocity = true;
        if (v1 > max_velocity) v1_gt_max_velocity = true;
    }

    /// < 1.the first case
    int i = 0;
    while (!feasible) {
        double max_acc_reach_1 = sqrt(std::abs(v1 - v0) / max_jerk);
        double max_acc_reach_2 = max_acceleration / max_jerk;

        // acceleration can reach the max, there is zero acc period
        auto reachable_jerk_time = std::min(max_acc_reach_1, max_acc_reach_2);
        if (reachable_jerk_time < max_acc_reach_2) {
            if (q1 - q0 >= reachable_jerk_time * (v0 + v1)) {
                feasible = true;
            }
        }
        // The deceleration-side jerk limit is active.
        else if (q1 - q0 >=
                 0.5 * (v0 + v1) *
                         (reachable_jerk_time + std::abs(v1 - v0) / max_acceleration)) {
            feasible = true;
        }
        //
        if (!feasible) {
            // make v1 and v0 more close to feasibe double s curve
            if (v1 > v0) {
                v1 += (v0 - v1) * 0.1;  // v1 is too large
            }
            if (v1 < v0) {
                v0 += (v1 - v0) * 0.01;  // v0 is tool large
                need_reduce_v0 = true;
            }
        }
        if (std::abs(v1 - v0) < Epsilon || i++ > 1000) {
            break;
        }
    }

    if (need_reduce_v0 && !allow_concave) {
        holistic_motion::utility::LogDebug("Need to reduce {}th segment V0( init speed )!",
                                seg_no);
    }

    // compute acceleration period jerk_accel_time
    if (std::abs(max_velocity - v0) * max_jerk <
        max_acceleration * max_acceleration) {
        // if max_acceleration is not reach
        jerk_accel_time = std::sqrt(std::abs(max_velocity - v0) / max_jerk);
        ta = 2 * jerk_accel_time;
    } else {
        jerk_accel_time = max_acceleration / max_jerk;
        ta = jerk_accel_time + std::abs(max_velocity - v0) / max_acceleration;
    }

    // compute deceleration period jerk_decel_time
    if (std::abs(max_velocity - v1) * max_jerk <
        max_acceleration * max_acceleration) {
        // if min_acceleration is not reach
        jerk_decel_time = std::sqrt(std::abs(max_velocity - v1) / max_jerk);
        td = 2 * jerk_decel_time;
    } else {
        jerk_decel_time = max_acceleration / max_jerk;
        td = jerk_decel_time + std::abs(max_velocity - v1) / max_acceleration;
    }

    // compute constant speed period
    tv = (q1 - q0) / max_velocity - 0.5 * ta * (1 + v0 / max_velocity) -
         0.5 * td * (1 + v1 / max_velocity);

    /// < 2.the second case, if tv <= 0.0, there is no constant speed period
    constexpr int maximum_acceleration_reductions = 1000;
    int acceleration_reductions = 0;
    while (tv <= 0.0) {
        jerk_accel_time = jerk_decel_time = jerk_time =
                max_acceleration / max_jerk;
        delta = std::sqrt(
                std::pow(max_acceleration, 4) / std::pow(max_jerk, 2) +
                2 * (v0 * v0 + v1 * v1) +
                max_acceleration *
                        (4 * (q1 - q0) -
                         2 * max_acceleration / max_jerk * (v0 + v1)));
        ta = (max_acceleration * max_acceleration / max_jerk - 2 * v0 + delta) /
             (2 * max_acceleration);
        td = (max_acceleration * max_acceleration / max_jerk - 2 * v1 + delta) /
             (2 * max_acceleration);
        tv = 0.0;

        holistic_motion::utility::LogDebug(
                "There is no constant speed period: jerk_accel_time:{}, ta:{}, tv:{}, "
                "td:{}, jerk_decel_time: {}, duration:{}",
                jerk_accel_time, ta, tv, td, jerk_decel_time, ta + tv + td);

        // ta or td is negative during the recursion process
        if (ta < 0.0)  // if ta < 0.0, there is only a deceleration period
        {
            td = 2 * (q1 - q0) / (v1 + v0);
            jerk_decel_time = (max_jerk * (q1 - q0) -
                   std::sqrt(std::abs(max_jerk *
                                      (max_jerk * std::pow(q1 - q0, 2) +
                                       std::pow(v1 + v0, 2) * (v1 - v0))))) /
                  (max_jerk * (v1 + v0));
            ta = jerk_accel_time = 0.0;
            break;
        }

        if (td < 0.0)  // if td < 0.0, there is only a acceleration period
        {
            ta = 2 * (q1 - q0) / (v1 + v0);
            jerk_accel_time = (max_jerk * (q1 - q0) -
                   std::sqrt(std::abs(max_jerk *
                                      (max_jerk * std::pow(q1 - q0, 2) -
                                       std::pow(v1 + v0, 2) * (v1 - v0))))) /
                  (max_jerk * (v1 + v0));
            td = jerk_decel_time = 0.0;
            break;
        }
        // TODO: need to adjust the maximum acceleration ratio to improve the
        // calculation success rate
        if ((ta < 2 * jerk_time) || (td < 2 * jerk_time)) {
            max_acceleration *= lower_scale;  // amax is not reached, set a
                                              // lower amax limit;
            if (++acceleration_reductions > maximum_acceleration_reductions ||
                !std::isfinite(max_acceleration) ||
                max_acceleration <= std::numeric_limits<double>::min()) {
                return false;
            }
            holistic_motion::utility::LogDebug("max_acceleration lower to [{}]",
                                    max_acceleration);
        } else {
            break;
        }
    }

    holistic_motion::utility::LogDebug(
            "DoubleS time, jerk_accel_time:{}, ta:{}, tv:{}, jerk_decel_time:{}, td:{} "
            "delta:{}, jerk_time:{}, duration:{}",
            jerk_accel_time, ta, tv, jerk_decel_time, td, delta, jerk_time,
            ta + tv + td);

    holistic_motion::utility::LogDebug("Begin to add trajectory seg...");

    if (allow_concave && v0_gt_max_velocity) {
        traj_seg.push_back(TrajectorySeg(
                seg_no, t0, q0, v0, acc_init,
                -max_jerk));  // reach max_jerk until reach max_acceleration
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), jerk_accel_time, 0.0,
                seg_no)));  // reach max_acceleration and the jerk = 0
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), (ta - 2.0 * jerk_accel_time), max_jerk,
                seg_no)));  // reach -max_jerk until zero acceleration and
                            // max_velocity
    } else {
        traj_seg.push_back(TrajectorySeg(
                seg_no, t0, q0, v0, acc_init,
                max_jerk));  // reach max_jerk until reach max_acceleration
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), jerk_accel_time, 0.0,
                seg_no)));  // reach max_acceleration and the jerk = 0
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), (ta - 2.0 * jerk_accel_time), -max_jerk,
                seg_no)));  // reach -max_jerk until zero acceleration and
                            // max_velocity
    }

    traj_seg.push_back(_ComputeNextTrajStep(traj_seg.back(), jerk_accel_time, 0.0,
                                            seg_no));  // vel = vmax

    if (allow_concave && v1_gt_max_velocity) {
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), tv, max_jerk,
                seg_no)));  // reach max_jerk until reach max_acceleration
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), jerk_decel_time, 0.0,
                seg_no)));  // reach max_acceleration and the jerk = 0
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), (td - 2.0 * jerk_decel_time), -max_jerk,
                seg_no)));  // reach -max_jerk until zero acceleration and
                            // max_velocity
    } else {
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), tv, -max_jerk,
                seg_no)));  // reach max_jerk until reach max_acceleration
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), jerk_decel_time, 0.0,
                seg_no)));  // reach max_acceleration and the jerk = 0
        traj_seg.push_back(TrajectorySeg(_ComputeNextTrajStep(
                traj_seg.back(), (td - 2.0 * jerk_decel_time), max_jerk,
                seg_no)));  // reach -max_jerk until zero acceleration and
                            // max_velocity
    }

    traj_seg.push_back(_ComputeNextTrajStep(traj_seg.back(), jerk_decel_time, 0.0,
                                            seg_no));  // vel = vmax

    holistic_motion::utility::LogDebug("Finsh to Add trajectory seg!");

    return !need_reduce_v0 || allow_concave;
}

template <typename LieGroup>
TrajectorySeg TrajectoryDoubleS<LieGroup>::_ComputeNextTrajStep(
        const TrajectorySeg &segment,
        const double &t,
        const double &next_jerk,
        const int &next_seg_no) {
    double acc = segment.acc + segment.jerk * t;
    double vel = segment.vel + segment.acc * t + 0.5 * segment.jerk * t * t;
    double pos = segment.pos + segment.vel * t + 0.5 * segment.acc * t * t +
                 segment.jerk * std::pow(t, 3) / 6.0;
    // holistic_motion::utility::LogDebug("TrajectorySeg << seg_no:{}, t:{}, pos:{},
    // vel:{}, acc:{}", next_seg_no, (segment.timestamp + t), pos, vel, acc,
    //                      next_jerk);
    return TrajectorySeg(next_seg_no, (segment.timestamp + t), pos, vel, acc,
                         next_jerk);
}

template <typename LieGroup>
bool TrajectoryDoubleS<LieGroup>::_ReverseWithMaxJerk(
        std::list<TrajectorySeg> &traj_seg) {
    holistic_motion::utility::LogDebug("_AdjustReverseWithMaxJerk");
    const int linear_jerk_step_count =
            8;  // one linear segment have 8 trajectory steps typically
    if (traj_seg.size() < linear_jerk_step_count) {
        return false;
    }
    // pop up the linear segment traj steps and re-compute
    int seg_no = traj_seg.back().seg_no;
    double q0{0.0}, q1{0.0}, v0{0.0}, v1{0.0}, vmax{0.0}, amax{0.0}, jmax{0.0},
            t0{0.0};
    for (int i = 0; i < linear_jerk_step_count; i++) {
        auto step = traj_seg.end();
        step--;
        if (step->seg_no != seg_no) {
            break;
        }
        if (i == 0) {
            q1 = step->pos;
            v1 = step->vel;
        } else if (i == linear_jerk_step_count - 1) {
            q0 = step->pos;
            v0 = step->vel;
            t0 = step->timestamp;
        }
        vmax = std::max(std::abs(step->vel), vmax);
        amax = std::max(std::abs(step->acc), amax);
        jmax = std::max(std::abs(step->jerk), jmax);
        traj_seg.pop_back();
    }
    std::list<TrajectorySeg> seg_traj_seg;
    if (_ComputeDoubleSProfile(q0, q1, v0, v1, vmax, amax, jmax, t0,
                               seg_traj_seg, seg_no)) {
        traj_seg.splice(traj_seg.end(), seg_traj_seg);
    } else {
        if (traj_seg.size() < linear_jerk_step_count) {
            return false;
        }
        traj_seg.back().vel = v0;
        if (_ReverseWithMaxJerk(traj_seg)) {
            double dt = (seg_traj_seg.front().pos - traj_seg.back().pos) /
                                traj_seg.back().vel -
                        (seg_traj_seg.front().timestamp -
                         traj_seg.back().timestamp);
            for (auto &step : seg_traj_seg) step.timestamp += dt;
            traj_seg.splice(traj_seg.end(), seg_traj_seg);
        } else {
            return false;
        }
    }
    return true;
}

}  // namespace robotics
}  // namespace holistic_motion
