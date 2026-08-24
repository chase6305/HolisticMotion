#include "holistic_motion/trajectory/TrajectoryBase.h"

namespace holistic_motion {
namespace robotics {

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(TrajectoryBase)

template <typename LieGroup>
std::vector<double> TrajectoryBase<LieGroup>::GetBreakpoints() const {
    if (!trajectory_pspline_) return {};
    std::vector<double> result = trajectory_pspline_->GetKnots();
    for (double& time : result) time *= time_scale_;
    return result;
}

template <typename LieGroup>
bool TrajectoryBase<LieGroup>::SetMinimumDuration(double duration) {
    if (!std::isfinite(duration) || duration < 0.0 || !trajectory_pspline_) {
        return false;
    }
    const double current_duration = GetDuration();
    if (duration > current_duration) {
        const double requested_scale =
                duration / trajectory_pspline_->GetLastTimeStamp();
        if (!std::isfinite(requested_scale)) return false;
        time_scale_ = requested_scale;
    }
    return true;
}

template <typename LieGroup>
LieGroup TrajectoryBase<LieGroup>::GetPosition(double t) const {
    if (!valid_ || !trajectory_pspline_ || !path_) {
        throw std::logic_error("cannot query an invalid trajectory");
    }
    if (!std::isfinite(t)) {
        throw std::invalid_argument("trajectory time must be finite");
    }
    t = clamp(t, 0.0, GetDuration());
    t /= time_scale_;

    holistic_motion::utility::LogDebug("[GetPosition], t:{}", t);

    double position_at_t = this->trajectory_pspline_->ComputeValueAtS(t);
    holistic_motion::utility::LogDebug(
            "[GetPosition] After ComputeValueAtS, position_at_t:{}",
            position_at_t);

    auto p = this->path_->GetConfig(position_at_t);
    return p;
}

template <typename LieGroup>
typename LieGroup::Tangent TrajectoryBase<LieGroup>::GetVelocity(
        double t) const {
    if (!valid_ || !trajectory_pspline_ || !path_) {
        throw std::logic_error("cannot query an invalid trajectory");
    }
    if (!std::isfinite(t)) {
        throw std::invalid_argument("trajectory time must be finite");
    }
    t = clamp(t, 0.0, GetDuration());
    t /= time_scale_;
    holistic_motion::utility::LogDebug("[GetVelocity], t:{}", t);

    double position_at_t = this->trajectory_pspline_->ComputeValueAtS(t);
    holistic_motion::utility::LogDebug(
            "[GetVelocity] After ComputeValueAtS, position_at_t:{}",
            position_at_t);

    auto tangent = this->path_->GetTangent(position_at_t);
    auto v = tangent * this->trajectory_pspline_->ComputeValueAtS(t, 1) /
             time_scale_;
    return v;
}

template <typename LieGroup>
typename LieGroup::Tangent TrajectoryBase<LieGroup>::GetAcceleration(
        double t) const {
    if (!valid_ || !trajectory_pspline_ || !path_) {
        throw std::logic_error("cannot query an invalid trajectory");
    }
    if (!std::isfinite(t)) {
        throw std::invalid_argument("trajectory time must be finite");
    }
    t = clamp(t, 0.0, GetDuration());
    t /= time_scale_;

    holistic_motion::utility::LogDebug("[GetAcceleration], t:{}", t);

    double position_at_t = this->trajectory_pspline_->ComputeValueAtS(t);
    holistic_motion::utility::LogDebug(
            "[GetAcceleration] After ComputeValueAtS, position_at_t:{}",
            position_at_t);
    auto curvature = this->path_->GetCurvature(position_at_t);
    auto tangent = this->path_->GetTangent(position_at_t);
    double velocity_at_t = this->trajectory_pspline_->ComputeValueAtS(t, 1);
    double acceleration_at_t = this->trajectory_pspline_->ComputeValueAtS(t, 2);
    auto c = tangent * acceleration_at_t +
             curvature * std::pow(velocity_at_t, 2);
    return c / std::pow(time_scale_, 2);
}

template <typename LieGroup>
typename LieGroup::Tangent TrajectoryBase<LieGroup>::GetJerk(double t) const {
    if (!valid_ || !trajectory_pspline_ || !path_) {
        throw std::logic_error("cannot query an invalid trajectory");
    }
    if (!std::isfinite(t)) {
        throw std::invalid_argument("trajectory time must be finite");
    }
    t = clamp(t, 0.0, GetDuration());
    t /= time_scale_;
    const double position = trajectory_pspline_->ComputeValueAtS(t);
    const double velocity = trajectory_pspline_->ComputeValueAtS(t, 1);
    const double acceleration = trajectory_pspline_->ComputeValueAtS(t, 2);
    const double jerk = trajectory_pspline_->ComputeValueAtS(t, 3);
    const auto tangent = path_->GetTangent(position);
    const auto curvature = path_->GetCurvature(position);
    const auto torsion = path_->GetTorsion(position);
    return (tangent * jerk + 3.0 * curvature * velocity * acceleration +
            torsion * std::pow(velocity, 3)) /
           std::pow(time_scale_, 3);
}

template <typename LieGroup>
typename TrajectoryBase<LieGroup>::State TrajectoryBase<LieGroup>::GetState(
        double t) const {
    if (!valid_ || !trajectory_pspline_ || !path_) {
        throw std::logic_error("cannot query an invalid trajectory");
    }
    if (!std::isfinite(t)) {
        throw std::invalid_argument("trajectory time must be finite");
    }
    t = clamp(t, 0.0, GetDuration()) / time_scale_;
    const auto path_jet = trajectory_pspline_->ComputeJetAtS(t);
    const double path_position = path_jet[0];
    const double path_velocity = path_jet[1];
    const double path_acceleration = path_jet[2];
    const double path_jerk = path_jet[3];
    const auto tangent = path_->GetTangent(path_position);
    const auto curvature = path_->GetCurvature(path_position);
    const auto torsion = path_->GetTorsion(path_position);
    const double inverse_scale = 1.0 / time_scale_;

    State state;
    state.position = path_->GetConfig(path_position);
    state.velocity = tangent * path_velocity * inverse_scale;
    state.acceleration =
            (tangent * path_acceleration +
             curvature * std::pow(path_velocity, 2)) *
            std::pow(inverse_scale, 2);
    state.jerk =
            (tangent * path_jerk +
             3.0 * curvature * path_velocity * path_acceleration +
             torsion * std::pow(path_velocity, 3)) *
            std::pow(inverse_scale, 3);
    return state;
}

template <typename LieGroup>
typename TrajectoryBase<LieGroup>::ConstraintReport
TrajectoryBase<LieGroup>::GetConstraintReport(std::size_t samples) const {
    if (!valid_ || !trajectory_pspline_ || !path_) {
        throw std::logic_error("cannot inspect an invalid trajectory");
    }
    if (samples < 2) {
        throw std::invalid_argument("constraint report requires at least 2 samples");
    }

    ConstraintReport report;
    report.peak_velocity = Eigen::VectorXd::Zero(dof_);
    report.peak_acceleration = Eigen::VectorXd::Zero(dof_);
    report.peak_jerk = Eigen::VectorXd::Zero(dof_);
    report.maximum_velocity_jump = Eigen::VectorXd::Zero(dof_);
    report.maximum_acceleration_jump = Eigen::VectorXd::Zero(dof_);
    const auto accumulate_state = [&](const State& state) {
        report.peak_velocity = report.peak_velocity.cwiseMax(
                state.velocity.Coeffs().cwiseAbs());
        report.peak_acceleration = report.peak_acceleration.cwiseMax(
                state.acceleration.Coeffs().cwiseAbs());
        report.peak_jerk = report.peak_jerk.cwiseMax(
                state.jerk.Coeffs().cwiseAbs());
    };
    const auto accumulate = [&](double time) {
        accumulate_state(GetState(time));
    };

    const double duration = GetDuration();
    for (std::size_t sample = 0; sample < samples; ++sample) {
        accumulate(duration * static_cast<double>(sample) /
                   static_cast<double>(samples - 1));
    }
    const auto breakpoints = GetBreakpoints();
    for (std::size_t index = 1; index + 1 < breakpoints.size(); ++index) {
        const double breakpoint = breakpoints[index];
        const double previous = breakpoints[index - 1];
        const double floating_offset =
                256.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, duration);
        const double left_offset = std::min(
                0.5 * (breakpoint - previous),
                std::max(floating_offset,
                         1e-9 * (breakpoint - previous)));
        const auto left_state = GetState(breakpoint - left_offset);
        const auto right_state = GetState(breakpoint);
        accumulate_state(left_state);
        accumulate_state(right_state);
        report.maximum_velocity_jump = report.maximum_velocity_jump.cwiseMax(
                (right_state.velocity.Coeffs() -
                 left_state.velocity.Coeffs()).cwiseAbs());
        report.maximum_acceleration_jump =
                report.maximum_acceleration_jump.cwiseMax(
                        (right_state.acceleration.Coeffs() -
                         left_state.acceleration.Coeffs()).cwiseAbs());
    }

    report.velocity_utilization =
            report.peak_velocity.cwiseQuotient(max_velocity_);
    report.acceleration_utilization =
            report.peak_acceleration.cwiseQuotient(max_acceleration_);
    report.jerk_utilization = report.peak_jerk.cwiseQuotient(max_jerk_);
    report.maximum_utilization = std::max({
            report.velocity_utilization.maxCoeff(),
            report.acceleration_utilization.maxCoeff(),
            report.jerk_utilization.maxCoeff()});
    report.within_limits = std::isfinite(report.maximum_utilization) &&
                           report.maximum_utilization <= 1.0 + 1e-12;
    const Eigen::VectorXd velocity_tolerance =
            max_velocity_.cwiseMax(Eigen::VectorXd::Ones(dof_)) * 1e-7;
    const Eigen::VectorXd acceleration_tolerance =
            max_acceleration_.cwiseMax(Eigen::VectorXd::Ones(dof_)) * 1e-7;
    report.velocity_continuous =
            (report.maximum_velocity_jump.array() <=
             velocity_tolerance.array()).all();
    report.acceleration_continuous =
            (report.maximum_acceleration_jump.array() <=
             acceleration_tolerance.array()).all();
    return report;
}

template <typename LieGroup>
bool TrajectoryBase<LieGroup>::EnforceJointLimits(
        const Eigen::VectorXd& velocity_limits,
        const Eigen::VectorXd& acceleration_limits,
        const Eigen::VectorXd& jerk_limits) {
    constexpr int target_samples = 2001;
    constexpr int minimum_samples_per_segment = 9;
    time_scale_ = 1.0;
    const double duration = trajectory_pspline_->GetLastTimeStamp();
    double required_scale = 0.0;
    const double target_step = duration / (target_samples - 1.0);
    const auto evaluate = [&](double time) {
        const auto state = GetState(time);
        if (!state.position.Coeffs().allFinite() ||
            !state.velocity.Coeffs().allFinite() ||
            !state.acceleration.Coeffs().allFinite() ||
            !state.jerk.Coeffs().allFinite()) {
            required_scale = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        for (Eigen::Index i = 0; i < velocity_limits.size(); ++i) {
            required_scale = std::max(
                    required_scale,
                    std::abs(state.velocity[i]) / velocity_limits[i]);
            required_scale = std::max(
                    required_scale,
                    std::sqrt(std::abs(state.acceleration[i]) /
                              acceleration_limits[i]));
            required_scale = std::max(
                    required_scale,
                    std::cbrt(std::abs(state.jerk[i]) / jerk_limits[i]));
        }
    };
    const auto& knots = trajectory_pspline_->GetKnots();
    for (std::size_t segment = 1; segment < knots.size(); ++segment) {
        const double start = knots[segment - 1];
        const double end = knots[segment];
        const int samples = std::max(
                minimum_samples_per_segment,
                static_cast<int>(std::ceil((end - start) / target_step)) + 1);
        for (int sample = 0; sample < samples; ++sample) {
            double time =
                    start + (end - start) * sample / (samples - 1.0);
            // Internal knots are right-continuous. Sample immediately before
            // the knot as well, so a jerk change cannot hide the left limit.
            if (sample == samples - 1 && segment + 1 < knots.size()) {
                const double floating_offset =
                        256.0 * std::numeric_limits<double>::epsilon() *
                        std::max(1.0, duration);
                const double left_offset = std::min(
                        0.5 * (end - start),
                        std::max(floating_offset,
                                 1e-9 * (end - start)));
                time = end - left_offset;
            }
            evaluate(time);
            if (!std::isfinite(required_scale)) {
                valid_ = false;
                return false;
            }
        }
    }
    // Leave a margin only when sampled utilization approaches a constraint;
    // trajectories already comfortably below every limit are not slowed.
    time_scale_ = std::max(1.0, required_scale * 1.01);
    if (!std::isfinite(time_scale_)) {
        valid_ = false;
        return false;
    }
    return true;
}

template <typename LieGroup>
std::shared_ptr<PSpline> TrajectoryBase<LieGroup>::InterpolateToPSpline(
        const std::list<TrajectorySeg>& traj_segs) const {
    std::shared_ptr<PSpline> psline = std::make_shared<PSpline>();
    Eigen::Vector4d data;
    holistic_motion::utility::LogDebug("InterpolateToPSpline Begin!");
    for (auto it = traj_segs.begin(); it != traj_segs.end();) {
        auto t0 = it->timestamp;
        data << it->pos, it->vel, it->acc / 2.0, it->jerk / 6.0;
        ++it;
        auto t1 = it == traj_segs.end() ? t0 : it->timestamp;
        auto T = t1 - t0;
        holistic_motion::utility::LogDebug("Polynomial:{},{},{},{}, T:{}", data[0],
                                data[1], data[2], data[3], T);

        if (T > Epsilon) {
            auto polynomial = std::make_shared<Polynomial>(data);
            psline->PushBack(polynomial, T);
        }
    }
    holistic_motion::utility::LogDebug("InterpolateToPSpline Finish!");
    return psline;
}

template <typename LieGroup>
bool TrajectoryBase<LieGroup>::GetLimitFromConstraintProfile(
        const std::shared_ptr<TrajectoryConstraints>& constraints,
        Eigen::VectorXd& velocity_limits,
        Eigen::VectorXd& acceleration_limits,
        Eigen::VectorXd& jerk_limits) {
    if (nullptr != constraints) {
        if (constraints->IsValid()) {
            velocity_limits = constraints->GetMaxVelocityConstraints();
            acceleration_limits = constraints->GetMaxAccelerationConstraints();
            jerk_limits = constraints->GetMaxJerkConstraints();
            this->dof_ = velocity_limits.size();
            if (!path_ || path_->GetWaypoints().size() < 2 ||
                static_cast<Eigen::Index>(
                        (path_->GetWaypoints()[1] - path_->GetWaypoints()[0])
                                .size()) != velocity_limits.size()) {
                holistic_motion::utility::LogWarning(
                        "Trajectory constraint dimension does not match path");
                return false;
            }
            if (PathType::Bezier5thCartesianSpace == this->path_type_) {
                const double ratio = 1.0;
                velocity_limits *= ratio;
                acceleration_limits *= ratio;
                jerk_limits *= ratio;
            }
            max_velocity_ = velocity_limits;
            max_acceleration_ = acceleration_limits;
            max_jerk_ = jerk_limits;
        } else {
            holistic_motion::utility::LogWarning("Trajectory constraints is invalid!");
            return false;
        }
    } else {
        holistic_motion::utility::LogWarning(
                "Trajectory constraints need to be set and passed in!");
        return false;
    }

    return true;
}

}  // namespace robotics
}  // namespace holistic_motion
