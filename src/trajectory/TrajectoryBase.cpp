#include "holistic_motion/trajectory/TrajectoryBase.h"

namespace holistic_motion {
namespace robotics {

namespace {
template <unsigned Power, typename Tangent>
Tangent ScaleByPower(const Tangent &value, double speed, double speed_power) {
    if (std::isnormal(speed_power) || speed == 0.0)
        return value * speed_power;
    // A speed power may overflow or underflow even when its product with a
    // geometric derivative is representable. Starting from that derivative
    // keeps every intermediate between it and the final result in magnitude.
    Tangent result = value;
    for (unsigned order = 0; order < Power; ++order)
        result *= speed;
    return result;
}

double SampleBeforeKnot(double start, double end) {
    // Step beyond PSpline's snapping tolerance using the local knot scale.
    // A long later phase must not move this sample into the interval interior.
    const double span = end - start;
    const double floating_offset =
        256.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, end);
    const double offset =
        std::min(0.5 * span, std::max(floating_offset, 1e-9 * span));
    return end - offset;
}
}  // namespace

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
        if (!std::isfinite(requested_scale) ||
            !std::isfinite(trajectory_pspline_->GetLastTimeStamp() *
                           requested_scale)) return false;
        time_scale_ = requested_scale;
    }
    return true;
}

template <typename LieGroup>
bool TrajectoryBase<LieGroup>::InitializePhasePathSegments() {
    phase_path_segments_.clear();
    const auto segments = path_->GetPathSegments();
    // Blended time phases can span several geometric segments. Keep their
    // position-based lookup; an all-linear path has one owner per phase.
    if (std::any_of(segments.begin(), segments.end(), [](const auto& segment) {
            return segment->GetPathSegType() != PathSegType::LinearSeg;
        }))
        return true;
    phase_path_segments_.reserve(trajectory_pspline_->GetKnots().size() - 1);
    for (auto it = trajectory_segments_.begin();
         it != trajectory_segments_.end(); ++it) {
        const auto next = std::next(it);
        if (next == trajectory_segments_.end()) break;
        if (next->timestamp <= it->timestamp) continue;
        if (it->seg_no < 0 ||
            static_cast<std::size_t>(it->seg_no) >= segments.size()) {
            phase_path_segments_.clear();
            return false;
        }
        phase_path_segments_.push_back(segments[it->seg_no]);
    }
    if (phase_path_segments_.size() + 1 !=
        trajectory_pspline_->GetKnots().size()) {
        phase_path_segments_.clear();
        return false;
    }
    return true;
}

template <typename LieGroup>
std::shared_ptr<PathSegmentBase<LieGroup>>
TrajectoryBase<LieGroup>::EvaluatePathJet(double time,
                                          std::array<double, 4>& jet) const {
    if (!valid_ || !trajectory_pspline_ || !path_) {
        throw std::logic_error("cannot query an invalid trajectory");
    }
    if (!std::isfinite(time)) {
        throw std::invalid_argument("trajectory time must be finite");
    }
    time = clamp(time, 0.0, GetDuration()) / time_scale_;
    std::size_t phase;
    jet = trajectory_pspline_->ComputeJetAtS(time, phase);
    if (!std::isfinite(jet[0])) {
        throw std::runtime_error(
            "trajectory evaluated a non-finite path parameter");
    }
    // At a stop, the path position can round to the next segment while time
    // still belongs to the incoming phase. Its derivatives need that phase's
    // geometry, including at right-continuous time knots and after rescaling.
    const auto segment = phase_path_segments_.empty()
                             ? path_->GetPathSegmentAtS(jet[0])
                             : phase_path_segments_[phase];
    if (!segment) throw std::logic_error("cannot query an invalid path");
    return segment;
}

template <typename LieGroup>
LieGroup TrajectoryBase<LieGroup>::GetPosition(double t) const {
    std::array<double, 4> jet;
    const auto segment = EvaluatePathJet(t, jet);
    return segment->GetConfig(jet[0]);
}

template <typename LieGroup>
typename LieGroup::Tangent TrajectoryBase<LieGroup>::GetVelocity(
    double t) const {
    std::array<double, 4> jet;
    const auto segment = EvaluatePathJet(t, jet);
    return segment->GetTangent(jet[0]) * jet[1] / time_scale_;
}

template <typename LieGroup>
typename LieGroup::Tangent TrajectoryBase<LieGroup>::GetAcceleration(
    double t) const {
    std::array<double, 4> jet;
    const auto segment = EvaluatePathJet(t, jet);
    const auto tangent = segment->GetTangent(jet[0]);
    const auto curvature = segment->GetCurvature(jet[0]);
    // Scale stepwise to preserve representable derivatives at large scales.
    const double inverse_scale = 1.0 / time_scale_;
    const double speed_squared = jet[1] * jet[1];
    return (tangent * jet[2] + ScaleByPower<2>(curvature, jet[1], speed_squared)) *
           inverse_scale * inverse_scale;
}

template <typename LieGroup>
typename LieGroup::Tangent TrajectoryBase<LieGroup>::GetJerk(double t) const {
    std::array<double, 4> jet;
    const auto segment = EvaluatePathJet(t, jet);
    const auto tangent = segment->GetTangent(jet[0]);
    const auto curvature = segment->GetCurvature(jet[0]);
    const auto torsion = segment->GetTorsion(jet[0]);
    const double inverse_scale = 1.0 / time_scale_;
    const double speed_squared = jet[1] * jet[1];
    return (tangent * jet[3] + 3.0 * curvature * jet[1] * jet[2] +
            ScaleByPower<3>(torsion, jet[1], speed_squared * jet[1])) *
           inverse_scale * inverse_scale * inverse_scale;
}

template <typename LieGroup>
typename TrajectoryBase<LieGroup>::State TrajectoryBase<LieGroup>::GetState(
    double t) const {
    std::array<double, 4> jet;
    const auto segment = EvaluatePathJet(t, jet);
    State state;
    typename LieGroup::Tangent tangent, curvature, torsion;
    segment->ComputeJet(jet[0], state.position, tangent, curvature, torsion);
    const double inverse_scale = 1.0 / time_scale_;
    const double speed_squared = jet[1] * jet[1];

    state.velocity = tangent * jet[1] * inverse_scale;
    state.acceleration =
        (tangent * jet[2] + ScaleByPower<2>(curvature, jet[1], speed_squared)) *
        inverse_scale * inverse_scale;
    state.jerk = (tangent * jet[3] + 3.0 * curvature * jet[1] * jet[2] +
                  ScaleByPower<3>(torsion, jet[1], speed_squared * jet[1])) *
                 inverse_scale * inverse_scale * inverse_scale;
    return state;
}

template <typename LieGroup>
typename TrajectoryBase<LieGroup>::ConstraintReport
TrajectoryBase<LieGroup>::GetConstraintReport(std::size_t samples) const {
    if (!valid_ || !trajectory_pspline_ || !path_) {
        throw std::logic_error("cannot inspect an invalid trajectory");
    }
    if (samples < 2) {
        throw std::invalid_argument(
            "constraint report requires at least 2 samples");
    }

    ConstraintReport report;
    report.peak_velocity = Eigen::VectorXd::Zero(dof_);
    report.peak_acceleration = Eigen::VectorXd::Zero(dof_);
    report.peak_jerk = Eigen::VectorXd::Zero(dof_);
    report.maximum_velocity_jump = Eigen::VectorXd::Zero(dof_);
    report.maximum_acceleration_jump = Eigen::VectorXd::Zero(dof_);
    const auto accumulate_state = [&](const State& state) {
        // A maximum reduction can discard NaNs and leave a plausible zero peak.
        // Validate all state components before accumulating any diagnostics.
        if (!state.position.Coeffs().allFinite() ||
            !state.velocity.Coeffs().allFinite() ||
            !state.acceleration.Coeffs().allFinite() ||
            !state.jerk.Coeffs().allFinite()) {
            throw std::runtime_error(
                "constraint report encountered a non-finite trajectory state");
        }
        report.peak_velocity =
            report.peak_velocity.cwiseMax(state.velocity.Coeffs().cwiseAbs());
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
        const double fraction =
            static_cast<double>(sample) / static_cast<double>(samples - 1);
        accumulate(duration * fraction);
    }
    const auto breakpoints = GetBreakpoints();
    for (std::size_t index = 1; index + 1 < breakpoints.size(); ++index) {
        const double breakpoint = breakpoints[index];
        const double previous = breakpoints[index - 1];
        const auto left_state =
            GetState(SampleBeforeKnot(previous, breakpoint));
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
    constexpr int minimum_samples_per_segment = 65;
    time_scale_ = 1.0;
    const double duration = trajectory_pspline_->GetLastTimeStamp();
    double required_scale = 0.0;
    double maximum_acceleration_utilization = 0.0;
    double maximum_jerk_utilization = 0.0;
    using LimitVector = Eigen::Matrix<double, LieGroup::DoF, 1>;
    const LimitVector inverse_velocity_limits = velocity_limits.cwiseInverse();
    const LimitVector inverse_acceleration_limits = acceleration_limits.cwiseInverse();
    const LimitVector inverse_jerk_limits = jerk_limits.cwiseInverse();
    const bool finite_reciprocals = inverse_velocity_limits.allFinite() &&
                                    inverse_acceleration_limits.allFinite() &&
                                    inverse_jerk_limits.allFinite();
    const auto evaluate = [&](double time) {
        State state;
        try {
            state = GetState(time);
        } catch (const std::runtime_error&) {
            // Internal evaluation failure invalidates construction, just like
            // a returned nonfinite state. Public diagnostics keep the error.
            required_scale = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        if (!state.position.Coeffs().allFinite() ||
            !state.velocity.Coeffs().allFinite() ||
            !state.acceleration.Coeffs().allFinite() ||
            !state.jerk.Coeffs().allFinite()) {
            required_scale = std::numeric_limits<double>::quiet_NaN();
            return;
        }
        if (finite_reciprocals) {
            // Vectorize the common finite case and reduce before checking for
            // overflow. Nonnegative ratios cannot hide infinity in a maximum.
            const double velocity_utilization = (state.velocity.Coeffs().array().abs() *
                                                 inverse_velocity_limits.array())
                                                    .maxCoeff();
            const double acceleration_utilization =
                (state.acceleration.Coeffs().array().abs() *
                 inverse_acceleration_limits.array())
                    .maxCoeff();
            const double jerk_utilization =
                (state.jerk.Coeffs().array().abs() * inverse_jerk_limits.array())
                    .maxCoeff();
            if (std::isfinite(velocity_utilization) &&
                std::isfinite(acceleration_utilization) &&
                std::isfinite(jerk_utilization)) {
                required_scale = std::max(required_scale, velocity_utilization);
                maximum_acceleration_utilization = std::max(
                    maximum_acceleration_utilization, acceleration_utilization);
                maximum_jerk_utilization =
                    std::max(maximum_jerk_utilization, jerk_utilization);
                return;
            }
        }
        // Keep direct division and root-before-ratio handling when a reciprocal
        // or a product overflows, including a zero value with a tiny limit.
        for (Eigen::Index i = 0; i < velocity_limits.size(); ++i) {
            required_scale = std::max(
                    required_scale,
                    std::abs(state.velocity[i]) / velocity_limits[i]);
            const double acceleration = std::abs(state.acceleration[i]);
            const double jerk = std::abs(state.jerk[i]);
            const double acceleration_utilization =
                acceleration / acceleration_limits[i];
            const double jerk_utilization = jerk / jerk_limits[i];
            // Positive roots preserve ordering, so reduce finite ratios
            // before taking roots once after all samples. An overflowing
            // ratio can still have a finite root; retain that fallback.
            if (std::isfinite(acceleration_utilization))
                maximum_acceleration_utilization = std::max(
                    maximum_acceleration_utilization, acceleration_utilization);
            else
                required_scale =
                    std::max(required_scale, std::sqrt(acceleration) /
                                                 std::sqrt(acceleration_limits[i]));
            if (std::isfinite(jerk_utilization))
                maximum_jerk_utilization =
                    std::max(maximum_jerk_utilization, jerk_utilization);
            else
                required_scale = std::max(required_scale,
                                          std::cbrt(jerk) / std::cbrt(jerk_limits[i]));
        }
    };
    const auto& knots = trajectory_pspline_->GetKnots();
    std::vector<std::shared_ptr<PathSegmentBase<LieGroup>>> geometric_segments;
    if (phase_path_segments_.empty())
        geometric_segments = path_->GetPathSegments();
    for (std::size_t segment = 1; segment < knots.size(); ++segment) {
        const double start = knots[segment - 1];
        const double end = knots[segment];
        const double left_end =
            segment + 1 < knots.size() ? SampleBeforeKnot(start, end) : end;
        const auto initial_jet = trajectory_pspline_->ComputeJetAtS(start);
        const double stationary =
            initial_jet[3] != 0.0 ? -initial_jet[2] / initial_jet[3] : -1.0;
        const double stationary_time = start + stationary;
        bool linear_phase = !phase_path_segments_.empty();
        bool monotone = false;
        std::array<double, 4> final_jet{};
        if (!linear_phase) {
            final_jet = trajectory_pspline_->ComputeJetAtS(left_end);
            // A cubic time law is monotone iff its quadratic velocity is
            // nonnegative at both ends and any interior minimum.
            double minimum_velocity = std::min(initial_jet[1], final_jet[1]);
            if (stationary > 0.0 && stationary < end - start) {
                const double velocity =
                    initial_jet[1] +
                    stationary * (initial_jet[2] + 0.5 * stationary * initial_jet[3]);
                minimum_velocity = std::min(minimum_velocity, velocity);
            }
            monotone = minimum_velocity >= 0.0 && std::isfinite(initial_jet[0]) &&
                       std::isfinite(final_jet[0]);
            if (monotone) {
                const auto first = path_->GetPathSegmentAtS(initial_jet[0]);
                linear_phase = first &&
                               first->GetPathSegType() == PathSegType::LinearSeg &&
                               first == path_->GetPathSegmentAtS(final_jet[0]);
            }
        }
        const auto sample_interval = [&](double interval_start, double interval_end,
                                         double sample_end, bool linear) {
            if (linear) {
                // On a line, joint velocity is quadratic, acceleration linear,
                // and jerk constant. Endpoints and a zero of acceleration
                // contain every derivative maximum.
                evaluate(interval_start);
                evaluate(sample_end);
                if (stationary > 0.0 && stationary_time > interval_start &&
                    stationary_time < interval_end)
                    evaluate(stationary_time);
                return std::isfinite(required_scale);
            }
            // Normalize first to avoid overflowing the numerator or losing
            // a subnormal time step. Retain at least 65 checks per interval.
            const int samples = std::max(
                minimum_samples_per_segment,
                static_cast<int>(std::ceil((interval_end - interval_start) / duration *
                                           (target_samples - 1.0))) +
                    1);
            for (int sample = 0; sample < samples; ++sample) {
                const double fraction = sample / (samples - 1.0);
                const double time =
                    sample == samples - 1
                        ? sample_end
                        : interval_start + (interval_end - interval_start) * fraction;
                evaluate(time);
                if (!std::isfinite(required_scale))
                    return false;
            }
            return true;
        };
        const auto interval_is_linear = [&](double interval_start, double sample_end) {
            const auto first_jet = trajectory_pspline_->ComputeJetAtS(interval_start);
            const auto last_jet = trajectory_pspline_->ComputeJetAtS(sample_end);
            if (!std::isfinite(first_jet[0]) || !std::isfinite(last_jet[0]))
                return false;
            const auto first = path_->GetPathSegmentAtS(first_jet[0]);
            return first && first->GetPathSegType() == PathSegType::LinearSeg &&
                   first == path_->GetPathSegmentAtS(last_jet[0]);
        };
        double interval_start = start;
        if (monotone && !linear_phase) {
            // One time phase can cross several geometric segments of very
            // different lengths. A grid over the complete phase can miss a
            // short curve entirely. Split at each interior geometric boundary
            // before applying the same local sampling density.
            auto boundary = std::upper_bound(
                geometric_segments.begin(), geometric_segments.end(), initial_jet[0],
                [](double position, const auto &geometry) {
                    return position < geometry->GetStartParameter();
                });
            for (; boundary != geometric_segments.end() &&
                   (*boundary)->GetStartParameter() < final_jet[0];
                 ++boundary) {
                const double position = (*boundary)->GetStartParameter();
                double low = interval_start, high = left_end;
                double middle = low + 0.5 * (high - low);
                if (initial_jet[2] == 0.0 && initial_jet[3] == 0.0 &&
                    initial_jet[1] > 0.0) {
                    // Most curves are traversed at constant path speed. Start
                    // from its inverse, then verify against actual evaluation.
                    const double candidate =
                        start + (position - initial_jet[0]) / initial_jet[1];
                    if (candidate > low && candidate < high)
                        middle = candidate;
                }
                // Bisect the actual spline evaluation, preserving its rounding
                // and knot behavior. Stop at the boundary or adjacent timestamps,
                // not an absolute tolerance that could erase a short curve.
                while (true) {
                    if (middle <= low || middle >= high)
                        break;
                    const double middle_position =
                        trajectory_pspline_->ComputeJetAtS(middle)[0];
                    if (!std::isfinite(middle_position)) {
                        valid_ = false;
                        return false;
                    }
                    if (middle_position == position) {
                        high = middle;
                        break;
                    }
                    if (middle_position < position)
                        low = middle;
                    else
                        high = middle;
                    middle = low + 0.5 * (high - low);
                }
                if (high <= interval_start || high >= left_end)
                    continue;
                const double sample_end = SampleBeforeKnot(interval_start, high);
                if (!sample_interval(interval_start, high, sample_end,
                                     interval_is_linear(interval_start, sample_end))) {
                    valid_ = false;
                    return false;
                }
                interval_start = high;
            }
        }
        const bool remaining_linear =
            linear_phase || (monotone && interval_start > start &&
                             interval_is_linear(interval_start, left_end));
        if (!sample_interval(interval_start, end, left_end, remaining_linear)) {
            valid_ = false;
            return false;
        }
    }
    required_scale =
        std::max({required_scale, std::sqrt(maximum_acceleration_utilization),
                  std::cbrt(maximum_jerk_utilization)});
    // Leave a margin only when sampled utilization approaches a constraint;
    // trajectories already comfortably below every limit are not slowed.
    time_scale_ = std::max(1.0, required_scale * 1.01);
    if (!std::isfinite(time_scale_) || !std::isfinite(duration * time_scale_)) {
        valid_ = false;
        return false;
    }
    return true;
}

template <typename LieGroup>
std::shared_ptr<PSpline> TrajectoryBase<LieGroup>::InterpolateToPSpline(
    const std::list<TrajectorySeg> &traj_segs) const {
    std::shared_ptr<PSpline> psline = std::make_shared<PSpline>();
    Eigen::Vector4d data;
    holistic_motion::utility::LogDebug("InterpolateToPSpline Begin!");
    for (auto it = traj_segs.begin(); it != traj_segs.end();) {
        auto t0 = it->timestamp;
        data << it->pos, it->vel, it->acc / 2.0, it->jerk / 6.0;
        // Validate every state, including the terminal state and zero-duration
        // phases that do not create a polynomial. Never return a valid prefix
        // when a later state contains invalid kinematic data.
        if (!data.allFinite()) {
            holistic_motion::utility::LogWarning(
                    "Trajectory interpolation requires finite phase states at time {}",
                    t0);
            return std::make_shared<PSpline>();
        }
        ++it;
        auto t1 = it == traj_segs.end() ? t0 : it->timestamp;
        auto T = t1 - t0;
        const double timestamp_tolerance =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max({1.0, std::abs(t0), std::abs(t1)});
        if (!std::isfinite(t0) || !std::isfinite(t1) ||
            !std::isfinite(T) || T < -timestamp_tolerance) {
            holistic_motion::utility::LogWarning(
                    "Trajectory interpolation requires finite ordered timestamps: "
                    "start={}, end={}, duration={}", t0, t1, T);
            return std::make_shared<PSpline>();
        }
        holistic_motion::utility::LogDebug("Polynomial:{},{},{},{}, T:{}", data[0],
                                data[1], data[2], data[3], T);

        // A positive phase carries both elapsed time and its state change,
        // even when shorter than the geometric path tolerance.
        if (T > 0.0) {
            auto polynomial = std::make_shared<Polynomial>(data);
            if (!psline->PushBack(polynomial, T)) {
                holistic_motion::utility::LogWarning(
                        "Trajectory interpolation cannot advance the spline timestamp");
                return std::make_shared<PSpline>();
            }
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
