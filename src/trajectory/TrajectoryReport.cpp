#include "holistic_motion/trajectory/TrajectoryBase.h"

#include "TrajectoryStateSampler.h"

namespace holistic_motion::robotics {

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
    const auto accumulate_state = [&](const State &state) {
        // A maximum reduction can discard NaNs and leave a plausible zero peak.
        // Validate all state components before accumulating any diagnostics.
        if (!state.position.Coeffs().allFinite() ||
            !state.velocity.Coeffs().allFinite() ||
            !state.acceleration.Coeffs().allFinite() ||
            !state.jerk.Coeffs().allFinite()) {
            throw std::runtime_error(
                "constraint report encountered a non-finite trajectory state");
        }
        // Reports expose dynamic vectors, but each state has a fixed dimension.
        // Retain that dimension during the hot reduction without extra storage
        // or stronger alignment requirements on the report's allocations.
        using Vector = Eigen::Matrix<double, LieGroup::DoF, 1>;
        Eigen::Map<Vector> velocity_peak(report.peak_velocity.data());
        Eigen::Map<Vector> acceleration_peak(report.peak_acceleration.data());
        Eigen::Map<Vector> jerk_peak(report.peak_jerk.data());
        velocity_peak =
            velocity_peak.cwiseMax(state.velocity.Coeffs().cwiseAbs());
        acceleration_peak =
            acceleration_peak.cwiseMax(state.acceleration.Coeffs().cwiseAbs());
        jerk_peak = jerk_peak.cwiseMax(state.jerk.Coeffs().cwiseAbs());
    };
    detail::TrajectoryStateSampler<LieGroup> sampler(*this);
    const auto accumulate = [&](double time) {
        accumulate_state(sampler.GetState(time));
    };

    const double duration = GetDuration();
    for (std::size_t sample = 0; sample < samples; ++sample) {
        const double fraction =
            static_cast<double>(sample) / static_cast<double>(samples - 1);
        accumulate(duration * fraction);
    }
    if (!valid_ || !trajectory_pspline_ || !path_)
        throw std::logic_error("cannot inspect an invalid trajectory");
    // A custom geometry query may replace the clock or release the path.
    // Keep the referenced knots alive and reject a changed phase layout before
    // using the next endpoint index, including in-place spline replacement.
    const auto endpoint_clock = trajectory_pspline_;
    const std::size_t breakpoint_count = endpoint_clock->GetKnots().size();
    const auto validate_endpoint_owner = [&] {
        if (!valid_ || !path_ || trajectory_pspline_ != endpoint_clock ||
            endpoint_clock->GetKnots().size() != breakpoint_count)
            throw std::logic_error("trajectory changed during constraint "
                                   "report endpoint sampling");
    };
    for (std::size_t index = 1; index + 1 < breakpoint_count; ++index) {
        const auto left_state = GetPhaseEndpoint(index - 1, true);
        validate_endpoint_owner();
        const auto right_state = GetPhaseEndpoint(index, false);
        validate_endpoint_owner();
        accumulate_state(left_state);
        accumulate_state(right_state);
        report.maximum_velocity_jump = report.maximum_velocity_jump.cwiseMax(
            (right_state.velocity.Coeffs() - left_state.velocity.Coeffs()).cwiseAbs());
        report.maximum_acceleration_jump = report.maximum_acceleration_jump.cwiseMax(
            (right_state.acceleration.Coeffs() - left_state.acceleration.Coeffs())
                .cwiseAbs());
    }

    report.velocity_utilization = report.peak_velocity.cwiseQuotient(max_velocity_);
    report.acceleration_utilization =
        report.peak_acceleration.cwiseQuotient(max_acceleration_);
    report.jerk_utilization = report.peak_jerk.cwiseQuotient(max_jerk_);
    report.maximum_utilization = std::max({report.velocity_utilization.maxCoeff(),
                                           report.acceleration_utilization.maxCoeff(),
                                           report.jerk_utilization.maxCoeff()});
    report.within_limits = std::isfinite(report.maximum_utilization) &&
                           report.maximum_utilization <= 1.0 + 1e-12;
    // Evaluate the tolerances inside the comparisons. Materializing them as
    // dynamic vectors adds two allocations to even the shortest report.
    report.velocity_continuous = (report.maximum_velocity_jump.array() <=
                                  max_velocity_.array().max(1.0) * 1e-7)
                                     .all();
    report.acceleration_continuous =
        (report.maximum_acceleration_jump.array() <=
         max_acceleration_.array().max(1.0) * 1e-7)
            .all();
    return report;
}

// Instantiate only this member here. The remaining class implementation lives
// in TrajectoryBase.cpp, keeping repeated report sampling out of its compile unit.
#define INSTANTIATE_REPORT(Group)                                                      \
    template TrajectoryBase<Group>::ConstraintReport                                   \
        TrajectoryBase<Group>::GetConstraintReport(std::size_t) const;

INSTANTIATE_REPORT(R1d)
INSTANTIATE_REPORT(R2d)
INSTANTIATE_REPORT(R3d)
INSTANTIATE_REPORT(R4d)
INSTANTIATE_REPORT(R5d)
INSTANTIATE_REPORT(R6d)
INSTANTIATE_REPORT(R7d)
INSTANTIATE_REPORT(R8d)
INSTANTIATE_REPORT(R9d)
INSTANTIATE_REPORT(R10d)
INSTANTIATE_REPORT(R11d)
INSTANTIATE_REPORT(R12d)
INSTANTIATE_REPORT(R13d)
INSTANTIATE_REPORT(R14d)
INSTANTIATE_REPORT(R15d)
INSTANTIATE_REPORT(R16d)
INSTANTIATE_REPORT(R17d)
INSTANTIATE_REPORT(R18d)
INSTANTIATE_REPORT(R19d)
INSTANTIATE_REPORT(R20d)
INSTANTIATE_REPORT(R21d)
INSTANTIATE_REPORT(R22d)
INSTANTIATE_REPORT(R23d)
INSTANTIATE_REPORT(R24d)
INSTANTIATE_REPORT(R25d)
INSTANTIATE_REPORT(R26d)
INSTANTIATE_REPORT(R27d)
INSTANTIATE_REPORT(R28d)
INSTANTIATE_REPORT(R29d)
INSTANTIATE_REPORT(R30d)
INSTANTIATE_REPORT(R31d)
INSTANTIATE_REPORT(R32d)
INSTANTIATE_REPORT(SE3d)

#undef INSTANTIATE_REPORT

} // namespace holistic_motion::robotics
