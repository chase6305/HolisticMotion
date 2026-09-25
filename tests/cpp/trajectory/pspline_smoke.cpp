#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "holistic_motion/trajectory/Polynomial.h"
#include "holistic_motion/trajectory/TrajectoryBase.h"
#include "holistic_motion/trajectory/TrajectoryDoubleS.h"
#include "holistic_motion/trajectory/TrajectoryTrapezium.h"

using namespace holistic_motion::robotics;

namespace {
void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

std::shared_ptr<Polynomial> Constant(double value) {
    return std::make_shared<Polynomial>(Eigen::Vector4d(value, 0.0, 0.0, 0.0));
}

void CheckLocalBoundaries() {
    for (double duration : {1e-100, 1.0, 1e100}) {
        PSpline scaled;
        Require(scaled.PushBack(std::make_shared<Polynomial>(
                                   Eigen::Vector4d(0.0, 1.0 / duration, 0.0, 0.0)),
                               duration),
                "append scaled linear time law");
        for (double fraction : {0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99})
            Require(std::abs(scaled.ComputeValueAtS(fraction * duration) - fraction) <
                        1e-14,
                    "knot snapping must not depend on the time unit");
    }
    PSpline spline;
    Require(spline.PushBack(Constant(1.0), 0.001), "first segment");
    Require(spline.PushBack(Constant(2.0), 0.001), "second segment");
    Require(spline.PushBack(Constant(3.0), 1e15), "long final segment");
    Require(spline.ComputeValueAtS(0.0005) == 1.0,
            "long tail must not consume first segment");
    Require(spline.ComputeValueAtS(0.0015) == 2.0,
            "long tail must not consume second segment");
    Require(spline.ComputeValueAtS(0.001) == 2.0,
            "internal knots must be right-continuous");
    Require(spline.ComputeValueAtS(-1.0) == 1.0, "clamp before start");
    Require(spline.ComputeValueAtS(2e15) == 3.0, "clamp after end");

    PSpline narrow;
    const double start = 1e8;
    const double ulp = std::nextafter(start, INFINITY) - start;
    Require(narrow.PushBack(Constant(1.0), start), "large first knot");
    Require(narrow.PushBack(Constant(2.0), 8 * ulp), "narrow middle segment");
    Require(narrow.PushBack(Constant(3.0), 1.0), "final segment");
    Require(narrow.ComputeValueAtS(start + 4 * ulp) == 2.0,
            "tolerance must not consume narrow segment interior");
}

void CheckAppendValidation() {
    PSpline spline;
    Require(spline.PushBack(Constant(1.0), 1e308), "large finite duration");
    const auto original = spline.GetKnots();
    Require(!spline.PushBack(Constant(2.0), 1e308), "reject duration overflow");
    Require(!spline.PushBack(Constant(2.0), 1.0), "reject non-advancing knot");
    Require(!spline.PushBack(nullptr, 1.0), "reject null polynomial");
    Require(spline.GetKnots() == original,
            "failed appends must preserve knots");
    Require(spline.ComputeValueAtS(1e308) == 1.0,
            "failed appends must preserve segment list");
}

void CheckJetAndInvalidTimes() {
    PSpline spline;
    const auto first =
        std::make_shared<Polynomial>(Eigen::Vector4d(1, 2, 3, 4));
    const auto second =
        std::make_shared<Polynomial>(Eigen::Vector4d(5, 6, 7, 8));
    Require(spline.PushBack(first, 2.0), "first cubic");
    Require(spline.PushBack(second, 3.0), "second cubic");
    for (double time : {-1.0, 0.0, 0.5, 2.0, 3.0, 5.0, 6.0}) {
        const double clamped = std::clamp(time, 0.0, 5.0);
        const auto& polynomial = clamped < 2.0 ? first : second;
        const double local = clamped < 2.0 ? clamped : clamped - 2.0;
        std::size_t phase;
        const auto jet = spline.ComputeJetAtS(time, phase);
        Require(phase == (clamped < 2.0 ? 0u : 1u),
                "jet must identify its right-continuous phase");
        Require(jet == spline.ComputeJetAtS(time),
                "phase-aware jet must preserve ordinary evaluation");
        for (unsigned order = 0; order < 4; ++order) {
            const double expected =
                polynomial->ComputePolyValueAtS(local, order);
            Require(jet[order] == expected, "jet must use correct local time");
            Require(spline.ComputeValueAtS(time, order) == expected,
                    "scalar must match jet");
        }
    }
    for (double time : {NAN, INFINITY, -INFINITY}) {
        bool scalar_rejected = false;
        bool jet_rejected = false;
        try {
            spline.ComputeValueAtS(time);
        } catch (const std::invalid_argument&) {
            scalar_rejected = true;
        }
        try {
            spline.ComputeJetAtS(time);
        } catch (const std::invalid_argument&) {
            jet_rejected = true;
        }
        Require(scalar_rejected && jet_rejected, "reject non-finite time");
    }
}

void CheckJetExtremeCoefficients() {
    const double tiny = std::numeric_limits<double>::denorm_min();
    const double huge = std::numeric_limits<double>::max();
    const std::vector<Eigen::Vector4d> coefficients{
        Eigen::Vector4d(-0.0, -0.0, -0.0, -0.0),
        Eigen::Vector4d(tiny, -tiny, 3 * tiny, -7 * tiny),
        Eigen::Vector4d(huge / 8, -huge / 16, huge / 32, -huge / 64),
        Eigen::Vector4d(1.0, -1e100, 1e200, -1e300),
        Eigen::Vector4d(huge, huge, huge, huge)};
    for (const auto &data : coefficients) {
        PSpline spline;
        Require(spline.PushBack(std::make_shared<Polynomial>(data), 2.0),
                "append extreme coefficients");
        for (double time : {0.0, tiny, 0.125, 0.5, 1.0, 1.5, 2.0}) {
            const auto jet = spline.ComputeJetAtS(time);
            for (unsigned order = 0; order < 4; ++order) {
                const double expected = spline.ComputeValueAtS(time, order);
                Require((std::isnan(jet[order]) && std::isnan(expected)) ||
                            (jet[order] == expected &&
                             std::signbit(jet[order]) == std::signbit(expected)),
                        "fused jet must preserve scalar rounding and zero signs");
            }
        }
    }
    PSpline empty_polynomial;
    Require(empty_polynomial.PushBack(std::make_shared<Polynomial>()),
            "append default polynomial");
    Require(empty_polynomial.ComputeJetAtS(0.5) == std::array<double, 4>{},
            "default polynomial jet must remain zero");
}

void CheckInvalidProfilesAreNotTruncated() {
    struct Builder : TrajectoryBase<Rn<double, 2>> {
        using TrajectoryBase<Rn<double, 2>>::InterpolateToPSpline;
    } builder;
    for (double scale : {1e-100, 0.1, 1.0}) {
        const std::list<TrajectorySeg> backwards{
            TrajectorySeg(0, 0.0, 0.0, 0.0, 0.0, 0.0),
            TrajectorySeg(0, scale, 1.0, 0.0, 0.0, 0.0),
            TrajectorySeg(0, 0.5 * scale, 1.0, 0.0, 0.0, 0.0),
            TrajectorySeg(0, 2.0 * scale, 2.0, 0.0, 0.0, 0.0)};
        Require(builder.InterpolateToPSpline(backwards)->GetKnots().size() == 1,
                "a backwards phase must not become roundoff in smaller time units");
        const std::list<TrajectorySeg> rounded{
            TrajectorySeg(0, 0.0, 0.0, 0.0, 0.0, 0.0),
            TrajectorySeg(0, scale, 1.0, 0.0, 0.0, 0.0),
            TrajectorySeg(0, std::nextafter(scale, 0.0), 1.0, 0.0, 0.0, 0.0),
            TrajectorySeg(0, 2.0 * scale, 2.0, 0.0, 0.0, 0.0)};
        Require(builder.InterpolateToPSpline(rounded)->GetKnots().size() == 3,
                "one-ulp backwards rounding must remain accepted in any time unit");
    }
    const std::list<TrajectorySeg> rounded_profile{
        TrajectorySeg(0, 0.0, 0.0, 0.0, 0.0, 0.0),
        TrajectorySeg(0, 1.0, 1.0, 0.0, 0.0, 0.0),
        TrajectorySeg(0, std::nextafter(1.0, 0.0), 1.0, 0.0, 0.0, 0.0),
        TrajectorySeg(0, 2.0, 2.0, 0.0, 0.0, 0.0)};
    Require(
        builder.InterpolateToPSpline(rounded_profile)->GetKnots().size() == 3,
        "rounding at a zero-duration phase must remain accepted");
    for (double end : {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN(), -1.0}) {
        const std::list<TrajectorySeg> profile{
            TrajectorySeg(0, 0.0, 0.0, 0.0, 0.0, 0.0),
            TrajectorySeg(0, 1.0, 1.0, 0.0, 0.0, 0.0),
            TrajectorySeg(0, end, 2.0, 0.0, 0.0, 0.0)};
        const auto spline = builder.InterpolateToPSpline(profile);
        Require(spline->GetKnots().size() == 1,
                "invalid profile must not return a valid prefix");
    }
}

void CheckAllPhaseStatesAreFinite() {
    struct Builder : TrajectoryBase<Rn<double, 2>> {
        using TrajectoryBase<Rn<double, 2>>::InterpolateToPSpline;
    } builder;
    // Constant-velocity motion with a finite zero-duration jerk transition.
    const std::list<TrajectorySeg> valid{
        TrajectorySeg(0, 0.0, 0.0, 1.0, 0.0, 0.0),
        TrajectorySeg(0, 1.0, 1.0, 1.0, 0.0, 5.0),
        TrajectorySeg(0, 1.0, 1.0, 1.0, 0.0, 0.0),
        TrajectorySeg(0, 2.0, 2.0, 1.0, 0.0, 0.0)};
    const auto finite = builder.InterpolateToPSpline(valid);
    Require(finite->GetKnots().size() == 3 &&
                finite->GetLastTimeStamp() == 2.0 &&
                finite->ComputeValueAtS(2.0) == 2.0 &&
                finite->ComputeValueAtS(1.5, 1) == 1.0,
            "finite zero-duration transitions must remain accepted");
    for (int phase : {3, 1, 0}) {
        for (int component = 0; component < 4; ++component) {
            for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity()}) {
                auto states = valid;
                auto& state = *std::next(states.begin(), phase);
                switch (component) {
                    case 0:
                        state.pos = invalid;
                        break;
                    case 1:
                        state.vel = invalid;
                        break;
                    case 2:
                        state.acc = invalid;
                        break;
                    case 3:
                        state.jerk = invalid;
                        break;
                }
                const auto spline = builder.InterpolateToPSpline(states);
                Require(
                    spline->GetKnots().size() == 1 &&
                        spline->GetLastTimeStamp() == 0.0,
                    "nonfinite phase state must invalidate the whole spline");
            }
        }
    }
}

void CheckShortPositivePhasesArePreserved() {
    struct Builder : TrajectoryBase<Rn<double, 2>> {
        using TrajectoryBase<Rn<double, 2>>::InterpolateToPSpline;
    } builder;
    // A single cubic sampled at phase boundaries is an independent reference
    // for the same polynomial expressed in successive local time frames.
    const auto position = [](double t) {
        return 1.0 + 2.0 * t + 3.0 * t * t + 4.0 * t * t * t;
    };
    const auto velocity = [](double t) { return 2.0 + 6.0 * t + 12.0 * t * t; };
    const auto acceleration = [](double t) { return 6.0 + 24.0 * t; };
    const std::vector<double> times{0.0, 1e-7, 3e-7, 1e-3, 1e-3 + 1e-7};
    std::list<TrajectorySeg> phases;
    for (double time : times)
        phases.emplace_back(0, time, position(time), velocity(time),
                            acceleration(time), 24.0);
    const auto spline = builder.InterpolateToPSpline(phases);
    Require(spline->GetKnots() == times,
            "short phases must retain their timestamps");
    for (std::size_t i = 1; i < times.size(); ++i) {
        for (double time :
             {times[i - 1], (times[i - 1] + times[i]) / 2, times[i]}) {
            const auto jet = spline->ComputeJetAtS(time);
            Require(std::abs(jet[0] - position(time)) < 1e-14 &&
                        std::abs(jet[1] - velocity(time)) < 1e-14 &&
                        std::abs(jet[2] - acceleration(time)) < 1e-14 &&
                        jet[3] == 24.0,
                    "short phase interpolation must reproduce the cubic and "
                    "derivatives");
        }
    }
    const std::list<TrajectorySeg> short_profile{
        TrajectorySeg(0, 0.0, 0.0, 1.0, 0.0, 0.0),
        TrajectorySeg(0, 1e-7, 1e-7, 1.0, 0.0, 0.0)};
    Require(
        builder.InterpolateToPSpline(short_profile)->GetLastTimeStamp() == 1e-7,
        "a wholly short profile must not become empty");
}

void CheckTimeScalingRejectsOverflowWithoutMutation() {
    struct TimedTrajectory : TrajectoryBase<Rn<double, 2>> {
        TimedTrajectory() {
            trajectory_pspline_ = std::make_shared<PSpline>();
            trajectory_pspline_->PushBack(Constant(1.0), 1.04);
        }
    } trajectory;
    const auto original_knots = trajectory.GetBreakpoints();
    Require(!trajectory.SetMinimumDuration(std::numeric_limits<double>::max()),
            "finite scale must not produce infinite duration");
    Require(trajectory.GetTimeScale() == 1.0 &&
                trajectory.GetDuration() == 1.04 &&
                trajectory.GetBreakpoints() == original_knots,
            "rejected time scaling must preserve the prior trajectory");
}

struct SplineTrajectory : TrajectoryBase<Rn<double, 2>> {
    explicit SplineTrajectory(const std::shared_ptr<PSpline>& spline) {
        std::vector<Rn<double, 2>> waypoints(2);
        waypoints[0].Coeffs() << 0.0, 0.0;
        waypoints[1].Coeffs() << 1.0, 0.0;
        path_ = std::make_shared<PathBezierCurve<Rn<double, 2>>>(waypoints, 5,
                                                                 false, 0.0);
        trajectory_pspline_ = spline;
        dof_ = 2;
        max_velocity_ = max_acceleration_ = max_jerk_ =
            Eigen::VectorXd::Constant(2, 4.0);
        valid_ = true;
    }

    bool Enforce() {
        return EnforceJointLimits(max_velocity_, max_acceleration_, max_jerk_);
    }

    void SetDerivativeLimit(unsigned order, double limit) {
        (order == 2 ? max_acceleration_ : max_jerk_).setConstant(limit);
    }
};

void CheckContinuityUsesLocalTimeScale() {
    for (double tail_duration : {1.0, 1e15}) {
        auto spline = std::make_shared<PSpline>();
        // p=t-t^2+t^3/3 reaches p=1/3 with zero velocity and acceleration.
        Require(spline->PushBack(std::make_shared<Polynomial>(Eigen::Vector4d(
                                     0.0, 1.0, -1.0, 1.0 / 3.0)),
                                 1.0),
                "append smooth stopping phase");
        Require(spline->PushBack(Constant(1.0 / 3.0), tail_duration),
                "append stationary tail");
        SplineTrajectory trajectory(spline);
        const auto report = trajectory.GetConstraintReport();
        Require(report.within_limits && report.velocity_continuous &&
                    report.acceleration_continuous,
                "a long stationary tail must not create a false discontinuity");
        Require(std::abs(report.peak_velocity[0] - 1.0) < 1e-12 &&
                    std::abs(report.peak_acceleration[0] - 2.0) < 1e-12,
                "report must retain initial derivative peaks");
    }
}

void CheckLimitSamplingSupportsExtremeDurations() {
    for (double duration : {std::numeric_limits<double>::denorm_min(), 1e308}) {
        auto spline = std::make_shared<PSpline>();
        Require(spline->PushBack(Constant(0.25), duration),
                "append finite stationary phase");
        SplineTrajectory trajectory(spline);
        Require(trajectory.Enforce() && trajectory.GetTimeScale() == 1.0 &&
                    trajectory.GetDuration() == duration,
                "stationary trajectory must need no scaling at any duration");
        const auto report = trajectory.GetConstraintReport();
        Require(report.within_limits && report.peak_velocity.isZero() &&
                    report.peak_acceleration.isZero() &&
                    report.peak_jerk.isZero(),
                "stationary trajectory must report zero derivative peaks");
    }
}

void CheckReportRejectsNonFiniteStates() {
    for (unsigned order = 0; order < 4; ++order) {
        auto spline = std::make_shared<PSpline>();
        Eigen::Vector4d coefficients = Eigen::Vector4d::Zero();
        coefficients[order] = std::numeric_limits<double>::quiet_NaN();
        Require(
            spline->PushBack(std::make_shared<Polynomial>(coefficients), 1.0),
            "append diagnostic non-finite profile");
        SplineTrajectory trajectory(spline);
        bool rejected = false;
        try {
            trajectory.GetConstraintReport();
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        Require(rejected,
                "non-finite states must not produce a successful constraint "
                "report");
        Require(!trajectory.Enforce() && !trajectory.IsValid(),
                "internal evaluation failure must invalidate construction");
    }
}

void CheckSmallLimitsAllowRepresentableTimeScaling() {
    for (unsigned order : {2u, 3u}) {
        auto spline = std::make_shared<PSpline>();
        Eigen::Vector4d coefficients = Eigen::Vector4d::Zero();
        coefficients[order] = order == 2 ? 0.5 : 1.0 / 6.0;
        Require(
            spline->PushBack(std::make_shared<Polynomial>(coefficients), 0.1),
            "append unit derivative profile");
        SplineTrajectory trajectory(spline);
        const double limit = std::numeric_limits<double>::denorm_min();
        trajectory.SetDerivativeLimit(order, limit);
        const auto endpoint = trajectory.GetPosition(trajectory.GetDuration());
        Require(trajectory.Enforce() && trajectory.IsValid(),
                "a representable time scale must survive utilization overflow");
        Require(std::isfinite(trajectory.GetDuration()) &&
                    trajectory.GetTimeScale() > 1.0,
                "small positive limits require finite slowing");
        Require((trajectory.GetPosition(trajectory.GetDuration()) - endpoint)
                        .Coeffs()
                        .norm() < 1e-12,
                "limit enforcement must preserve the path endpoint");
        const auto report = trajectory.GetConstraintReport();
        const auto& peak =
            order == 2 ? report.peak_acceleration : report.peak_jerk;
        Require(
            report.within_limits && peak[0] > 0.0 && peak.maxCoeff() <= limit,
            "scaled unit derivative must remain representable and bounded");
    }
}

template <template <typename> class Profile>
void CheckCornerPhaseOwnership() {
    using Group = Rn<double, 2>;
    std::vector<Group> waypoints(3);
    waypoints[0].Coeffs() << 0.0, 0.0;
    waypoints[1].Coeffs() << 0.1, 0.0;
    waypoints[2].Coeffs() << 0.1, 0.2;
    auto path =
        std::make_shared<PathBezierCurve<Group>>(waypoints, 5, false, 0.0);
    const Eigen::Vector2d limits(1.0, 0.2);
    auto constraints =
        std::make_shared<TrajectoryConstraints>(limits, limits, limits);
    Profile<Group> trajectory(path, constraints);
    Require(trajectory.IsValid(), "construct stopped corner");
    for (double scale : {1.0, 1.7}) {
        Require(trajectory.SetMinimumDuration(scale * trajectory.GetDuration()),
                "slow stopped corner");
        const auto knots = trajectory.GetBreakpoints();
        bool found = false;
        for (std::size_t i = 1; i + 1 < knots.size(); ++i) {
            if ((trajectory.GetPosition(knots[i]) - waypoints[1])
                        .Coeffs()
                        .norm() > 1e-14 ||
                trajectory.GetVelocity(knots[i]).Coeffs().norm() > 1e-13)
                continue;
            const double step = 1e-8 * std::min(knots[i] - knots[i - 1],
                                                knots[i + 1] - knots[i]);
            const auto left = trajectory.GetState(knots[i] - step);
            const auto right = trajectory.GetState(knots[i] + step);
            Require(
                left.acceleration[0] < 0.0 && left.acceleration[1] == 0.0 &&
                    right.acceleration[0] == 0.0 && right.acceleration[1] > 0.0,
                "corner acceleration must use the active phase's direction");
            found = true;
            break;
        }
        Require(found, "locate stopped corner knot");
    }
}

void CheckLinearPathPhaseOwnership() {
    CheckCornerPhaseOwnership<TrajectoryDoubleS>();
    CheckCornerPhaseOwnership<TrajectoryTrapezium>();
}
void CheckLinearPhaseVelocityExtremum() {
    struct LinearPhase : TrajectoryBase<Rn<double, 2>> {
        explicit LinearPhase(bool cache_owner) {
            std::vector<Rn<double, 2>> points(2);
            points[0].Coeffs() << 0.0, 0.0;
            points[1].Coeffs() << 1.0, 0.0;
            path_ =
                std::make_shared<PathBezierCurve<Rn<double, 2>>>(points, 5, false, 0.0);
            constexpr double acceleration = 0.742468;
            trajectory_segments_ = {
                TrajectorySeg(0, 0.0, 0.0, 0.7, acceleration, -2.0),
                TrajectorySeg(0, 1.0, 0.7 + acceleration / 2.0 - 1.0 / 3.0,
                              0.7 + acceleration - 1.0, acceleration - 2.0, -2.0)};
            trajectory_pspline_ = InterpolateToPSpline(trajectory_segments_);
            dof_ = 2;
            max_velocity_ = Eigen::VectorXd::Constant(2, 0.5);
            max_acceleration_ = max_jerk_ = Eigen::VectorXd::Constant(2, 100.0);
            valid_ = InitializePhasePathSegments();
            if (!cache_owner)
                phase_path_segments_.clear();
            valid_ = valid_ &&
                     EnforceJointLimits(max_velocity_, max_acceleration_, max_jerk_);
        }
    };
    constexpr double stationary = 0.742468 / 2.0;
    const double expected = 1.01 * (0.7 + stationary * stationary) / 0.5;
    for (bool cache_owner : {false, true}) {
        const LinearPhase trajectory(cache_owner);
        Require(trajectory.IsValid() &&
                    std::abs(trajectory.GetTimeScale() - expected) < 2e-14,
                "linear phase must include the interior velocity extremum");
    }
}

void CheckReturningPhaseStillSamplesCurvedExcursion() {
    struct ReturningPhase : SplineTrajectory {
        explicit ReturningPhase(const std::shared_ptr<PSpline> &spline)
            : SplineTrajectory(spline) {
            std::vector<Rn<double, 2>> points(3);
            points[0].Coeffs() << 0.0, 0.0;
            points[1].Coeffs() << 1.0, 0.0;
            points[2].Coeffs() << 1.0, 1.0;
            path_ =
                std::make_shared<PathBezierCurve<Rn<double, 2>>>(points, 5, false, 0.1);
        }
    };
    auto spline = std::make_shared<PSpline>();
    // Both endpoints lie on the first line, but the interior reaches the
    // curve and returns. Equal owners alone do not establish a linear phase.
    Require(
        spline->PushBack(
            std::make_shared<Polynomial>(Eigen::Vector4d(0.25, 4.0, -4.0, 0.0)), 1.0),
        "append returning phase");
    ReturningPhase trajectory(spline);
    Require(!trajectory.GetConstraintReport(10001).within_limits,
            "curved excursion should exceed the original limits");
    Require(trajectory.Enforce() && trajectory.GetConstraintReport(10001).within_limits,
            "nonmonotone phase must retain curved-path limit checks");
}

}  // namespace

int main() {
    int failures = 0;
    for (const auto check :
         {CheckLinearPhaseVelocityExtremum,
          CheckReturningPhaseStillSamplesCurvedExcursion, CheckLocalBoundaries,
          CheckAppendValidation, CheckJetAndInvalidTimes, CheckJetExtremeCoefficients,
          CheckInvalidProfilesAreNotTruncated, CheckAllPhaseStatesAreFinite,
          CheckShortPositivePhasesArePreserved,
          CheckTimeScalingRejectsOverflowWithoutMutation,
          CheckContinuityUsesLocalTimeScale, CheckLimitSamplingSupportsExtremeDurations,
          CheckReportRejectsNonFiniteStates,
          CheckSmallLimitsAllowRepresentableTimeScaling,
          CheckLinearPathPhaseOwnership}) {
        try {
            check();
        } catch (const std::exception& error) {
            std::cerr << error.what() << '\n';
            ++failures;
        }
    }
    return failures ? 1 : 0;
}
