#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "holistic_motion/trajectory/TrajectoryTrapezium.h"

using namespace holistic_motion::robotics;
using Group = Rn<double, 2>;

namespace {
void CheckValue(double actual, double expected) {
    if (!std::isfinite(actual) ||
        (expected == 0.0 ? actual != 0.0 : std::abs(actual / expected - 1.0) > 2e-13))
        throw std::runtime_error("chain rule lost a representable derivative");
}

struct Monomial : PathSegmentBase<Group> {
    int degree;
    double inverse_length;
    explicit Monomial(int power, double scale = 0.0)
        : degree(power),
          inverse_length(scale != 0.0 ? scale : (power == 2 ? 1e150 : 1e100)) {
        length_ = 1.0 / inverse_length;
        path_seg_type_ = PathSegType::Bezier5thSeg;
    }
    Group GetConfig(double s) const override {
        Group result;
        result.Coeffs() << std::pow(s * inverse_length, degree), 0.0;
        return result;
    }
    Group::Tangent GetTangent(double s) const override {
        Group::Tangent result;
        result.Coeffs() << degree * inverse_length *
                               std::pow(s * inverse_length, degree - 1),
            0.0;
        return result;
    }
    Group::Tangent GetCurvature(double s) const override {
        Group::Tangent result;
        result.Coeffs() << degree * (degree - 1) * inverse_length * inverse_length *
                               std::pow(s * inverse_length, degree - 2),
            0.0;
        return result;
    }
    Group::Tangent GetTorsion(double) const override {
        Group::Tangent result = Group::Tangent::ZeroHelper();
        if (degree == 3)
            result.Coeffs()[0] = 6.0 * inverse_length * inverse_length * inverse_length;
        return result;
    }
};

struct SingleSegmentPath : PathBase<Group> {
    explicit SingleSegmentPath(std::shared_ptr<PathSegmentBase<Group>> segment) {
        path_segments_.push_back(segment);
        length_ = segment->GetLength();
        valid_ = segment->IsValid();
    }
};

struct QueryProbe : TrajectoryBase<Group> {
    QueryProbe(std::shared_ptr<PathSegmentBase<Group>> segment, double speed,
               double duration)
        : QueryProbe(segment, Eigen::Vector4d(0.0, speed, 0.0, 0.0), duration) {}
    QueryProbe(std::shared_ptr<PathSegmentBase<Group>> segment,
               const Eigen::Vector4d &coefficients, double duration)
        : QueryProbe(std::make_shared<SingleSegmentPath>(segment), coefficients,
                     duration) {}
    QueryProbe(std::shared_ptr<PathBase<Group>> path,
               const Eigen::Vector4d &coefficients, double duration) {
        path_ = std::move(path);
        trajectory_pspline_ = std::make_shared<PSpline>();
        trajectory_pspline_->PushBack(std::make_shared<Polynomial>(coefficients),
                                      duration);
        dof_ = 2;
        valid_ = true;
    }
    using TrajectoryBase::EnforceJointLimits;
};

void CheckSmallSpeedPowers() {
    for (int degree : {2, 3}) {
        auto segment = std::make_shared<Monomial>(degree);
        const double speed = degree == 2 ? 1e-200 : 1e-150;
        const double duration = segment->GetLength() / speed;
        QueryProbe trajectory(segment, speed, duration);
        // q(t)=(t/T)^degree is an independent time-domain oracle. Its
        // acceleration/jerk remain nonzero even though speed^degree underflows.
        for (double fraction : {0.0, 0.125, 0.5, 0.875, 1.0}) {
            const auto state = trajectory.GetState(fraction * duration);
            const double velocity = degree * std::pow(fraction, degree - 1) / duration;
            const double acceleration = degree * (degree - 1) *
                                        std::pow(fraction, degree - 2) / duration /
                                        duration;
            const double jerk =
                degree == 2 ? 0.0 : 6.0 / duration / duration / duration;
            CheckValue(state.position.Coeffs()[0], std::pow(fraction, degree));
            CheckValue(state.velocity.Coeffs()[0], velocity);
            CheckValue(state.acceleration.Coeffs()[0], acceleration);
            CheckValue(state.jerk.Coeffs()[0], jerk);
            CheckValue(trajectory.GetAcceleration(fraction * duration).Coeffs()[0],
                       acceleration);
            CheckValue(trajectory.GetJerk(fraction * duration).Coeffs()[0], jerk);
        }
    }
}

void CheckMixedJerkProduct() {
    for (bool large_curvature : {false, true}) {
        const double inverse_length = large_curvature ? 9e153 : 1e150;
        const double linear = large_curvature ? 1e-155 : 1e-200;
        const double quadratic = 0.5 * linear;
        auto segment = std::make_shared<Monomial>(2, inverse_length);
        QueryProbe trajectory(segment, Eigen::Vector4d(0.0, linear, quadratic, 0.0),
                              1.0);
        // q(t)=(a*t+b*t^2)^2 has jerk 12*a*b+24*b^2*t. All derivatives
        // are representable even when 3*curvature overflows or speed*acc
        // underflows. Form the reference entirely in normalized coordinates.
        const double a = inverse_length * linear;
        const double b = inverse_length * quadratic;
        for (double time : {0.0, 0.125, 0.5, 0.875, 1.0}) {
            const double expected = 12.0 * a * b + 24.0 * b * b * time;
            CheckValue(trajectory.GetJerk(time).Coeffs()[0], expected);
            CheckValue(trajectory.GetState(time).jerk.Coeffs()[0], expected);
        }
    }
}

void CheckFastLinearMotion() {
    std::vector<Group> points(2);
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << 1.0, 0.0;
    auto path = std::make_shared<PathBezierCurve<Group>>(points, 5, false, 0.0);
    auto limits = std::make_shared<TrajectoryConstraints>(
        Eigen::Vector2d::Constant(1e150), Eigen::Vector2d::Constant(1e300),
        Eigen::Vector2d::Constant(1e300));
    TrajectoryTrapezium<Group> trajectory(path, limits);
    if (!trajectory.IsValid())
        throw std::runtime_error("zero torsion multiplied an overflowing speed cube");
    for (double fraction : {0.0, 0.1, 0.5, 0.9, 1.0}) {
        const auto state = trajectory.GetState(fraction * trajectory.GetDuration());
        if (!state.position.Coeffs().allFinite() ||
            !state.velocity.Coeffs().allFinite() ||
            !state.acceleration.Coeffs().allFinite() ||
            state.velocity.Coeffs().cwiseAbs().maxCoeff() > 1e150 ||
            state.acceleration.Coeffs().cwiseAbs().maxCoeff() > 1e300)
            throw std::runtime_error("fast linear motion violates finite limits");
        CheckValue(state.jerk.Coeffs()[0], 0.0);
        CheckValue(trajectory.GetJerk(fraction * trajectory.GetDuration()).Coeffs()[0],
                   0.0);
    }
    CheckValue(trajectory.GetPosition(trajectory.GetDuration()).Coeffs()[0], 1.0);
}

void CheckTinyLimitReciprocal() {
    std::array<Group, 2> points;
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << 1.0, 0.0;
    QueryProbe trajectory(std::make_shared<PathSegLinear<Group>>(points), 1e-310, 1.0);
    if (!trajectory.EnforceJointLimits(Eigen::Vector2d::Constant(1e-310),
                                       Eigen::Vector2d::Ones(),
                                       Eigen::Vector2d::Ones()))
        throw std::runtime_error("an overflowing reciprocal rejected a finite ratio");
    CheckValue(trajectory.GetTimeScale(), 1.01);
}
void CheckOverflowingUtilizationRoots() {
    for (bool cubic : {false, true}) {
        const double derivative = cubic ? 1e80 : 1e100;
        Eigen::Vector4d coefficients = Eigen::Vector4d::Zero();
        coefficients[cubic ? 3 : 2] = derivative / (cubic ? 6.0 : 2.0);
        std::array<Group, 2> points;
        points[0].Coeffs() << 0.0, 0.0;
        points[1].Coeffs() << coefficients.sum(), 0.0;
        QueryProbe trajectory(std::make_shared<PathSegLinear<Group>>(points),
                              coefficients, 1.0);
        const Eigen::Vector2d velocity = Eigen::Vector2d::Constant(1e200);
        const Eigen::Vector2d acceleration =
            Eigen::Vector2d::Constant(cubic ? 1e200 : 1e-250);
        const Eigen::Vector2d jerk = Eigen::Vector2d::Constant(cubic ? 1e-250 : 1e200);
        if (!trajectory.EnforceJointLimits(velocity, acceleration, jerk))
            throw std::runtime_error("an overflowing ratio lost its finite root");
        CheckValue(trajectory.GetTimeScale() / (cubic ? 1e110 : 1e175), 1.01);
    }
}
void CheckCachedCurveLimitSampling() {
    using Curve = PathSegBezierCurve5th<Group>;
    struct VirtualCurve : Curve {
        explicit VirtualCurve(const Curve &curve) : Curve(curve) {}
        Group GetConfig(double s) const override {
            ++calls[0];
            return Curve::GetConfig(s);
        }
        Group::Tangent GetTangent(double s) const override {
            ++calls[1];
            return Curve::GetTangent(s);
        }
        Group::Tangent GetCurvature(double s) const override {
            ++calls[2];
            return Curve::GetCurvature(s);
        }
        Group::Tangent GetTorsion(double s) const override {
            ++calls[3];
            return Curve::GetTorsion(s);
        }
        mutable std::array<unsigned, 4> calls{};
    };
    for (double scale : {1e-4, 1.0, 1e4}) {
        std::array<Group, 3> points;
        points[0].Coeffs() << -0.2 * scale, 0.3 * scale;
        points[1].Coeffs() << 0.7 * scale, -0.4 * scale;
        points[2].Coeffs() << 1.1 * scale, 0.6 * scale;
        // Nonzero endpoint curvature exercises all stored control differences.
        auto native =
            std::make_shared<Curve>(points, 0.0, 0.8, 0.7 / scale, 1.2, -0.4 / scale);
        if (!native->IsValid())
            throw std::runtime_error("invalid cached-limit curve fixture");
        const double length = native->GetLength();
        for (const Eigen::Vector4d &coefficients :
             {Eigen::Vector4d(0.0, length, 0.0, 0.0),
              Eigen::Vector4d(length, -length, 0.0, 0.0),
              Eigen::Vector4d(0.0, 4.0 * length, -4.0 * length, 0.0),
              Eigen::Vector4d(0.0, 0.0, 0.0, length)}) {
            for (unsigned order = 0; order < 3; ++order) {
                auto scalar = std::make_shared<VirtualCurve>(*native);
                QueryProbe cached(native, coefficients, 1.0);
                QueryProbe reference(scalar, coefficients, 1.0);
                std::array<Eigen::Vector2d, 3> limits{
                    Eigen::Vector2d::Constant(scale),
                    Eigen::Vector2d::Constant(2.0 * scale),
                    Eigen::Vector2d::Constant(5.0 * scale)};
                limits[order] *= 1e-3;
                if (!cached.EnforceJointLimits(limits[0], limits[1], limits[2]) ||
                    !reference.EnforceJointLimits(limits[0], limits[1], limits[2]) ||
                    cached.GetTimeScale() != reference.GetTimeScale())
                    throw std::runtime_error(
                        "cached limit sampling differs from scalar geometry");
                if (scalar->calls[0] < 65 ||
                    !std::all_of(
                        scalar->calls.begin(), scalar->calls.end(),
                        [&](unsigned calls) { return calls == scalar->calls[0]; }))
                    throw std::runtime_error("limit sampling bypassed virtual queries");
            }
        }
    }
}

void CheckLimitSamplingRejectsNonfinitePosition() {
    using Curve = PathSegBezierCurve5th<Group>;
    struct InvalidPositionCurve : Curve {
        using Curve::Curve;
        Group GetConfig(double) const override {
            Group result;
            result.Coeffs().setConstant(std::numeric_limits<double>::infinity());
            return result;
        }
        Group::Tangent GetTangent(double) const override {
            return Group::Tangent::ZeroHelper();
        }
        Group::Tangent GetCurvature(double) const override {
            return Group::Tangent::ZeroHelper();
        }
        Group::Tangent GetTorsion(double) const override {
            return Group::Tangent::ZeroHelper();
        }
    };
    std::array<Group, 3> points;
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << 1.0, 0.0;
    points[2].Coeffs() << 1.0, 1.0;
    auto curve = std::make_shared<InvalidPositionCurve>(points, 0.0);
    QueryProbe trajectory(curve, curve->GetLength(), 1.0);
    if (trajectory.EnforceJointLimits(Eigen::Vector2d::Ones(), Eigen::Vector2d::Ones(),
                                      Eigen::Vector2d::Ones()))
        throw std::runtime_error("finite derivatives hid a nonfinite position");
}

void CheckLimitSamplingValidatesGeometryBeforeCaching() {
    std::array<Group, 3> points;
    for (auto &point : points)
        point.Coeffs().setZero();
    auto invalid = std::make_shared<PathSegBezierCurve5th<Group>>(points, 0.0);
    if (invalid->IsValid())
        throw std::runtime_error("degenerate curve fixture unexpectedly valid");
    struct InconsistentPath : SingleSegmentPath {
        explicit InconsistentPath(std::shared_ptr<PathSegmentBase<Group>> segment)
            : SingleSegmentPath(std::move(segment)) {
            // Even a custom path claiming validity must not cause the sampler
            // to read controls missing from a rejected native curve.
            valid_ = true;
            length_ = 1.0;
        }
    };
    QueryProbe trajectory(std::make_shared<InconsistentPath>(invalid),
                          Eigen::Vector4d(0.0, 1.0, 0.0, 0.0), 1.0);
    try {
        trajectory.EnforceJointLimits(Eigen::Vector2d::Ones(), Eigen::Vector2d::Ones(),
                                      Eigen::Vector2d::Ones());
    } catch (const std::logic_error &) {
        return;
    }
    throw std::runtime_error("limit sampling accepted invalid curve geometry");
}

void CheckShortCurvesWithNonlinearTimeLaws() {
    std::vector<Group> points(4);
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << 0.513, 0.0;
    points[2].Coeffs() << 0.51305, 0.00005;
    points[3].Coeffs() << 1.0, 1.0;
    auto path = std::make_shared<PathBezierCurve<Group>>(points, 5, false, 1e-4);
    if (!path->IsValid())
        throw std::runtime_error("invalid short-curve fixture");
    const auto segments = path->GetPathSegments();
    if (std::none_of(segments.begin(), segments.end(), [](const auto &segment) {
            return segment->GetPathSegType() == PathSegType::Bezier5thSeg;
        }))
        throw std::runtime_error("short-curve fixture lost its curves");
    for (int power : {1, 2, 3}) {
        // One monotone phase crosses two curves much shorter than its global
        // sampling step. Quadratic/cubic laws require a nonlinear inverse.
        Eigen::Vector4d coefficients = Eigen::Vector4d::Zero();
        coefficients[power] = path->GetLength();
        QueryProbe trajectory(path, coefficients, 1.0);
        if (!trajectory.EnforceJointLimits(Eigen::Vector2d::Ones(),
                                           Eigen::Vector2d::Ones(),
                                           Eigen::Vector2d::Ones()))
            throw std::runtime_error("nonlinear time-law limit enforcement failed");
        // Sample independently in geometry and invert s=L*t^power directly;
        // this grid cannot miss either short curve inside the shared phase.
        for (const auto &segment : segments) {
            for (int sample = 0; sample <= 1000; ++sample) {
                const double position = segment->GetStartParameter() +
                                        segment->GetLength() * (sample / 1000.0);
                const double time =
                    std::pow(position / path->GetLength(), 1.0 / power) *
                    trajectory.GetDuration();
                const auto state = trajectory.GetState(time);
                for (const auto &derivative :
                     {state.velocity.Coeffs(), state.acceleration.Coeffs(),
                      state.jerk.Coeffs()}) {
                    if (!derivative.allFinite() ||
                        derivative.cwiseAbs().maxCoeff() > 1.0 + 1e-10)
                        throw std::runtime_error(
                            "shared nonlinear time phase missed a short-curve peak");
                }
            }
        }
    }
}
} // namespace

int main() {
    try {
        CheckSmallSpeedPowers();
        CheckMixedJerkProduct();
        CheckFastLinearMotion();
        CheckTinyLimitReciprocal();
        CheckOverflowingUtilizationRoots();
        CheckShortCurvesWithNonlinearTimeLaws();
        CheckCachedCurveLimitSampling();
        CheckLimitSamplingRejectsNonfinitePosition();
        CheckLimitSamplingValidatesGeometryBeforeCaching();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
