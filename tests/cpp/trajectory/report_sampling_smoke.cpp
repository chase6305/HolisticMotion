#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>
#include <vector>

#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

using namespace holistic_motion::robotics;

namespace {
template <typename Group, typename Curve = PathSegBezierCurve5th<Group>>
struct ScalarCurve : Curve {
    using Tangent = typename Group::Tangent;
    explicit ScalarCurve(const Curve &curve) : Curve(curve) {}
    Group GetConfig(double s) const override {
        ++calls[0];
        return Curve::GetConfig(s);
    }
    Tangent GetTangent(double s) const override {
        ++calls[1];
        return Curve::GetTangent(s);
    }
    Tangent GetCurvature(double s) const override {
        ++calls[2];
        return Curve::GetCurvature(s);
    }
    Tangent GetTorsion(double s) const override {
        ++calls[3];
        return Curve::GetTorsion(s);
    }
    mutable std::array<unsigned, 4> calls{};
};

template <typename Group, typename Curve = PathSegBezierCurve5th<Group>>
struct ScalarPath : PathBase<Group> {
    std::vector<std::shared_ptr<ScalarCurve<Group, Curve>>> curves;
    explicit ScalarPath(const PathBase<Group> &path) {
        for (const auto &segment : path.GetPathSegments()) {
            if (typeid(*segment) == typeid(Curve)) {
                auto curve = std::make_shared<ScalarCurve<Group, Curve>>(
                    static_cast<const Curve &>(*segment));
                curves.push_back(curve);
                this->path_segments_.push_back(curve);
            } else if (typeid(*segment) == typeid(PathSegLinear<Group>)) {
                this->path_segments_.push_back(
                    std::make_shared<ScalarCurve<Group, PathSegLinear<Group>>>(
                        static_cast<const PathSegLinear<Group> &>(*segment)));
            } else {
                this->path_segments_.push_back(segment);
            }
        }
        this->path_type_ = path.GetType();
        this->blend_tolerance_ = path.GetBlendTolerance();
        this->waypoints_ = path.GetWaypoints();
        this->length_ = path.GetLength();
        this->valid_ = path.IsValid();
    }
};

template <typename Report> void EqualReports(const Report &a, const Report &b) {
    if (a.peak_velocity != b.peak_velocity ||
        a.peak_acceleration != b.peak_acceleration || a.peak_jerk != b.peak_jerk ||
        a.velocity_utilization != b.velocity_utilization ||
        a.acceleration_utilization != b.acceleration_utilization ||
        a.jerk_utilization != b.jerk_utilization ||
        a.maximum_velocity_jump != b.maximum_velocity_jump ||
        a.maximum_acceleration_jump != b.maximum_acceleration_jump ||
        a.maximum_utilization != b.maximum_utilization ||
        a.within_limits != b.within_limits ||
        a.velocity_continuous != b.velocity_continuous ||
        a.acceleration_continuous != b.acceleration_continuous)
        throw std::runtime_error("cached report differs from virtual geometry");
}

template <typename Group> struct CustomClockProbe : TrajectoryBase<Group> {
    CustomClockProbe(const std::shared_ptr<PathBase<Group>> &path,
                     const Eigen::Vector4d &coefficients) {
        this->path_ = path;
        this->trajectory_pspline_ = std::make_shared<PSpline>();
        this->trajectory_pspline_->PushBack(std::make_shared<Polynomial>(coefficients),
                                            1.0);
        this->dof_ = Group::DoF;
        this->max_velocity_ = this->max_acceleration_ = this->max_jerk_ =
            Eigen::VectorXd::Ones(Group::DoF);
        this->valid_ = true;
    }
};

template <typename Group, int Degree = 5> void CheckMultiSegmentReports() {
    using Curve = std::conditional_t<Degree == 2, PathSegBezierCurve2nd<Group>,
                                     PathSegBezierCurve5th<Group>>;
    for (double scale : {1e-2, 1.0, 1e4}) {
        std::vector<Group> points(8);
        for (std::size_t i = 0; i < points.size(); ++i) {
            if constexpr (std::is_same_v<Group, SE3d>) {
                points[i] =
                    SE3d(Eigen::Vector3d(0.2 * i * scale, scale * std::sin(0.7 * i),
                                         scale * std::cos(0.3 * i)),
                         SO3d(0.1 * std::sin(i), 0.1 * std::cos(i), 0.02 * i));
            } else {
                for (int j = 0; j < Group::DoF; ++j)
                    points[i].Coeffs()[j] = scale * std::sin(0.7 * i + 0.3 * j);
            }
        }
        auto native = std::make_shared<PathBezierCurve<Group>>(points, Degree, false,
                                                               0.005 * scale);
        auto scalar = std::make_shared<ScalarPath<Group, Curve>>(*native);
        if (!native->IsValid() || scalar->curves.empty()) {
            std::cerr << "DoF=" << Group::DoF
                      << " pose=" << std::is_same_v<Group, SE3d> << " scale=" << scale
                      << " valid=" << native->IsValid()
                      << " curves=" << scalar->curves.size() << "\n";
            throw std::runtime_error("report fixture must contain valid curves");
        }
        // Exercise automatically fitted clocks where these fixtures support
        // them; explicit clocks below also cover quadratic geometry directly.
        if constexpr (Degree == 5) {
            const Eigen::VectorXd limit = Eigen::VectorXd::Constant(Group::DoF, scale);
            auto limits = std::make_shared<TrajectoryConstraints>(limit, 2.0 * limit,
                                                                  5.0 * limit);
            TrajectoryDoubleS<Group> cached(native, limits), reference(scalar, limits);
            if (!cached.IsValid() || !reference.IsValid() ||
                cached.GetDuration() != reference.GetDuration()) {
                std::cerr << "trajectory DoF=" << Group::DoF << " scale=" << scale
                          << " native=" << cached.IsValid() << ","
                          << cached.GetDuration() << " virtual=" << reference.IsValid()
                          << "," << reference.GetDuration() << "\n";
                throw std::runtime_error("report fixture trajectories differ");
            }
            for (double slowdown : {1.0, 1.7}) {
                const double duration = slowdown * cached.GetDuration();
                if (!cached.SetMinimumDuration(duration) ||
                    !reference.SetMinimumDuration(duration))
                    throw std::runtime_error("report fixture cannot slow down");
                for (std::size_t samples : {2, 3, 65, 257}) {
                    for (const auto &curve : scalar->curves)
                        curve->calls.fill(0);
                    EqualReports(cached.GetConstraintReport(samples),
                                 reference.GetConstraintReport(samples));
                    unsigned total = 0;
                    for (const auto &curve : scalar->curves) {
                        total += curve->calls[0];
                        for (unsigned calls : curve->calls)
                            if (calls != curve->calls[0])
                                throw std::runtime_error(
                                    "report bypassed virtual query");
                    }
                    if (samples == 257 && total == 0)
                        throw std::runtime_error("report fixture missed every curve");
                }
            }
        }
        const double length = native->GetLength();
        // Revisit earlier geometry and cross several curves inside one phase.
        // A report-local cache must follow the actual query, not assume that
        // increasing time means increasing path position.
        for (const auto &coefficients :
             {Eigen::Vector4d(length, -length, 0.0, 0.0),
              Eigen::Vector4d(0.0, 4.0 * length, -4.0 * length, 0.0),
              Eigen::Vector4d(0.0, 0.0, 0.0, length),
              Eigen::Vector4d(0.4 * length, 0.0, 0.0, 0.0)}) {
            CustomClockProbe<Group> native_probe(native, coefficients);
            CustomClockProbe<Group> scalar_probe(scalar, coefficients);
            for (std::size_t samples : {2, 65, 257})
                EqualReports(native_probe.GetConstraintReport(samples),
                             scalar_probe.GetConstraintReport(samples));
            if (!native_probe.SetMinimumDuration(1.7) ||
                !scalar_probe.SetMinimumDuration(1.7))
                throw std::runtime_error("cannot rescale custom report clock");
            EqualReports(native_probe.GetConstraintReport(257),
                         scalar_probe.GetConstraintReport(257));
        }
    }
}

using Group = Rn<double, 2>;
struct SinglePath : PathBase<Group> {
    explicit SinglePath(const std::shared_ptr<PathSegmentBase<Group>> &segment) {
        path_segments_.push_back(segment);
        length_ = 1.0;
        valid_ = true;
    }
};
struct ReportProbe : TrajectoryBase<Group> {
    explicit ReportProbe(const std::shared_ptr<PathSegmentBase<Group>> &segment) {
        path_ = std::make_shared<SinglePath>(segment);
        trajectory_pspline_ = std::make_shared<PSpline>();
        trajectory_pspline_->PushBack(
            std::make_shared<Polynomial>(Eigen::Vector4d(0.0, 1.0, 0.0, 0.0)), 1.0);
        dof_ = 2;
        max_velocity_ = max_acceleration_ = max_jerk_ = Eigen::Vector2d::Ones();
        valid_ = true;
    }
};

void CheckLinearReportQueries() {
    struct ChangingLine : PathSegLinear<Group> {
        using PathSegLinear<Group>::PathSegLinear;
        mutable unsigned calls{0};
        bool nonfinite_position{false};
        Group GetConfig(double s) const override {
            if (nonfinite_position) {
                Group result;
                result.Coeffs().setConstant(std::numeric_limits<double>::infinity());
                return result;
            }
            return PathSegLinear<Group>::GetConfig(s);
        }
        Group::Tangent GetTangent(double) const override {
            return Group::Tangent(Eigen::Vector2d::Constant(++calls));
        }
    };
    std::array<Group, 2> points;
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << 1.0, 0.0;
    auto custom = std::make_shared<ChangingLine>(points);
    ReportProbe custom_probe(custom);
    if (custom_probe.GetConstraintReport(65).peak_velocity[0] != 65.0 ||
        custom->calls != 65 ||
        custom_probe.GetConstraintReport(3).peak_velocity[0] != 68.0 ||
        custom->calls != 68)
        throw std::runtime_error("report cached a custom line's changing derivative");
    custom->nonfinite_position = true;
    bool rejected = false;
    try {
        custom_probe.GetConstraintReport(3);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    if (!rejected)
        throw std::runtime_error("report accepted a nonfinite custom line position");

    auto native = std::make_shared<PathSegLinear<Group>>(points);
    ReportProbe native_probe(native);
    if (native_probe.GetConstraintReport(65).peak_velocity != Eigen::Vector2d(1, 0))
        throw std::runtime_error("incorrect native linear report");
    // A workspace may cache a line only for the duration of one report.
    points[1].Coeffs() << 0.0, 1.0;
    *native = PathSegLinear<Group>(points);
    if (native_probe.GetConstraintReport(65).peak_velocity != Eigen::Vector2d(0, 1))
        throw std::runtime_error("report reused a stale native linear tangent");
    points[1] = points[0];
    *native = PathSegLinear<Group>(points);
    const auto constant = native_probe.GetConstraintReport(65);
    if (!constant.within_limits || !constant.peak_velocity.isZero(0.0) ||
        !constant.peak_acceleration.isZero(0.0) || !constant.peak_jerk.isZero(0.0))
        throw std::runtime_error("report divided by a zero-length line");
}

void CheckCustomQueriesAndFailures() {
    struct ChangingCurve : PathSegBezierCurve5th<Group> {
        using PathSegBezierCurve5th<Group>::PathSegBezierCurve5th;
        mutable unsigned calls{0};
        bool nonfinite_position{false};
        Group GetConfig(double s) const override {
            if (nonfinite_position) {
                Group result;
                result.Coeffs().setConstant(std::numeric_limits<double>::infinity());
                return result;
            }
            return PathSegBezierCurve5th<Group>::GetConfig(s);
        }
        Group::Tangent GetTangent(double) const override {
            return Group::Tangent(Eigen::Vector2d::Constant(++calls));
        }
    };
    std::array<Group, 3> points;
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << 1.0, 0.0;
    points[2].Coeffs() << 1.0, 1.0;
    auto curve = std::make_shared<ChangingCurve>(points, 0.0);
    ReportProbe probe(curve);
    if (probe.GetConstraintReport(65).peak_velocity[0] != 65.0 || curve->calls != 65 ||
        probe.GetConstraintReport(3).peak_velocity[0] != 68.0 || curve->calls != 68)
        throw std::runtime_error("report cached a custom curve's changing derivative");
    curve->nonfinite_position = true;
    try {
        probe.GetConstraintReport(3);
        throw std::logic_error("report accepted a nonfinite custom position");
    } catch (const std::runtime_error &) {
    }

    points[1].Coeffs()[0] = std::numeric_limits<double>::infinity();
    auto invalid = std::make_shared<PathSegBezierCurve5th<Group>>(points, 0.0);
    if (invalid->IsValid())
        throw std::runtime_error("invalid report fixture was accepted");
    ReportProbe invalid_probe(invalid);
    try {
        invalid_probe.GetConstraintReport(3);
    } catch (const std::logic_error &) {
        return;
    }
    throw std::runtime_error("report accessed invalid native curve controls");
}
} // namespace

int main() {
    holistic_motion::utility::SetVerbosityLevel(
        holistic_motion::utility::VerbosityLevel::Error);
    try {
        CheckMultiSegmentReports<Rn<double, 2>>();
        CheckMultiSegmentReports<Rn<double, 7>>();
        CheckMultiSegmentReports<Rn<double, 32>>();
        CheckMultiSegmentReports<SE3d>();
        CheckMultiSegmentReports<SE3d, 2>();
        CheckLinearReportQueries();
        CheckCustomQueriesAndFailures();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
