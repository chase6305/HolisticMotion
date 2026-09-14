#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "holistic_motion/trajectory/PathBezierCurve.h"

using namespace holistic_motion::robotics;
using Group = Rn<double, 2>;

std::vector<Group> Corners(double scale = 1.0) {
    std::vector<Group> points(4);
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << scale, 0.0;
    points[2].Coeffs() << scale, scale;
    points[3].Coeffs() << 2.0 * scale, scale;
    return points;
}

void CheckQuadraticCorners() {
    const auto points = Corners();
    PathBezierCurve<Group> path(points, 2, false, 0.01);
    if (!path.IsValid())
        throw std::runtime_error("quadratic path is invalid");
    int corner = 1;
    for (const auto &segment : path.GetPathSegments()) {
        if (segment->GetPathSegType() != PathSegType::Bezier2ndSeg)
            continue;
        const auto middle = segment->GetConfig(segment->GetStartParameter() +
                                               0.5 * segment->GetLength());
        const double distance = (middle - points.at(corner++)).Coeffs().norm();
        if (std::abs(distance - 0.01) > 1e-12)
            throw std::runtime_error(
                "quadratic blend used the wrong corner or tolerance");
    }
    if (corner != 3)
        throw std::runtime_error("missing quadratic corner");
    for (double scale : {0.25, 8.0}) {
        PathBezierCurve<Group> scaled(Corners(scale), 2, false, 0.01 * scale);
        if (!scaled.IsValid() ||
            std::abs(scaled.GetLength() / scale - path.GetLength()) > 1e-12)
            throw std::runtime_error("quadratic blend changed shape with length units");
        for (int index = 0; index <= 100; ++index) {
            const double fraction = index / 100.0;
            if ((scaled.GetConfig(fraction * scaled.GetLength()).Coeffs() / scale -
                 path.GetConfig(fraction * path.GetLength()).Coeffs())
                    .norm() > 1e-12)
                throw std::runtime_error("quadratic samples changed with length units");
        }
    }
}

void CheckReversal(int degree) {
    auto points = Corners();
    points[2].Coeffs() << 0.0, 0.0;
    points[3].Coeffs() << 0.0, 1.0;
    PathBezierCurve<Group> path(points, degree, false, 0.01);
    if (!path.IsValid())
        throw std::runtime_error("reversing path must remain valid");
    if ((path.GetConfig(1.0) - points[1]).Coeffs().norm() > 1e-12)
        throw std::runtime_error("reversal must retain its exact waypoint");
    const auto segments = path.GetPathSegments();
    if (segments.size() < 2 ||
        segments[0]->GetPathSegType() != PathSegType::LinearSeg ||
        segments[1]->GetPathSegType() != PathSegType::LinearSeg)
        throw std::runtime_error("reversal must join two linear segments");
    for (int index = 0; index <= 100; ++index)
        if (!path.GetConfig(path.GetLength() * index / 100.0).Coeffs().allFinite())
            throw std::runtime_error("reversing path contains non-finite samples");
}

void CheckFinalDebugRecord() {
    using namespace holistic_motion::utility;
    auto &logger = Logger::GetInstance();
    int triples = 0;
    struct ResetLogger {
        Logger &logger;
        ~ResetLogger() {
            logger.ResetRecordFunction();
            logger.SetVerbosityLevel(VerbosityLevel::Warning);
        }
    } reset{logger};
    logger.SetRecordFunction([&](const LogRecord &record) {
        if (record.message.find("waypoint0:") == 0 &&
            record.message.find("waypoint2:") != std::string::npos)
            ++triples;
    });
    logger.SetVerbosityLevel(VerbosityLevel::Debug);
    {
        PathBezierCurve<Group> path(Corners(), 5, false, 0.01);
        if (!path.IsValid())
            throw std::runtime_error("debug path is invalid");
    }
    if (triples != 2)
        throw std::runtime_error(
            "final segment must not read a nonexistent third waypoint");
}

void CheckShortReversalBlend() {
    std::vector<Group> points(3);
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << 0.001, 0.0;
    points[2].Coeffs() << 0.0, 0.00001;
    PathBezierCurve<Group> path(points, 5, false, 0.00003);
    if (!path.IsValid() || path.GetNumOfPathSegments() != 2 ||
        (path.GetConfig(0.001) - points[1]).Coeffs().norm() > 1e-12)
        throw std::runtime_error("short reversal blend must retain its waypoint");
    for (const auto &segment : path.GetPathSegments())
        if (segment->GetPathSegType() != PathSegType::LinearSeg)
            throw std::runtime_error("unresolvable blend must become a linear join");
}

void CheckStraightSegmentDerivatives() {
    for (double step : {0x1p-16, 1.0, 0x1p16}) {
        std::array<Group, 3> points;
        points[0].Coeffs() << 0.0, 0.0;
        points[1].Coeffs() << 2.0 * step, 0.0;
        points[2].Coeffs() << 5.0 * step, 0.0;
        PathSegBezierCurve5th<Group> segment(points, 0.0);
        if (!segment.IsValid() || segment.GetLength() != 5.0 * step)
            throw std::runtime_error("straight Bezier fixture changed length");
        for (double fraction : {0.0, 0.13, 0.33, 0.7, 0.9, 1.0}) {
            const double s = fraction * segment.GetLength();
            const auto curvature = segment.GetCurvature(s).Coeffs();
            const auto torsion = segment.GetTorsion(s).Coeffs();
            if (!curvature.isZero(0.0) || !torsion.isZero(0.0)) {
                std::cerr << "step=" << step << " fraction=" << fraction
                          << " curvature=" << curvature.transpose()
                          << " torsion=" << torsion.transpose() << '\n';
                throw std::runtime_error("straight Bezier acquired false derivatives");
            }
        }
    }
}

void CheckCurveLengthNearLinearEquation() {
    for (double scale : {0x1p-10, 1.0, 0x1p10}) {
        std::array<Group, 3> points;
        points[0].Coeffs() << 0.0, 0.0;
        points[1].Coeffs() << 2.0 * scale, 0.0;
        points[2].Coeffs() << 5.0 * scale, 0.0;
        for (double norm : {1.1428549107121058, 1.142857120535714, 1.1428571426339287,
                            1.1428571428125, 8.0 / 7.0, -1.0, 0.0, 1.0, 2.0}) {
            const PathSegBezierCurve5th<Group> segment(points, 0.0, norm, 0.0, norm,
                                                       0.0);
            // For collinear endpoints separated by D with equal tangent norms n,
            // the length polynomial factors as
            // ((16+14n)L-30D) * ((16-14n)L+30D).
            // This analytic oracle avoids subtracting nearly equal quantities.
            const long double n = norm;
            const long double d = 5.0L * scale;
            long double expected = 30.0L * d / (16.0L + 14.0L * n);
            if (norm == 2.0)
                expected = 30.0L * d / (14.0L * n - 16.0L);
            const double relative_error =
                std::abs(static_cast<double>(segment.GetLength() / expected - 1.0L));
            if (!segment.IsValid() ||
                relative_error > 32.0 * std::numeric_limits<double>::epsilon()) {
                std::cerr << "norm=" << norm << " scale=" << scale
                          << " length relative error=" << relative_error << '\n';
                throw std::runtime_error("Bezier length disagrees with factored root");
            }
        }
    }
}

void CheckLargeCurveDerivatives() {
    const auto points = [](double scale) {
        std::array<Group, 3> result;
        result[0].Coeffs() << 0.0, 0.0;
        result[1].Coeffs() << scale, 0.0;
        result[2].Coeffs() << scale, scale;
        return result;
    };
    const auto check = [](const PathSegmentBase<Group> &reference,
                          const PathSegmentBase<Group> &scaled, int exponent) {
        if (!reference.IsValid() || !scaled.IsValid())
            throw std::runtime_error("large curve fixture must be valid");
        for (double t : {0.0, 0.13, 0.33, 0.7, 0.9, 1.0}) {
            const double s = t * reference.GetLength();
            const double scaled_s = t * scaled.GetLength();
            const std::array<Eigen::Vector2d, 4> expected{
                reference.GetConfig(s).Coeffs(), reference.GetTangent(s).Coeffs(),
                reference.GetCurvature(s).Coeffs(), reference.GetTorsion(s).Coeffs()};
            const std::array<Eigen::Vector2d, 4> actual{
                scaled.GetConfig(scaled_s).Coeffs(),
                scaled.GetTangent(scaled_s).Coeffs(),
                scaled.GetCurvature(scaled_s).Coeffs(),
                scaled.GetTorsion(scaled_s).Coeffs()};
            // Under a uniform geometric scale k, q^(j)(s) scales as k^(1-j).
            // ldexp rescales without forming k^2, which itself could overflow.
            for (int order = 0; order < 4; ++order) {
                Eigen::Vector2d normalized;
                for (int axis = 0; axis < 2; ++axis)
                    normalized[axis] =
                        std::ldexp(actual[order][axis], (order - 1) * exponent);
                if (!normalized.allFinite() ||
                    (normalized - expected[order]).norm() > 1e-11) {
                    std::cerr << "scale exponent=" << exponent << " order=" << order
                              << " expected=" << expected[order].transpose()
                              << " rescaled=" << normalized.transpose() << '\n';
                    throw std::runtime_error(
                        "large curve lost a representable derivative");
                }
            }
        }
    };
    const PathSegBezierCurve2nd<Group> quadratic(points(1.0));
    const PathSegBezierCurve2nd<Group> large_quadratic(points(0x1p511));
    check(quadratic, large_quadratic, 511);
    const PathSegBezierCurve5th<Group> quintic(points(1.0), 0.0);
    const PathSegBezierCurve5th<Group> large_quintic(points(0x1p400), 0.0);
    check(quintic, large_quintic, 400);
}

void CheckCubicSegmentDerivatives() {
    struct CubicSegment : PathSegBezierCurve5th<Group> {
        static std::array<Group, 3> StraightPoints() {
            std::array<Group, 3> points;
            points[0].Coeffs() << 0.0, 0.0;
            points[1].Coeffs() << 2.0, 0.0;
            points[2].Coeffs() << 5.0, 0.0;
            return points;
        }
        CubicSegment() : PathSegBezierCurve5th(StraightPoints(), 7.0) {
            // Degree elevation of q(t)=(5*t, t^3), t in [0,1]. Its analytic
            // derivatives provide an oracle independent of evaluation basis.
            for (int i = 0; i < 6; ++i)
                control_points_[i].Coeffs() << i, i * (i - 1) * (i - 2) / 60.0;
        }
    };
    const CubicSegment segment;
    for (double t : {0.0, 0.13, 0.33, 0.7, 0.9, 1.0}) {
        const double s = 7.0 + 5.0 * t;
        if ((segment.GetConfig(s).Coeffs() - Eigen::Vector2d(5.0 * t, t * t * t))
                    .norm() > 1e-13 ||
            (segment.GetTangent(s).Coeffs() - Eigen::Vector2d(1.0, 0.6 * t * t))
                    .norm() > 1e-13 ||
            (segment.GetCurvature(s).Coeffs() - Eigen::Vector2d(0.0, 0.24 * t)).norm() >
                1e-13 ||
            (segment.GetTorsion(s).Coeffs() - Eigen::Vector2d(0.0, 0.048)).norm() >
                1e-13)
            throw std::runtime_error("Bezier derivatives disagree with analytic cubic");
    }
}

template <typename Exception>
void CheckRejectedQueries(const PathSegmentBase<Group> &segment, double s) {
    for (int query = 0; query < 4; ++query) {
        try {
            switch (query) {
            case 0:
                segment.GetConfig(s);
                break;
            case 1:
                segment.GetTangent(s);
                break;
            case 2:
                segment.GetCurvature(s);
                break;
            case 3:
                segment.GetTorsion(s);
                break;
            }
        } catch (const Exception &) {
            continue;
        }
        throw std::runtime_error("path segment query did not reject invalid input");
    }
}

void CheckSegmentQueryValidation() {
    std::array<Group, 3> points;
    points[0].Coeffs() << 0.0, 0.0;
    points[1].Coeffs() << 2.0, 0.0;
    points[2].Coeffs() << 5.0, 0.0;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    PathSegLinear<Group> linear({points[0], points[2]}, 7.0);
    PathSegBezierCurve2nd<Group> quadratic(points, 7.0);
    PathSegBezierCurve5th<Group> quintic(points, 7.0);
    for (const PathSegmentBase<Group> *segment :
         std::array<const PathSegmentBase<Group> *, 3>{&linear, &quadratic, &quintic}) {
        for (double s : {nan, inf, -inf})
            CheckRejectedQueries<std::invalid_argument>(*segment, s);
        if ((segment->GetConfig(-100.0) - points[0]).Coeffs().norm() > 1e-12 ||
            (segment->GetConfig(100.0) - points[2]).Coeffs().norm() > 1e-12)
            throw std::runtime_error("finite segment queries must still clamp");
    }
    points.fill(points[0]);
    PathSegLinear<Group> constant({points[0], points[0]});
    if (!constant.IsValid() ||
        (constant.GetConfig(10.0) - points[0]).Coeffs().norm() != 0.0 ||
        !constant.GetTangent(10.0).Coeffs().isZero(0.0))
        throw std::runtime_error("constant linear segment must remain queryable");
    PathSegBezierCurve2nd<Group> zero_quadratic(points);
    PathSegBezierCurve5th<Group> zero_quintic(points, 0.0);
    PathSegLinear<Group> invalid_start({points[0], points[0]}, inf);
    for (const PathSegmentBase<Group> *segment :
         std::array<const PathSegmentBase<Group> *, 3>{&zero_quadratic, &zero_quintic,
                                                       &invalid_start}) {
        if (segment->IsValid())
            throw std::runtime_error("degenerate Bezier or invalid start was accepted");
        CheckRejectedQueries<std::logic_error>(*segment, 0.0);
    }
    points[1].Coeffs() << 2.0, 0.0;
    points[2].Coeffs() << 5.0, 0.0;
    for (int index = 0; index < 4; ++index) {
        for (double value : {nan, inf, -inf}) {
            std::array<double, 4> norms{1.0, 0.0, 1.0, 0.0};
            norms[index] = value;
            PathSegBezierCurve5th<Group> invalid_curve(points, 0.0, norms[0], norms[1],
                                                       norms[2], norms[3]);
            if (invalid_curve.IsValid())
                throw std::runtime_error("non-finite endpoint norm was accepted");
            CheckRejectedQueries<std::logic_error>(invalid_curve, 0.0);
        }
    }
    for (int index : {1, 3}) {
        std::array<double, 4> norms{1.0, 0.0, 1.0, 0.0};
        norms[index] = std::numeric_limits<double>::max();
        PathSegBezierCurve5th<Group> overflow_curve(points, 0.0, norms[0], norms[1],
                                                    norms[2], norms[3]);
        if (overflow_curve.IsValid())
            throw std::runtime_error("overflowing control points were accepted");
        CheckRejectedQueries<std::logic_error>(overflow_curve, 0.0);
    }
    // Opposite nonzero endpoint directions with coincident endpoints give a
    // double zero root, not a usable curve or a reason to divide by zero.
    points[2] = points[0];
    PathSegBezierCurve5th<Group> zero_root(points, 0.0);
    if (zero_root.IsValid())
        throw std::runtime_error("zero curve-length root was accepted");
    CheckRejectedQueries<std::logic_error>(zero_root, 0.0);
}

int main() {
    try {
        CheckQuadraticCorners();
        CheckReversal(2);
        CheckReversal(5);
        CheckFinalDebugRecord();
        CheckShortReversalBlend();
        CheckStraightSegmentDerivatives();
        CheckCurveLengthNearLinearEquation();
        CheckLargeCurveDerivatives();
        CheckCubicSegmentDerivatives();
        CheckSegmentQueryValidation();
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
