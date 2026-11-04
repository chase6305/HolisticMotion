#include "holistic_motion/manif/Rn.h"
#include "holistic_motion/manif/algorithms/bezier.h"
#include "holistic_motion/manif/algorithms/interpolation.h"

#include <cmath>
#include <iostream>
#include <limits>

using namespace holistic_motion::robotics;

static_assert(Ipow<int>(std::numeric_limits<int>::max(), 1) == std::numeric_limits<int>::max());
static_assert(Ipow(3, 5) == 243);
static_assert(BinomialCoefficient(5, 2) == 10);

namespace {
template <class Group>
bool CheckCubicTangents(const Group &start, const Group &end, const typename Group::Tangent &first,
                        const typename Group::Tangent &last) {
    constexpr double step = 1e-6;
    const Group near_start = InterpolateCubic(start, end, step, first, last);
    const Group near_end = InterpolateCubic(start, end, 1 - step, first, last);
    const auto start_velocity = (near_start.Rminus(start).Coeffs() / step).eval();
    const auto end_velocity = (end.Rminus(near_end).Coeffs() / step).eval();
    return InterpolateCubic(start, end, 0.0, first, last).IsApprox(start, 1e-14) &&
           InterpolateCubic(start, end, 1.0, first, last).IsApprox(end, 1e-14) &&
           (start_velocity - first.Coeffs()).norm() < 1e-4 &&
           (end_velocity - last.Coeffs()).norm() < 1e-4;
}
} // namespace

int main() {
    using Point = Rn<double, 2>;
    Point start, end;
    start.Coeffs() << 2, -1;
    end.Coeffs() << 5, 4;
    Point::Tangent first, last;
    first.Coeffs() << .4, -.2;
    last.Coeffs() << -.6, .8;
    if (!CheckCubicTangents(start, end, first, last))
        return 1;
    if (!InterpolateCubic(start, end, 0.0).IsApprox(start, 1e-14) ||
        !InterpolateCubic(start, end, 1.0).IsApprox(end, 1e-14))
        return 1;
    const Eigen::Map<const Point> mapped_start(start.Coeffs().data());
    const Eigen::Map<const Point> mapped_end(end.Coeffs().data());
    for (int sample = 0; sample <= 20; ++sample) {
        const double t = sample / 20.0, t2 = t * t, t3 = t2 * t;
        const Eigen::Vector2d expected =
            (2 * t3 - 3 * t2 + 1) * start.Coeffs() + (-2 * t3 + 3 * t2) * end.Coeffs() +
            (t3 - 2 * t2 + t) * first.Coeffs() + (t3 - t2) * last.Coeffs();
        if ((InterpolateCubic(start, end, t, first, last).Coeffs() - expected).norm() > 2e-14 ||
            (Interpolate(mapped_start, mapped_end, t, INTERP_METHOD::CUBIC, first, last).Coeffs() -
             expected)
                    .norm() > 2e-14)
            return 2;
    }

    SE3<double> pose_start(Eigen::Vector3d(.2, -.1, .3), SO3<double>(.1, -.2, .3));
    SE3<double> pose_end(Eigen::Vector3d(.5, .2, -.1), SO3<double>(-.3, .4, .2));
    SE3Tangent<double> velocity_start, velocity_end;
    velocity_start.Coeffs() << .1, -.2, .3, -.1, .2, .1;
    velocity_end.Coeffs() << -.2, .1, .2, .2, -.1, .3;
    if (!CheckCubicTangents(pose_start, pose_end, velocity_start, velocity_end))
        return 3;
    const auto midpoint = InterpolateLinearInSE3(pose_start, pose_end, .5);
    if (!midpoint.GetTranslation().isApprox(
            .5 * (pose_start.GetTranslation() + pose_end.GetTranslation()), 1e-14) ||
        !midpoint.GetQuat().isApprox(pose_start.GetQuat().slerp(.5, pose_end.GetQuat()), 1e-14))
        return 4;

    for (unsigned int order = 1; order <= 4; ++order) {
        for (int sample = 0; sample <= 100; ++sample) {
            const double t = sample / 100.0, value = SmoothingPhi(t, order);
            if (value < -1e-13 || value > 1 + 1e-13 ||
                std::abs(value + SmoothingPhi(1 - t, order) - 1) > 2e-13)
                return 5;
        }
    }

    std::vector<Point> controls(7);
    for (std::size_t i = 0; i < controls.size(); ++i)
        controls[i].Coeffs() << .2 * i, std::sin(.3 * i);
    for (unsigned int degree : {2, 3, 4, 5}) {
        const auto curve = Decasteljau(controls, degree, 3);
        const auto compatibility = ComputeBezierCurve(controls, degree, 3);
        const auto closed = Decasteljau(controls, degree, 3, true);
        if (curve.empty() || curve.size() != compatibility.size() || closed.empty() ||
            curve.back().Coeffs() != controls.back().Coeffs() ||
            closed.back().Coeffs() != controls.front().Coeffs())
            return 6;
        std::size_t output = 0;
        for (std::size_t begin = 0; begin + 1 < controls.size(); begin += degree - 1) {
            const std::size_t count = std::min<std::size_t>(degree, controls.size() - begin);
            const std::size_t samples = count == 2 ? 3 : 3 * count;
            for (std::size_t sample = 1; sample <= samples; ++sample) {
                const double t = double(sample) / samples;
                Eigen::Vector2d expected = Eigen::Vector2d::Zero();
                double binomial = 1;
                for (std::size_t point = 0; point < count; ++point) {
                    expected += binomial * std::pow(1 - t, count - point - 1) * std::pow(t, point) *
                                controls[begin + point].Coeffs();
                    binomial *= double(count - point - 1) / (point + 1);
                }
                if (output >= curve.size() || !curve[output].IsApprox(compatibility[output], 0.0) ||
                    (curve[output++].Coeffs() - expected).norm() > 2e-14)
                    return 7;
            }
        }
        if (output != curve.size())
            return 8;
    }
    if (Decasteljau(std::vector<Point>{start, end}, 2, 4).size() != 4)
        return 9;
    for (unsigned int degree : {0, 1, 8}) {
        try {
            Decasteljau(controls, degree, 3);
            return 10;
        } catch (const std::exception &) {
        }
    }
    return 0;
}
