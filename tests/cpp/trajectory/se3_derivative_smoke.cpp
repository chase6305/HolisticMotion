#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include "../../../src/trajectory/PathSegmentEvaluation.h"
#include "holistic_motion/trajectory/TrajectoryDoubleS.h"
#include "holistic_motion/trajectory/TrajectoryTrapezoidal.h"

using namespace holistic_motion::robotics;
using Vector6 = Eigen::Matrix<double, 6, 1>;

namespace {
void Check(const Vector6 &actual, const Vector6 &expected, double tolerance,
           const char *message) {
    const double error = (actual - expected).norm() / std::max(1.0, expected.norm());
    if (!actual.allFinite() || !expected.allFinite() || error > tolerance) {
        std::cerr << message << ": " << error << '\n';
        throw std::runtime_error(message);
    }
}

void CheckSegment(const PathSegmentBase<SE3d> &segment) {
    if (!segment.IsValid())
        throw std::runtime_error("invalid derivative fixture");
    detail::SegmentEvaluationSampler<SE3d> sampler(segment);
    const double length = segment.GetLength();
    for (double x : {0.0, 0.17, 0.43, 0.81, 1.0}) {
        const double s = segment.GetStartParameter() + x * length;
        const auto q = segment.GetConfig(s);
        const auto v = segment.GetTangent(s);
        const auto a = segment.GetCurvature(s);
        const auto j = segment.GetTorsion(s);
        SE3d position;
        SE3Tangentd tangent, curvature, torsion;
        sampler.ComputeJet(s, position, tangent, curvature, torsion);
        Check((position - q).Coeffs(), Vector6::Zero(), 1e-12, "cached position");
        Check(tangent.Coeffs(), v.Coeffs(), 1e-12, "cached tangent");
        Check(curvature.Coeffs(), a.Coeffs(), 1e-12, "cached curvature");
        Check(torsion.Coeffs(), j.Coeffs(), 1e-12, "cached torsion");
        sampler.Compute(s, tangent, curvature, torsion);
        Check(tangent.Coeffs(), v.Coeffs(), 1e-12, "speed-cap tangent");
        Check(curvature.Coeffs(), a.Coeffs(), 1e-12, "speed-cap curvature");
        Check(torsion.Coeffs(), j.Coeffs(), 1e-12, "speed-cap torsion");
        for (double step : {1e-4, 1e-5}) {
            const double h = step * length;
            Vector6 dv, da, dj;
            if (x == 0.0 || x == 1.0) {
                const double d = x == 0.0 ? h : -h;
                // Fourth-order one-sided stencil keeps endpoint truncation error
                // below roundoff-amplified curvature on short blends.
                dv = (48 * (segment.GetConfig(s + d) - q).Coeffs() -
                      36 * (segment.GetConfig(s + 2 * d) - q).Coeffs() +
                      16 * (segment.GetConfig(s + 3 * d) - q).Coeffs() -
                      3 * (segment.GetConfig(s + 4 * d) - q).Coeffs()) /
                     (12 * d);
                da = (-25 * v.Coeffs() + 48 * segment.GetTangent(s + d).Coeffs() -
                      36 * segment.GetTangent(s + 2 * d).Coeffs() +
                      16 * segment.GetTangent(s + 3 * d).Coeffs() -
                      3 * segment.GetTangent(s + 4 * d).Coeffs()) /
                     (12 * d);
                dj = (-25 * a.Coeffs() + 48 * segment.GetCurvature(s + d).Coeffs() -
                      36 * segment.GetCurvature(s + 2 * d).Coeffs() +
                      16 * segment.GetCurvature(s + 3 * d).Coeffs() -
                      3 * segment.GetCurvature(s + 4 * d).Coeffs()) /
                     (12 * d);
            } else {
                dv = ((segment.GetConfig(s - 2 * h) - q).Coeffs() -
                      8 * (segment.GetConfig(s - h) - q).Coeffs() +
                      8 * (segment.GetConfig(s + h) - q).Coeffs() -
                      (segment.GetConfig(s + 2 * h) - q).Coeffs()) /
                     (12 * h);
                da = (segment.GetTangent(s - 2 * h).Coeffs() -
                      8 * segment.GetTangent(s - h).Coeffs() +
                      8 * segment.GetTangent(s + h).Coeffs() -
                      segment.GetTangent(s + 2 * h).Coeffs()) /
                     (12 * h);
                dj = (segment.GetCurvature(s - 2 * h).Coeffs() -
                      8 * segment.GetCurvature(s - h).Coeffs() +
                      8 * segment.GetCurvature(s + h).Coeffs() -
                      segment.GetCurvature(s + 2 * h).Coeffs()) /
                     (12 * h);
            }
            Check(dv, v.Coeffs(), 2e-6, "pose finite difference");
            Check(da, a.Coeffs(), 2e-5, "tangent finite difference");
            Check(dj, j.Coeffs(), 2e-5, "curvature finite difference");
        }
    }
}

struct SingleSegmentPath : PathBase<SE3d> {
    explicit SingleSegmentPath(std::shared_ptr<PathSegmentBase<SE3d>> segment) {
        path_segments_.push_back(segment);
        length_ = segment->GetStartParameter() + segment->GetLength();
        valid_ = segment->IsValid();
    }
};

struct ClockProbe : TrajectoryBase<SE3d> {
    explicit ClockProbe(std::shared_ptr<PathSegmentBase<SE3d>> segment) {
        path_ = std::make_shared<SingleSegmentPath>(segment);
        trajectory_pspline_ = std::make_shared<PSpline>();
        const double length = segment->GetLength();
        trajectory_pspline_->PushBack(std::make_shared<Polynomial>(Eigen::Vector4d(
                                          segment->GetStartParameter(), 0.35 * length,
                                          0.2 * length, 0.1 * length)),
                                      1.0);
        dof_ = 6;
        max_velocity_ = max_acceleration_ = max_jerk_ = Eigen::VectorXd::Ones(6);
        valid_ = true;
    }
};

void CheckClock(std::shared_ptr<PathSegmentBase<SE3d>> segment) {
    ClockProbe trajectory(segment);
    for (double duration : {1.0, 1.7}) {
        if (!trajectory.SetMinimumDuration(duration))
            throw std::runtime_error("cannot slow test clock");
        for (double fraction : {0.1, 0.35, 0.75}) {
            const double t = duration * fraction;
            const auto state = trajectory.GetState(t);
            Check(state.velocity.Coeffs(), trajectory.GetVelocity(t).Coeffs(), 1e-12,
                  "state velocity");
            Check(state.acceleration.Coeffs(), trajectory.GetAcceleration(t).Coeffs(),
                  1e-12, "state acceleration");
            Check(state.jerk.Coeffs(), trajectory.GetJerk(t).Coeffs(), 1e-12,
                  "state jerk");
            for (double step : {1e-4, 1e-5}) {
                const double h = duration * step;
                const auto before = trajectory.GetState(t - h);
                const auto after = trajectory.GetState(t + h);
                Check(((after.position - state.position).Coeffs() -
                       (before.position - state.position).Coeffs()) /
                          (2 * h),
                      state.velocity.Coeffs(), 2e-6, "clock pose finite difference");
                Check((after.velocity.Coeffs() - before.velocity.Coeffs()) / (2 * h),
                      state.acceleration.Coeffs(), 2e-6,
                      "clock velocity finite difference");
                Check((after.acceleration.Coeffs() - before.acceleration.Coeffs()) /
                          (2 * h),
                      state.jerk.Coeffs(), 2e-6,
                      "clock acceleration finite difference");
            }
        }
        const auto report = trajectory.GetConstraintReport(257);
        Vector6 velocity = Vector6::Zero(), acceleration = Vector6::Zero(),
                jerk = Vector6::Zero();
        for (int i = 0; i < 257; ++i) {
            const double t = duration * (i / 256.0);
            velocity = velocity.cwiseMax(trajectory.GetVelocity(t).Coeffs().cwiseAbs());
            acceleration = acceleration.cwiseMax(
                trajectory.GetAcceleration(t).Coeffs().cwiseAbs());
            jerk = jerk.cwiseMax(trajectory.GetJerk(t).Coeffs().cwiseAbs());
        }
        Check(report.peak_velocity, velocity, 1e-12, "report velocity");
        Check(report.peak_acceleration, acceleration, 1e-12, "report acceleration");
        Check(report.peak_jerk, jerk, 1e-12, "report jerk");
    }
}

template <typename Curve> struct OverrideCurve : Curve {
    using Curve::Curve;
    SE3Tangentd GetTangent(double s) const override {
        return 1.25 * Curve::GetTangent(s);
    }
    SE3Tangentd GetCurvature(double s) const override {
        return 0.75 * Curve::GetCurvature(s);
    }
    SE3Tangentd GetTorsion(double s) const override {
        return 0.5 * Curve::GetTorsion(s);
    }
};

template <typename Curve> void CheckOverrides(const std::array<SE3d, 3> &points) {
    auto curve = std::make_shared<OverrideCurve<Curve>>(points, 0.0);
    detail::SegmentEvaluationSampler<SE3d> sampler(*curve);
    ClockProbe trajectory(curve);
    const double t = 0.4, length = curve->GetLength();
    const double s = length * (0.35 * t + 0.2 * t * t + 0.1 * t * t * t);
    const double speed = length * (0.35 + 0.4 * t + 0.3 * t * t);
    SE3d position;
    SE3Tangentd tangent, curvature, torsion;
    sampler.ComputeJet(s, position, tangent, curvature, torsion);
    Check(tangent.Coeffs(), curve->GetTangent(s).Coeffs(), 1e-12, "virtual tangent");
    Check(curvature.Coeffs(), curve->GetCurvature(s).Coeffs(), 1e-12,
          "virtual curvature");
    Check(torsion.Coeffs(), curve->GetTorsion(s).Coeffs(), 1e-12, "virtual torsion");
    Check(trajectory.GetState(t).velocity.Coeffs(), speed * tangent.Coeffs(), 1e-12,
          "virtual state");
}

void CheckUniformTranslations() {
    for (double delta : {1.0 / 65536, 1.0 / 256, 1.0, 256.0}) {
        std::vector<SE3d> points;
        for (int i = 0; i < 6; ++i)
            points.emplace_back(Eigen::Vector3d(i * delta, 0, 0), SO3d(0, 0, 0));
        for (double s : {0.0, 0.17, 0.43, 0.81, 1.0}) {
            const detail::QuinticEvaluation<SE3d> curve(points, s, 5 * delta);
            Check(curve.SecondDerivative().Coeffs(), Vector6::Zero(), 0.0,
                  "uniform translation curvature");
            Check(curve.ThirdDerivative().Coeffs(), Vector6::Zero(), 0.0,
                  "uniform translation torsion");
        }
    }
}
} // namespace

int main() {
    try {
        holistic_motion::utility::SetVerbosityLevel(
            holistic_motion::utility::VerbosityLevel::Error);
        CheckUniformTranslations();
        std::mt19937_64 generator(20260928);
        std::uniform_real_distribution<double> random(-0.5, 0.5);
        for (int sample = 0; sample < 48; ++sample) {
            std::array<SE3d, 3> points;
            for (auto &point : points) {
                Vector6 value;
                for (double &entry : value)
                    entry = random(generator);
                if (sample % 3 == 0)
                    value.tail<3>().setZero(); // Commuting translations.
                if (sample % 3 == 1) {
                    value.head<5>().setZero(); // Commuting rotations.
                }
                point = SE3Tangentd(value).Exp();
            }
            CheckSegment(PathSegBezierCurve2nd<SE3d>(points, 0.7, true));
            CheckSegment(PathSegBezierCurve5th<SE3d>(points, 0.7, 1, 0, 1, 0, true));
            if (sample < 6) {
                CheckClock(std::make_shared<PathSegBezierCurve2nd<SE3d>>(points, 0.7));
                CheckClock(std::make_shared<PathSegBezierCurve5th<SE3d>>(points, 0.7));
                CheckOverrides<PathSegBezierCurve2nd<SE3d>>(points);
                CheckOverrides<PathSegBezierCurve5th<SE3d>>(points);
            }
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
