#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "holistic_motion/trajectory/PathBase.h"
#include "holistic_motion/trajectory/PathSegment.h"

namespace sampling_checks {
using Group = holistic_motion::robotics::Rn<double, 2>;
using Segment = holistic_motion::robotics::PathSegmentBase<Group>;

class SampledSegment : public Segment {
public:
    SampledSegment(double start, double length, int bad_derivative = -1,
                   bool endpoint_peak = false)
        : bad_derivative_(bad_derivative), endpoint_peak_(endpoint_peak) {
        sp_ = start;
        length_ = length;
    }
    Group GetConfig(double) const override { return Group(); }
    Group::Tangent GetTangent(double s) const override {
        if (samples.size() >= maximum_samples)
            throw std::runtime_error("curve sampler repeated without progress");
        samples.push_back(s);
        return Derivative(s, 0, endpoint_peak_ && s == sp_ + length_ ? 4.0 : 2.0);
    }
    Group::Tangent GetCurvature(double s) const override {
        return Derivative(s, 1, 8.0);
    }
    Group::Tangent GetTorsion(double s) const override {
        return Derivative(s, 2, 64.0);
    }
    mutable std::vector<double> samples;
    std::size_t maximum_samples{8};

private:
    Group::Tangent Derivative(double s, int order, double value) const {
        Group::Tangent result;
        result.Coeffs() << value, 0.0;
        if (order == bad_derivative_ && s == sp_ + length_)
            result.Coeffs()[0] = std::numeric_limits<double>::quiet_NaN();
        return result;
    }
    int bad_derivative_;
    bool endpoint_peak_;
};

class ConstantDerivativeSegment : public Segment {
public:
    ConstantDerivativeSegment(int order, double value) : order_(order), value_(value) {
        length_ = 0.005;
    }
    Group GetConfig(double) const override { return Group(); }
    Group::Tangent GetTangent(double) const override { return Derivative(0); }
    Group::Tangent GetCurvature(double) const override { return Derivative(1); }
    Group::Tangent GetTorsion(double) const override { return Derivative(2); }

private:
    Group::Tangent Derivative(int order) const {
        Group::Tangent result;
        result.Coeffs() << (order == order_ ? value_ : 0.0), (order == 0 ? 1.0 : 0.0);
        return result;
    }
    int order_;
    double value_;
};

template <typename Sample> void CheckSegmentSampling(Sample sample) {
    {
        using Curve = holistic_motion::robotics::PathSegBezierCurve5th<Group>;
        struct OverriddenCurve : Curve {
            explicit OverriddenCurve(const std::array<Group, 3>& points)
                : Curve(points, 0.0) {}
            Group GetConfig(double) const override {
                throw std::runtime_error("speed-cap sampling must not query position");
            }
            Group::Tangent GetTangent(double) const override { return Value(0, 2.0); }
            Group::Tangent GetCurvature(double) const override { return Value(1, 128.0); }
            Group::Tangent GetTorsion(double) const override { return Value(2, 64.0); }
            Group::Tangent Value(unsigned order, double value) const {
                ++calls[order];
                Group::Tangent result;
                result.Coeffs() << value, 0.0;
                return result;
            }
            mutable std::array<unsigned, 3> calls{};
        };
        std::array<Group, 3> points;
        points[0].Coeffs() << 0.0, 0.0;
        points[1].Coeffs() << 0.01, 0.0;
        points[2].Coeffs() << 0.01, 0.01;
        auto curve = std::make_shared<OverriddenCurve>(points);
        const double speed =
            sample(curve, Eigen::Vector2d::Constant(4.0),
                   Eigen::Vector2d::Constant(32.0), Eigen::Vector2d::Constant(512.0));
        if (speed != 0.5 || curve->calls[0] < 2 || curve->calls[0] != curve->calls[1] ||
            curve->calls[1] != curve->calls[2])
            throw std::runtime_error("shared curve evaluation bypassed overrides");
    }
    {
        std::array<Group, 3> points;
        points[0].Coeffs() << 0.0, 0.0;
        points[1].Coeffs() << 2.0, 0.0;
        points[2].Coeffs() << 5.0, 0.0;
        auto line =
            std::make_shared<holistic_motion::robotics::PathSegBezierCurve5th<Group>>(
                points, 0.0);
        const Eigen::VectorXd v = Eigen::VectorXd::Ones(2);
        const Eigen::VectorXd tiny = Eigen::VectorXd::Constant(2, 1e-20);
        const double speed = sample(line, v, tiny, tiny);
        if (!std::isfinite(speed) || std::abs(speed - 1.0) > 1e-14)
            throw std::runtime_error("straight Bezier was slowed by false derivatives");
    }
    const Eigen::VectorXd velocity = Eigen::VectorXd::Constant(2, 4.0);
    const Eigen::VectorXd acceleration = Eigen::VectorXd::Constant(2, 32.0);
    const Eigen::VectorXd jerk = Eigen::VectorXd::Constant(2, 512.0);
    const auto speed = [&](const std::shared_ptr<SampledSegment> &segment) {
        return sample(segment, velocity, acceleration, jerk);
    };
    // The fixture aborts repeated sampling, so the old infinite loop fails
    // deterministically without relying on a wall-clock timeout.
    for (const auto &interval :
         {std::pair<double, double>{0x1p48, 1.0},
          {1e20, 1.0},
          {0.0, -1.0},
          {0.0, std::numeric_limits<double>::quiet_NaN()},
          {std::numeric_limits<double>::infinity(), 1.0},
          {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()}}) {
        auto segment =
            std::make_shared<SampledSegment>(interval.first, interval.second);
        if (speed(segment) != 0.0)
            throw std::runtime_error("invalid sampling interval returned a speed");
    }
    for (int order : {0, 1, 2}) {
        auto segment = std::make_shared<SampledSegment>(0.0, 0.025, order);
        if (speed(segment) != 0.0)
            throw std::runtime_error("nonfinite curve derivative was ignored");
    }
    for (double length : {0.025, 0.005}) {
        auto segment = std::make_shared<SampledSegment>(0.0, length, -1, true);
        if (speed(segment) != 1.0 || segment->samples.front() != 0.0 ||
            segment->samples.back() != length ||
            segment->samples.size() != (length < 0.01 ? 2 : 4))
            throw std::runtime_error("curve sampler missed its endpoint limit");
        if (length > 0.01 &&
            (segment->samples[1] != 0.01 || segment->samples[2] != 0.02))
            throw std::runtime_error("ordinary sampling grid changed");
    }
    // Fixed absolute steps would require 1e8--1e102 evaluations here.
    // Enforce a deterministic work bound rather than relying on a timeout.
    for (double length : {1e6, 1e100}) {
        auto segment = std::make_shared<SampledSegment>(0.0, length, -1, true);
        segment->maximum_samples = 4097;
        if (speed(segment) != 1.0 || segment->samples.size() != 4097 ||
            segment->samples.front() != 0.0 || segment->samples.back() != length) {
            throw std::runtime_error("large curve sampling exceeded its work bound");
        }
        for (std::size_t i = 1; i < segment->samples.size(); ++i) {
            if (!std::isfinite(segment->samples[i]) ||
                segment->samples[i] <= segment->samples[i - 1]) {
                throw std::runtime_error("large curve sampling did not advance");
            }
        }
    }
    struct LimitCase {
        int order;
        int derivative_exponent;
        int limit_exponent;
        int expected_speed_exponent;
    };
    // Powers of two give independent, exactly representable root references.
    for (const auto &input :
         {LimitCase{0, -40, -50, -10}, LimitCase{1, -40, -60, -10},
          LimitCase{2, -40, -70, -10}, LimitCase{1, -10, 1020, 515},
          LimitCase{2, -10, 1022, 344}, LimitCase{1, 800, -400, -600},
          LimitCase{2, 800, -400, -400}}) {
        for (double sign : {-1.0, 1.0}) {
            auto segment = std::make_shared<ConstantDerivativeSegment>(
                input.order, sign * std::scalbn(1.0, input.derivative_exponent));
            Eigen::VectorXd v =
                Eigen::VectorXd::Constant(2, std::numeric_limits<double>::max());
            Eigen::VectorXd a = v;
            Eigen::VectorXd j = v;
            (input.order == 0   ? v
             : input.order == 1 ? a
                                : j)[0] = std::scalbn(1.0, input.limit_exponent);
            const double actual = sample(segment, v, a, j);
            const double expected = std::scalbn(1.0, input.expected_speed_exponent);
            if (!std::isfinite(actual) || std::abs(actual / expected - 1.0) > 1e-13)
                throw std::runtime_error(
                    "curve limit lost a representable speed bound");
        }
    }
}
} // namespace sampling_checks
