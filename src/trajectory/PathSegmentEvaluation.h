#pragma once

#include "holistic_motion/trajectory/PathSegment.h"

#include <optional>
#include <typeinfo>

namespace holistic_motion::robotics::detail {
// Control differences are shared by a complete state or a construction-local
// sampling loop. Preserve the scalar API's Bernstein expressions and rounding.
template <typename LieGroup> struct QuinticEvaluation {
    using Tangent = typename LieGroup::Tangent;
    const LieGroup &origin;
    Tangent t0, t1, t2, t3, t4;
    double s, length, s2, s3, s4, s5, u, u2;

    QuinticEvaluation(const std::vector<LieGroup> &points, double parameter,
                      double segment_length)
        : origin(points[0]), t0(points[1] - points[0]), t1(points[2] - points[1]),
          t2(points[3] - points[2]), t3(points[4] - points[3]),
          t4(points[5] - points[4]), s(parameter), length(segment_length), s2(s * s),
          s3(s * s2), s4(s2 * s2), s5(s2 * s3), u(1.0 - s), u2(u * u) {}

    void SetParameter(double parameter) {
        s = parameter;
        s2 = s * s;
        s3 = s * s2;
        s4 = s2 * s2;
        s5 = s2 * s3;
        u = 1.0 - s;
        u2 = u * u;
    }

    LieGroup Position() const {
        return origin + (5.0 - 10.0 * s + 10.0 * s2 - 5.0 * s3 + s4) * s * t0 +
               (10.0 - 20.0 * s + 15.0 * s2 - 4.0 * s3) * s2 * t1 +
               (10.0 - 15.0 * s + 6.0 * s2) * s3 * t2 + (5.0 - 4.0 * s) * s4 * t3 +
               s5 * t4;
    }

    Tangent FirstDerivative() const {
        const auto ret = 5.0 * u2 * u2 * t0 + 20.0 * s * u2 * u * t1 +
                         30.0 * s2 * u2 * t2 + 20.0 * s2 * s * u * t3 + 5.0 * s4 * t4;
        return ret / length;
    }

    Tangent SecondDerivative() const {
        // Difference the controls before evaluating the basis, so an exact
        // line does not acquire curvature through basis cancellation.
        return SecondDerivativeFromDifferences(t1 - t0, t2 - t1, t3 - t2, t4 - t3);
    }

    Tangent SecondDerivativeFromDifferences(const Tangent &d0, const Tangent &d1,
                                            const Tangent &d2,
                                            const Tangent &d3) const {
        const auto ret = 20.0 * u * u * u * d0 + 60.0 * s * u * u * d1 +
                         60.0 * s2 * u * d2 + 20.0 * s2 * s * d3;
        return ret / (length * length);
    }

    Tangent ThirdDerivative() const {
        const auto d0 = t1 - t0;
        const auto d1 = t2 - t1;
        const auto d2 = t3 - t2;
        const auto d3 = t4 - t3;
        return ThirdDerivativeFromDifferences(d1 - d0, d2 - d1, d3 - d2);
    }

    Tangent ThirdDerivativeFromDifferences(const Tangent &e0, const Tangent &e1,
                                           const Tangent &e2) const {
        const auto ret = 60.0 * u * u * e0 + 120.0 * s * u * e1 + 60.0 * s2 * e2;
        const double length_cubed = length * length * length;
        if (std::isfinite(length_cubed))
            return ret / length_cubed;
        // A finite derivative can survive an overflowing length cubed.
        return ((ret / length) / length) / length;
    }
};

// Additional differences pay off only across repeated samples. Scalar queries
// keep their smaller evaluator and do not store this construction-only cache.
template <typename LieGroup>
struct CachedQuinticEvaluation : QuinticEvaluation<LieGroup> {
    using Base = QuinticEvaluation<LieGroup>;
    using Tangent = typename LieGroup::Tangent;
    Tangent d0, d1, d2, d3, e0, e1, e2;

    CachedQuinticEvaluation(const std::vector<LieGroup> &points, double length)
        : Base(points, 0.0, length), d0(this->t1 - this->t0), d1(this->t2 - this->t1),
          d2(this->t3 - this->t2), d3(this->t4 - this->t3), e0(d1 - d0), e1(d2 - d1),
          e2(d3 - d2) {}

    Tangent SecondDerivative() const {
        return this->SecondDerivativeFromDifferences(d0, d1, d2, d3);
    }

    Tangent ThirdDerivative() const {
        return this->ThirdDerivativeFromDifferences(e0, e1, e2);
    }
};

// A construction-local workspace. Only the exact built-in curve may reuse
// controls: derived segments retain every virtual scalar query.
template <typename LieGroup> class SegmentEvaluationSampler {
public:
    using Tangent = typename LieGroup::Tangent;

    explicit SegmentEvaluationSampler(const PathSegmentBase<LieGroup> &segment)
        : segment_(segment) {
        if (typeid(segment) == typeid(PathSegBezierCurve5th<LieGroup>) &&
            segment.IsValid()) {
            const auto &curve =
                static_cast<const PathSegBezierCurve5th<LieGroup> &>(segment);
            evaluation_.emplace(curve.control_points_, segment.GetLength());
        }
    }

    // The enclosing speed-cap loop validates geometry and finite sample parameters
    // before constructing this workspace and guarantees forward progress.
    void Compute(double s, Tangent &tangent, Tangent &curvature, Tangent &torsion) {
        if (evaluation_) {
            SetParameter(s);
            tangent = evaluation_->FirstDerivative();
            curvature = evaluation_->SecondDerivative();
            torsion = evaluation_->ThirdDerivative();
        } else {
            tangent = segment_.GetTangent(s);
            curvature = segment_.GetCurvature(s);
            torsion = segment_.GetTorsion(s);
        }
    }

    void ComputeJet(double s, LieGroup &position, Tangent &tangent, Tangent &curvature,
                    Tangent &torsion) {
        if (evaluation_) {
            segment_.ValidateQuery(s);
            SetParameter(s);
            position = evaluation_->Position();
            tangent = evaluation_->FirstDerivative();
            curvature = evaluation_->SecondDerivative();
            torsion = evaluation_->ThirdDerivative();
        } else {
            segment_.ComputeJet(s, position, tangent, curvature, torsion);
        }
    }

private:
    void SetParameter(double s) {
        const double length = segment_.GetLength();
        s = clamp(s - segment_.GetStartParameter(), 0.0, length);
        evaluation_->SetParameter(length > Epsilon ? s / length : 0.0);
    }

    const PathSegmentBase<LieGroup> &segment_;
    std::optional<CachedQuinticEvaluation<LieGroup>> evaluation_;
};
} // namespace holistic_motion::robotics::detail
