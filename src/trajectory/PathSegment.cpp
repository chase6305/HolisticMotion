#include "holistic_motion/trajectory/PathSegment.h"
namespace holistic_motion {
namespace robotics {

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(PathSegLinear)
HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(PathSegBezierCurve2nd)
HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(PathSegBezierCurve5th)

template <typename LieGroup>
PathSegLinear<LieGroup>::PathSegLinear(const std::array<LieGroup, 2> &waypoints,
                                       const double &sp,
                                       const bool &is_cartesian_space) {
    this->sp_ = sp;
    this->path_seg_type_ = PathSegType::LinearSeg;
    this->waypoints_.insert(this->waypoints_.end(), waypoints.cbegin(),
                            waypoints.cend());
    this->tangent_ = this->waypoints_[1] - this->waypoints_[0];
    // Path distances act on tangents (SE3 has 6 tangent coordinates but stores
    // 7 pose coefficients). Pose storage size is not the metric dimension.
    auto weights = GetWeights(LieGroup::DoF, is_cartesian_space);
    this->length_ = WeightedNorm(this->tangent_, weights);

    holistic_motion::utility::LogDebug("[PathSegLinear], sp:{}, length:{}", this->sp_,
                                       this->length_);
}

template <typename LieGroup>
LieGroup PathSegLinear<LieGroup>::GetConfig(double s) const {
    this->ValidateQuery(s);
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > 0.0 ? (s / this->length_) : 0;
    holistic_motion::utility::LogDebug("[PathSegLinear] s:{}, sp_:{}", s, this->sp_);

    return this->waypoints_[0] + s * this->tangent_;
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegLinear<LieGroup>::GetTangent(double s) const {
    this->ValidateQuery(s);
    return this->length_ > 0.0 ? this->tangent_ / this->length_ : this->tangent_;
}

template <typename LieGroup>
PathSegBezierCurve2nd<LieGroup>::PathSegBezierCurve2nd(
    const std::array<LieGroup, 3> &waypoints, const double &sp,
    const bool &is_cartesian_space) {
    this->path_seg_type_ = PathSegType::Bezier2ndSeg;
    this->sp_ = sp;
    this->waypoints_.insert(this->waypoints_.end(), waypoints.cbegin(),
                            waypoints.cend());
    this->control_points_ = this->waypoints_;
    auto weights = GetWeights(LieGroup::DoF, is_cartesian_space);
    // Calculate the relative distance before and after three points
    this->length_ =
        WeightedNorm(this->control_points_[1] - this->control_points_[0], weights) +
        WeightedNorm(this->control_points_[2] - this->control_points_[1], weights);
    if (this->length_ == 0.0)
        this->length_ = std::numeric_limits<double>::quiet_NaN();
    holistic_motion::utility::LogDebug(
        "[PathSegBezierCurve2nd], sp:{}, length:{}, dof:{}", this->sp_, this->length_,
        (*waypoints.begin()).size());
}

template <typename LieGroup>
LieGroup PathSegBezierCurve2nd<LieGroup>::GetConfig(double s) const {
    this->ValidateQuery(s);
    s = clamp(s - this->sp_, 0.0, this->length_);
    s /= this->length_;
    holistic_motion::utility::LogDebug("[PathSegBezierCurve2nd] s:{}, sp_:{}", s,
                                       this->sp_);

    // Ps = (1-t)^2P0 + 2(1-t)tP1 + t^2P2
    auto ps = this->control_points_[0] +
              s * (2 - s) * (this->control_points_[1] - this->control_points_[0]) +
              s * s * (this->control_points_[2] - this->control_points_[1]);
    return ps;
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegBezierCurve2nd<LieGroup>::GetTangent(double s) const {
    this->ValidateQuery(s);
    s = clamp(s - this->sp_, 0.0, this->length_);
    s /= this->length_;

    // Ps' = [(1-t)^2P0 + 2(1-t)tP1 + t^2P2]'
    auto ret = (2 - 2 * s) * (this->control_points_[1] - this->control_points_[0]) +
               2 * s * (this->control_points_[2] - this->control_points_[1]);
    return ret / this->length_;
}

template <typename LieGroup>
typename LieGroup::Tangent
PathSegBezierCurve2nd<LieGroup>::GetCurvature(double s) const {
    this->ValidateQuery(s);
    // (Ps')' = [[(1-t)^2P0 + 2(1-t)tP1 + t^2P2]']'
    auto ret = 2.0 * ((this->control_points_[2] - this->control_points_[1]) -
                      (this->control_points_[1] - this->control_points_[0]));
    const double length_squared = this->length_ * this->length_;
    if (std::isfinite(length_squared))
        return ret / length_squared;
    // The derivative can remain representable when the denominator overflows.
    return (ret / this->length_) / this->length_;
}

template <typename LieGroup>
PathSegBezierCurve5th<LieGroup>::PathSegBezierCurve5th(
    const std::array<LieGroup, 3> &waypoints, const double &sp,
    const double &tstart_norm, const double &cstart_norm, const double &tend_norm,
    const double &cend_norm, const bool &is_cartesian_space) {
    this->path_seg_type_ = PathSegType::Bezier5thSeg;
    this->sp_ = sp;
    this->waypoints_.insert(this->waypoints_.end(), waypoints.cbegin(),
                            waypoints.cend());
    this->tangent_ = waypoints[1] - waypoints[0];

    if (!std::isfinite(tstart_norm) || !std::isfinite(cstart_norm) ||
        !std::isfinite(tend_norm) || !std::isfinite(cend_norm)) {
        this->length_ = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    // compute weights...
    auto weights = GetWeights(LieGroup::DoF, is_cartesian_space);

    // compute trangent between waypoints0, waypoints1 and waypoints3
    auto tstart = waypoints[1] - waypoints[0];
    auto tend = waypoints[2] - waypoints[1];

    holistic_motion::utility::LogDebug(
        "[PathSegBezierCurve5th] waypoints0:[{}], waypoints1:[{}], "
        "waypoints2:[{}]",
        fmt::join(waypoints[0].Coeffs(), ","), fmt::join(waypoints[1].Coeffs(), ","),
        fmt::join(waypoints[2].Coeffs(), ","));

    // compute weights of the tstart or the tend
    const double tstart_length = WeightedNorm(tstart, weights);
    const double tend_length = WeightedNorm(tend, weights);
    if (!std::isfinite(tstart_length) || !std::isfinite(tend_length) ||
        tstart_length <= Epsilon || tend_length <= Epsilon) {
        this->length_ = std::numeric_limits<double>::quiet_NaN();
        holistic_motion::utility::LogWarning(
            "[PathSegBezierCurve5th] degenerate endpoint tangent");
        return;
    }
    auto cstart = tstart / tstart_length;
    auto cend = tend / tend_length;

    holistic_motion::utility::LogDebug(
        "[PathSegBezierCurve5th] tstart:[{}], tend:[{}], cstart:[{}], "
        "cend:[{}]",
        fmt::join(tstart.Coeffs(), ","), fmt::join(tend.Coeffs(), ","),
        fmt::join(cstart.Coeffs(), ","), fmt::join(cend.Coeffs(), ","));

    tstart = tstart_norm * cstart; // default 1.0
    tend = tend_norm * cend;       // default 1.0
    cstart *= cstart_norm;         // default 0.0
    cend *= cend_norm;             // default 0.0

    const auto &pstart = this->waypoints_[0];
    const auto &pend = this->waypoints_[2];

    // add notes
    double a = 256.0 - 49.0 * std::pow(WeightedNorm(tend + tstart, weights), 2);
    // Use the same tangent metric as a and c, including rotation. A fixed
    // three-coordinate block also reads beyond R1/R2 tangent storage.
    const double b = 420.0 * ((pend - pstart).Coeffs().array() * weights.array() *
                              (tstart + tend).Coeffs().array())
                                 .sum();
    double c = -900.0 * WeightedNorm(pend - pstart, weights) *
               WeightedNorm(pend - pstart, weights);
    double delta = b * b - 4 * a * c;
    holistic_motion::utility::LogDebug(
        "[PathSegBezierCurve5th] a:{}, b:{}, c:{}, delta:{}", a, b, c, delta);
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
        !std::isfinite(delta) || delta < 0.0) {
        this->length_ = std::numeric_limits<double>::quiet_NaN();
        holistic_motion::utility::LogWarning(
            "[PathSegBezierCurve5th] cannot compute control points!");
        return;
    }
    double root = std::numeric_limits<double>::quiet_NaN();
    if (std::abs(a) <= Small) {
        if (std::abs(b) > Small)
            root = -c / b;
    } else {
        // Compute the root numerator whose terms have matching signs, then
        // recover the other root from their product c/a. Direct subtraction
        // loses precision when the quadratic coefficient approaches zero.
        const double q = -0.5 * (b + std::copysign(std::sqrt(delta), b));
        root = q == 0.0 ? 0.0 : std::max(q / a, c / q);
    }
    if (!std::isfinite(root) || root <= Epsilon) {
        this->length_ = std::numeric_limits<double>::quiet_NaN();
        holistic_motion::utility::LogWarning(
            "[PathSegBezierCurve5th] invalid curve length");
        return;
    }

    auto p0 = pstart;
    auto p1 = pstart + 0.2 * root * tstart;
    auto p2 = p1 + (p1 - p0) + 0.05 * root * root * cstart;
    auto p5 = pend;
    auto p4 = p5 + (-0.2) * root * tend;
    auto p3 = p4 + (p4 - p5) + 0.05 * root * root * cend;

    if (!p0.Coeffs().allFinite() || !p1.Coeffs().allFinite() ||
        !p2.Coeffs().allFinite() || !p3.Coeffs().allFinite() ||
        !p4.Coeffs().allFinite() || !p5.Coeffs().allFinite()) {
        this->length_ = std::numeric_limits<double>::quiet_NaN();
        return;
    }

    this->control_points_.emplace_back(p0);
    this->control_points_.emplace_back(p1);
    this->control_points_.emplace_back(p2);
    this->control_points_.emplace_back(p3);
    this->control_points_.emplace_back(p4);
    this->control_points_.emplace_back(p5);
    this->length_ = root;

    holistic_motion::utility::LogDebug(
        "[PathSegBezierCurve5th] sp:{}, length:{}, dof:{}", this->sp_, this->length_,
        (*waypoints.begin()).size());
}

template <typename LieGroup>
LieGroup PathSegBezierCurve5th<LieGroup>::GetConfig(double s) const {
    this->ValidateQuery(s);
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > Epsilon ? (s / this->length_) : 0;
    holistic_motion::utility::LogDebug(
        "[PathSegBezierCurve5th<LieGroup>::GetConfig] s:{}, sp_:{}", s, this->sp_);

    auto t0 = this->control_points_[1] - this->control_points_[0];
    auto t1 = this->control_points_[2] - this->control_points_[1];
    auto t2 = this->control_points_[3] - this->control_points_[2];
    auto t3 = this->control_points_[4] - this->control_points_[3];
    auto t4 = this->control_points_[5] - this->control_points_[4];
    double s2 = s * s;
    double s3 = s * s2;
    double s4 = s2 * s2;
    double s5 = s2 * s3;

    // Bx(count+1)=P0(1)*(1-t)^5+5*P1(1)*t*(1-t)^4+10*P2(1)*t^2*(1-t)^3+10*P3(1)*t^3*(1-t)^2+5*P4(1)*t^4*(1-t)+P5(1)*t^5;
    return this->control_points_[0] +
           (5.0 - 10.0 * s + 10.0 * s2 - 5.0 * s3 + s4) * s * t0 +
           (10.0 - 20.0 * s + 15.0 * s2 - 4.0 * s3) * s2 * t1 +
           (10.0 - 15.0 * s + 6.0 * s2) * s3 * t2 + (5.0 - 4.0 * s) * s4 * t3 + s5 * t4;
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegBezierCurve5th<LieGroup>::GetTangent(double s) const {
    this->ValidateQuery(s);
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > Epsilon ? (s / this->length_) : 0;

    double s2 = s * s;
    double s4 = s2 * s2;
    const double u = 1.0 - s;
    const double u2 = u * u;
    auto t0 = this->control_points_[1] - this->control_points_[0];
    auto t1 = this->control_points_[2] - this->control_points_[1];
    auto t2 = this->control_points_[3] - this->control_points_[2];
    auto t3 = this->control_points_[4] - this->control_points_[3];
    auto t4 = this->control_points_[5] - this->control_points_[4];
    auto ret = 5.0 * u2 * u2 * t0 + 20.0 * s * u2 * u * t1 + 30.0 * s2 * u2 * t2 +
               20.0 * s2 * s * u * t3 + 5.0 * s4 * t4;
    return ret / this->length_;
}

template <typename LieGroup>
typename LieGroup::Tangent
PathSegBezierCurve5th<LieGroup>::GetCurvature(double s) const {
    this->ValidateQuery(s);
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > Epsilon ? (s / this->length_) : 0;

    double s2 = s * s;
    const double u = 1.0 - s;
    auto t0 = this->control_points_[1] - this->control_points_[0];
    auto t1 = this->control_points_[2] - this->control_points_[1];
    auto t2 = this->control_points_[3] - this->control_points_[2];
    auto t3 = this->control_points_[4] - this->control_points_[3];
    auto t4 = this->control_points_[5] - this->control_points_[4];
    // Differentiate control-point differences before evaluating the Bernstein
    // basis. Expanding the basis first invents curvature on an exact line,
    // and division by a short segment length amplifies that cancellation.
    auto ret = 20.0 * u * u * u * (t1 - t0) + 60.0 * s * u * u * (t2 - t1) +
               60.0 * s2 * u * (t3 - t2) + 20.0 * s2 * s * (t4 - t3);
    return ret / (this->length_ * this->length_);
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegBezierCurve5th<LieGroup>::GetTorsion(double s) const {
    this->ValidateQuery(s);
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > Epsilon ? (s / this->length_) : 0;

    double s2 = s * s;
    const double u = 1.0 - s;
    auto t0 = this->control_points_[1] - this->control_points_[0];
    auto t1 = this->control_points_[2] - this->control_points_[1];
    auto t2 = this->control_points_[3] - this->control_points_[2];
    auto t3 = this->control_points_[4] - this->control_points_[3];
    auto t4 = this->control_points_[5] - this->control_points_[4];
    const auto d0 = t1 - t0;
    const auto d1 = t2 - t1;
    const auto d2 = t3 - t2;
    const auto d3 = t4 - t3;
    auto ret =
        60.0 * u * u * (d1 - d0) + 120.0 * s * u * (d2 - d1) + 60.0 * s2 * (d3 - d2);
    const double length_cubed = this->length_ * this->length_ * this->length_;
    if (std::isfinite(length_cubed))
        return ret / length_cubed;
    // Preserve ordinary rounding for normal lengths, but do not turn a finite
    // nonzero derivative into zero merely because length^3 is not representable.
    return ((ret / this->length_) / this->length_) / this->length_;
}

} // namespace robotics
} // namespace holistic_motion
