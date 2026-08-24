#include "holistic_motion/trajectory/PathSegment.h"
namespace holistic_motion {
namespace robotics {

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(PathSegLinear)
HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(PathSegBezierCurve2nd)
HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(PathSegBezierCurve5th)

template <typename LieGroup>
PathSegLinear<LieGroup>::PathSegLinear(const std::array<LieGroup, 2>& waypoints,
                                       const double& sp,
                                       const bool& is_cartesian_space) {
    this->sp_ = sp;
    this->path_seg_type_ = PathSegType::LinearSeg;
    this->waypoints_.insert(this->waypoints_.end(), waypoints.cbegin(),
                            waypoints.cend());
    this->tangent_ = this->waypoints_[1] - this->waypoints_[0];
    auto weights = GetWeights(waypoints.front().size(), is_cartesian_space);
    this->length_ = WeightedNorm(this->tangent_, weights);

    holistic_motion::utility::LogDebug("[PathSegLinear], sp:{}, length:{}", this->sp_,
                            this->length_);
}

template <typename LieGroup>
LieGroup PathSegLinear<LieGroup>::GetConfig(double s) const {
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > Epsilon ? (s / this->length_) : 0;
    holistic_motion::utility::LogDebug("[PathSegLinear] s:{}, sp_:{}", s, this->sp_);

    return this->waypoints_[0] + s * this->tangent_;
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegLinear<LieGroup>::GetTangent(double /*s*/) const {
    return this->length_ > Epsilon ? this->tangent_ / this->length_
                                   : this->tangent_;
}

template <typename LieGroup>
PathSegBezierCurve2nd<LieGroup>::PathSegBezierCurve2nd(
        const std::array<LieGroup, 3>& waypoints,
        const double& sp,
        const bool& is_cartesian_space) {
    this->path_seg_type_ = PathSegType::Bezier2ndSeg;
    this->sp_ = sp;
    this->waypoints_.insert(this->waypoints_.end(), waypoints.cbegin(),
                            waypoints.cend());
    this->control_points_ = this->waypoints_;
    auto weights = GetWeights((*waypoints.begin()).size(), is_cartesian_space);
    // Calculate the relative distance before and after three points
    this->length_ =
            WeightedNorm(this->control_points_[1] - this->control_points_[0],
                         weights) +
            WeightedNorm(this->control_points_[2] - this->control_points_[1],
                         weights);
    holistic_motion::utility::LogDebug("[PathSegBezierCurve2nd], sp:{}, length:{}, dof:{}",
                            this->sp_, this->length_,
                            (*waypoints.begin()).size());
}

template <typename LieGroup>
LieGroup PathSegBezierCurve2nd<LieGroup>::GetConfig(double s) const {
    s = clamp(s - this->sp_, 0.0, this->length_);
    s /= this->length_;
    holistic_motion::utility::LogDebug("[PathSegBezierCurve2nd] s:{}, sp_:{}", s,
                            this->sp_);

    // Ps = (1-t)^2P0 + 2(1-t)tP1 + t^2P2
    auto ps = this->control_points_[0] +
              s * (2 - s) *
                      (this->control_points_[1] - this->control_points_[0]) +
              s * s * (this->control_points_[2] - this->control_points_[1]);
    return ps;
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegBezierCurve2nd<LieGroup>::GetTangent(
        double s) const {
    s = clamp(s - this->sp_, 0.0, this->length_);
    s /= this->length_;

    // Ps' = [(1-t)^2P0 + 2(1-t)tP1 + t^2P2]'
    auto ret = (2 - 2 * s) *
                       (this->control_points_[1] - this->control_points_[0]) +
               2 * s * (this->control_points_[2] - this->control_points_[1]);
    return ret / this->length_;
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegBezierCurve2nd<LieGroup>::GetCurvature(
        double /*s*/) const {
    // (Ps')' = [[(1-t)^2P0 + 2(1-t)tP1 + t^2P2]']'
    auto ret = 2.0 * ((this->control_points_[2] - this->control_points_[1]) -
                      (this->control_points_[1] - this->control_points_[0]));
    return ret / (this->length_ * this->length_);
}

template <typename LieGroup>
PathSegBezierCurve5th<LieGroup>::PathSegBezierCurve5th(
        const std::array<LieGroup, 3>& waypoints,
        const double& sp,
        const double& tstart_norm,
        const double& cstart_norm,
        const double& tend_norm,
        const double& cend_norm,
        const bool& is_cartesian_space) {
    this->path_seg_type_ = PathSegType::Bezier5thSeg;
    this->sp_ = sp;
    this->waypoints_.insert(this->waypoints_.end(), waypoints.cbegin(),
                            waypoints.cend());
    this->tangent_ = waypoints[1] - waypoints[0];

    // compute weights...
    auto weights = GetWeights((*waypoints.begin()).size(), is_cartesian_space);

    // compute trangent between waypoints0, waypoints1 and waypoints3
    auto tstart = waypoints[1] - waypoints[0];
    auto tend = waypoints[2] - waypoints[1];

    holistic_motion::utility::LogDebug(
            "[PathSegBezierCurve5th] waypoints0:[{}], waypoints1:[{}], "
            "waypoints2:[{}]",
            fmt::join(waypoints[0].Coeffs(), ","),
            fmt::join(waypoints[1].Coeffs(), ","),
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

    tstart = tstart_norm * cstart;  // default 1.0
    tend = tend_norm * cend;        // default 1.0
    cstart *= cstart_norm;          // default 0.0
    cend *= cend_norm;              // default 0.0

    const auto& pstart = this->waypoints_[0];
    const auto& pend = this->waypoints_[2];

    // add notes
    double a = 256.0 - 49.0 * std::pow(WeightedNorm(tend + tstart, weights), 2);
    double b = 420.0 * ((pend - pstart).Coeffs().dot((tstart + tend).Coeffs()));

    if (is_cartesian_space) {
        b = 420.0 *
            ((pend - pstart)
                     .Coeffs()
                     .template block<3, 1>(0, 0)
                     .dot((tstart + tend).Coeffs().template block<3, 1>(0, 0)));
    }
    double c = -900.0 * WeightedNorm(pend - pstart, weights) *
               WeightedNorm(pend - pstart, weights);
    double delta = b * b - 4 * a * c;
    holistic_motion::utility::LogDebug(
            "[PathSegBezierCurve5th] a:{}, b:{}, c:{}, delta:{}", a, b, c,
            delta);
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) ||
        !std::isfinite(delta) || delta < 0.0) {
        this->length_ = std::numeric_limits<double>::quiet_NaN();
        holistic_motion::utility::LogWarning(
                "[PathSegBezierCurve5th] cannot compute control points!");
        return;
    }
    double root = std::numeric_limits<double>::quiet_NaN();
    if (std::abs(a) <= Small) {
        if (std::abs(b) > Small) root = -c / b;
    } else {
        const double root1 = (-b + std::sqrt(delta)) / (2 * a);
        const double root2 = (-b - std::sqrt(delta)) / (2 * a);
        root = std::max(root1, root2);
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

    this->control_points_.emplace_back(p0);
    this->control_points_.emplace_back(p1);
    this->control_points_.emplace_back(p2);
    this->control_points_.emplace_back(p3);
    this->control_points_.emplace_back(p4);
    this->control_points_.emplace_back(p5);
    this->length_ = root;

    holistic_motion::utility::LogDebug("[PathSegBezierCurve5th] sp:{}, length:{}, dof:{}",
                            this->sp_, this->length_,
                            (*waypoints.begin()).size());
}

template <typename LieGroup>
LieGroup PathSegBezierCurve5th<LieGroup>::GetConfig(double s) const {
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > Epsilon ? (s / this->length_) : 0;
    holistic_motion::utility::LogDebug(
            "[PathSegBezierCurve5th<LieGroup>::GetConfig] s:{}, sp_:{}", s,
            this->sp_);

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
           (10.0 - 15.0 * s + 6.0 * s2) * s3 * t2 + (5.0 - 4.0 * s) * s4 * t3 +
           s5 * t4;
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegBezierCurve5th<LieGroup>::GetTangent(
        double s) const {
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > Epsilon ? (s / this->length_) : 0;

    double s2 = s * s;
    double s3 = s * s2;
    double s4 = s2 * s2;
    auto t0 = this->control_points_[1] - this->control_points_[0];
    auto t1 = this->control_points_[2] - this->control_points_[1];
    auto t2 = this->control_points_[3] - this->control_points_[2];
    auto t3 = this->control_points_[4] - this->control_points_[3];
    auto t4 = this->control_points_[5] - this->control_points_[4];
    auto ret = (5.0 - 20.0 * s + 30.0 * s2 - 20.0 * s3 + 5.0 * s4) * t0 +
               (20.0 - 60.0 * s + 60.0 * s2 - 20.0 * s3) * s * t1 +
               (30.0 - 60.0 * s + 30.0 * s2) * s2 * t2 +
               (20.0 - 20.0 * s) * s3 * t3 + 5.0 * s4 * t4;
    return ret / this->length_;
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegBezierCurve5th<LieGroup>::GetCurvature(
        double s) const {
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > Epsilon ? (s / this->length_) : 0;

    double s2 = s * s;
    double s3 = s * s2;
    auto t0 = this->control_points_[1] - this->control_points_[0];
    auto t1 = this->control_points_[2] - this->control_points_[1];
    auto t2 = this->control_points_[3] - this->control_points_[2];
    auto t3 = this->control_points_[4] - this->control_points_[3];
    auto t4 = this->control_points_[5] - this->control_points_[4];
    auto ret = (-20.0 + 60.0 * s - 60.0 * s2 + 20.0 * s3) * t0 +
               (20.0 - 120.0 * s + 180.0 * s2 - 80.0 * s3) * t1 +
               (60.0 - 180.0 * s + 120.0 * s2) * s * t2 +
               (60.0 - 80.0 * s) * s2 * t3 + 20.0 * s3 * t4;
    return ret / (this->length_ * this->length_);
}

template <typename LieGroup>
typename LieGroup::Tangent PathSegBezierCurve5th<LieGroup>::GetTorsion(
        double s) const {
    s = clamp(s - this->sp_, 0.0, this->length_);
    s = this->length_ > Epsilon ? (s / this->length_) : 0;

    double s2 = s * s;
    auto t0 = this->control_points_[1] - this->control_points_[0];
    auto t1 = this->control_points_[2] - this->control_points_[1];
    auto t2 = this->control_points_[3] - this->control_points_[2];
    auto t3 = this->control_points_[4] - this->control_points_[3];
    auto t4 = this->control_points_[5] - this->control_points_[4];
    auto ret = (60.0 - 120.0 * s + 60.0 * s2) * t0 +
               (-120.0 + 360.0 * s - 240.0 * s2) * t1 +
               (60.0 - 360.0 * s + 360.0 * s2) * t2 +
               (120.0 - 240.0 * s) * s * t3 + 60.0 * s2 * t4;
    return ret / (this->length_ * this->length_ * this->length_);
}

}  // namespace robotics
}  // namespace holistic_motion
