#pragma once

#include "holistic_motion/trajectory/PathBezierCurve.h"

#include <algorithm>
#include <type_traits>

namespace holistic_motion {
namespace robotics {
namespace {

template <typename Derived>
EIGEN_DONT_INLINE double RescaledTangentNorm(const TangentBase<Derived> &tangent, double norm) {
    const double scale = tangent.Coeffs().cwiseAbs().maxCoeff();
    if (scale == 0.0 || !std::isfinite(scale))
        return norm;
    const typename TangentBase<Derived>::Tangent normalized(tangent.Coeffs() / scale);
    return scale * normalized.WeightedNorm();
}

template <typename Derived> inline double StableTangentNorm(const TangentBase<Derived> &tangent) {
    const double norm = tangent.WeightedNorm();
    return norm > 0.0 && std::isfinite(norm) ? norm : RescaledTangentNorm(tangent, norm);
}

template <typename Derived>
EIGEN_DONT_INLINE double RescaledCoefficientRoundoff(const Eigen::MatrixBase<Derived> &value) {
    const double scale = value.cwiseAbs().maxCoeff();
    return (64.0 * std::numeric_limits<double>::epsilon() * scale) * (value / scale).norm();
}

template <typename Derived>
inline double CoefficientRoundoff(const Eigen::MatrixBase<Derived> &value) {
    const double norm = value.norm();
    return std::isfinite(norm) ? 64.0 * std::numeric_limits<double>::epsilon() * norm
                               : RescaledCoefficientRoundoff(value);
}
} // namespace

template <typename LieGroup>
PathBezierCurve<LieGroup>::PathBezierCurve(const std::vector<LieGroup> &waypoints, const int degree,
                                           const bool is_cartesian_space,
                                           const double blend_tolerance) {
    if ((degree != 2 && degree != 5) || !std::isfinite(blend_tolerance) || blend_tolerance < 0.0 ||
        waypoints.size() < 2 ||
        std::any_of(waypoints.begin(), waypoints.end(),
                    [](const auto &point) { return !point.Coeffs().allFinite(); })) {
        holistic_motion::utility::LogWarning(
            "Path requires degree 2 or 5, a finite non-negative blend, "
            "and at least two finite waypoints");
        return;
    }
    this->waypoints_ = waypoints;
    this->blend_tolerance_ = blend_tolerance;
    this->is_cartesian_space_ = is_cartesian_space;
    this->_CheckPathWaypoints(this->waypoints_);
    Build(degree);
}

template <typename LieGroup> void PathBezierCurve<LieGroup>::PathBezierCurve2nd() { Build(2); }

template <typename LieGroup> void PathBezierCurve<LieGroup>::PathBezierCurve5th() { Build(5); }

template <typename LieGroup> void PathBezierCurve<LieGroup>::Build(int degree) {
    this->valid_ = false;
    this->length_ = 0.0;
    this->path_segments_.clear();
    this->path_type_ = degree == 2 ? (this->is_cartesian_space_ ? PathType::Bezier2ndCartesianSpace
                                                                : PathType::Bezier2ndJointSpace)
                                   : (this->is_cartesian_space_ ? PathType::Bezier5thCartesianSpace
                                                                : PathType::Bezier5thJointSpace);
    if (this->waypoints_.size() < 2)
        return;
    this->path_segments_.reserve(this->waypoints_.size() - 1);
    double path_length = 0.0;
    LieGroup control_start = this->waypoints_.front();
    Tangent incoming;
    double incoming_length = 0.0;
    for (std::size_t index = 1; index < this->waypoints_.size(); ++index) {
        const auto &previous = this->waypoints_[index - 1];
        const auto &corner = this->waypoints_[index];
        const bool last = index + 1 == this->waypoints_.size();
        LieGroup control_end = corner;
        Tangent outgoing;
        double distance = 0.0;
        if (!last) {
            const auto &next = this->waypoints_[index + 1];
            holistic_motion::utility::LogDebug(
                "waypoint0:{}, waypoint1:{}, waypoint2:{}.", fmt::join(previous.Coeffs(), ","),
                fmt::join(corner.Coeffs(), ","), fmt::join(next.Coeffs(), ","));
            if (index == 1) {
                incoming = corner - previous;
                incoming_length = StableTangentNorm(incoming);
                incoming = 1.0 / incoming_length * incoming;
            }
            outgoing = next - corner;
            const double outgoing_length = StableTangentNorm(outgoing);
            outgoing = 1.0 / outgoing_length * outgoing;
            if (this->blend_tolerance_ > Epsilon) {
                const double direction_change = StableTangentNorm(outgoing - incoming);
                const double direction_sum = StableTangentNorm(incoming + outgoing);
                // Reversals retain a stopped waypoint instead of coincident blend ends.
                if (direction_change > Epsilon && direction_sum > Epsilon)
                    distance = 4.0 * this->blend_tolerance_ / direction_change;
                const double divisor = degree == 2 ? 3.0 : 2.0;
                distance = std::min(distance, incoming_length / divisor);
                distance = std::min(distance, outgoing_length / divisor);
                if constexpr (!std::is_same_v<LieGroup, SE3d>) {
                    // A quintic near a reversal can be shorter than its minimum
                    // valid length even when both incoming/outgoing legs are valid.
                    if (degree == 5 && distance > Epsilon &&
                        distance * (30.0 * direction_sum / (16.0 + 7.0 * direction_sum)) <= Epsilon)
                        distance = 0.0;
                }
                if (distance <= Epsilon)
                    distance = 0.0;
            }
            control_end = corner + (-distance) * incoming;
            // The next corner's incoming edge is this corner's outgoing edge.
            incoming = outgoing;
            incoming_length = outgoing_length;
        }
        auto linear = std::make_shared<PathSegLinear<LieGroup>>(
            std::array<LieGroup, 2>{control_start, control_end}, path_length,
            this->is_cartesian_space_);
        if (!linear->IsValid())
            return;
        const double linear_length = linear->GetLength();
        // Trimmed controls can cancel much larger source coordinates. Retain
        // real final legs while suppressing only roundoff at interior joins.
        const double join_roundoff = last ? 0.0
                                          : std::max({64.0 * std::numeric_limits<double>::epsilon(),
                                                      CoefficientRoundoff(control_start.Coeffs()),
                                                      CoefficientRoundoff(control_end.Coeffs()),
                                                      CoefficientRoundoff(previous.Coeffs()),
                                                      CoefficientRoundoff(corner.Coeffs())});
        if (linear_length > join_roundoff) {
            this->path_segments_.push_back(std::move(linear));
            path_length += linear_length;
        }
        if (last)
            break;
        if (distance == 0.0) {
            control_start = corner;
            continue;
        }
        control_start = control_end;
        control_end = corner + distance * outgoing;
        std::shared_ptr<PathSegmentBase<LieGroup>> blend;
        if (degree == 2) {
            blend = std::make_shared<PathSegBezierCurve2nd<LieGroup>>(
                std::array<LieGroup, 3>{control_start, corner, control_end}, path_length,
                this->is_cartesian_space_);
        } else {
            blend = std::make_shared<PathSegBezierCurve5th<LieGroup>>(
                std::array<LieGroup, 3>{control_start, corner, control_end}, path_length, 1.0, 0.0,
                1.0, 0.0, this->is_cartesian_space_);
        }
        if (!blend->IsValid())
            return;
        path_length += blend->GetLength();
        this->path_segments_.push_back(std::move(blend));
        control_start = control_end;
    }
    this->length_ = path_length;
    this->valid_ = path_length > 0.0;
}

} // namespace robotics
} // namespace holistic_motion
