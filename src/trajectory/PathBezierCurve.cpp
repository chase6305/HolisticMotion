#include "holistic_motion/trajectory/PathBezierCurve.h"

#include <algorithm>
#include <type_traits>

namespace holistic_motion {
namespace robotics {

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(PathBezierCurve)

template <typename LieGroup>
PathBezierCurve<LieGroup>::PathBezierCurve(const std::vector<LieGroup> &waypoints,
                                           const int degree,
                                           const bool is_cartesian_space,
                                           const double blend_tolerance) {
    holistic_motion::utility::LogDebug("Constucting Path...");
    if ((degree != 2 && degree != 5) || !std::isfinite(blend_tolerance) ||
        blend_tolerance < 0.0) {
        holistic_motion::utility::LogWarning(
            "Path requires degree 2 or 5 and a finite non-negative blend");
        return;
    }
    if (waypoints.size() < 2 ||
        std::any_of(waypoints.begin(), waypoints.end(),
                    [](const auto &point) { return !point.Coeffs().allFinite(); })) {
        holistic_motion::utility::LogWarning(
            "Path requires at least two finite waypoints");
        return;
    }
    this->waypoints_ = waypoints;
    this->blend_tolerance_ = blend_tolerance;
    this->is_cartesian_space_ = is_cartesian_space;
    if (degree == 2) {
        this->path_type_ = this->is_cartesian_space_ == true
                               ? PathType::Bezier2ndCartesianSpace
                               : PathType::Bezier2ndJointSpace;
    } else if (degree == 5) {
        this->path_type_ = this->is_cartesian_space_ == true
                               ? PathType::Bezier5thCartesianSpace
                               : PathType::Bezier5thJointSpace;
    }
    if (holistic_motion::utility::GetVerbosityLevel() >=
        holistic_motion::utility::VerbosityLevel::Debug) {
        std::string waypoints_str;
        for (const auto &p : this->waypoints_)
            waypoints_str += fmt::format("[{}],", fmt::join(p.Coeffs(), ","));
        holistic_motion::utility::LogDebug("[PathBezierCurve] input waypoints list:{}",
                                           waypoints_str);
    }

    this->_CheckPathWaypoints(this->waypoints_);

    if (holistic_motion::utility::GetVerbosityLevel() >=
        holistic_motion::utility::VerbosityLevel::Debug) {
        std::string waypoints_str;
        for (const auto &p : this->waypoints_)
            waypoints_str += fmt::format("[{}],", fmt::join(p.Coeffs(), ","));
        holistic_motion::utility::LogDebug("[PathBezierCurve] check waypoints list:{}",
                                           waypoints_str);
    }

    if (this->waypoints_.size() == 0 || this->waypoints_.size() == 1) {
        holistic_motion::utility::LogWarning("Input: at least 2 waypoints.");
        return;
    }
    this->weights_ = GetWeights(LieGroup::DoF, is_cartesian_space);
    // move linear from first point to end point
    if (this->waypoints_.size() == 2) {
        // construct a linear path
        holistic_motion::utility::LogDebug("Line construction begin...\n");
        std::shared_ptr<PathSegmentBase<LieGroup>> path_seg =
            std::make_shared<PathSegLinear<LieGroup>>(
                std::array<LieGroup, 2>{waypoints.front(), waypoints.back()}, 0.0,
                this->is_cartesian_space_);
        // Check path length after construct straight path segments
        if (path_seg->IsValid()) {
            this->valid_ = true;
            this->length_ += path_seg->GetLength();
            this->path_segments_.push_back(path_seg);
            if (this->length_ <= 0) {
                this->valid_ = false;
                auto str_joint_front = fmt::format(
                    "joint_front:[{}]", fmt::join(waypoints.front().Coeffs(), " , "));
                auto str_joint_back = fmt::format(
                    "joint_back:[{}]", fmt::join(waypoints.back().Coeffs(), " ,"));
                holistic_motion::utility::LogDebug(
                    "Invalid line construction, cause path lenth is zero, "
                    "{} , {}\n",
                    str_joint_front, str_joint_back);
                return;
            }
            holistic_motion::utility::LogDebug("Line construction success!\n");
            return;
        } else {
            this->valid_ = false;
            holistic_motion::utility::LogWarning("Invalid line construction!\n");
            return;
        }
    }

    // TODO: may not be appropriate to choose a different method
    switch (degree) {
    case 2:
        this->PathBezierCurve2nd();
        break;
    case 5:
        this->PathBezierCurve5th();
        break;

    default:
        holistic_motion::utility::LogWarning("The method corresponding to "
                                             "the order has not yet been implemented!");
        break;
    }
}

template <typename LieGroup> void PathBezierCurve<LieGroup>::PathBezierCurve2nd() {
    holistic_motion::utility::LogDebug("Constucting PathBezierCurve2nd...");

    double path_len = 0.0;
    auto waypoint0 = this->waypoints_.begin();
    auto waypoint1 = waypoint0;
    waypoint1++;
    auto waypoint2 = waypoint1;
    waypoint2++;

    auto control_0 = this->waypoints_.front();
    auto control_1 = *waypoint1;
    auto control_2 = control_1;

    // distance of Bezier curve first and second control points
    double dis = 0.0;
    int i = 0;
    Tangent tangent_1_0;
    Tangent tangent_2_1;
    Tangent tangent_2_0;

    std::shared_ptr<PathSegmentBase<LieGroup>> path_seg;

    while (waypoint2 != this->waypoints_.end()) {
        control_1 = *waypoint1;
        // calculate the tangent between waypoint1 and waypoint2;
        holistic_motion::utility::LogDebug("waypoint0:{}, waypoint1:{}, waypoint2:{}.",
                                           fmt::join((*waypoint0).Coeffs(), ","),
                                           fmt::join((*waypoint1).Coeffs(), ","),
                                           fmt::join((*waypoint2).Coeffs(), ","));

        tangent_1_0 = *waypoint1 - *waypoint0;
        double length_1_0 = tangent_1_0.WeightedNorm();
        tangent_1_0 = 1.0 / length_1_0 * tangent_1_0;

        tangent_2_1 = *waypoint2 - *waypoint1;
        double length_2_1 = tangent_2_1.WeightedNorm();
        tangent_2_1 = 1.0 / length_2_1 * tangent_2_1;

        tangent_2_0 = tangent_2_1 - tangent_1_0;
        double length_2_0 = tangent_2_0.WeightedNorm();

        const bool reverses = (tangent_1_0 + tangent_2_1).WeightedNorm() <= Epsilon;
        dis = (this->blend_tolerance_ <= Epsilon || length_2_0 <= Epsilon || reverses)
                  ? 0.0
                  : 4.0 * this->blend_tolerance_ / length_2_0;
        dis = std::min(dis, length_1_0 / 3.0);
        dis = std::min(dis, length_2_1 / 3.0);
        if (dis <= Epsilon)
            dis = 0.0;

        holistic_motion::utility::LogDebug("length_1_0:{}, length_2_1:{}, "
                                           "length_2_0:{}, dis:{}",
                                           length_1_0, length_2_1, length_2_0, dis);

        control_2 = (control_1 + (-dis) * tangent_1_0);
        path_seg = std::make_shared<PathSegLinear<LieGroup>>(
            std::array<LieGroup, 2>{control_0, control_2}, path_len,
            this->is_cartesian_space_);
        if (!path_seg->IsValid()) {
            this->valid_ = false;
            holistic_motion::utility::LogWarning("Invalid line construction!");
            return;
        }

        double len_path1 = path_seg->GetLength();
        const double join_roundoff =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, control_0.Coeffs().norm(), control_2.Coeffs().norm(),
                      waypoint0->Coeffs().norm(), waypoint1->Coeffs().norm()});
        if (len_path1 > join_roundoff) {
            // store path_seg object pointer
            this->path_segments_.push_back(path_seg);
            path_len += len_path1;
            holistic_motion::utility::LogDebug(
                "The {}th segment, startposition:{}, length:{} ", i,
                path_seg->GetStartParameter(), len_path1);
        } else {
            holistic_motion::utility::LogWarning("Invalid line construction!");
        }

        if (dis <= Epsilon) {
            holistic_motion::utility::LogDebug("Distance[{}] is close to 0.0!", dis);
            control_0 = control_1;
            control_2 = control_1;
            waypoint0++;
            waypoint1++;
            waypoint2++;
            i++;
            continue;
        }

        control_0 = control_2;
        control_2 = (*waypoint1 + dis * tangent_2_1);
        path_seg = std::make_shared<PathSegBezierCurve2nd<LieGroup>>(
            std::array<LieGroup, 3>{control_0, control_1, control_2}, path_len,
            this->is_cartesian_space_);
        if (!path_seg->IsValid()) {
            holistic_motion::utility::LogWarning("Invalid 2nd-BezierCurve "
                                                 "construction!");
            return;
        }
        holistic_motion::utility::LogDebug(
            "The {}th segment,{} startposition:{}, length:{}", i,
            to_underlying_type(path_seg->GetPathSegType()),
            path_seg->GetStartParameter(), path_seg->GetLength());
        this->path_segments_.push_back(path_seg);
        path_len += path_seg->GetLength();

        control_0 = control_2;
        waypoint0++;
        waypoint1++;
        waypoint2++;
        i++;
    }

    // the last straight line segment
    path_seg = std::make_shared<PathSegLinear<LieGroup>>(
        std::array<LieGroup, 2>{control_2, this->waypoints_.back()}, path_len,
        this->is_cartesian_space_);
    if (!path_seg->IsValid()) {
        this->valid_ = false;
        holistic_motion::utility::LogWarning("Invalid last line construction!");
        return;
    }

    holistic_motion::utility::LogDebug(
        "The {}th segment,{} startposition:{}, length:{}", i,
        to_underlying_type(path_seg->GetPathSegType()), path_seg->GetStartParameter(),
        path_seg->GetLength());

    this->path_segments_.push_back(path_seg);
    path_len += path_seg->GetLength();
    this->length_ = path_len;
    this->valid_ = true;
    holistic_motion::utility::LogDebug("Construct PathBezierCurve2nd success!\n");
}

template <typename LieGroup> void PathBezierCurve<LieGroup>::PathBezierCurve5th() {
    holistic_motion::utility::LogDebug("Constucting PathBezierCurve5th...");

    double path_len = 0.0;

    auto waypoint0 = this->waypoints_.begin();
    auto waypoint1 = waypoint0;
    waypoint1++;
    auto waypoint2 = waypoint1;
    waypoint2++;

    auto control_0 = this->waypoints_.front();
    auto control_1 = *waypoint1;
    auto control_2 = control_1;

    // distance of Bezier curve first and second control points
    double dis = 0.0;
    int i = 0;
    Tangent tangent_1_0;
    Tangent tangent_2_1;
    Tangent tangent_2_0;

    // Solve for coefficient ratios for control points
    const double cstart_norm = 0.0;
    const double cend_norm = 0.0;
    const double tstart_norm = 1.0;
    const double tend_norm = 1.0;

    std::shared_ptr<PathSegmentBase<LieGroup>> path_seg;

    // compute bezier curve
    while (true) {
        // The final segment has only two waypoints. Never dereference end(),
        // even as an argument to a disabled debug call.
        bool last_loop = (waypoint2 == this->waypoints_.end());
        if (!last_loop) {
            holistic_motion::utility::LogDebug(
                "waypoint0:{}, waypoint1:{}, waypoint2:{}.",
                fmt::join((*waypoint0).Coeffs(), ","),
                fmt::join((*waypoint1).Coeffs(), ","),
                fmt::join((*waypoint2).Coeffs(), ","));
        }
        control_1 = *waypoint1;
        // calculate the tangent between waypoint1 and waypoint2;
        tangent_1_0 = *waypoint1 - *waypoint0;
        double length_1_0 = tangent_1_0.WeightedNorm();
        tangent_1_0 = 1.0 / length_1_0 * tangent_1_0;

        // The last group of points needs to consider the problem of calculating
        // tangent space
        if (!last_loop) {
            tangent_2_1 = *waypoint2 - *waypoint1;
        } else {
            tangent_2_1 = tangent_1_0;
        }

        double length_2_1 = tangent_2_1.WeightedNorm();
        tangent_2_1 = 1.0 / length_2_1 * tangent_2_1;

        tangent_2_0 = tangent_2_1 - tangent_1_0;
        double length_2_0 = tangent_2_0.WeightedNorm();

        // A reversal is a stopped waypoint, not a blend with coincident ends.
        const double direction_sum = (tangent_1_0 + tangent_2_1).WeightedNorm();
        const bool reverses = direction_sum <= Epsilon;
        dis = (this->blend_tolerance_ <= Epsilon || length_2_0 <= Epsilon || reverses)
                  ? 0.0
                  : 4.0 * this->blend_tolerance_ / length_2_0;
        dis = std::min(dis, length_1_0 / 2.0);
        dis = std::min(dis, length_2_1 / 2.0);

        if constexpr (!std::is_same_v<LieGroup, SE3d>) {
            // For symmetric Rn controls and unit endpoint tangents, the
            // positive length root simplifies to 30*d*u/(16+7*u), where
            // u is the norm of the sum of the two directions. A near reversal
            // can make this smaller than the segment's minimum length even
            // though both legs are valid. Retain a stopped waypoint then.
            if (dis > Epsilon) {
                const double blend_length =
                    dis * (30.0 * direction_sum / (16.0 + 7.0 * direction_sum));
                if (blend_length <= Epsilon)
                    dis = 0.0;
            }
        }

        holistic_motion::utility::LogDebug("length_1_0:{}, length_2_1:{}, "
                                           "length_2_0:{}, dis:{}",
                                           length_1_0, length_2_1, length_2_0, dis);

        if (dis <= Epsilon)
            dis = 0.0;
        if (last_loop) {
            control_2 = control_1;
        } else {
            control_2 = (control_1 + (-dis) * tangent_1_0);
        }

        path_seg = std::make_shared<PathSegLinear<LieGroup>>(
            std::array<LieGroup, 2>{control_0, control_2}, path_len,
            this->is_cartesian_space_);
        if (!path_seg->IsValid()) {
            holistic_motion::utility::LogWarning("Invalid line construction!");
            return;
        }
        auto len_path1 = path_seg->GetLength();
        // Neighboring blends can meet up to coordinate roundoff. Do not
        // introduce a microscopic timing phase for that numerical residue,
        // but always retain a real final leg to the requested endpoint.
        // Include the source waypoints: trimmed controls near the origin can
        // result from cancellation of much larger coordinates and offsets.
        const double join_roundoff =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, control_0.Coeffs().norm(), control_2.Coeffs().norm(),
                      waypoint0->Coeffs().norm(), waypoint1->Coeffs().norm()});
        if (len_path1 > join_roundoff || (last_loop && len_path1 > 0.0)) {
            // store path_seg object pointer
            this->path_segments_.push_back(path_seg);
            path_len += len_path1;
            holistic_motion::utility::LogDebug(
                "The {}th segment, startposition:{}, length:{} ", i,
                path_seg->GetStartParameter(), len_path1);
        } else {
            holistic_motion::utility::LogDebug("Invalid line construction!");
        }
        if (last_loop)
            break;

        if (dis <= Epsilon) {
            holistic_motion::utility::LogDebug("Distance[{}] is close to 0.0!", dis);
            control_0 = control_1;
            waypoint0++;
            waypoint1++;
            waypoint2++;
            i++;
            continue;
        }

        control_0 = control_2;
        control_2 = (control_1 + dis * tangent_2_1);

        holistic_motion::utility::LogDebug(
            "path_len:{}, tstart_norm:{}, tend_norm:{}, cstart_norm:{}, "
            "cend_norm:{}, control_0:[{}], control_1:[{}], control_2:[{}]",
            path_len, tstart_norm, tend_norm, cstart_norm, cend_norm,
            fmt::join(control_0.Coeffs(), ","), fmt::join(control_1.Coeffs(), ","),
            fmt::join(control_2.Coeffs(), ","));
        path_seg = std::make_shared<PathSegBezierCurve5th<LieGroup>>(
            std::array<LieGroup, 3>{control_0, control_1, control_2}, path_len,
            tstart_norm, cstart_norm, tend_norm, cend_norm, this->is_cartesian_space_);
        if (!path_seg->IsValid()) {
            holistic_motion::utility::LogWarning("Invalid 5th-BezierCurve "
                                                 "construction!");
            return;
        }

        holistic_motion::utility::LogDebug(
            "The {}th segment,{} start position:{}, length:{}", i,
            to_underlying_type(path_seg->GetPathSegType()),
            path_seg->GetStartParameter(), path_seg->GetLength());
        this->path_segments_.push_back(path_seg);
        path_len += path_seg->GetLength();

        control_0 = control_2;
        waypoint0++;
        waypoint1++;
        waypoint2++;
        i++;
    }

    // the last straight line segment
    this->length_ = path_len;
    this->valid_ = true;
    holistic_motion::utility::LogDebug("Construct PathBezierCurve5th success!\n");
}

} // namespace robotics
} // namespace holistic_motion
