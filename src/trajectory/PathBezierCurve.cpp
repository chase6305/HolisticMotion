#include "holistic_motion/trajectory/PathBezierCurve.h"

#include <algorithm>

namespace holistic_motion {
namespace robotics {

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(PathBezierCurve)

template <typename LieGroup>
PathBezierCurve<LieGroup>::PathBezierCurve(
        const std::vector<LieGroup> &waypoints,
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
        std::any_of(waypoints.begin(), waypoints.end(), [](const auto& point) {
            return !point.Coeffs().allFinite();
        })) {
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
    std::string waypoints_str;
    for (auto p : this->waypoints_)
        waypoints_str += fmt::format("[{}],", fmt::join(p.Coeffs(), ","));
    holistic_motion::utility::LogDebug("[PathBezierCurve] input waypoints list:{}",
                            waypoints_str);

    this->_CheckPathWaypoints(this->waypoints_);

    waypoints_str = "";
    for (auto p : this->waypoints_)
        waypoints_str += fmt::format("[{}],", fmt::join(p.Coeffs(), ","));
    holistic_motion::utility::LogDebug("[PathBezierCurve] check waypoints list:{}",
                            waypoints_str);

    if (this->waypoints_.size() == 0 || this->waypoints_.size() == 1) {
        holistic_motion::utility::LogWarning("Input: at least 2 waypoints.");
        return;
    }
    this->weights_ = GetWeights(int(this->waypoints_.front().size()),
                                is_cartesian_space);
    // move linear from first point to end point
    if (this->waypoints_.size() == 2) {
        // construct a linear path
        holistic_motion::utility::LogDebug("Line construction begin...\n");
        std::shared_ptr<PathSegmentBase<LieGroup>> path_seg =
                std::make_shared<PathSegLinear<LieGroup>>(
                        std::array<LieGroup, 2>{waypoints.front(),
                                                waypoints.back()},
                        0.0, this->is_cartesian_space_);
        // Check path length after construct straight path segments
        if (path_seg->IsValid()) {
            this->valid_ = true;
            this->length_ += path_seg->GetLength();
            this->path_segments_.push_back(path_seg);
            if (this->length_ <= 0) {
                this->valid_ = false;
                auto str_joint_front = fmt::format(
                        "joint_front:[{}]",
                        fmt::join(waypoints.front().Coeffs(), " , "));
                auto str_joint_back =
                        fmt::format("joint_back:[{}]",
                                    fmt::join(waypoints.back().Coeffs(), " ,"));
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
            holistic_motion::utility::LogWarning(
                    "The method corresponding to "
                    "the order has not yet been implemented!");
            break;
    }
}

template <typename LieGroup>
void PathBezierCurve<LieGroup>::PathBezierCurve2nd() {
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

        tangent_2_0 = *waypoint2 - *waypoint0;
        double length_2_0 = tangent_2_0.WeightedNorm();

        dis = this->blend_tolerance_ <= Epsilon
                      ? 0.0
                      : 4.0 * this->blend_tolerance_ / length_2_0;
        dis = std::min(dis, length_1_0 / 3.0);
        dis = std::min(dis, length_2_1 / 3.0);

        holistic_motion::utility::LogDebug(
                "length_1_0:{}, length_2_1:{}, "
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
        if (len_path1 > Epsilon) {
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
                std::array<LieGroup, 3>{control_0, control_1, control_2},
                path_len, this->is_cartesian_space_);
        if (!path_seg->IsValid()) {
            holistic_motion::utility::LogWarning(
                    "Invalid 2nd-BezierCurve "
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
            std::array<LieGroup, 2>{control_2, this->waypoints_.back()},
            path_len, this->is_cartesian_space_);
    if (!path_seg->IsValid()) {
        this->valid_ = false;
        holistic_motion::utility::LogWarning("Invalid last line construction!");
        return;
    }

    holistic_motion::utility::LogDebug("The {}th segment,{} startposition:{}, length:{}",
                            i, to_underlying_type(path_seg->GetPathSegType()),
                            path_seg->GetStartParameter(),
                            path_seg->GetLength());

    this->path_segments_.push_back(path_seg);
    path_len += path_seg->GetLength();
    this->length_ = path_len;
    this->valid_ = true;
    holistic_motion::utility::LogDebug("Construct PathBezierCurve2nd success!\n");
}

template <typename LieGroup>
void PathBezierCurve<LieGroup>::PathBezierCurve5th() {
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
        holistic_motion::utility::LogDebug("waypoint0:{}, waypoint1:{}, waypoint2:{}.",
                                fmt::join((*waypoint0).Coeffs(), ","),
                                fmt::join((*waypoint1).Coeffs(), ","),
                                fmt::join((*waypoint2).Coeffs(), ","));
        // Determine whether the traversal is complete
        bool last_loop = (waypoint2 == this->waypoints_.end());
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

        dis = (this->blend_tolerance_ <= Epsilon ||
               length_2_0 <= Epsilon)
                      ? 0.0
                      : 4.0 * this->blend_tolerance_ / length_2_0;
        dis = std::min(dis, length_1_0 / 2.0);
        dis = std::min(dis, length_2_1 / 2.0);

        holistic_motion::utility::LogDebug(
                "length_1_0:{}, length_2_1:{}, "
                "length_2_0:{}, dis:{}",
                length_1_0, length_2_1, length_2_0, dis);

        // TODO: may opposite direction, may exchange waypoint0 and waypoint2 or
        // use linear segment

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
        if (len_path1 > Epsilon) {
            // store path_seg object pointer
            this->path_segments_.push_back(path_seg);
            path_len += len_path1;
            holistic_motion::utility::LogDebug(
                    "The {}th segment, startposition:{}, length:{} ", i,
                    path_seg->GetStartParameter(), len_path1);
        } else {
            holistic_motion::utility::LogDebug("Invalid line construction!");
        }
        if (last_loop) break;

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
                fmt::join(control_0.Coeffs(), ","),
                fmt::join(control_1.Coeffs(), ","),
                fmt::join(control_2.Coeffs(), ","));
        path_seg = std::make_shared<PathSegBezierCurve5th<LieGroup>>(
                std::array<LieGroup, 3>{control_0, control_1, control_2},
                path_len, tstart_norm, cstart_norm, tend_norm, cend_norm,
                this->is_cartesian_space_);
        if (!path_seg->IsValid()) {
            holistic_motion::utility::LogWarning(
                    "Invalid 5th-BezierCurve "
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

}  // namespace robotics
}  // namespace holistic_motion
