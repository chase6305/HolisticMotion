#include "holistic_motion/trajectory/PathBase.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace holistic_motion {
namespace robotics {

HOLISTIC_MOTION_TRAJECTORY_GROUP_INSTANTIATIONS(PathBase)

template <typename LieGroup>
void PathBase<LieGroup>::_CheckPathWaypoints(std::vector<LieGroup>& waypoints) {
    if (waypoints.size() < 2) return;

    std::vector<LieGroup> filtered;
    filtered.reserve(waypoints.size());
    filtered.push_back(waypoints.front());
    for (auto waypoint = std::next(waypoints.begin());
         waypoint != waypoints.end(); ++waypoint) {
        const auto difference = *waypoint - filtered.back();
        if (difference.Coeffs().norm() > Epsilon) {
            filtered.push_back(*waypoint);
        }
    }
    waypoints.swap(filtered);
}

template <typename LieGroup>
std::shared_ptr<PathSegmentBase<LieGroup>>
PathBase<LieGroup>::GetPathSegmentAtS(double s) const {
    if (path_segments_.empty()) return nullptr;
    holistic_motion::utility::LogDebug("GetPathSegmentAtS, s:{}", s);
    s = clamp(s, 0.0, length_);

    holistic_motion::utility::LogDebug("GetPathSegmentAtS after clamp, s:{} length_:{}", s,
                            length_);

    // Segments are ordered by their path parameter. Preserve the existing
    // right-continuous boundary behavior while reducing lookup to O(log N).
    auto path_seg = std::upper_bound(
            path_segments_.begin(), path_segments_.end(), s,
            [](double parameter,
               const std::shared_ptr<PathSegmentBase<LieGroup>>& segment) {
                return parameter < segment->GetStartParameter() +
                                           segment->GetLength();
            });
    if (path_seg == path_segments_.end()) {
        path_seg--;
    }
    holistic_motion::utility::LogDebug("GetPathSegmentAtS success");

    return *path_seg;
}

template <typename LieGroup>
std::shared_ptr<PathSegmentBase<LieGroup>>
PathBase<LieGroup>::GetPathSegmentByIndex(const int& index) const {
    if (path_segments_.empty()) return nullptr;
    if (index <= 0) {
        return path_segments_.front();
    } else if (index >= (int)path_segments_.size()) {
        return path_segments_.back();
    } else {
        return path_segments_[index];
    }
}

template <typename LieGroup>
LieGroup PathBase<LieGroup>::GetConfig(double s) const {
    auto seg = GetPathSegmentAtS(s);
    if (!seg) throw std::logic_error("cannot query an invalid path");
    return seg->GetConfig(s);
}

template <typename LieGroup>
typename LieGroup::Tangent PathBase<LieGroup>::GetTangent(double s) const {
    auto seg = GetPathSegmentAtS(s);
    if (!seg) throw std::logic_error("cannot query an invalid path");
    return seg->GetTangent(s);
}

template <typename LieGroup>
typename LieGroup::Tangent PathBase<LieGroup>::GetCurvature(double s) const {
    auto seg = GetPathSegmentAtS(s);
    if (!seg) throw std::logic_error("cannot query an invalid path");
    return seg->GetCurvature(s);
}

template <typename LieGroup>
typename LieGroup::Tangent PathBase<LieGroup>::GetTorsion(double s) const {
    auto seg = GetPathSegmentAtS(s);
    if (!seg) throw std::logic_error("cannot query an invalid path");
    return seg->GetTorsion(s);
}

}  // namespace robotics
}  // namespace holistic_motion
