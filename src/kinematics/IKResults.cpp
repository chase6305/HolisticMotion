#include "holistic_motion/kinematics/utility.h"

#include "IKDeduplication.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace holistic_motion::robotics {

// add joints vector to ik_joints
bool IkRtn::PushBack(const Eigen::VectorXd &joints) {
    if (!joints.allFinite())
        return false;
    if (!ik_joints.empty() && joints.size() != this->dof) {
        holistic_motion::utility::LogWarning(
            "Ignoring IK with inconsistent DOF: expected {}, got {}", this->dof,
            joints.size());
        return false;
    }
    this->dof = joints.size();
    ik_joints.push_back(joints);
    ik_number = static_cast<int>(ik_joints.size());
    return true;
};

bool IkRtn::RemoveRepeatedIK(const double &eps) {
    const double tolerance = eps;
    // Public result fields may be edited by callers. Validate before sorting,
    // and use the actual storage rather than trusting its cached count.
    joint_states.resize(0, 0);
    const auto fail = [this]() {
        ik_joints.clear();
        ik_number = 0;
        success = false;
        return false;
    };
    if (!std::isfinite(tolerance) || tolerance < 0.0 || dof < 0 ||
        ik_joints.empty() ||
        ik_joints.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        std::any_of(ik_joints.begin(), ik_joints.end(), [&](const auto &q) {
            return q.size() != dof || !q.allFinite();
        }))
        return fail();

    const auto lexicographic = [](const auto &first, const auto &second) {
        for (Eigen::Index joint = 0; joint < first.size(); ++joint) {
            if (first[joint] != second[joint])
                return first[joint] < second[joint];
        }
        return false;
    };
    std::sort(ik_joints.begin(), ik_joints.end(), lexicographic);
    if (dof == 0) {
        ik_joints.resize(1);
    } else if (tolerance == 0.0) {
        ik_joints.erase(std::unique(ik_joints.begin(), ik_joints.end(),
                                    [](const auto &first, const auto &second) {
                                        return first == second;
                                    }),
                        ik_joints.end());
    } else {
        // Approximate equality is not transitive, so it cannot be used as an
        // ordered-container comparator. Query only previously retained points
        // while preserving lexicographically first representatives.
        detail::RetainedSolutionIndex index(ik_joints);
        std::vector<std::size_t> retained;
        retained.reserve(ik_joints.size());
        for (std::size_t candidate = 0; candidate < ik_joints.size();
             ++candidate) {
            if (!index.HasNeighbor(ik_joints[candidate], tolerance)) {
                retained.push_back(candidate);
                index.Retain(candidate);
            }
        }
        // The spatial partition references original coordinates. Compact only
        // after every query has finished.
        for (std::size_t kept = 0; kept < retained.size(); ++kept) {
            if (kept != retained[kept])
                ik_joints[kept] = std::move(ik_joints[retained[kept]]);
        }
        ik_joints.resize(retained.size());
    }
    ik_number = static_cast<int>(ik_joints.size());
    success = true;
    return true;
}

bool IkRtn::SortMinMovement(const Eigen::VectorXd &target_joints,
                            std::vector<double> &dist) {
    joint_states.resize(0, 0);
    const auto fail = [this, &dist]() {
        ik_joints.clear();
        dist.clear();
        ik_number = 0;
        success = false;
        return false;
    };
    if (dof < 0 || target_joints.size() != dof || !target_joints.allFinite() ||
        ik_joints.empty() || dist.size() != ik_joints.size() ||
        ik_joints.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        std::any_of(
            ik_joints.begin(), ik_joints.end(),
            [&](const auto &q) { return q.size() != dof || !q.allFinite(); }) ||
        std::any_of(dist.begin(), dist.end(),
                    [](double value) { return !std::isfinite(value); }))
        return fail();

    std::vector<std::size_t> indices(ik_joints.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
                     [&dist](std::size_t first, std::size_t second) {
                         return dist[first] < dist[second];
                     });
    std::vector<Eigen::VectorXd> sorted;
    std::vector<double> sorted_distances;
    sorted.reserve(indices.size());
    sorted_distances.reserve(indices.size());
    for (std::size_t index : indices) {
        sorted.push_back(std::move(ik_joints[index]));
        sorted_distances.push_back(dist[index]);
    }
    ik_joints = std::move(sorted);
    dist = std::move(sorted_distances);
    ik_number = static_cast<int>(ik_joints.size());
    success = true;
    return true;
}

bool IkRtn::FilterJointSolutions(const JointFilterManager &filter) {
    joint_states.resize(0, 0);
    const auto fail = [this]() {
        ik_joints.clear();
        ik_number = 0;
        success = false;
        return false;
    };
    if (dof < 0 || filter.Size() != static_cast<std::size_t>(dof) ||
        ik_joints.empty() ||
        ik_joints.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        std::any_of(ik_joints.begin(), ik_joints.end(), [&](const auto &q) {
            return q.size() != dof || !q.allFinite();
        }))
        return fail();

    ik_joints.erase(
        std::remove_if(ik_joints.begin(), ik_joints.end(),
                       [&](const auto &q) {
                           for (int joint = 0; joint < dof; ++joint) {
                               if (!filter.IsValidJointAngle(joint, q[joint]))
                                   return true;
                           }
                           return false;
                       }),
        ik_joints.end());
    ik_number = static_cast<int>(ik_joints.size());
    success = ik_number > 0;
    return success;
}

} // namespace holistic_motion::robotics
