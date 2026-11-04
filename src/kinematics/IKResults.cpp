#include "holistic_motion/kinematics/utility.h"
#include "holistic_motion/utility/Logging.h"

#include "IKDeduplication.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace holistic_motion::robotics {

// add joints vector to ik_joints
bool IkRtn::PushBack(const Eigen::VectorXd &joints) {
    if (!joints.allFinite())
        return false;
    if (!ik_joints.empty() && joints.size() != this->dof) {
        holistic_motion::utility::LogWarning(
            "Ignoring IK with inconsistent DOF: expected {}, got {}", this->dof, joints.size());
        return false;
    }
    this->dof = joints.size();
    ik_joints.push_back(joints);
    ik_number = static_cast<int>(ik_joints.size());
    return true;
}

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
    if (!std::isfinite(tolerance) || tolerance < 0.0 || dof < 0 || ik_joints.empty() ||
        ik_joints.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        std::any_of(ik_joints.begin(), ik_joints.end(),
                    [&](const auto &q) { return q.size() != dof || !q.allFinite(); }))
        return fail();

    if (dof == 0) {
        ik_joints.resize(1);
    } else if (ik_joints.size() > 1) {
        const auto lexicographic = [](const auto &first, const auto &second) {
            for (Eigen::Index joint = 0; joint < first.size(); ++joint) {
                if (first[joint] != second[joint])
                    return first[joint] < second[joint];
            }
            return false;
        };
        std::sort(ik_joints.begin(), ik_joints.end(), lexicographic);
        if (tolerance == 0.0) {
            ik_joints.erase(
                std::unique(ik_joints.begin(), ik_joints.end(),
                            [](const auto &first, const auto &second) { return first == second; }),
                ik_joints.end());
        } else if (ik_joints.size() <= 8) {
            // Analytic solvers emit at most eight candidates; compact them directly.
            std::size_t kept = 0;
            for (auto &candidate : ik_joints) {
                if (std::any_of(ik_joints.begin(), ik_joints.begin() + kept,
                                [&](const auto &point) {
                                    return ((candidate - point).array().abs() < tolerance).all();
                                }))
                    continue;
                if (&candidate != &ik_joints[kept])
                    ik_joints[kept] = std::move(candidate);
                ++kept;
            }
            ik_joints.resize(kept);
        } else {
            // Approximate equality is not transitive. Query only retained
            // points, preserving lexicographically first representatives.
            detail::RetainedSolutionIndex index(ik_joints);
            for (std::size_t candidate = 0; candidate < ik_joints.size(); ++candidate)
                if (!index.HasNeighbor(ik_joints[candidate], tolerance))
                    index.Retain(candidate);
            // Coordinates stay fixed until all spatial queries have finished.
            std::size_t kept = 0;
            for (std::size_t candidate = 0; candidate < ik_joints.size(); ++candidate) {
                if (!index.IsRetained(candidate))
                    continue;
                if (kept != candidate)
                    ik_joints[kept] = std::move(ik_joints[candidate]);
                ++kept;
            }
            ik_joints.resize(kept);
        }
    }
    ik_number = static_cast<int>(ik_joints.size());
    success = true;
    return true;
}

bool IkRtn::SortMinMovement(const Eigen::VectorXd &target_joints, std::vector<double> &dist) {
    joint_states.resize(0, 0);
    const auto fail = [this, &dist]() {
        ik_joints.clear();
        dist.clear();
        ik_number = 0;
        success = false;
        return false;
    };
    if (dof < 0 || target_joints.size() != dof || !target_joints.allFinite() || ik_joints.empty() ||
        dist.size() != ik_joints.size() ||
        ik_joints.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        std::any_of(ik_joints.begin(), ik_joints.end(),
                    [&](const auto &q) { return q.size() != dof || !q.allFinite(); }) ||
        std::any_of(dist.begin(), dist.end(), [](double value) { return !std::isfinite(value); }))
        return fail();

    if (ik_joints.size() > 1) {
        std::vector<std::size_t> indices(ik_joints.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&dist](std::size_t first, std::size_t second) {
            return dist[first] != dist[second] ? dist[first] < dist[second] : first < second;
        });
        // Rotate each permutation cycle in place. Original indices break distance
        // ties, retaining stable order without allocating whole result copies.
        for (std::size_t start = 0; start < indices.size(); ++start) {
            if (indices[start] == start)
                continue;
            Eigen::VectorXd first = std::move(ik_joints[start]);
            const double first_distance = dist[start];
            std::size_t destination = start;
            while (indices[destination] != start) {
                const std::size_t source = indices[destination];
                ik_joints[destination] = std::move(ik_joints[source]);
                dist[destination] = dist[source];
                indices[destination] = destination;
                destination = source;
            }
            ik_joints[destination] = std::move(first);
            dist[destination] = first_distance;
            indices[destination] = destination;
        }
    }
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
    if (dof < 0 || filter.Size() != static_cast<std::size_t>(dof) || ik_joints.empty() ||
        ik_joints.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        std::any_of(ik_joints.begin(), ik_joints.end(),
                    [&](const auto &q) { return q.size() != dof || !q.allFinite(); }))
        return fail();

    ik_joints.erase(std::remove_if(ik_joints.begin(), ik_joints.end(),
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
