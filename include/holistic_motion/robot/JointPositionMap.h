#pragma once

#include <cmath>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace holistic_motion::robotics {

/// Coordinate block in a model configuration (q) and its tangent vector (v).
/// Continuous revolute joints use [cos(angle), sin(angle)] when nq=2, nv=1.
/// Scalar chain models instead use nq=nv=1, including continuous angles.
struct JointCoordinate {
    std::string name;
    int idx_q{0};
    int nq{1};
    int idx_v{0};
    int nv{1};
};

/// Immutable, backend-independent mapping of ordered scalar joint positions.
/// Build once per selected chain. All operations are const and own their
/// output; an instance can be shared between workers without sharing model
/// workspaces. Multi-DOF blocks (e.g. a free flyer) may remain in the reference
/// configuration but cannot be selected as scalar joint positions.
class JointPositionMap {
  public:
    JointPositionMap(int nq, int nv, std::vector<JointCoordinate> joints)
        : nq_(nq), nv_(nv), joints_(std::move(joints)) {
        if (nq < 0 || nv < 0)
            throw std::invalid_argument("model nq and nv must be non-negative");
        std::set<std::string> names;
        std::set<int> q_indices, v_indices;
        for (const auto &joint : joints_) {
            if (joint.name.empty() || !names.insert(joint.name).second)
                throw std::invalid_argument(
                    "joint names must be nonempty and unique");
            if (joint.nv != 1 || (joint.nq != 1 && joint.nq != 2))
                throw std::invalid_argument(
                    "joint position mapping requires a scalar joint: " +
                    joint.name);
            if (joint.idx_q < 0 || joint.idx_q > nq - joint.nq ||
                joint.idx_v < 0 || joint.idx_v >= nv)
                throw std::invalid_argument(
                    "joint coordinate block is outside model nq/nv: " +
                    joint.name);
            for (int i = 0; i < joint.nq; ++i)
                if (!q_indices.insert(joint.idx_q + i).second)
                    throw std::invalid_argument(
                        "joint configuration blocks must not overlap");
            if (!v_indices.insert(joint.idx_v).second)
                throw std::invalid_argument(
                    "joint velocity blocks must not overlap");
        }
    }

    int GetConfigurationSize() const noexcept { return nq_; }
    int GetVelocitySize() const noexcept { return nv_; }
    const std::vector<JointCoordinate> &GetJoints() const noexcept {
        return joints_;
    }

    Eigen::VectorXd
    ConfigurationWithJointPositions(const Eigen::VectorXd &reference,
                                    const Eigen::VectorXd &positions) const {
        Validate(reference, nq_, "reference configuration");
        Validate(positions, static_cast<Eigen::Index>(joints_.size()),
                 "joint positions");
        Eigen::VectorXd result = reference;
        for (std::size_t i = 0; i < joints_.size(); ++i) {
            const auto &joint = joints_[i];
            const double value = positions[static_cast<Eigen::Index>(i)];
            if (joint.nq == 1) {
                result[joint.idx_q] = value;
            } else {
                result[joint.idx_q] = std::cos(value);
                result[joint.idx_q + 1] = std::sin(value);
            }
        }
        return result;
    }

    /// Extract scalar positions; continuous pairs return the principal angle.
    Eigen::VectorXd JointPositions(const Eigen::VectorXd &configuration) const {
        Validate(configuration, nq_, "configuration");
        Eigen::VectorXd result(static_cast<Eigen::Index>(joints_.size()));
        for (std::size_t i = 0; i < joints_.size(); ++i) {
            const auto &joint = joints_[i];
            if (joint.nq == 1) {
                result[static_cast<Eigen::Index>(i)] =
                    configuration[joint.idx_q];
            } else {
                const double c = configuration[joint.idx_q];
                const double s = configuration[joint.idx_q + 1];
                if (std::abs(std::hypot(c, s) - 1.0) > 1e-8)
                    throw std::invalid_argument(
                        "continuous joint coordinates must form a unit pair: " +
                        joint.name);
                result[static_cast<Eigen::Index>(i)] = std::atan2(s, c);
            }
        }
        return result;
    }

    /// Restrict a full tangent vector or tangent-space gradient, never q
    /// indices.
    Eigen::VectorXd JointVelocities(const Eigen::VectorXd &velocity) const {
        Validate(velocity, nv_, "velocity/gradient");
        Eigen::VectorXd result(static_cast<Eigen::Index>(joints_.size()));
        for (std::size_t i = 0; i < joints_.size(); ++i)
            result[static_cast<Eigen::Index>(i)] = velocity[joints_[i].idx_v];
        return result;
    }

  private:
    static void Validate(const Eigen::VectorXd &values, Eigen::Index size,
                         const char *name) {
        if (values.size() != size || !values.allFinite())
            throw std::invalid_argument(
                std::string(name) +
                " must be finite and match its coordinate size");
    }

    int nq_, nv_;
    std::vector<JointCoordinate> joints_;
};

} // namespace holistic_motion::robotics
