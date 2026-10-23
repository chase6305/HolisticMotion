#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace holistic_motion::robotics::planning {

/// Immutable scalar planning coordinates, independent of a robot or collision
/// backend. Optional names define the vector order; they do not reorder input.
/// Share a value snapshot between planning and optimization to retain the same
/// limits, metric and periodic topology. Validators remain separate.
/// Copies share immutable coordinate data without allocation. Builders publish
/// new data; existing spaces, planners and results retain their snapshots.
class JointSpace {
  public:
    JointSpace(Eigen::VectorXd lower_limits, Eigen::VectorXd upper_limits,
               std::vector<std::string> joint_names = {});

    JointSpace(const JointSpace &) noexcept = default;
    JointSpace &operator=(const JointSpace &) noexcept = default;
    // Immutable snapshots remain usable after a move as well as a copy.
    JointSpace(JointSpace &&other) noexcept : data_(other.data_) {}
    JointSpace &operator=(JointSpace &&other) noexcept {
        data_ = other.data_;
        return *this;
    }

    Eigen::Index GetDOF() const noexcept;
    const Eigen::VectorXd &GetLowerLimits() const noexcept;
    const Eigen::VectorXd &GetUpperLimits() const noexcept;
    const Eigen::VectorXd &GetJointWeights() const noexcept;
    const std::vector<std::string> &GetJointNames() const noexcept;
    const std::vector<bool> &GetContinuousMask() const noexcept;
    std::vector<std::size_t> GetContinuousJoints() const;

    /// Return a new space; neither this value nor existing consumers change.
    JointSpace WithJointWeights(const Eigen::VectorXd &weights) const;
    /// Periodic scalar angles have period 2*pi and need a representable full
    /// revolution within their limits. Duplicate indices are accepted.
    JointSpace WithContinuousJoints(const std::vector<std::size_t> &indices) const;

    /// Lift each periodic edge to its shortest arc before scalar trajectory
    /// interpolation. Bounded coordinates remain exact. Input rows must be in
    /// this space; lifted periodic values may leave its representative limits.
    /// This preserves the planned polyline topology, not spline/blend geometry
    /// or continuous collision validity. Reject unrepresentable lifted steps or
    /// shifts that change a source angle's sine/cosine by more than 1e-10.
    std::vector<Eigen::VectorXd>
    UnwrapPath(const std::vector<Eigen::VectorXd> &path) const;

  private:
    struct Data;
    explicit JointSpace(std::shared_ptr<const Data> data) : data_(std::move(data)) {}
    std::shared_ptr<const Data> data_;
};

} // namespace holistic_motion::robotics::planning
