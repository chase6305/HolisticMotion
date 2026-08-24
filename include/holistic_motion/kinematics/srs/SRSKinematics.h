#pragma once

#include "holistic_motion/kinematics/NumericalKinematics.h"

#include <optional>
#include <utility>

namespace holistic_motion::robotics {

enum class SRSSolveMethod {
    SEEDED_NUMERICAL,
    CONFIGURATION,
    ALL_CONFIGURATIONS,
    NEAREST_REDUNDANCY
};

struct SRSConfiguration {
    int shoulder{1};
    int elbow{1};
    int wrist{1};
    double redundancy{0.0};
};

struct SRSGeometryAnalysis {
    bool structurally_compatible{false};
    bool closed_form_compatible{false};
    Eigen::Vector3d shoulder_center{Eigen::Vector3d::Zero()};
    Eigen::Vector3d wrist_center{Eigen::Vector3d::Zero()};
    double upper_arm_length{0.0};
    double forearm_length{0.0};
    double shoulder_axis_residual{0.0};
    double wrist_axis_residual{0.0};
    double shoulder_orthogonality_residual{0.0};
    double wrist_orthogonality_residual{0.0};
    double elbow_angle_offset{0.0};
    double elbow_angle_direction{1.0};
};

// Solver utilities for redundant 7R manipulators with spherical
// shoulder/wrist layouts. All inputs use radians and SI units.
class SRSKinematics final : public NumericalKinematics {
public:
    explicit SRSKinematics(const std::vector<JointNode>& joint_nodes)
        : NumericalKinematics(joint_nodes) {}

    bool IsCompatible() const noexcept;

    SRSGeometryAnalysis AnalyzeGeometry(
            double axis_tolerance = 1e-4) const;

    bool GetNullSpaceVelocity(const Eigen::VectorXd& joints,
                              const Eigen::VectorXd& preferred_velocity,
                              Eigen::VectorXd& velocity) const;

    SRSConfiguration GetConfiguration(const Eigen::VectorXd& joints) const;

    double GetArmAngle(const Eigen::VectorXd& joints) const;

    bool Solve(const SE3d& target,
               const Eigen::VectorXd& seed,
               SRSSolveMethod method,
               std::vector<Eigen::VectorXd>& solutions) const;

    bool SolveConfiguration(const SE3d& target,
                            const SRSConfiguration& configuration,
                            const Eigen::VectorXd& seed,
                            Eigen::VectorXd& solution) const;

    bool GetAnalyticElbowSeed(const SE3d& target,
                              const SRSConfiguration& configuration,
                              const Eigen::VectorXd& seed,
                              Eigen::VectorXd& analytic_seed) const;

    bool GetAnalyticSolution(const SE3d& target,
                             const SRSConfiguration& configuration,
                             const Eigen::VectorXd& seed,
                             Eigen::VectorXd& solution) const;

private:
    mutable std::optional<std::pair<double, SRSGeometryAnalysis>>
            geometry_analysis_cache_;
};

}  // namespace holistic_motion::robotics
