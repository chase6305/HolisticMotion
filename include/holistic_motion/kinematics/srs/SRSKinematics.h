#pragma once

#include <memory>

#include "holistic_motion/kinematics/NumericalKinematics.h"

namespace holistic_motion::robotics {

namespace srs_detail {
struct CanonicalModel;
}

enum class SRSSolveMethod {
    SEEDED_NUMERICAL,
    CONFIGURATION,
    ALL_CONFIGURATIONS,
    NEAREST_REDUNDANCY
};

enum class SRSSolveStatus {
    SUCCESS,
    INVALID_INPUT,
    INCOMPATIBLE_MODEL,
    NO_SOLUTION
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

struct SRSSolveReport {
    SRSSolveStatus status{SRSSolveStatus::NO_SOLUTION};
    SRSSolveMethod method{SRSSolveMethod::NEAREST_REDUNDANCY};
    bool closed_form_compatible{false};
    std::vector<Eigen::VectorXd> solutions;
    std::vector<SRSConfiguration> configurations;
    std::vector<double> minimum_singular_values;
    std::vector<double> minimum_joint_limit_margins;
    std::vector<Eigen::VectorXd> joint_limit_margins;
    std::vector<bool> near_singularities;
    std::vector<bool> joint_limit_hits;
};

// Solver utilities for redundant 7R manipulators with spherical
// shoulder/wrist layouts. All inputs use radians and SI units.
/// Geometry is prepared during construction and model updates. Read-only
/// solving/analysis calls may share an instance with separate seeds/outputs;
/// model, frame and limit updates require external synchronization.
class SRSKinematics final : public NumericalKinematics {
public:
  KinematicsCapabilities GetCapabilities() const override;

  explicit SRSKinematics(const std::vector<JointNode> &joint_nodes)
      : NumericalKinematics(joint_nodes) {
      OnKinematicModelChanged();
  }

    bool IsCompatible() const noexcept;

    SRSGeometryAnalysis AnalyzeGeometry(
            double axis_tolerance = 1e-4) const;

    bool GetNullSpaceVelocity(const Eigen::VectorXd& joints,
                              const Eigen::VectorXd& preferred_velocity,
                              Eigen::VectorXd& velocity) const;

    SRSConfiguration GetConfiguration(const Eigen::VectorXd& joints) const;

    double GetArmAngle(const Eigen::VectorXd& joints) const;

    /// A seed may refer to an element of the previous solutions vector.
    bool Solve(const SE3d& target,
               const Eigen::VectorXd& seed,
               SRSSolveMethod method,
               std::vector<Eigen::VectorXd>& solutions) const;

    SRSSolveReport SolveDetailed(const SE3d& target,
                                 const Eigen::VectorXd& seed,
                                 SRSSolveMethod method) const;

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
    bool UsesDefaultForwardKinematics() const noexcept override { return true; }

    void OnKinematicModelChanged() override;
    SRSGeometryAnalysis ComputeGeometryAnalysis() const;

    bool GetCanonicalModel(srs_detail::CanonicalModel &model) const;

    SRSGeometryAnalysis geometry_analysis_;
    std::shared_ptr<const srs_detail::CanonicalModel> canonical_model_;
};

}  // namespace holistic_motion::robotics
