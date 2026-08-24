#pragma once

#include "holistic_motion/kinematics/NumericalKinematics.h"

namespace holistic_motion::robotics {

enum class FEPSolveMethod {
    SEEDED_NUMERICAL,
    CONFIGURATION,
    ALL_CONFIGURATIONS,
    NEAREST_REDUNDANCY,
    COMPATIBILITY
};

enum class FEPBackend { AUTO, CPU, CUDA };

struct FEPConfiguration {
    int shoulder{1};
    int elbow{1};
    int wrist{1};
    double redundancy{0.0};
};

// Solver facade for offset 7R manipulators. Analytical backends are enabled
// only after their geometry model has been validated; all current methods use
// the shared URDF FK and validated numerical correction path.
class FEPKinematics final : public NumericalKinematics {
public:
    explicit FEPKinematics(const std::vector<JointNode>& joint_nodes)
        : NumericalKinematics(joint_nodes) {}

    bool IsCompatible() const noexcept;
    static bool HasCudaBackend() noexcept;
    static bool IsCudaCompiled() noexcept;
    FEPBackend ResolveBackend(FEPBackend requested,
                              std::size_t batch_size) const noexcept;
    bool ForwardBatch(const Eigen::MatrixXd& joints,
                      FEPBackend backend,
                      std::vector<Eigen::Matrix4d>& poses) const;
    FEPConfiguration GetConfiguration(const Eigen::VectorXd& joints) const;

    bool Solve(const SE3d& target,
               const Eigen::VectorXd& seed,
               FEPSolveMethod method,
               std::vector<Eigen::VectorXd>& solutions) const;

    bool SolveConfiguration(const SE3d& target,
                            const FEPConfiguration& configuration,
                            const Eigen::VectorXd& seed,
                            Eigen::VectorXd& solution) const;

private:
    bool SolveSeed(const SE3d& target,
                   Eigen::VectorXd seed,
                   Eigen::VectorXd& solution) const;
};

}  // namespace holistic_motion::robotics
