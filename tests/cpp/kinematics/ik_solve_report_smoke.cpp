#include <cmath>
#include <limits>
#include <vector>

#include "holistic_motion/kinematics/NumericalKinematics.h"
#include "holistic_motion/kinematics/OPWKinematics.h"
#include "holistic_motion/kinematics/URKinematics.h"

using namespace holistic_motion::robotics;

namespace {

template <class Base> class InheritedSolver : public Base {
  public:
    using Base::Base;
};

class ConcurrentNumerical : public NumericalKinematics {
  public:
    using NumericalKinematics::NumericalKinematics;
    KinematicsCapabilities GetCapabilities() const override {
        auto result = NumericalKinematics::GetCapabilities();
        result.supports_concurrent_queries = true;
        return result;
    }
};

// Existing custom solvers need not implement new virtual methods. A malformed
// successful legacy output must not become a successful common report.
class CustomSolver : public KinematicsBase {
  public:
    CustomSolver() { dof_ = 1; }
    bool GetFK(const Eigen::VectorXd &, SE3d &) const override { return false; }
    bool GetIK(const SE3d &, IkRtn &result, Eigen::VectorXd &,
               std::vector<double> &distances) const override {
        result.ik_joints.push_back(Eigen::VectorXd::Zero(2));
        distances.push_back(0.0);
        return true;
    }
    bool GetNearstIK(const SE3d &, IkRtn &result, Eigen::VectorXd &,
                     double &distance) const override {
        result.ik_joints.push_back(Eigen::VectorXd::Zero(1));
        distance = std::numeric_limits<double>::quiet_NaN();
        return true;
    }
};

bool EmptyFailure(const IKSolveReport &report, IKSolveStatus status) {
    return !report.Success() && report.status == status && report.solutions.empty() &&
           report.distances.empty();
}

} // namespace

int main() {
    const JointNode terminal;
    NumericalKinematics fixed({terminal});
    const Eigen::VectorXd empty;
    for (auto selection : {IKSelection::NEAREST, IKSelection::ALL}) {
        const auto report = fixed.SolveIK(SE3d(), empty, selection);
        if (!report.Success() || report.solutions.size() != 1 ||
            report.solutions.front().size() != 0 || report.distances.size() != 1 ||
            report.distances.front() != 0.0 ||
            report.algorithm != IKAlgorithm::NUMERICAL)
            return 1;
        if (!EmptyFailure(
                fixed.SolveIK(SE3d(Eigen::Vector3d(1, 0, 0), SO3d()), empty, selection),
                IKSolveStatus::NO_SOLUTION))
            return 2;
    }

    std::vector<JointNode> nodes{
        {SE3d(), Eigen::Vector3d::UnitZ(), JointType::REVOLUTE, -1e6, 1e6}, terminal};
    NumericalKinematics solver(nodes);
    const InheritedSolver<NumericalKinematics> inherited(nodes);
    const ConcurrentNumerical opted_in(nodes);
    if (!solver.GetCapabilities().supports_concurrent_queries ||
        inherited.GetCapabilities().supports_concurrent_queries ||
        !opted_in.GetCapabilities().supports_concurrent_queries)
        return 8;
    const Eigen::VectorXd seed = Eigen::VectorXd::Constant(1, 0.3);
    SE3d target;
    if (!solver.GetFK(seed, target))
        return 3;
    const auto nearest = solver.SolveIK(target, seed);
    if (!nearest.Success() || nearest.solutions.size() != 1 ||
        !nearest.solutions.front().isApprox(seed, 1e-12) || seed[0] != 0.3)
        return 4;
    // ALL exhausts the existing periodic-expansion budget, while NEAREST
    // requests one representation and remains useful for wide limits.
    if (!EmptyFailure(solver.SolveIK(target, seed, IKSelection::ALL),
                      IKSolveStatus::NO_SOLUTION))
        return 5;
    SE3d invalid = target;
    invalid.Coeffs().tail<4>() *= 0.5;
    if (!EmptyFailure(solver.SolveIK(invalid, seed), IKSolveStatus::INVALID_INPUT) ||
        !EmptyFailure(solver.SolveIK(target, empty), IKSolveStatus::INVALID_INPUT) ||
        !EmptyFailure(solver.SolveIK(target, seed, static_cast<IKSelection>(42)),
                      IKSolveStatus::INVALID_INPUT))
        return 6;

    CustomSolver custom;
    const auto capabilities = custom.GetCapabilities();
    if (capabilities.family != KinematicsFamily::CUSTOM ||
        capabilities.default_ik_algorithm != IKAlgorithm::CUSTOM ||
        capabilities.supports_concurrent_queries ||
        !EmptyFailure(custom.SolveIK(SE3d(), seed), IKSolveStatus::NO_SOLUTION) ||
        !EmptyFailure(custom.SolveIK(SE3d(), seed, IKSelection::ALL),
                      IKSolveStatus::NO_SOLUTION))
        return 7;
    std::vector<JointNode> analytic_nodes(6, nodes.front());
    analytic_nodes.push_back(terminal);
    OPWParameters opw_parameters;
    opw_parameters.c2 = 0.6;
    opw_parameters.c3 = 0.1;
    opw_parameters.rotation_directions.fill(1);
    URParameters ur_parameters;
    ur_parameters.d1 = 0.1;
    ur_parameters.a2 = -0.4;
    ur_parameters.a3 = -0.3;
    ur_parameters.d4 = 0.1;
    ur_parameters.d5 = 0.1;
    ur_parameters.d6 = 0.1;
    ur_parameters.rotation_directions.fill(1);
    const InheritedSolver<OPWKinematics> inherited_opw(opw_parameters, analytic_nodes);
    const InheritedSolver<URKinematics> inherited_ur(ur_parameters, analytic_nodes);
    if (inherited_opw.GetCapabilities().supports_concurrent_queries ||
        inherited_ur.GetCapabilities().supports_concurrent_queries)
        return 9;
    return 0;
}
