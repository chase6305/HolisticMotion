#include "holistic_motion/kinematics/srs/SRSKinematics.h"

#include <algorithm>

#include <Eigen/SVD>

namespace holistic_motion::robotics {

bool SRSKinematics::GetNullSpaceVelocity(
    const Eigen::VectorXd &joints, const Eigen::VectorXd &preferred_velocity,
    Eigen::VectorXd &velocity) const {
    if (!IsCompatible() || joints.size() != 7 ||
        preferred_velocity.size() != 7 || !joints.allFinite() ||
        !preferred_velocity.allFinite())
        return false;
    Eigen::MatrixXd jacobian;
    if (!GetJacobian(joints, jacobian))
        return false;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinV);
    const auto &singular = svd.singularValues();
    if (svd.info() != Eigen::Success || singular.size() == 0 ||
        !singular.allFinite())
        return false;
    // Treat numerically weak task directions as singular. A machine-epsilon
    // cutoff can invert near-zero singular values and inject large joint
    // velocities into an otherwise bounded null-space request.
    const double threshold = std::max(1e-10, singular[0] * 1e-8);
    // The thin V is 7-by-6. Project onto its retained row-space directions
    // directly instead of forming an incorrectly sized 7-by-6 inverse or
    // amplifying roundoff through singular-value inversion and multiplication
    // by J. Compute all coefficients first to support in-place output.
    Eigen::VectorXd components = svd.matrixV().transpose() * preferred_velocity;
    for (Eigen::Index i = 0; i < singular.size(); ++i) {
        if (singular[i] <= threshold)
            components[i] = 0.0;
    }
    velocity = preferred_velocity - svd.matrixV() * components;
    return velocity.allFinite();
}

} // namespace holistic_motion::robotics
