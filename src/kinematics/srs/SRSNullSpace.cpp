#include "holistic_motion/kinematics/srs/SRSKinematics.h"

#include <algorithm>
#include <limits>

#include <Eigen/SVD>

namespace holistic_motion::robotics {

bool SRSKinematics::GetNullSpaceVelocity(const Eigen::VectorXd &joints,
                                         const Eigen::VectorXd &preferred_velocity,
                                         Eigen::VectorXd &velocity) const {
    if (!IsCompatible() || joints.size() != 7 || preferred_velocity.size() != 7 ||
        !joints.allFinite() || !preferred_velocity.allFinite())
        return false;
    Eigen::MatrixXd jacobian;
    if (!GetJacobian(joints, jacobian))
        return false;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(jacobian, Eigen::ComputeThinV);
    const auto &singular = svd.singularValues();
    if (svd.info() != Eigen::Success || singular.size() == 0 || !singular.allFinite())
        return false;
    // Treat numerically weak task directions as singular. A machine-epsilon
    // cutoff can invert near-zero singular values and inject large joint
    // velocities into an otherwise bounded null-space request.
    const double threshold = std::max(1e-10, singular[0] * 1e-8);
    // The thin V is 7-by-6. Project onto its retained row-space directions
    // directly instead of forming an incorrectly sized 7-by-6 inverse or
    // amplifying roundoff through singular-value inversion and multiplication
    // by J. Compute all coefficients first to support in-place output.
    // The model has exactly seven coordinates and six task directions. Keep
    // projection storage on the stack and scale only inputs whose dot products
    // could overflow before row-space cancellation produces a finite result.
    Eigen::Matrix<double, 6, 1> components;
    Eigen::Matrix<double, 7, 1> scaled_preferred;
    const double maximum = preferred_velocity.cwiseAbs().maxCoeff();
    const bool scale_input = maximum > std::numeric_limits<double>::max() / 16.0;
    if (scale_input) {
        scaled_preferred = preferred_velocity / maximum;
        components.noalias() = svd.matrixV().transpose() * scaled_preferred;
    } else {
        components.noalias() = svd.matrixV().transpose() * preferred_velocity;
    }
    for (Eigen::Index i = 0; i < singular.size(); ++i) {
        if (singular[i] <= threshold)
            components[i] = 0.0;
    }
    if (scale_input)
        velocity = (scaled_preferred - svd.matrixV() * components) * maximum;
    else
        velocity = preferred_velocity - svd.matrixV() * components;
    return velocity.allFinite();
}

} // namespace holistic_motion::robotics
