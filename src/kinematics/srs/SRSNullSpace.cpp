#include "holistic_motion/kinematics/srs/SRSKinematics.h"

#include <algorithm>
#include <limits>

#include <Eigen/SVD>

namespace holistic_motion::robotics {

bool SRSKinematics::GetNullSpaceVelocity(
        const Eigen::VectorXd& joints,
        const Eigen::VectorXd& preferred_velocity,
        Eigen::VectorXd& velocity) const {
    if (!IsCompatible() || joints.size() != 7 ||
        preferred_velocity.size() != 7) return false;
    Eigen::MatrixXd jacobian;
    if (!GetJacobian(joints, jacobian)) return false;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto& singular = svd.singularValues();
    Eigen::MatrixXd inverse =
            Eigen::MatrixXd::Zero(jacobian.cols(), jacobian.rows());
    const double threshold = std::max(
            1e-10, singular[0] * std::numeric_limits<double>::epsilon() *
                           std::max(jacobian.rows(), jacobian.cols()) * 16.0);
    for (Eigen::Index i = 0; i < singular.size(); ++i) {
        if (singular[i] > threshold) inverse(i, i) = 1.0 / singular[i];
    }
    const Eigen::MatrixXd pseudo_inverse =
            svd.matrixV() * inverse * svd.matrixU().transpose();
    velocity = (Eigen::MatrixXd::Identity(7, 7) -
                pseudo_inverse * jacobian) * preferred_velocity;
    return velocity.allFinite();
}

}  // namespace holistic_motion::robotics
