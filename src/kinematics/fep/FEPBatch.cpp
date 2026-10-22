#include "holistic_motion/kinematics/fep/FEPKinematics.h"

#include "FEPBatchCUDA.h"
#include "FEPSelection.h"
#include "../JointMotion.h"

#include <array>
#include <limits>

namespace holistic_motion::robotics {

FEPBackend FEPKinematics::ResolveBackend(FEPBackend requested,
                                         std::size_t batch_size) const noexcept {
    if (requested == FEPBackend::CPU) return FEPBackend::CPU;
    if (requested == FEPBackend::CUDA)
        return HasCudaBackend() ? FEPBackend::CUDA : FEPBackend::CPU;
    // Small AUTO batches stay on CPU without initializing the CUDA runtime.
    return batch_size >= 128 && HasCudaBackend() ? FEPBackend::CUDA
                                                 : FEPBackend::CPU;
}

bool FEPKinematics::ForwardBatch(
        const Eigen::MatrixXd& joints,
        FEPBackend backend,
        std::vector<Eigen::Matrix4d>& poses) const {
    poses.clear();
    if (!IsCompatible() || joints.cols() != 7 || !joints.allFinite())
        return false;
    for (Eigen::Index row = 0; row < joints.rows(); ++row) {
        for (Eigen::Index joint = 0; joint < 7; ++joint) {
            if (joints(row, joint) < joint_nodes_[joint].lower_limit ||
                joints(row, joint) > joint_nodes_[joint].upper_limit)
                return false;
        }
    }
    const auto resolved = ResolveBackend(
            backend, static_cast<std::size_t>(joints.rows()));
    if (backend == FEPBackend::CUDA && resolved != FEPBackend::CUDA)
        return false;
#ifndef HOLISTICMOTION_HAS_CUDA
    (void)resolved;
#endif
    // Prepare the axis scale once on the host for every backend. Recomputing
    // its norm on the device can differ by an ulp and change a large q's phase.
    std::array<Eigen::Vector3d, 7> axes;
    std::array<double, 7> axis_norms;
    for (int joint = 0; joint < 7; ++joint) {
        axis_norms[joint] = detail::JointAxisNorm(joint_nodes_[joint].axis);
        axes[joint] = joint_nodes_[joint].axis;
        if (axis_norms[joint] > 0.0) axes[joint] /= axis_norms[joint];
    }
#ifdef HOLISTICMOTION_HAS_CUDA
    if (resolved == FEPBackend::CUDA && joints.rows() > 0 &&
        joints.rows() <= (std::numeric_limits<int>::max)()) {
        std::vector<double> packed_joints(
                static_cast<std::size_t>(joints.rows()) * 7);
        std::array<double, 8 * 16> origins{};
        std::array<double, 7 * 4> packed_axes{};
        std::array<double, 16> tcp{};
        std::vector<double> output(
                static_cast<std::size_t>(joints.rows()) * 16);
        for (Eigen::Index row = 0; row < joints.rows(); ++row)
            for (int joint = 0; joint < 7; ++joint)
                packed_joints[static_cast<std::size_t>(row) * 7 + joint] =
                        joints(row, joint);
        for (int node = 0; node < 8; ++node) {
            const auto matrix = joint_nodes_[node].origin_pose.GetTransform();
            for (int row = 0; row < 4; ++row)
                for (int col = 0; col < 4; ++col)
                    origins[node * 16 + row * 4 + col] = matrix(row, col);
            if (node < 7) {
                for (int axis = 0; axis < 3; ++axis)
                    packed_axes[node * 4 + axis] = axes[node][axis];
                packed_axes[node * 4 + 3] = axis_norms[node];
            }
        }
        const auto tcp_matrix = GetTCP().GetTransform();
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                tcp[row * 4 + col] = tcp_matrix(row, col);
        if (fep_internal::ForwardBatchCUDA(
                    packed_joints.data(), static_cast<int>(joints.rows()),
                    origins.data(), packed_axes.data(), tcp.data(), output.data())) {
            poses.resize(static_cast<std::size_t>(joints.rows()));
            bool finite_poses = true;
            for (Eigen::Index pose = 0; pose < joints.rows(); ++pose) {
                for (int row = 0; row < 4; ++row)
                    for (int col = 0; col < 4; ++col)
                        poses[static_cast<std::size_t>(pose)](row, col) =
                                output[static_cast<std::size_t>(pose) * 16 +
                                       row * 4 + col];
                if (!poses[static_cast<std::size_t>(pose)].allFinite()) {
                    finite_poses = false;
                    break;
                }
            }
            if (finite_poses) return true;
            poses.clear();
        }
        if (backend == FEPBackend::CUDA) return false;
    }
    if (backend == FEPBackend::CUDA &&
        joints.rows() > (std::numeric_limits<int>::max)())
        return false;
#else
    if (backend == FEPBackend::CUDA) return false;
#endif
    // Prepare the fixed model data once per call. This also observes model/TCP
    // updates without a persistent cache and avoids per-row joint/pose vectors.
    const SE3d terminal = joint_nodes_.back().origin_pose * GetTCP();
    poses.reserve(static_cast<std::size_t>(joints.rows()));
    for (Eigen::Index row = 0; row < joints.rows(); ++row) {
        SE3d pose;
        for (int joint = 0; joint < 7; ++joint) {
            pose = pose * joint_nodes_[joint].origin_pose;
            if (joints(row, joint) != 0.0) {
                const SO3d rotation(Eigen::AngleAxisd(
                        joints(row, joint) * axis_norms[joint], axes[joint]));
                pose = SE3d(pose.GetTranslation(),
                           SO3d(pose.GetQuat()) * rotation);
            }
        }
        poses.push_back((pose * terminal).GetTransform());
        if (!poses.back().allFinite()) {
            poses.clear();
            return false;
        }
    }
    return true;
}

}  // namespace holistic_motion::robotics
