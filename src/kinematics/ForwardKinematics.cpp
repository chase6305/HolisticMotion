#include "ForwardKinematics.h"

#include "JointMotion.h"
#include "holistic_motion/kinematics/KinematicsBase.h"

namespace holistic_motion::robotics::detail {
namespace {
template <typename Visitor, typename Prepare>
bool Evaluate(const KinematicsBase &solver, const Eigen::VectorXd &target_joint,
              Visitor &&visit, Prepare &&prepare, SE3d &terminal_pose) {
    const auto &nodes = solver.GetJointNodesView();
    const int dof = solver.GetDOF();
    const auto tcp = solver.GetTCP();

    const std::size_t num = nodes.size();
    if (num == 0 || dof < 0 || static_cast<std::size_t>(dof) > num ||
        target_joint.size() != dof || !target_joint.allFinite()) {
        holistic_motion::utility::LogWarning(
            "Invalid FK model or joint vector (DOF {}, nodes {}, joints {})", dof, num,
            target_joint.size());
        return false;
    }
    prepare(num);
    SE3d pre_pose = SE3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    for (std::size_t i = 0; i < num; ++i) {
        const auto type = nodes[i].joint_type;
        if ((type != JointType::FIXED && type != JointType::UNKNOWN &&
             i >= static_cast<std::size_t>(dof)) ||
            type == JointType::PLANAR || type == JointType::FLOATING) {
            return false;
        }
        // Fixed nodes and exactly zero joint motions only compose their
        // origin. Avoid constructing and multiplying a second identity SE3.
        // Keep malformed protected-model axes on the validation path below.
        if (type == JointType::FIXED || type == JointType::UNKNOWN ||
            (i < static_cast<std::size_t>(dof) && target_joint[i] == 0.0 &&
             nodes[i].axis.allFinite())) {
            pre_pose = pre_pose * nodes[i].origin_pose;
            visit(i, pre_pose);
            continue;
        }
        SE3d transform = SE3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        Eigen::Vector3d xyz(0.0, 0.0, 0.0);
        SO3d rot = SO3d(0.0, 0.0, 0.0, 1.0);
        if (JointType::REVOLUTE == nodes[i].joint_type ||
            JointType::CONTINUOUS == nodes[i].joint_type) {
            // SO3d(Vector3d) interprets Euler angles, not a rotation vector.
            const Eigen::Vector3d &axis = nodes[i].axis;
            const double axis_norm = detail::JointAxisNorm(axis);
            // Compute the model's axis scale before applying q, as CPU/CUDA
            // batch FK does. Reconstructing the norm from q*axis can change a
            // large angle by whole turns through component rounding alone.
            const double angle = target_joint[i] * axis_norm;
            if (!std::isfinite(angle)) {
                return false;
            }
            if (angle != 0.0) {
                rot = SO3d(Eigen::AngleAxisd(angle, axis / axis_norm));
            }
        }
        if (JointType::PRISMATIC == nodes[i].joint_type) {
            // generate translation vector
            xyz = nodes[i].axis * target_joint[i];
            if (!xyz.allFinite()) {
                return false;
            }
        }
        transform = SE3d(xyz, rot);
        SE3d pose = pre_pose * nodes[i].origin_pose;
        if (type == JointType::REVOLUTE || type == JointType::CONTINUOUS) {
            // Revolute motion has no translation. Compose the same SO3
            // factors without forming a rotation matrix just to multiply zero.
            pose = SE3d(pose.GetTranslation(), SO3d(pose.GetQuat()) * rot);
        } else {
            pose = pose * transform;
        }
        pre_pose = pose;
        visit(i, pre_pose);
        // holistic_motion::utility::LogDebug("Joint{} pose:[SE3({})].", i + 1,
        //                         fmt::join(pre_pose.Coeffs(), ","));
    }
    // update tcp pose
    const SE3d terminal = pre_pose * tcp;
    if (!terminal.Coeffs().allFinite()) {
        return false;
    }
    terminal_pose = terminal;
    return true;
}
} // namespace

bool ComputeTerminalFK(const KinematicsBase &solver, const Eigen::VectorXd &joints,
                       SE3d &pose) {
    return Evaluate(
        solver, joints, [](std::size_t, const SE3d &) {}, [](std::size_t) {}, pose);
}

bool ComputeAllFK(const KinematicsBase &solver, const Eigen::VectorXd &joints,
                  std::vector<SE3d> &poses) {
    poses.clear();
    SE3d terminal;
    if (!Evaluate(
            solver, joints,
            [&poses](std::size_t, const SE3d &pose) { poses.push_back(pose); },
            [&poses](std::size_t count) { poses.reserve(count + 1); }, terminal)) {
        poses.clear();
        return false;
    }
    poses.push_back(terminal);
    return true;
}

bool ComputeJacobian(const KinematicsBase &solver, const Eigen::VectorXd &joints,
                     Eigen::Ref<Eigen::MatrixXd> jacobian) {
    const auto &nodes = solver.GetJointNodesView();
    const int dof = solver.GetDOF();
    if (jacobian.rows() != 6 || jacobian.cols() != dof)
        return false;
    SE3d terminal;
    if (!Evaluate(
            solver, joints,
            [&](std::size_t index, const SE3d &pose) {
                if (index < static_cast<std::size_t>(dof)) {
                    jacobian.col(index).head<3>() = pose.GetTranslation();
                    jacobian.col(index).tail<3>() =
                        pose.GetRotation() * nodes[index].axis;
                }
            },
            [](std::size_t) {}, terminal))
        return false;
    const Eigen::Vector3d end = terminal.GetTranslation();
    for (int index = 0; index < dof; ++index) {
        const auto type = nodes[static_cast<std::size_t>(index)].joint_type;
        const Eigen::Vector3d axis = jacobian.col(index).tail<3>();
        if (type == JointType::REVOLUTE || type == JointType::CONTINUOUS) {
            const Eigen::Vector3d origin = jacobian.col(index).head<3>();
            jacobian.col(index).head<3>() = axis.cross(end - origin);
        } else {
            jacobian.col(index).setZero();
            if (type == JointType::PRISMATIC)
                jacobian.col(index).head<3>() = axis;
        }
    }
    return true;
}
} // namespace holistic_motion::robotics::detail
