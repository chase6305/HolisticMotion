#include "holistic_motion/robot/Robot.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

#include "holistic_motion/kinematics/fep/FEPKinematics.h"
#include "holistic_motion/kinematics/srs/SRSKinematics.h"

namespace holistic_motion::robotics {
namespace {

bool IsSupportedJoint(const Joint &joint) {
    return joint.mimic_joint.empty() &&
           (joint.joint_type == JointType::FIXED ||
            joint.joint_type == JointType::REVOLUTE ||
            joint.joint_type == JointType::CONTINUOUS ||
            joint.joint_type == JointType::PRISMATIC);
}

std::vector<std::shared_ptr<Joint>> SupportedChain(const RobotModel &model,
                                                   const std::string &base,
                                                   const std::string &tip) {
    auto path = model.GetChainJoints(base, tip);
    if (!std::all_of(path.begin(), path.end(), [](const auto &joint) {
            return IsSupportedJoint(*joint);
        }))
        return {};
    return path;
}

std::vector<std::shared_ptr<Joint>> DefaultChain(const RobotModel &model) {
    // Loading publishes parent joints before descendants. Select the default
    // chain from the fresh model without rebuilding its adjacency lists.
    std::unordered_map<std::string_view, std::size_t> active_counts;
    active_counts.reserve(model.GetLinks().size());
    active_counts.emplace(model.GetRootLinkName(), 0);
    for (const auto &joint : model.GetJoints()) {
        const auto parent = active_counts.find(joint->parent_link);
        if (parent != active_counts.end() && IsSupportedJoint(*joint))
            active_counts.emplace(joint->child_link,
                                  parent->second +
                                      (joint->joint_type != JointType::FIXED));
    }
    const Link *best_tip = nullptr;
    std::size_t best_count = 0;
    for (const auto &link : model.GetLinks()) {
        const auto count = active_counts.find(link->name);
        if (link->child_joints.empty() && count != active_counts.end() &&
            count->second > best_count) {
            best_count = count->second;
            best_tip = link.get();
        }
    }
    return best_tip
               ? model.GetChainJoints(model.GetRootLinkName(), best_tip->name)
               : std::vector<std::shared_ptr<Joint>>{};
}

std::vector<JointNode>
ChainNodes(const std::vector<std::shared_ptr<Joint>> &path) {
    std::vector<JointNode> nodes;
    const auto active_joints =
        std::count_if(path.begin(), path.end(), [](const auto &joint) {
            return joint->joint_type != JointType::FIXED;
        });
    if (active_joints)
        nodes.reserve(active_joints + 1);
    SE3d fixed_transform;
    for (const auto &joint : path) {
        if (joint->joint_type == JointType::FIXED) {
            fixed_transform = fixed_transform * joint->origin_pose;
            continue;
        }
        const JointType type = joint->joint_type == JointType::PRISMATIC
                                   ? JointType::PRISMATIC
                                   : JointType::REVOLUTE;
        nodes.emplace_back(fixed_transform * joint->origin_pose, joint->axis,
                           type, joint->limit.lower_limit,
                           joint->limit.upper_limit);
        fixed_transform = SE3d();
    }
    if (!nodes.empty()) {
        JointNode terminal;
        terminal.origin_pose = fixed_transform;
        terminal.joint_type = JointType::FIXED;
        nodes.push_back(terminal);
    }
    return nodes;
}

std::shared_ptr<NumericalKinematics>
MakeKinematics(const std::vector<std::shared_ptr<Joint>> &path) {
    const auto nodes = ChainNodes(path);
    if (nodes.empty())
        return nullptr;

    if (nodes.size() == 8 && std::all_of(nodes.begin(), nodes.begin() + 7,
                                         [](const JointNode &node) {
                                             return node.joint_type ==
                                                    JointType::REVOLUTE;
                                         })) {
        return std::make_shared<SRSKinematics>(nodes);
    }
    return std::make_shared<NumericalKinematics>(nodes);
}

} // namespace

Robot::Robot(const std::string &urdf_path, bool load_visuals) {
    if (!LoadURDF(urdf_path, load_visuals))
        throw std::runtime_error("Unable to load URDF: " + urdf_path);
}

std::shared_ptr<Robot> Robot::Create(const std::string &urdf_path,
                                     bool load_visuals) {
    return std::make_shared<Robot>(urdf_path, load_visuals);
}

bool Robot::LoadURDF(const std::string &urdf_path, bool load_visuals) {
    RobotModel model;
    if (!model.LoadURDF(urdf_path, load_visuals))
        return false;
    auto path = DefaultChain(model);
    auto solver = MakeKinematics(path);
    path.erase(std::remove_if(path.begin(), path.end(),
                              [](const auto &joint) {
                                  return joint->joint_type == JointType::FIXED;
                              }),
               path.end());
    model_ = std::move(model);
    actuated_joints_ = std::move(path);
    kinematics_ = std::move(solver);
    return true;
}

bool Robot::LoadVisuals() { return model_.LoadVisuals(); }
std::shared_ptr<Link> Robot::GetLink(const std::string &name) const noexcept {
    return model_.GetLink(name);
}
std::shared_ptr<Joint> Robot::GetJoint(const std::string &name) const noexcept {
    return model_.GetJoint(name);
}
std::vector<JointCoordinate> Robot::GetJointCoordinates() const {
    return model_.GetJointCoordinates();
}
JointPositionMap
Robot::CreateJointPositionMap(const std::vector<std::string> &names) const {
    return model_.CreateJointPositionMap(names);
}
std::vector<std::string>
Robot::GetChainJointNames(const std::string &base,
                          const std::string &tip) const {
    std::vector<std::string> names;
    for (const auto &joint : SupportedChain(model_, base, tip))
        if (joint->joint_type != JointType::FIXED)
            names.push_back(joint->name);
    return names;
}
std::shared_ptr<NumericalKinematics>
Robot::CreateKinematics(const std::string &base, const std::string &tip) const {
    return MakeKinematics(SupportedChain(model_, base, tip));
}
std::shared_ptr<NumericalKinematics>
Robot::CreateFEPKinematics(const std::string &base,
                           const std::string &tip) const {
    const auto nodes = ChainNodes(SupportedChain(model_, base, tip));
    if (nodes.empty())
        return nullptr;
    auto solver = std::make_shared<FEPKinematics>(nodes);
    return solver->IsCompatible() ? solver : nullptr;
}

} // namespace holistic_motion::robotics
