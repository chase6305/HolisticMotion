#include "holistic_motion/robot/Robot.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_map>

#include <urdf_parser/urdf_parser.h>

#include "holistic_motion/kinematics/srs/SRSKinematics.h"
#include "holistic_motion/kinematics/fep/FEPKinematics.h"

namespace holistic_motion::robotics {
namespace {

Eigen::Vector3d ToEigen(const urdf::Vector3& value) {
    return {value.x, value.y, value.z};
}

SE3d ToSE3(const urdf::Pose& pose) {
    Eigen::Quaterniond rotation(pose.rotation.w, pose.rotation.x,
                                pose.rotation.y, pose.rotation.z);
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform.topLeftCorner<3, 3>() = rotation.normalized().toRotationMatrix();
    transform.topRightCorner<3, 1>() = ToEigen(pose.position);
    return SE3d(transform);
}

JointType ToJointType(int type) {
    switch (type) {
        case urdf::Joint::REVOLUTE:
            return JointType::REVOLUTE;
        case urdf::Joint::CONTINUOUS:
            return JointType::CONTINUOUS;
        case urdf::Joint::PRISMATIC:
            return JointType::PRISMATIC;
        case urdf::Joint::FLOATING:
            return JointType::FLOATING;
        case urdf::Joint::PLANAR:
            return JointType::PLANAR;
        case urdf::Joint::FIXED:
            return JointType::FIXED;
        default:
            return JointType::UNKNOWN;
    }
}

std::shared_ptr<Joint> ConvertJoint(const urdf::Joint& source) {
    constexpr double kPi = 3.14159265358979323846;
    const double lower = source.limits ? source.limits->lower : -kPi;
    const double upper = source.limits ? source.limits->upper : kPi;
    auto joint = std::make_shared<Joint>(
            source.name, source.parent_link_name, source.child_link_name,
            ToSE3(source.parent_to_joint_origin_transform),
            ToJointType(source.type), ToEigen(source.axis), lower, upper,
            source.limits ? source.limits->velocity : 0.0, 0.0, 0.0,
            source.limits ? source.limits->effort : 0.0);
    if (source.mimic) {
        joint->mimic_joint = source.mimic->joint_name;
        joint->mimic_multiplier = source.mimic->multiplier;
        joint->mimic_offset = source.mimic->offset;
    }
    return joint;
}

void ParseVisual(const urdf::Visual& source, VisualGeometry& result) {
    result.name = source.name;
    result.origin = ToSE3(source.origin);
    result.material_name = source.material_name;
    if (source.material) {
        result.texture_path = source.material->texture_filename;
        result.color = {source.material->color.r, source.material->color.g,
                        source.material->color.b, source.material->color.a};
        result.has_color = true;
    }
    if (!source.geometry) return;
    switch (source.geometry->type) {
        case urdf::Geometry::MESH: {
            const auto& mesh =
                    static_cast<const urdf::Mesh&>(*source.geometry);
            result.type = GeometryType::MESH;
            result.mesh_path = mesh.filename;
            result.scale = ToEigen(mesh.scale);
            break;
        }
        case urdf::Geometry::BOX: {
            const auto& box = static_cast<const urdf::Box&>(*source.geometry);
            result.type = GeometryType::BOX;
            result.size = ToEigen(box.dim);
            break;
        }
        case urdf::Geometry::CYLINDER: {
            const auto& cylinder =
                    static_cast<const urdf::Cylinder&>(*source.geometry);
            result.type = GeometryType::CYLINDER;
            result.radius = cylinder.radius;
            result.length = cylinder.length;
            break;
        }
        case urdf::Geometry::SPHERE: {
            const auto& sphere =
                    static_cast<const urdf::Sphere&>(*source.geometry);
            result.type = GeometryType::SPHERE;
            result.radius = sphere.radius;
            break;
        }
    }
}

void PopulateVisuals(const urdf::Link& source, Link& target) {
    target.visuals.clear();
    target.visuals.reserve(source.visual_array.size());
    for (const auto& visual : source.visual_array) {
        if (!visual) continue;
        VisualGeometry geometry;
        ParseVisual(*visual, geometry);
        target.visuals.push_back(std::move(geometry));
    }
}

using UrdfJointPath = std::vector<urdf::JointConstSharedPtr>;

UrdfJointPath LongestActuatedPath(const urdf::LinkConstSharedPtr& root) {
    UrdfJointPath best;
    UrdfJointPath current;
    std::function<void(const urdf::LinkConstSharedPtr&)> visit =
            [&](const urdf::LinkConstSharedPtr& link) {
                if (!link) return;
                if (link->child_links.empty()) {
                    const auto supported = [](const auto& joint) {
                        return joint &&
                               (joint->type == urdf::Joint::FIXED ||
                                (!joint->mimic &&
                                 (joint->type == urdf::Joint::REVOLUTE ||
                                  joint->type == urdf::Joint::CONTINUOUS ||
                                  joint->type == urdf::Joint::PRISMATIC)));
                    };
                    if (!std::all_of(current.begin(), current.end(), supported)) {
                        return;
                    }
                    const auto active = [](const auto& joint) {
                        return joint && joint->type != urdf::Joint::FIXED;
                    };
                    if (std::count_if(current.begin(), current.end(), active) >
                        std::count_if(best.begin(), best.end(), active)) {
                        best = current;
                    }
                    return;
                }
                for (const auto& child : link->child_links) {
                    current.push_back(child->parent_joint);
                    visit(child);
                    current.pop_back();
                }
            };
    visit(root);
    return best;
}

}  // namespace

Robot::Robot(const std::string& urdf_path, bool load_visuals) {
    if (!LoadURDF(urdf_path, load_visuals)) {
        throw std::runtime_error("Unable to load URDF: " + urdf_path);
    }
}

std::shared_ptr<Robot> Robot::Create(const std::string& urdf_path,
                                     bool load_visuals) {
    return std::make_shared<Robot>(urdf_path, load_visuals);
}

bool Robot::LoadURDF(const std::string& urdf_path, bool load_visuals) {
    const auto model = urdf::parseURDFFile(urdf_path);
    if (!model || !model->getRoot()) return false;

    name_ = model->getName();
    urdf_path_ = urdf_path;
    root_link_name_ = model->getRoot()->name;
    joints_.clear();
    actuated_joints_.clear();
    all_actuated_joints_.clear();
    links_.clear();
    joint_nodes_.clear();

    std::unordered_map<std::string, std::shared_ptr<Link>> links_by_name;
    std::unordered_map<std::string, std::shared_ptr<Joint>> joints_by_name;
    std::function<void(const urdf::LinkConstSharedPtr&)> visit =
            [&](const urdf::LinkConstSharedPtr& source) {
                auto link = std::make_shared<Link>(source->name);
                if (source->inertial) {
                    link->inertial.present = true;
                    link->inertial.mass = source->inertial->mass;
                    link->inertial.origin = ToSE3(source->inertial->origin);
                    link->inertial.inertia <<
                            source->inertial->ixx, source->inertial->ixy,
                            source->inertial->ixz, source->inertial->ixy,
                            source->inertial->iyy, source->inertial->iyz,
                            source->inertial->ixz, source->inertial->iyz,
                            source->inertial->izz;
                }
                if (source->parent_joint) {
                    link->parent_joint = source->parent_joint->name;
                }
                for (const auto& child_joint : source->child_joints) {
                    link->child_joints.push_back(child_joint->name);
                    auto joint = ConvertJoint(*child_joint);
                    if (joint->mimic_joint.empty() &&
                        (joint->joint_type == JointType::REVOLUTE ||
                         joint->joint_type == JointType::CONTINUOUS ||
                         joint->joint_type == JointType::PRISMATIC)) {
                        all_actuated_joints_.push_back(joint);
                    }
                    joints_by_name.emplace(joint->name, joint);
                    joints_.push_back(std::move(joint));
                }
                if (load_visuals) PopulateVisuals(*source, *link);
                links_by_name.emplace(link->name, link);
                links_.push_back(std::move(link));
                for (const auto& child : source->child_links) visit(child);
            };
    visit(model->getRoot());

    SE3d pending_fixed_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    for (const auto& source : LongestActuatedPath(model->getRoot())) {
        const SE3d origin = ToSE3(source->parent_to_joint_origin_transform);
        if (source->type == urdf::Joint::FIXED) {
            pending_fixed_transform = pending_fixed_transform * origin;
            continue;
        }
        // LongestActuatedPath only returns chains supported by the numerical
        // solver. Other URDF joint types remain available in joints_/links_.
        const auto converted = joints_by_name.find(source->name);
        if (converted == joints_by_name.end()) return false;
        actuated_joints_.push_back(converted->second);
        const double lower = source->limits ? source->limits->lower : -3.14159265358979323846;
        const double upper = source->limits ? source->limits->upper : 3.14159265358979323846;
        const JointType solver_type = source->type == urdf::Joint::PRISMATIC
                                              ? JointType::PRISMATIC
                                              : JointType::REVOLUTE;
        joint_nodes_.emplace_back(pending_fixed_transform * origin,
                                  ToEigen(source->axis), solver_type, lower,
                                  upper);
        pending_fixed_transform =
                SE3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    }
    if (joint_nodes_.empty()) {
        kinematics_.reset();
    } else {
        JointNode terminal;
        terminal.origin_pose = pending_fixed_transform;
        terminal.joint_type = JointType::FIXED;
        joint_nodes_.push_back(terminal);
        const bool is_seven_revolute =
                joint_nodes_.size() == 8 &&
                std::all_of(joint_nodes_.begin(), joint_nodes_.begin() + 7,
                            [](const JointNode& node) {
                                return node.joint_type == JointType::REVOLUTE;
                            });
        if (is_seven_revolute) {
            kinematics_ = std::make_shared<SRSKinematics>(joint_nodes_);
        } else {
            kinematics_ = std::make_shared<NumericalKinematics>(joint_nodes_);
        }
    }
    visuals_loaded_ = load_visuals;
    return true;
}

bool Robot::LoadVisuals() {
    if (visuals_loaded_) return true;
    const auto model = urdf::parseURDFFile(urdf_path_);
    if (!model) return false;
    std::unordered_map<std::string, std::shared_ptr<Link>> links_by_name;
    for (const auto& link : links_) links_by_name.emplace(link->name, link);
    std::vector<urdf::LinkSharedPtr> parsed_links;
    model->getLinks(parsed_links);
    for (const auto& source : parsed_links) {
        const auto target = links_by_name.find(source->name);
        if (target != links_by_name.end()) {
            PopulateVisuals(*source, *target->second);
        }
    }
    visuals_loaded_ = true;
    return true;
}

std::shared_ptr<NumericalKinematics> Robot::CreateKinematics(
        const std::string& base_link, const std::string& tip_link) const {
    if (!GetLink(base_link)) return nullptr;
    auto current = GetLink(tip_link);
    if (!current) return nullptr;

    std::vector<std::shared_ptr<Joint>> path;
    while (current && current->name != base_link) {
        if (current->parent_joint.empty()) return nullptr;
        auto joint = GetJoint(current->parent_joint);
        if (!joint) return nullptr;
        path.push_back(joint);
        current = GetLink(joint->parent_link);
    }
    if (!current || current->name != base_link) return nullptr;
    std::reverse(path.begin(), path.end());

    std::vector<JointNode> nodes;
    SE3d fixed_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    for (const auto& joint : path) {
        if (!joint->mimic_joint.empty()) return nullptr;
        if (joint->joint_type == JointType::FIXED) {
            fixed_transform = fixed_transform * joint->origin_pose;
            continue;
        }
        if (joint->joint_type != JointType::REVOLUTE &&
            joint->joint_type != JointType::CONTINUOUS &&
            joint->joint_type != JointType::PRISMATIC) {
            return nullptr;
        }
        const JointType type = joint->joint_type == JointType::PRISMATIC
                                       ? JointType::PRISMATIC
                                       : JointType::REVOLUTE;
        nodes.emplace_back(fixed_transform * joint->origin_pose, joint->axis,
                           type, joint->limit.lower_limit,
                           joint->limit.upper_limit);
        fixed_transform =
                SE3d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    }
    if (nodes.empty()) return nullptr;
    JointNode terminal;
    terminal.origin_pose = fixed_transform;
    terminal.joint_type = JointType::FIXED;
    nodes.push_back(terminal);

    if (nodes.size() == 8 &&
        std::all_of(nodes.begin(), nodes.begin() + 7,
                    [](const JointNode& node) {
                        return node.joint_type == JointType::REVOLUTE;
                    })) {
        return std::make_shared<SRSKinematics>(nodes);
    }
    return std::make_shared<NumericalKinematics>(nodes);
}

std::shared_ptr<NumericalKinematics> Robot::CreateFEPKinematics(
        const std::string& base_link, const std::string& tip_link) const {
    const auto generic = CreateKinematics(base_link, tip_link);
    if (!generic) return nullptr;
    const auto nodes = generic->GetJointNode();
    auto solver = std::make_shared<FEPKinematics>(nodes);
    return solver->IsCompatible() ? solver : nullptr;
}

std::shared_ptr<Link> Robot::GetLink(const std::string& name) const noexcept {
    const auto found = std::find_if(
            links_.begin(), links_.end(), [&](const auto& link) {
                return link && link->name == name;
            });
    return found == links_.end() ? nullptr : *found;
}

std::shared_ptr<Joint> Robot::GetJoint(const std::string& name) const noexcept {
    const auto found = std::find_if(
            joints_.begin(), joints_.end(), [&](const auto& joint) {
                return joint && joint->name == name;
            });
    return found == joints_.end() ? nullptr : *found;
}

}  // namespace holistic_motion::robotics
