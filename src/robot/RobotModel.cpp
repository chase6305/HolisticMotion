#include "holistic_motion/robot/RobotModel.h"
#include "../utility/BuildIdentity.h"
#include "URDFModel.h"

#include <algorithm>
#include <filesystem>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace holistic_motion::robotics {
namespace {
Eigen::Vector3d ToEigen(const urdf::Vector3 &value) {
    return {value.x, value.y, value.z};
}

SE3d ToSE3(const urdf::Pose &pose) {
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

std::shared_ptr<Joint> ConvertJoint(const urdf::Joint &source) {
    constexpr double kPi = 3.14159265358979323846;
    // A continuous joint may have velocity/effort limits without position
    // limits. Use the solver's canonical full-turn interval in that case too.
    const bool bounded =
        source.type != urdf::Joint::CONTINUOUS && source.limits;
    const double lower = bounded ? source.limits->lower : -kPi;
    const double upper = bounded ? source.limits->upper : kPi;
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

void ParseVisual(const urdf::Visual &source, VisualGeometry &result) {
    result.name = source.name;
    result.origin = ToSE3(source.origin);
    result.material_name = source.material_name;
    if (source.material) {
        result.texture_path = source.material->texture_filename;
        result.color = {source.material->color.r, source.material->color.g,
                        source.material->color.b, source.material->color.a};
        result.has_color = true;
    }
    if (!source.geometry)
        return;
    switch (source.geometry->type) {
    case urdf::Geometry::MESH: {
        const auto &mesh = static_cast<const urdf::Mesh &>(*source.geometry);
        result.type = GeometryType::MESH;
        result.mesh_path = mesh.filename;
        result.scale = ToEigen(mesh.scale);
        break;
    }
    case urdf::Geometry::BOX: {
        const auto &box = static_cast<const urdf::Box &>(*source.geometry);
        result.type = GeometryType::BOX;
        result.size = ToEigen(box.dim);
        break;
    }
    case urdf::Geometry::CYLINDER: {
        const auto &cylinder =
            static_cast<const urdf::Cylinder &>(*source.geometry);
        result.type = GeometryType::CYLINDER;
        result.radius = cylinder.radius;
        result.length = cylinder.length;
        break;
    }
    case urdf::Geometry::SPHERE: {
        const auto &sphere =
            static_cast<const urdf::Sphere &>(*source.geometry);
        result.type = GeometryType::SPHERE;
        result.radius = sphere.radius;
        break;
    }
    }
}

std::vector<VisualGeometry> ParseVisuals(const urdf::Link &source) {
    std::vector<VisualGeometry> result;
    result.reserve(source.visual_array.size());
    for (const auto &visual : source.visual_array) {
        if (!visual)
            continue;
        VisualGeometry geometry;
        ParseVisual(*visual, geometry);
        result.push_back(std::move(geometry));
    }
    return result;
}

template <class Node>
std::shared_ptr<Node>
FindIndexed(const std::vector<std::shared_ptr<Node>> &nodes,
            std::string_view name,
            std::unordered_map<std::string_view, std::shared_ptr<Node>> &index,
            std::size_t &scanned) {
    const auto found = index.find(name);
    if (found != index.end())
        return found->second;
    // Scan each model entry at most once. A chain near the beginning of a
    // large branched model need not index all unrelated branches.
    while (scanned < nodes.size()) {
        const auto &node = nodes[scanned++];
        if (!node)
            continue;
        index.emplace(node->name, node);
        if (node->name == name)
            return node;
    }
    return nullptr;
}

} // namespace

RobotModel::RobotModel(const std::string &urdf_path, bool load_visuals) {
    if (!LoadURDF(urdf_path, load_visuals))
        throw std::runtime_error("Unable to load URDF: " + urdf_path);
}

bool RobotModel::LoadURDF(const std::string &urdf_path, bool load_visuals) {
    std::error_code path_error;
    const auto resolved_path = std::filesystem::absolute(urdf_path, path_error);
    if (path_error)
        return false;
    const auto model = urdf_internal::LoadModel(resolved_path.string());
    if (!model || !model->getRoot())
        return false;

    RobotModel loaded;
    loaded.name_ = model->getName();
    loaded.urdf_path_ = urdf_path;
    loaded.root_link_name_ = model->getRoot()->name;

    std::function<void(const urdf::LinkConstSharedPtr &)> visit =
        [&](const urdf::LinkConstSharedPtr &source) {
            auto link = std::make_shared<Link>(source->name);
            if (source->inertial) {
                link->inertial.present = true;
                link->inertial.mass = source->inertial->mass;
                link->inertial.origin = ToSE3(source->inertial->origin);
                link->inertial.inertia << source->inertial->ixx,
                    source->inertial->ixy, source->inertial->ixz,
                    source->inertial->ixy, source->inertial->iyy,
                    source->inertial->iyz, source->inertial->ixz,
                    source->inertial->iyz, source->inertial->izz;
            }
            if (source->parent_joint) {
                link->parent_joint = source->parent_joint->name;
            }
            for (const auto &child_joint : source->child_joints) {
                link->child_joints.push_back(child_joint->name);
                auto joint = ConvertJoint(*child_joint);
                if (joint->mimic_joint.empty() &&
                    (joint->joint_type == JointType::REVOLUTE ||
                     joint->joint_type == JointType::CONTINUOUS ||
                     joint->joint_type == JointType::PRISMATIC)) {
                    loaded.all_actuated_joints_.push_back(joint);
                }
                loaded.joints_.push_back(std::move(joint));
            }
            if (load_visuals) {
                link->visuals = ParseVisuals(*source);
            } else if (!source->visual_array.empty()) {
                loaded.pending_visuals_.emplace_back(loaded.links_.size(),
                                                     ParseVisuals(*source));
            }
            loaded.links_.push_back(std::move(link));
            for (const auto &child : source->child_links)
                visit(child);
        };
    visit(model->getRoot());

    loaded.visuals_loaded_ = load_visuals;
    *this = std::move(loaded);
    return true;
}

bool RobotModel::LoadVisuals() {
    if (visuals_loaded_)
        return true;
    if (links_.empty())
        return false;
    // All metadata belongs to the same model parse. Publishing uses
    // noexcept vector moves; no second file parse can mix model generations.
    for (const auto &link : links_)
        link->visuals.clear();
    for (auto &entry : pending_visuals_)
        links_[entry.first]->visuals = std::move(entry.second);
    decltype(pending_visuals_)().swap(pending_visuals_);
    visuals_loaded_ = true;
    return true;
}

std::vector<std::shared_ptr<Joint>>
RobotModel::GetChainJoints(const std::string &base_link,
                           const std::string &tip_link) const {
    if (!GetLink(base_link))
        return {};
    auto current = GetLink(tip_link);
    if (!current)
        return {};

    // Public C++ model fields are mutable. Build a temporary index only when
    // repeated scans on a long path justify it; each call sees current names.
    std::unordered_map<std::string_view, std::shared_ptr<Link>> links_by_name;
    std::unordered_map<std::string_view, std::shared_ptr<Joint>> joints_by_name;
    std::size_t scanned_links = 0, scanned_joints = 0;
    bool indexed = false;
    std::vector<std::shared_ptr<Joint>> path;
    while (current && current->name != base_link) {
        // A mutated parent relation may create a cycle. A tree path cannot
        // visit more links than the model contains.
        if (path.size() >= links_.size())
            return {};
        if (!indexed && path.size() == 8 && links_.size() > 64) {
            links_by_name.reserve(std::min<std::size_t>(links_.size(), 128));
            joints_by_name.reserve(std::min<std::size_t>(joints_.size(), 128));
            indexed = true;
        }
        if (current->parent_joint.empty())
            return {};
        const auto joint = indexed ? FindIndexed(joints_, current->parent_joint,
                                                 joints_by_name, scanned_joints)
                                   : GetJoint(current->parent_joint);
        if (!joint)
            return {};
        path.push_back(joint);
        current = indexed ? FindIndexed(links_, joint->parent_link,
                                        links_by_name, scanned_links)
                          : GetLink(joint->parent_link);
    }
    if (!current || current->name != base_link)
        return {};
    std::reverse(path.begin(), path.end());

    return path;
}

std::vector<JointCoordinate> RobotModel::GetJointCoordinates() const {
    std::vector<JointCoordinate> result;
    result.reserve(all_actuated_joints_.size());
    int index = 0;
    for (const auto &joint : all_actuated_joints_) {
        result.push_back({joint->name, index, 1, index, 1});
        ++index;
    }
    return result;
}

JointPositionMap RobotModel::CreateJointPositionMap(
    const std::vector<std::string> &names) const {
    std::unordered_map<std::string_view, int> by_name;
    by_name.reserve(all_actuated_joints_.size());
    for (std::size_t index = 0; index < all_actuated_joints_.size(); ++index) {
        const auto &name = all_actuated_joints_[index]->name;
        if (name.empty() ||
            !by_name.emplace(name, static_cast<int>(index)).second)
            throw std::invalid_argument(
                "model joint names must be nonempty and unique: " + name);
    }
    std::vector<JointCoordinate> selected;
    selected.reserve(names.size());
    for (const auto &name : names) {
        const auto found = by_name.find(name);
        if (found == by_name.end())
            throw std::invalid_argument("unknown scalar model joint: " + name);
        selected.push_back({name, found->second, 1, found->second, 1});
    }
    return JointPositionMap(GetModelDoF(), GetModelDoF(), std::move(selected));
}

std::shared_ptr<Link>
RobotModel::GetLink(const std::string &name) const noexcept {
    const auto found =
        std::find_if(links_.begin(), links_.end(), [&](const auto &link) {
            return link && link->name == name;
        });
    return found == links_.end() ? nullptr : *found;
}

std::shared_ptr<Joint>
RobotModel::GetJoint(const std::string &name) const noexcept {
    const auto found =
        std::find_if(joints_.begin(), joints_.end(), [&](const auto &joint) {
            return joint && joint->name == name;
        });
    return found == joints_.end() ? nullptr : *found;
}

} // namespace holistic_motion::robotics

namespace holistic_motion::utility::detail {
const char *ModelBuildIdentity() noexcept {
    return HOLISTICMOTION_NATIVE_BUILD_ID;
}
} // namespace holistic_motion::utility::detail
