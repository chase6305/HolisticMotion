#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "holistic_motion/kinematics/Types.h"

namespace holistic_motion::robotics {

struct Inertial {
    bool present{false};
    double mass{0.0};
    Eigen::Matrix3d inertia{Eigen::Matrix3d::Zero()};
    SE3d origin{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
};

enum class GeometryType { MESH, BOX, CYLINDER, SPHERE };

struct VisualGeometry {
    std::string name;
    GeometryType type{GeometryType::MESH};
    std::string mesh_path;
    std::string material_name;
    std::string texture_path;
    Eigen::Vector4d color{Eigen::Vector4d::Zero()};
    bool has_color{false};
    Eigen::Vector3d scale{Eigen::Vector3d::Ones()};
    Eigen::Vector3d size{Eigen::Vector3d::Zero()};
    double radius{0.0};
    double length{0.0};
    SE3d origin{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
};

class Link {
public:
    explicit Link(std::string link_name) : name(std::move(link_name)) {}

    std::string name;
    Inertial inertial;
    std::vector<VisualGeometry> visuals;
    std::string parent_joint;
    std::vector<std::string> child_joints;
};

}  // namespace holistic_motion::robotics
