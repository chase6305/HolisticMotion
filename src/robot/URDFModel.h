#pragma once

#include <urdf_parser/urdf_parser.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace holistic_motion::robotics::urdf_internal {

// Parse once for model construction. Normalize scalar directions before either
// native chain conversion or Pinocchio's joint classification sees the axes.
// No backend may interpret axis magnitude as a scale on joint coordinates.
inline urdf::ModelInterfaceSharedPtr
LoadModel(const std::string &path, std::string *xml_snapshot = nullptr) {
    std::ifstream input(path);
    if (!input)
        return {};
    std::string xml;
    try {
        xml.assign(std::istreambuf_iterator<char>(input),
                   std::istreambuf_iterator<char>());
    } catch (const std::ios_base::failure &) {
        return {};
    }
    if (input.bad())
        return {};
    auto model = urdf::parseURDF(xml);
    if (!model || !model->getRoot())
        return {};
    for (const auto &entry : model->joints_) {
        auto &joint = *entry.second;
        if (joint.type != urdf::Joint::REVOLUTE &&
            joint.type != urdf::Joint::CONTINUOUS &&
            joint.type != urdf::Joint::PRISMATIC)
            continue;
        if (joint.type != urdf::Joint::CONTINUOUS && joint.limits &&
            (std::isnan(joint.limits->lower) ||
             std::isnan(joint.limits->upper) ||
             joint.limits->lower > joint.limits->upper))
            throw std::invalid_argument(
                "URDF joint position limits must be ordered and not NaN: " +
                joint.name);
        const auto &axis = joint.axis;
        const double scale =
            std::max({std::abs(axis.x), std::abs(axis.y), std::abs(axis.z)});
        if (!std::isfinite(axis.x) || !std::isfinite(axis.y) ||
            !std::isfinite(axis.z) || !(scale > 0.0))
            throw std::invalid_argument(
                "URDF joint axis must be finite and nonzero: " + joint.name);
        const double x = axis.x / scale, y = axis.y / scale, z = axis.z / scale;
        Eigen::Vector3d unit(x, y, z);
        unit.normalize();
        // Preserve unit-coordinate semantics in native screw motion too:
        // a norm of 1 +/- one ulp changes a large angle's phase completely.
        // Adjust only the largest component, where a one-ulp correction has
        // the greatest effect, until Eigen also measures exactly one.
        Eigen::Index largest;
        unit.cwiseAbs().maxCoeff(&largest);
        for (int correction = 0; correction < 16 && unit.norm() != 1.0;
             ++correction) {
            const double toward = unit.norm() < 1.0
                                      ? std::copysign(INFINITY, unit[largest])
                                      : 0.0;
            unit[largest] = std::nextafter(unit[largest], toward);
        }
        if (unit.norm() != 1.0)
            throw std::invalid_argument("Cannot normalize URDF joint axis: " +
                                        joint.name);
        joint.axis = urdf::Vector3(unit.x(), unit.y(), unit.z());
    }
    if (xml_snapshot)
        *xml_snapshot = std::move(xml);
    return model;
}

} // namespace holistic_motion::robotics::urdf_internal
