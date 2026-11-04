#pragma once

#include "holistic_motion/robot/JointPositionMap.h"

#include <map>
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace holistic_motion::robotics::collision::detail {

inline void ValidateConfiguration(const pinocchio::Model &model, const Eigen::VectorXd &q) {
    if (q.size() != model.nq)
        throw std::invalid_argument("configuration size must equal model.nq (" +
                                    std::to_string(model.nq) + ")");
    if (!q.allFinite())
        throw std::invalid_argument("configuration must contain finite values");
    if (model.nq != model.nv && !pinocchio::isNormalized(model, q, 1e-8))
        throw std::invalid_argument("configuration must satisfy the model manifold (normalized "
                                    "continuous pairs/quaternions)");
}

// Shared full-model coordinates for mesh and sphere collision queries.
inline std::vector<JointCoordinate> JointCoordinates(const pinocchio::Model &model) {
    std::vector<JointCoordinate> result;
    result.reserve(model.joints.size());
    for (std::size_t id = 1; id < model.joints.size(); ++id) {
        result.push_back({model.names[id], model.idx_qs[id], model.nqs[id],
                          model.idx_vs[id], model.nvs[id]});
    }
    return result;
}

inline JointPositionMap PositionMap(const pinocchio::Model &model,
                                    const std::vector<std::string> &names) {
    std::vector<JointCoordinate> selected;
    selected.reserve(names.size());
    for (const auto &name : names) {
        const auto id = model.getJointId(name);
        if (id == 0 || id >= model.joints.size())
            throw std::invalid_argument("unknown Pinocchio joint: " + name);
        selected.push_back({name, model.idx_qs[id], model.nqs[id],
                            model.idx_vs[id], model.nvs[id]});
    }
    return JointPositionMap(model.nq, model.nv, std::move(selected));
}

inline Eigen::VectorXd JointLimits(const pinocchio::Model &model,
                                   const std::vector<std::string> &names, bool upper) {
    const auto mapping = PositionMap(model, names);
    Eigen::VectorXd result(static_cast<Eigen::Index>(names.size()));
    for (std::size_t i = 0; i < names.size(); ++i) {
        const auto &joint = mapping.GetJoints()[i];
        // Continuous angle limits are not limits on its cosine/sine pair.
        result[static_cast<Eigen::Index>(i)] =
            joint.nq == 2 ? (upper ? std::acos(-1.0) : -std::acos(-1.0))
                          : (upper ? model.upperPositionLimit[joint.idx_q]
                                   : model.lowerPositionLimit[joint.idx_q]);
    }
    return result;
}

inline Eigen::VectorXd
ConfigurationFromJointPositions(const pinocchio::Model &model,
                                const std::map<std::string, double> &positions) {
    std::vector<std::string> names;
    Eigen::VectorXd values(static_cast<Eigen::Index>(positions.size()));
    names.reserve(positions.size());
    Eigen::Index index = 0;
    for (const auto &entry : positions) {
        names.push_back(entry.first);
        values[index++] = entry.second;
    }
    return PositionMap(model, names)
        .ConfigurationWithJointPositions(pinocchio::neutral(model), values);
}

inline Eigen::VectorXd ConfigurationWithJointPositions(const pinocchio::Model &model,
                                                       const Eigen::VectorXd &reference,
                                                       const std::vector<std::string> &names,
                                                       const Eigen::VectorXd &positions) {
    auto result = PositionMap(model, names).ConfigurationWithJointPositions(reference, positions);
    ValidateConfiguration(model, result);
    return result;
}

} // namespace holistic_motion::robotics::collision::detail
