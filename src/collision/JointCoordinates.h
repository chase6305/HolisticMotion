#pragma once

#include "holistic_motion/robot/JointPositionMap.h"

namespace holistic_motion::robotics::collision::detail {

// Templated to avoid exposing a Pinocchio header in the public coordinate API.
template <class Model>
std::vector<JointCoordinate> JointCoordinates(const Model &model) {
    std::vector<JointCoordinate> result;
    result.reserve(model.joints.size());
    for (std::size_t id = 1; id < model.joints.size(); ++id) {
        result.push_back({model.names[id], model.idx_qs[id], model.nqs[id],
                          model.idx_vs[id], model.nvs[id]});
    }
    return result;
}

template <class Model>
JointPositionMap PositionMap(const Model &model,
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

template <class Model>
Eigen::VectorXd JointLimits(const Model &model,
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

} // namespace holistic_motion::robotics::collision::detail
