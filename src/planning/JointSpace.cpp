#include "holistic_motion/planning/JointSpace.h"

#include <set>
#include <stdexcept>
#include <utility>

#include "../utility/AnglePhase.h"
#include "JointSpaceTopology.h"

namespace holistic_motion::robotics::planning {

struct JointSpace::Data {
    Eigen::VectorXd lower_limits, upper_limits, weights;
    std::vector<bool> continuous;
    std::vector<std::string> joint_names;
};

JointSpace::JointSpace(Eigen::VectorXd lower_limits, Eigen::VectorXd upper_limits,
                       std::vector<std::string> joint_names) {
    auto data = std::make_shared<Data>();
    data->lower_limits = std::move(lower_limits);
    data->upper_limits = std::move(upper_limits);
    data->joint_names = std::move(joint_names);
    if (data->lower_limits.size() == 0 ||
        data->lower_limits.size() != data->upper_limits.size() ||
        !data->lower_limits.allFinite() || !data->upper_limits.allFinite() ||
        (data->lower_limits.array() >= data->upper_limits.array()).any())
        throw std::invalid_argument(
            "joint limits must be finite, ordered, and non-empty");
    const Eigen::VectorXd ranges = data->upper_limits - data->lower_limits;
    data->weights = detail::DefaultJointWeights(ranges);
    if (!ranges.allFinite() || !data->weights.allFinite() ||
        (data->weights.array() <= 0.0).any())
        throw std::invalid_argument(
            "joint limit ranges must produce finite positive default weights");
    if (!data->joint_names.empty()) {
        if (data->joint_names.size() !=
            static_cast<std::size_t>(data->lower_limits.size()))
            throw std::invalid_argument(
                "joint names must match the planning dimension");
        std::set<std::string> names;
        for (const auto &name : data->joint_names)
            if (name.empty() || !names.insert(name).second)
                throw std::invalid_argument("joint names must be nonempty and unique");
    }
    data->continuous.assign(static_cast<std::size_t>(data->lower_limits.size()), false);
    data_ = std::move(data);
}

Eigen::Index JointSpace::GetDOF() const noexcept { return data_->lower_limits.size(); }
const Eigen::VectorXd &JointSpace::GetLowerLimits() const noexcept {
    return data_->lower_limits;
}
const Eigen::VectorXd &JointSpace::GetUpperLimits() const noexcept {
    return data_->upper_limits;
}
const Eigen::VectorXd &JointSpace::GetJointWeights() const noexcept {
    return data_->weights;
}
const std::vector<std::string> &JointSpace::GetJointNames() const noexcept {
    return data_->joint_names;
}
const std::vector<bool> &JointSpace::GetContinuousMask() const noexcept {
    return data_->continuous;
}

std::vector<std::size_t> JointSpace::GetContinuousJoints() const {
    std::vector<std::size_t> indices;
    for (std::size_t index = 0; index < data_->continuous.size(); ++index)
        if (data_->continuous[index])
            indices.push_back(index);
    return indices;
}

JointSpace JointSpace::WithJointWeights(const Eigen::VectorXd &weights) const {
    if (weights.size() != GetDOF() || !weights.allFinite() ||
        (weights.array() <= 0.0).any())
        throw std::invalid_argument(
            "joint weights must be finite, positive, and dimensionally compatible");
    auto data = std::make_shared<Data>(*data_);
    data->weights = weights;
    return JointSpace(std::move(data));
}

JointSpace
JointSpace::WithContinuousJoints(const std::vector<std::size_t> &indices) const {
    std::vector<bool> continuous(static_cast<std::size_t>(GetDOF()), false);
    for (std::size_t index : indices) {
        if (index >= continuous.size())
            throw std::out_of_range("continuous joint index");
        const auto axis = static_cast<Eigen::Index>(index);
        if (data_->upper_limits[axis] - data_->lower_limits[axis] <
            2.0 * detail::kPi - 1e-9)
            throw std::invalid_argument(
                "continuous joints require a range of at least 2*pi");
        if (!detail::HasRepresentablePeriod(data_->lower_limits[axis],
                                            data_->upper_limits[axis]))
            throw std::invalid_argument(
                "continuous joint limits cannot represent a full revolution");
        continuous[index] = true;
    }
    auto data = std::make_shared<Data>(*data_);
    data->continuous = std::move(continuous);
    return JointSpace(std::move(data));
}

std::vector<Eigen::VectorXd>
JointSpace::UnwrapPath(const std::vector<Eigen::VectorXd> &path) const {
    if (path.empty())
        throw std::invalid_argument("path must contain at least one state");
    std::vector<Eigen::VectorXd> result;
    result.reserve(path.size());
    for (std::size_t row = 0; row < path.size(); ++row) {
        const auto &state = path[row];
        if (state.size() != GetDOF() || !state.allFinite() ||
            (state.array() < data_->lower_limits.array()).any() ||
            (state.array() > data_->upper_limits.array()).any())
            throw std::invalid_argument("path contains an invalid state");
        Eigen::VectorXd lifted = state;
        if (row != 0) {
            for (Eigen::Index axis = 0; axis < GetDOF(); ++axis) {
                if (!data_->continuous[static_cast<std::size_t>(axis)])
                    continue;
                const double delta =
                    detail::WrappedDifference(state[axis] - path[row - 1][axis]);
                const double previous = result.back()[axis];
                lifted[axis] = previous + delta;
                if (!std::isfinite(lifted[axis]) ||
                    (delta != 0.0 && lifted[axis] == previous))
                    throw std::overflow_error(
                        "periodic path step cannot be represented");
                if (lifted[axis] != state[axis] &&
                    !::holistic_motion::robotics::detail::SameAnglePhase(lifted[axis],
                                                                         state[axis]))
                    throw std::overflow_error(
                        "periodic path step cannot preserve its source phase");
            }
        }
        result.push_back(std::move(lifted));
    }
    return result;
}

} // namespace holistic_motion::robotics::planning
