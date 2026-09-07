#pragma once

#include "JointSpaceTopology.h"

namespace holistic_motion::robotics::planning::detail {

// One workspace per optimization call. Only the two adjacent edges change
// during a waypoint's line search; the outer edges remain cached until the
// next waypoint is selected, including when the sweep direction changes.
class PathGeometryWorkspace {
public:
    PathGeometryWorkspace(const Eigen::VectorXd& weights,
                          const std::vector<bool>& continuous,
                          double length_weight, double smoothness_weight)
        : weights_(weights),
          continuous_(continuous),
          has_continuous_(std::any_of(continuous.begin(), continuous.end(),
                                      [](bool value) { return value; })),
          length_weight_(length_weight),
          smoothness_weight_(smoothness_weight),
          left_(weights.size()),
          right_(weights.size()),
          previous_(weights.size()),
          next_(weights.size()),
          acceleration_(weights.size()) {}

    void SetWaypoint(const std::vector<Eigen::VectorXd>& path,
                     std::size_t index) {
        DifferenceInto(path[index - 1], path[index], left_);
        DifferenceInto(path[index], path[index + 1], right_);
        has_previous_ = index > 1;
        has_next_ = index + 2 < path.size();
        if (has_previous_)
            DifferenceInto(path[index - 2], path[index - 1], previous_);
        if (has_next_) DifferenceInto(path[index + 1], path[index + 2], next_);
    }

    double LocalObjective() {
        const double length =
            std::sqrt(SquaredNorm(left_)) + std::sqrt(SquaredNorm(right_));
        acceleration_ = right_ - left_;
        double smoothness = SquaredNorm(acceleration_);
        if (has_previous_) {
            acceleration_ = left_ - previous_;
            smoothness += SquaredNorm(acceleration_);
        }
        if (has_next_) {
            acceleration_ = next_ - right_;
            smoothness += SquaredNorm(acceleration_);
        }
        return length_weight_ * length + smoothness_weight_ * smoothness;
    }

    double CandidateObjective(const Eigen::VectorXd& before,
                              const Eigen::VectorXd& candidate,
                              const Eigen::VectorXd& after) {
        DifferenceInto(before, candidate, left_);
        DifferenceInto(candidate, after, right_);
        return LocalObjective();
    }

    void Gradient(Eigen::VectorXd& gradient) {
        gradient.setZero(weights_.size());
        const double left_length = std::sqrt(SquaredNorm(left_));
        const double right_length = std::sqrt(SquaredNorm(right_));
        if (left_length > 1e-15)
            gradient.array() +=
                length_weight_ * weights_.array() * left_.array() / left_length;
        if (right_length > 1e-15)
            gradient.array() -= length_weight_ * weights_.array() *
                                right_.array() / right_length;
        if (has_previous_) {
            acceleration_ = left_ - previous_;
            gradient.array() += 2.0 * smoothness_weight_ * weights_.array() *
                                acceleration_.array();
        }
        acceleration_ = right_ - left_;
        gradient.array() -=
            4.0 * smoothness_weight_ * weights_.array() * acceleration_.array();
        if (has_next_) {
            acceleration_ = next_ - right_;
            gradient.array() += 2.0 * smoothness_weight_ * weights_.array() *
                                acceleration_.array();
        }
    }

private:
    void DifferenceInto(const Eigen::VectorXd& from, const Eigen::VectorXd& to,
                        Eigen::VectorXd& difference) const {
        difference = to - from;
        if (has_continuous_)
            for (Eigen::Index i = 0; i < difference.size(); ++i)
                if (continuous_[static_cast<std::size_t>(i)])
                    difference[i] = WrappedDifference(difference[i]);
    }

    double SquaredNorm(const Eigen::VectorXd& value) const {
        return (weights_.array() * value.array().square()).sum();
    }

    const Eigen::VectorXd& weights_;
    const std::vector<bool>& continuous_;
    const bool has_continuous_;
    const double length_weight_;
    const double smoothness_weight_;
    Eigen::VectorXd left_, right_, previous_, next_, acceleration_;
    bool has_previous_{false};
    bool has_next_{false};
};

}  // namespace holistic_motion::robotics::planning::detail
