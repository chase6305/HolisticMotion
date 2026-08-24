#pragma once
#include <numeric>
#include <set>

#include "holistic_motion/kinematics/Types.h"
#include "holistic_motion/utility/Logging.h"

namespace holistic_motion {
namespace robotics {

class JointFilterManager {
public:
    enum class FilterMode {
        NONE = 0,      // No filtering
        POSITIVE = 1,  // Keep only positive angles
        NEGATIVE = -1  // Keep only negative angles
    };

    /// \brief Constructor that creates filter based on degrees of freedom
    ///
    /// \param dof Robot degrees of freedom
    explicit JointFilterManager(size_t dof = 6) : filters_(dof) {
        InitializeThresholds(dof);
        Reset();
    }

    /// \brief Resize the filter array
    ///
    /// \param new_dof New degrees of freedom
    void Resize(size_t new_dof) {
        filters_.resize(new_dof, FilterMode::NONE);
        angle_thresholds_.resize(new_dof, 0.0);
    }

    /// \brief Get the size of filter array
    size_t Size() const { return filters_.size(); }

    /// \brief Set filter mode for a single joint
    ///
    /// \param joint_index Index of the joint
    /// \param mode Filter mode to set
    void SetJointFilter(size_t joint_index, FilterMode mode) {
        if (joint_index < filters_.size()) {
            filters_[joint_index] = mode;
        }
    }

    /// \brief Set same filter mode for all joints
    ///
    /// \param mode Filter mode to set for all joints
    void SetAllJointFilters(FilterMode mode) {
        std::fill(filters_.begin(), filters_.end(), mode);
    }

    /// \brief Set filter modes for multiple specified joints
    ///
    /// \param settings Vector of joint index and mode pairs
    void SetMultipleJointFilters(
            const std::vector<std::pair<size_t, FilterMode>>& settings) {
        for (const auto& [joint_index, mode] : settings) {
            SetJointFilter(joint_index, mode);
        }
    }

    /// \brief Set filter modes for all joints using a vector
    ///
    /// \param modes Vector containing filter modes for all joints
    void SetJointFilters(const std::vector<FilterMode>& modes) {
        if (modes.size() != filters_.size()) {
            Resize(modes.size());
        }
        std::copy(modes.begin(), modes.end(), filters_.begin());
    }

    /// \brief Set angle thresholds for all joints using a vector
    ///
    /// \param thresholds Vector containing threshold values for all joints
    /// \return true if successful, false if vector size doesn't match
    bool SetJointAngleThresholds(const std::vector<double>& thresholds) {
        if (thresholds.size() != angle_thresholds_.size()) {
            holistic_motion::utility::LogWarning(
                    "Thresholds size ({}) does not match robot DOF ({})",
                    thresholds.size(), angle_thresholds_.size());
            return false;
        }
        std::copy(thresholds.begin(), thresholds.end(),
                  angle_thresholds_.begin());
        return true;
    }

    /// \brief Set angle threshold for a specific joint
    ///
    /// \param joint_index Index of the joint
    /// \param threshold Threshold value for the joint
    /// \return true if successful, false if index out of range
    bool SetJointAngleThreshold(size_t joint_index, double threshold) {
        if (joint_index >= angle_thresholds_.size()) {
            holistic_motion::utility::LogWarning("Joint index {} out of range (max {})",
                                      joint_index,
                                      angle_thresholds_.size() - 1);
            return false;
        }
        angle_thresholds_[joint_index] = threshold;
        return true;
    }

    /// \brief Get angle threshold for a specific joint
    ///
    /// \param joint_index Index of the joint
    /// \return Threshold value for the specified joint, returns 0.0 if index is
    /// out of range
    double GetJointAngleThreshold(size_t joint_index) const {
        if (joint_index >= angle_thresholds_.size()) {
            holistic_motion::utility::LogWarning("Joint index {} out of range (max {})",
                                      joint_index,
                                      angle_thresholds_.size() - 1);
            return 0.0;
        }
        return angle_thresholds_[joint_index];
    }

    /// \brief Get current angle thresholds for all joints
    ///
    /// \return Vector of current threshold values
    const std::vector<double>& GetJointAngleThresholds() const {
        return angle_thresholds_;
    }

    /// \brief Check if joint angle is valid according to filter mode
    ///
    /// \param joint_index Index of the joint to check
    /// \param angle Angle value to validate
    /// \return true if angle is valid, false otherwise
    bool IsValidJointAngle(size_t joint_index, double angle) const {
        if (joint_index >= filters_.size()) {
            return true;
        }

        const double threshold = angle_thresholds_[joint_index];
        switch (filters_[joint_index]) {
            case FilterMode::POSITIVE:
                return angle >= threshold;
            case FilterMode::NEGATIVE:
                return angle <= -threshold;
            case FilterMode::NONE:
            default:
                return true;
        }
    }

    /// \brief Reset all joint filters to NONE mode
    void Reset() {
        std::fill(filters_.begin(), filters_.end(), FilterMode::NONE);
    }

    /// \brief Get filter mode of specified joint
    ///
    /// \param joint_index Index of the joint
    /// \return FilterMode of the specified joint
    FilterMode GetJointFilter(size_t joint_index) const {
        if (joint_index < filters_.size()) {
            return filters_[joint_index];
        }
        return FilterMode::NONE;
    }

private:
    /// \brief Initialize angle thresholds in constructor
    void InitializeThresholds(size_t dof) {
        angle_thresholds_.resize(dof, 0.0);
    }

private:
    std::vector<FilterMode> filters_;
    std::vector<double> angle_thresholds_;
};

}  // namespace robotics
}  // namespace holistic_motion
