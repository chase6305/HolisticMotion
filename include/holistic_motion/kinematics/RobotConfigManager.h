#pragma once
#include <numeric>
#include <set>

#include "holistic_motion/kinematics/Types.h"
#include "holistic_motion/utility/Logging.h"

namespace holistic_motion {
namespace robotics {

/// \brief get the sign of the input number
///
/// \param T scalar
/// \param x input number
/// \return int 1, -1 or 0
template <typename T>
static inline int Sign(const T& x) {
    return (x > T(0)) ? 1 : ((x < T(0)) ? -1 : 0);
}

// Define bitmask enum class to represent different robot configuration options
enum class RobotConfig {
    NONE = 0,                 // 0: No configuration
    BASE_FRONT = 1 << 1,      // 10: Base facing front
    BASE_BACK = 1 << 2,       // 100: Base facing back
    ELBOW_UP = 1 << 3,        // 1000: Elbow facing up
    ELBOW_DOWN = 1 << 4,      // 10000: Elbow facing down
    WRIST_FLIP = 1 << 5,      // 100000: Wrist flip
    WRIST_NOT_FLIP = 1 << 6,  // 1000000: Wrist not flip
};

/// \brief Bitwise OR operator for RobotConfig.
///
/// \param lhs The left-hand side RobotConfig value.
/// \param rhs The right-hand side RobotConfig value.
/// \return A RobotConfig value that is the result of a bitwise OR operation
/// between lhs and rhs.
constexpr RobotConfig operator|(RobotConfig lhs, RobotConfig rhs) {
    return static_cast<RobotConfig>(static_cast<int>(lhs) |
                                    static_cast<int>(rhs));
}

/// \brief Bitwise AND operator for RobotConfig.
///
/// \param lhs The left-hand side RobotConfig value.
/// \param rhs The right-hand side RobotConfig value.
/// \return A RobotConfig value that is the result of a bitwise AND operation
/// between lhs and rhs.
constexpr RobotConfig operator&(RobotConfig lhs, RobotConfig rhs) {
    return static_cast<RobotConfig>(static_cast<int>(lhs) &
                                    static_cast<int>(rhs));
}

/// \brief Bitwise NOT operator for RobotConfig.
///
/// \param rhs The RobotConfig value to be negated.
/// \return A RobotConfig value that is the result of a bitwise NOT operation on
/// rhs.
constexpr RobotConfig operator~(RobotConfig rhs) {
    return static_cast<RobotConfig>(~static_cast<int>(rhs));
}

class RobotConfigManager {
public:
    /// \brief Set the base orientation to front or back.
    ///
    /// \param enable Set to true to face front, false to face back.
    void SetBaseFront(bool enable) {
        if (enable) {
            config_ = (config_ | RobotConfig::BASE_FRONT) &
                      ~RobotConfig::BASE_BACK;
        } else {
            config_ = (config_ & ~RobotConfig::BASE_FRONT) |
                      RobotConfig::BASE_BACK;
        }
        UpdateEnableConfig();
    }

    /// \brief Set the elbow orientation to up or down.
    ///
    /// \param enable Set to true to face up, false to face down.
    void SetElbowUp(bool enable) {
        if (enable) {
            config_ = (config_ | RobotConfig::ELBOW_UP) &
                      ~RobotConfig::ELBOW_DOWN;
        } else {
            config_ = (config_ & ~RobotConfig::ELBOW_UP) |
                      RobotConfig::ELBOW_DOWN;
        }
        UpdateEnableConfig();
    }

    /// \brief Set the wrist orientation to flip or not flip.
    ///
    /// \param enable Set to true to flip, false to not flip.
    void SetWristFlip(bool enable) {
        if (enable) {
            config_ = (config_ | RobotConfig::WRIST_FLIP) &
                      ~RobotConfig::WRIST_NOT_FLIP;
        } else {
            config_ = (config_ & ~RobotConfig::WRIST_FLIP) |
                      RobotConfig::WRIST_NOT_FLIP;
        }
        UpdateEnableConfig();
    }

    /// \brief Get the base orientation.
    /// \return True if facing front, false if facing back.
    bool IsBaseFront() const {
        return this->IsFlagSet(config_, RobotConfig::BASE_FRONT);
    }

    /// \brief Get the base orientation.
    /// \return True if facing back, false if facing front.
    bool IsBaseBack() const {
        return this->IsFlagSet(config_, RobotConfig::BASE_BACK);
    }

    /// \brief Get the elbow orientation.
    /// \return True if facing up, false if facing down.
    bool IsElbowUp() const {
        return this->IsFlagSet(config_, RobotConfig::ELBOW_UP);
    }

    /// \brief Get the elbow orientation.
    /// \return True if facing down, false if facing up.
    bool IsElbowDown() const {
        return this->IsFlagSet(config_, RobotConfig::ELBOW_DOWN);
    }

    /// \brief Get the wrist orientation.
    /// \return True if flipped, false if not flipped.
    bool IsWristFlip() const {
        return this->IsFlagSet(config_, RobotConfig::WRIST_FLIP);
    }

    /// \brief Get the wrist orientation.
    /// \return True if not flipped, false if flipped.
    bool IsWristNotFlip() const {
        return this->IsFlagSet(config_, RobotConfig::WRIST_NOT_FLIP);
    }

    /// \brief Get disable or not.
    /// \return True if Disable, false if valid.
    bool IsDisable() const { return config_ == RobotConfig::NONE; }

    /// \brief Disables all configurations.
    void Disable() {
        config_ = RobotConfig::NONE;
        UpdateEnableConfig();
    }

private:
    /// \brief Check if a specific flag is set in the configuration.
    ///
    /// \param config The current configuration to check.
    /// \param flag The flag to check within the configuration.
    /// \return True if the specified flag is set in the configuration, false
    /// otherwise.
    inline bool IsFlagSet(RobotConfig config, RobotConfig flag) const {
        return static_cast<int>(config & flag) != 0;
    }

    /// \brief Update the enable configuration based on the current settings.
    ///
    /// This function checks if any of the specific configuration flags
    /// (BASE_FRONT, BASE_BACK, ELBOW_UP, ELBOW_DOWN, WRIST_FLIP,
    /// WRIST_NOT_FLIP) are set in the current configuration. If any of these
    /// flags are set, the ENABLE_CONFIG flag is added to the configuration.
    /// Otherwise, the ENABLE_CONFIG flag is removed from the configuration.
    void UpdateEnableConfig() {
        if (static_cast<int>(config_) &
            static_cast<int>(RobotConfig::BASE_FRONT | RobotConfig::BASE_BACK |
                             RobotConfig::ELBOW_UP | RobotConfig::ELBOW_DOWN |
                             RobotConfig::WRIST_FLIP |
                             RobotConfig::WRIST_NOT_FLIP)) {
            config_ = config_;
        } else {
            config_ = config_;
        }
    }

    RobotConfig config_ = RobotConfig::NONE;
};

}  // namespace robotics
}  // namespace holistic_motion
