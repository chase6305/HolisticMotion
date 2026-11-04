#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "holistic_motion/kinematics/JointFilterManager.h"
#include "holistic_motion/kinematics/RobotConfigManager.h"
#include "holistic_motion/kinematics/Types.h"

namespace holistic_motion {
namespace robotics {

inline double GetRAngle(const Eigen::Matrix3d &rotation) {
    const double cosine = clamp((rotation.trace() - 1.0) * 0.5, -1.0, 1.0);
    // atan2 retains small rotations whose trace already rounds to three.
    // hypot also preserves tiny skew coefficients when their squares vanish.
    const double x = rotation(2, 1) - rotation(1, 2);
    const double y = rotation(0, 2) - rotation(2, 0);
    const double z = rotation(1, 0) - rotation(0, 1);
    if (x == 0.0 && y == 0.0 && z == 0.0)
        return cosine < 0.0 ? M_PI : 0.0;
    const double sine = 0.5 * std::hypot(x, y, z);
    return std::atan2(sine, cosine);
}

/// \brief OPW available through papers: "An Analytical Solution of the Inverse
/// Kinematics Problem of Industrial Serial Manipulators with an Ortho-parallel
/// Basis and a Spherical Wrist".
struct OPWParameters {
    OPWParameters() : a1{0}, a2{0}, c1{0}, c2{0}, c3{0}, c4{0}, b{0} {
        // debug logger
    }

    /// c1, c2, c3, c4 : the main arm lengths.
    /// a1, a2 : the arm-offsets.
    //  b : the the lateral offset of the third arm in y0-direction.
    double a1, a2, c1, c2, c3, c4, b;

    SE3d T_b_ob;  ///< from "base" frame to opsw_base frame
    SE3d T_oe_e;  ///< from opsw_end_effector frame to "end_effector" frame

    std::array<double, 6> offsets{};  ///< joint offsets in rad
    std::array<signed char, 6>
            rotation_directions{};  ///< rotation direction (about z-axis), 1
                                    ///< for clockwise, and -1 for
                                    ///< counterclockwise

    /// \brief report self
    ///
    /// \param OPWParameters
    /// \return std::string
    friend std::string _report(const OPWParameters &params);
};

struct URParameters {
    URParameters() : d1{0}, a2{0}, a3{0}, d4{0}, d5{0}, d6{0} {
        // debug logger
    }

    /// read from https://github.com/pyni/ur_inverse_solutions
    double d1, a2, a3, d4, d5, d6;

    SE3d T_b_ob;  ///< from "base" frame to ur_base frame
    SE3d T_oe_e;  ///< from ur_end_effector frame to "end_effector" frame

    std::array<double, 6> offsets{};  ///< joint offsets in rad
    std::array<signed char, 6>
            rotation_directions{};  ///< rotation direction (about z-axis), 1
                                    ///< for clockwise, and -1 for
                                    ///< counterclockwise

    /// \brief report self
    ///
    /// \param URParameters
    /// \return std::string
    friend std::string _report(const URParameters &params);
};

/// \brief FEP available through papers: "Analytical Inverse Kinematic
/// Computation for 7-DOF Redundant Manipulators With Joint Limits and Its
/// Application to Redundancy Resolution".
struct FEPParameters {
    double d1{0.0}, d3{0.0}, d5{0.0}, d7e{0.0};
    double a4{0.0}, a5{0.0}, a7{0.0};
    double LL24{0.0}, LL46{0.0}, L24{0.0}, L46{0.0};
    double thetaH46{0.0}, theta342{0.0}, theta46H{0.0};
    std::array<double, 7> q_min{}, q_max{};
    double q4_minus_angle{0.0};
    double q7_start{0.0}, q7_end{0.0};
};

struct FrankaParameters : FEPParameters {
    FrankaParameters();

    /// \brief report self
    ///
    /// \param FrankaParameters
    /// \return std::string
    friend std::string _report(const FrankaParameters &params);
};

// TODO: read from robot urdf params, extend to other type robot
struct AgileDiana7Parameters : FEPParameters {
    AgileDiana7Parameters();

    double a6;

    /// \brief report self
    ///
    /// \param AgileDiana7Parameters
    /// \return std::string
    friend std::string _report(const AgileDiana7Parameters &params);
};

struct IkRtn {
    bool success;

    // may define a enum
    int state;  // state 0: success; 1: exceed link length; 2: exceed joint
                // limit;
    Eigen::MatrixXd joint_states; // [result_num, control_joint_num]
                                  // inverse kinematic joint states
    int dof;
    int ik_number;
    std::vector<Eigen::VectorXd> ik_joints;  // ik

    // may add and modify joint_limit
    IkRtn() : success(false), state(-1) {
        dof = 6;
        ik_number = 0;
    }
    IkRtn(bool success, int state, const Eigen::MatrixXd& joint_states)
        : success(success), state(state), joint_states(joint_states) {
        dof = 6;
        ik_number = 0;
    }

    /// \brief Clear all members to initial state
    void Clear() {
        success = false;
        state = -1;
        dof = 6;
        if (joint_states.rows() != 0 || joint_states.cols() != 0) {
            joint_states.resize(0, 0);
        }
        ik_joints.clear();
        ik_number = 0;
    }

    /// \brief Add new joint solution vector to ik_joints
    ///
    /// \param joints Joint angle vector to be added
    /// An empty vector represents the single configuration of a zero-DOF model.
    /// Subsequent solutions must have the same dimension as the first.
    /// \return true if successfully added, false otherwise
    bool PushBack(const Eigen::VectorXd& joints);

    /// \brief Remove approximately equal IK solutions within tolerance
    ///
    /// Keeps lexicographically first representatives in a deterministic greedy
    /// pass. All coordinate differences must be strictly below eps to merge;
    /// eps == 0 merges exact duplicates. Approximate equality is not transitive.
    /// Invalid tolerance or malformed candidates clear solution outputs.
    /// Cached ik_number is reconciled from ik_joints and joint_states cleared.
    /// \param eps Finite, nonnegative coordinate tolerance
    /// \return true if operation successful, false otherwise
    bool RemoveRepeatedIK(const double& eps = 1e-4);

    /// \brief Expand periodic solutions inside inclusive joint limits.
    /// At most 65536 results are generated. Excessive expansion or a turn
    /// offset that cannot be represented fails and clears solution outputs.
    /// Nonperiodic coordinates are checked without angle wrapping.
    /// Rotational coordinate periods follow the model's axis length:
    /// 2*pi/axis.norm(), preserving the FK convention for nonunit axes.
    ///
    /// \param joint_nodes Vector containing joint limit information
    /// \return true if valid solutions found, false otherwise
    bool GetLimitsIK(const std::vector<JointNode>& joint_nodes);

    /// Explicit expansion budget, from 1 through INT_MAX. No partial result is
    /// returned on failure. The budget counts generated configurations before
    /// any later deduplication.
    bool GetLimitsIK(const std::vector<JointNode>& joint_nodes,
                     std::size_t max_solutions);

    /// Map each periodic IK branch into limits using the representation nearest
    /// to a reference state, without generating equivalent multi-turn copies.
    /// Nonperiodic coordinates are only checked against their limits. Invalid
    /// input or an unrepresentable selected turn/phase clears solution outputs.
    bool WrapToLimitsNear(const std::vector<JointNode>& joint_nodes,
                          const Eigen::VectorXd& reference);

    /// \brief Sort solutions by minimum movement from reference joint state
    ///
    /// Sorts the supplied finite distances with stable ties; distances must
    /// correspond to ik_joints. The reference is validated, not recomputed.
    /// Malformed input clears solutions and distances. Clears joint_states.
    /// \param joints Finite reference state with the configured DOF
    /// \param dist Output vector storing sorted distances
    /// \return true if sorting successful, false otherwise
    bool SortMinMovement(const Eigen::VectorXd& joints,
                         std::vector<double>& dist);

    /// \brief Filter solutions based on robot configuration constraints
    ///
    /// \param joint_nodes Vector of joint nodes
    /// \param robot_config Robot configuration manager with constraints
    /// \return true if valid solutions remain after filtering
    /// Six-DoF elbow filtering uses scalar-axis FK for its first four nodes.
    /// Base/wrist constraints and other dimensions are unsupported and clear
    /// the solutions rather than silently ignoring the requested constraint.
    /// Invalid geometry or candidates clear the outputs; clears joint_states.
    bool LimitRobotConfig(const std::vector<JointNode>& joint_nodes,
                          const RobotConfigManager& robot_config);

    /// \brief Apply joint solution filters based on filter manager rules
    ///
    /// Invalid dimensions or nonfinite candidates clear solution outputs.
    /// Reconciles ik_number and clears joint_states after filtering.
    /// \param filter Joint filter manager containing filtering rules
    /// \return true if filtering successful, false otherwise
    bool FilterJointSolutions(const JointFilterManager& filter);

    /// \brief Calculate angle between two 3D vectors
    ///
    /// \param v1 First vector
    /// \param v2 Second vector
    /// \return Angle between vectors in radians
    double _CalculateAngleBetweenVectors(const Eigen::Vector3d& v1,
                                         const Eigen::Vector3d& v2);

    /// \brief Generate formatted report of IkRtn state
    ///
    /// \param report IkRtn object to report
    /// \return Formatted string containing object state information
    friend std::string _report(const IkRtn &report);
};

}  // namespace robotics
}  // namespace holistic_motion
