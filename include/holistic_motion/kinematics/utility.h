#pragma once
#include <numeric>
#include <set>

#include "holistic_motion/kinematics/JointFilterManager.h"
#include "holistic_motion/kinematics/RobotConfigManager.h"
#include "holistic_motion/kinematics/Types.h"
#include "holistic_motion/utility/Logging.h"

namespace holistic_motion {
namespace robotics {

inline double GetRAngle(const Eigen::Matrix3d& rotation) {
    const double cosine = clamp((rotation.trace() - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(cosine);
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
    friend std::string _report(const OPWParameters& params) {
        std::string res;
        res += "\t OPWParameters: \n";
        res += fmt::format(
                "\t - [ a1 = {}, a2 = {}, b = {}, c1 = {}, c2 = "
                "{}, c3 = {}, c4 = {} ]\n"
                "\t - T_b_ob : SE3d({})\n"
                "\t - T_oe_e : SE3d({})\n"
                "\t - offsets: ({})\n"
                "\t - rotation_directions: ({})\n",
                params.a1, params.a2, params.b, params.c1, params.c2, params.c3,
                params.c4, fmt::join(params.T_b_ob.Coeffs(), " , "),
                fmt::join(params.T_oe_e.Coeffs(), " , "),
                fmt::join(params.offsets, " , "),
                fmt::join(params.rotation_directions, " , "));
        return res;
    };
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
    friend std::string _report(const URParameters& params) {
        std::string res;
        res += "\t URParameters: \n";
        res += fmt::format(
                "\t - [ d1 = {}, a2 = {}, a3 = {}, d4 = {}, a5 = "
                "{}, d6 = {} ]\n"
                "\t - T_b_ob : SE3d({})\n"
                "\t - T_oe_e : SE3d({})\n"
                "\t - offsets: ({})\n"
                "\t - rotation_directions: ({})\n",
                params.d1, params.a2, params.a3, params.d4, params.d5,
                params.d6, fmt::join(params.T_b_ob.Coeffs(), " , "),
                fmt::join(params.T_oe_e.Coeffs(), " , "),
                fmt::join(params.offsets, " , "),
                fmt::join(params.rotation_directions, " , "));
        return res;
    };
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
    FrankaParameters() {
        d1 = 0.3330;
        d3 = 0.3160;
        d5 = 0.3840;
        // const double d7e = 0.2104;  // use Hand Data (0.107 + 0.1034)
        d7e = 0.107;
        a4 = 0.0825;  // a4 = abs(-a5)
        a5 = 0.0825;
        a7 = 0.0880;

        LL24 = 0.10666225;     // a4^2 + d3^2
        LL46 = 0.15426225;     // a4^2 + d5^2
        L24 = 0.326591870689;  // sqrt(LL24)
        L46 = 0.392762332715;  // sqrt(LL46)

        thetaH46 = 1.35916951803;   // atan(d5/a4);
        theta342 = 1.31542071191;   // atan(d3/a4);
        theta46H = 0.211626808766;  // acot(d5/a4); np.pi/2 - atan(d5/a4)

        q_min = {{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175,
                  -2.8973}};
        q_max = {{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};

        q7_start = q_min[6];
        q7_end = q_max[6];

        holistic_motion::utility::LogDebug(
                "\nLoad Franka params:\n\t- [ d1 = {}, d3 = {}, d5 = {}, d7e = "
                "{}, a4 = {}, a7 = {}]"
                "\t - [ LL24 = {}, LL46 = {}, L24 = {}, L46 = {}]\n"
                "\t - [ thetaH46 = {}, theta342 = {}, theta46H = {}]\n",
                d1, d3, d5, d7e, a4, a7, LL24, LL46, L24, L46, thetaH46,
                theta342, theta46H);
    }

    /// \brief report self
    ///
    /// \param FrankaParameters
    /// \return std::string
    friend std::string _report(const FrankaParameters& params) {
        std::string res;
        res += "\t FrankaParameters: \n";
        res += fmt::format(
                "\t - [ d1 = {}, d3 = {}, d5 = {}, d7e = {}, a4 = {}, a7 = "
                "{}]\n"
                "\t - [ LL24 = {}, LL46 = {}, L24 = {}, L46 = {}]\n"
                "\t - [ thetaH46 = {}, theta342 = {}, theta46H = {}]\n",
                params.d1, params.d3, params.d5, params.d7e, params.a4,
                params.a7, params.LL24, params.LL46, params.L24, params.L46,
                params.thetaH46, params.theta342, params.theta46H);
        return res;
    };
};

// TODO: read from robot urdf params, extend to other type robot
struct AgileDiana7Parameters : FEPParameters {
    AgileDiana7Parameters() {
        d1 = 0.2856;
        d3 = 0.4586;
        d5 = 0.4554;
        d7e = 0.1169;
        a4 = 0.0650;  // a4 = abs(-a5 - a6)
        a5 = 0.0528;
        a6 = 0.0122;
        a7 = 0.0870;

        LL24 = 0.21453896;         // a4^2 + d3^2
        LL46 = 0.21161416;         // a4^2 + d5^2
        L24 = 0.4631835057512303;  // sqrt(LL24)
        L46 = 0.4600153910468649;  // sqrt(LL46)

        thetaH46 = 1.4290222431689619;  // atan(d5/a4);
        theta342 = 1.429998441264436;   // atan(d3/a4);
        theta46H = 0.1417740836259348;  // atan(a4/d5)

        q_min = {{-3.12413, -1.57079, -3.12413, 0, -3.12413, -3.12413,
                  -3.12413}};
        q_max = {{3.12413, 1.57079, 3.12413, 3.05432, 3.12413, 3.12413,
                  3.12413}};

        q7_start = q_min[6];
        q7_end = q_max[6];

        holistic_motion::utility::LogDebug(
                "\nLoad AgileDiana7 params:\n\t- [ d1 = {}, d3 = {}, d5 = {}, "
                "d7e = "
                "{}, a4 = {}, a7 = {}]\n"
                "\t - [ LL24 = {}, LL46 = {}, L24 = {}, L46 = {}]\n"
                "\t - [ thetaH46 = {}, theta342 = {}, theta46H = {}]\n",
                d1, d3, d5, d7e, a4, a7, LL24, LL46, L24, L46, thetaH46,
                theta342, theta46H);
    }

    double a6;

    /// \brief report self
    ///
    /// \param AgileDiana7Parameters
    /// \return std::string
    friend std::string _report(const AgileDiana7Parameters& params) {
        std::string res;
        res += "\t AgileDiana7Parameters: \n";
        res += fmt::format(
                "\t - [ d1 = {}, d3 = {}, d5 = {}, d7e = {}, a4 = {}, a7 = "
                "{}]\n"
                "\t - [ LL24 = {}, LL46 = {}, L24 = {}, L46 = {}]\n"
                "\t - [ thetaH46 = {}, theta342 = {}, theta46H = {}]\n",
                params.d1, params.d3, params.d5, params.d7e, params.a4,
                params.a7, params.LL24, params.LL46, params.L24, params.L46,
                params.thetaH46, params.theta342, params.theta46H);
        return res;
    };
};

struct IkRtn {
    bool success;

    // may define a enum
    int state;  // state 0: success; 1: exceed link length; 2: exceed joint
                // limit;
    Eigen::MatrixXd joint_states;  // [result_num, control_joint_num] of
                                   // float32. inverse kinematic joint states
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
        ik_joints.clear();
        ik_number = 0;
    }

    /// \brief Add new joint solution vector to ik_joints
    ///
    /// \param joints Joint angle vector to be added
    /// \return true if successfully added, false otherwise
    bool PushBack(const Eigen::VectorXd& joints);

    /// \brief Remove approximately equal IK solutions within tolerance
    ///
    /// \param eps Tolerance threshold for comparing solutions
    /// \return true if operation successful, false otherwise
    bool RemoveRepeatedIK(const double& eps = 1e-4);

    /// \brief Filter solutions to keep only those within joint limits
    ///
    /// \param joint_nodes Vector containing joint limit information
    /// \return true if valid solutions found, false otherwise
    bool GetLimitsIK(const std::vector<JointNode>& joint_nodes);

    /// Map each periodic IK branch into limits using the representation nearest
    /// to a reference state, without generating equivalent multi-turn copies.
    bool WrapToLimitsNear(const std::vector<JointNode>& joint_nodes,
                          const Eigen::VectorXd& reference);

    /// \brief Sort solutions by minimum movement from reference joint state
    ///
    /// \param joints Reference joint state to compare against
    /// \param dist Output vector storing sorted distances
    /// \return true if sorting successful, false otherwise
    bool SortMinMovement(const Eigen::VectorXd& joints,
                         std::vector<double>& dist);

    /// \brief Filter solutions based on robot configuration constraints
    ///
    /// \param joint_nodes Vector of joint nodes
    /// \param robot_config Robot configuration manager with constraints
    /// \return true if valid solutions remain after filtering
    bool LimitRobotConfig(const std::vector<JointNode>& joint_nodes,
                          const RobotConfigManager& robot_config);

    /// \brief Apply joint solution filters based on filter manager rules
    ///
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
    friend std::string _report(const IkRtn& report) {
        std::string res;
        res += "\t IkRtn: \n";
        res += fmt::format(
                "\t - success: {0} \n"
                "\t - ik number: {1}\n"
                "\t - ik joints:\n"
                "\t   [\n",
                report.success, report.ik_number);
        for (int i = 0; i < report.ik_number; ++i) {
            if (i == (report.ik_number - 1))
                res += fmt::format("\t\t[{0}]\n",
                                   fmt::join(report.ik_joints[i], ","));
            else {
                res += fmt::format("\t\t[{0}],\n",
                                   fmt::join(report.ik_joints[i], ","));
            }
        }
        res += fmt::format("\t   ]");
        return res;
    };
};

}  // namespace robotics
}  // namespace holistic_motion
