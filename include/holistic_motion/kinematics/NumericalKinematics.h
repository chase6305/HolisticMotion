#pragma once
#include "holistic_motion/kinematics/KinematicsBase.h"

namespace holistic_motion {
namespace robotics {

enum class IkMethod { SVD, DAMP_SVD, JACOBIAN_TRANSPOSE, CCD };

class NumericalKinematics : public KinematicsBase {
public:
    /// The last node is a fixed tool transform (UNKNOWN is also accepted).
    /// Empty or invalid models throw std::invalid_argument. A single fixed node
    /// represents a zero-coordinate model.
    explicit NumericalKinematics(const std::vector<JointNode>& joint_node);
    virtual ~NumericalKinematics() {
        holistic_motion::utility::LogDebug("Destructing NumericalKinematics...");
    };

    /// Compatibility setter: only the model-derived DOF is accepted. Other
    /// values throw std::invalid_argument without changing model-sized state.
    void SetDOF(int dof);

    /// \brief get the forward kinematics
    ///
    /// \param target_joint
    /// \param pose
    /// \return bool success or not
    virtual bool GetFK(const Eigen::VectorXd& target_joint,
                       SE3d& pose) const override final;

    /// \brief get all inverse kinematics
    ///
    /// \param target_pose
    /// \param ik_solutions
    /// \param joints_seed
    /// \return bool success or not
    virtual bool GetIK(const SE3d& target_pose,
                       IkRtn& ik_solutions,
                       Eigen::VectorXd& joint_seed,
                       std::vector<double>& dist) const override final;

    /// \brief get the nearst inverse kinematics
    ///
    /// \param target_pose
    /// \param ik_solutions
    /// \param joints_seed
    /// \return bool success or not
    virtual bool GetNearstIK(const SE3d& target_pose,
                             IkRtn& ik_solutions,
                             Eigen::VectorXd& joint_seed,
                             double& min_dist) const override final;

    bool GetJacobian(const Eigen::VectorXd& joint_pos,
                     Eigen::MatrixXd& jacobian) const;

    bool SetMaxIterNum(const int& max_iter_num);

    bool GetMaxIterNum(int& max_iter_num) const;

    bool SetTransErrTh(const double& trans_err_th);

    bool GetTransErrTh(double& trans_err_th) const;

    bool SetAngleErrTh(const double& angle_err_th);

    bool GetAngleErrTh(double& angle_err_th) const;

    bool SetStepSize(const double& step_size);

    bool GetStepSize(double& step_size) const;

    bool SetDamp(const double& damp);

    bool GetDamp(double& damp) const;

private:
    /// \brief Computes the kinematic Jacobian matrix for the robotic
    /// manipulator.
    ///
    /// @param jacobian An output parameter that will be filled with the
    /// calculated Jacobian matrix.
    /// @param pose_list A list of poses representing the configuration of the
    /// robotic manipulator's segments.
    /// @return Returns true if the Jacobian matrix was successfully calculated.
    bool _GetKinJacobian(Eigen::MatrixXd& jacobian,
                         const std::vector<SE3d>& pose_list) const;

    /// \brief Retrieves the bounds for controllable joints of the robotic
    /// manipulator.
    ///
    /// \return bool success or not
    bool _GetControllableJointBounds() const;

private:
    double eps_ = 1e-4;  ///< About the iteration accuracy of the displacement
                         ///< and rotation components of the iterative pose
    int max_iter_num_ = 100;             ///< The maximum number of iterations
    double trans_err_th_ = 5e-4;         ///< XYZ Error Level
    double angle_err_th_ = M_PI / 1800;  ///< rotation Error Level
    double step_size_ = 0.95;            ///< step size
    double damp_coeff_ = 0.05;
};

}  // namespace robotics
}  // namespace holistic_motion
