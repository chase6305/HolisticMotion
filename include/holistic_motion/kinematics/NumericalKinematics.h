#pragma once
#include "holistic_motion/kinematics/KinematicsBase.h"

namespace holistic_motion {
namespace robotics {

enum class IkMethod { SVD, DAMP_SVD, JACOBIAN_TRANSPOSE, CCD };

class NumericalKinematics : public KinematicsBase {
public:
    explicit NumericalKinematics(const std::vector<JointNode>& joint_node) {
        this->joint_nodes_ = joint_node;

        this->dof_ = joint_node.size() - 1;
        this->initalize_ = true;
        this->home_joints_ = Eigen::VectorXd::Zero(this->dof_);
        this->ik_nearst_weight_ = Eigen::VectorXd::Ones(this->dof_);
        this->joint_filter_config_.Resize(this->dof_);
        holistic_motion::utility::LogDebug("Constructing NumericalKinematics...");
    };
    virtual ~NumericalKinematics() {
        holistic_motion::utility::LogDebug("Destructing NumericalKinematics...");
    };

    void SetDOF(const int dof) { this->dof_ = dof; };

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

    /// \brief Calculates the current axes of the robotic manipulator's joints
    /// in the world coordinate frame.
    ///
    /// @param axis_list An output parameter that will be filled with the axes
    /// of the joints in the world coordinate frame.
    /// @param pose_list A list of poses representing the configuration of the
    /// robotic manipulator's segments.
    /// @return Returns true if the joint axes were successfully calculated.
    bool _GetCurrentJointAxisInWorld(Eigen::MatrixXd& axis_list,
                                     const std::vector<SE3d>& pose_list) const;

    /// \brief Calculates the change in joint angles (delta theta) based on a
    /// desired change in end-effector position/orientation (delta x), using a
    /// specified inverse kinematics method.
    ///
    /// @param delta_x A transformation representing the desired change in
    /// end-effector position and orientation.
    /// @param jacobian The Jacobian matrix of the robotic manipulator.
    /// @param delta_theta Output parameter that will be filled with the
    /// calculated change in joint angles.
    /// @param method The inverse kinematics method to be used for calculating
    /// delta theta. Defaults to Damped SVD.
    /// @return Returns true if the calculation was successful.
    bool _CalDeltaTheta(SE3d& delta_x,
                        Eigen::MatrixXd& jacobian,
                        Eigen::VectorXd& delta_theta,
                        const IkMethod& method = IkMethod::DAMP_SVD) const;

    /// \brief Calculates the change in joint angles (delta theta) using the
    /// Singular Value Decomposition (SVD) method, based on a desired change in
    /// end-effector position/orientation (delta x) and the manipulator's
    /// Jacobian matrix.
    ///
    /// @param delta_x A transformation representing the desired change in
    /// end-effector position and orientation.
    /// @param jacobian The Jacobian matrix of the robotic manipulator.
    /// @param delta_theta Output parameter that will be filled with the
    /// calculated change in joint angles.
    /// @return Returns true if the calculation was successful.
    bool _GetDeltaThetaSVD(SE3d& delta_x,
                           const Eigen::MatrixXd& jacobian,
                           Eigen::VectorXd& delta_theta) const;

    /// \brief Calculates the change in joint angles (delta theta) using the
    /// Damped Singular Value Decomposition (SVD) method, based on a desired
    /// change in end-effector position/orientation (delta x), the manipulator's
    /// Jacobian matrix, and a damping coefficient to improve numerical
    /// stability.
    ///
    /// @param delta_x A transformation representing the desired change in
    /// end-effector position and orientation.
    /// @param jacobian The Jacobian matrix of the robotic manipulator.
    /// @param delta_theta Output parameter that will be filled with the
    /// calculated change in joint angles.
    /// @param damp_coeffs The damping coefficient used to improve the stability
    /// of the inverse calculation. Defaults to 0.0.
    /// @return Returns true if the calculation was successful.
    ///
    bool _GetDeltaThetaDampedSVD(SE3d& delta_x,
                                 const Eigen::MatrixXd& jacobian,
                                 Eigen::VectorXd& delta_theta,
                                 const double& damp_coeffs = 0.0) const;

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
