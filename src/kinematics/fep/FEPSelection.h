#pragma once

#include <Eigen/Core>
#include <vector>

namespace holistic_motion::robotics::fep_internal {

std::vector<double> ScoreCandidatesCPU(
        const std::vector<Eigen::VectorXd>& candidates,
        const Eigen::VectorXd& seed);

#ifdef HOLISTICMOTION_HAS_CUDA
bool CudaBackendAvailable() noexcept;
bool ScoreCandidatesCUDA(const std::vector<Eigen::VectorXd>& candidates,
                         const Eigen::VectorXd& seed,
                         std::vector<double>& scores);
#endif

}  // namespace holistic_motion::robotics::fep_internal
