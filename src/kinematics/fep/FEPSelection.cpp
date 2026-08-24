#include "FEPSelection.h"

#include <cmath>

namespace holistic_motion::robotics::fep_internal {

std::vector<double> ScoreCandidatesCPU(
        const std::vector<Eigen::VectorXd>& candidates,
        const Eigen::VectorXd& seed) {
    std::vector<double> scores(candidates.size(), 0.0);
    for (std::size_t row = 0; row < candidates.size(); ++row) {
        for (Eigen::Index joint = 0; joint < seed.size(); ++joint) {
            const double delta = std::remainder(
                    candidates[row][joint] - seed[joint], 2.0 * M_PI);
            scores[row] += delta * delta;
        }
    }
    return scores;
}

}  // namespace holistic_motion::robotics::fep_internal
