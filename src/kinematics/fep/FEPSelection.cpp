#include "FEPSelection.h"

#include <cmath>

namespace holistic_motion::robotics::fep_internal {

std::vector<double> ScoreCandidatesCPU(const std::vector<Eigen::VectorXd> &candidates,
                                       const Eigen::VectorXd &seed,
                                       const CoordinatePeriods &periods) {
    if (seed.size() != 7 || !seed.allFinite() || !ValidCoordinatePeriods(periods))
        return {};
    std::vector<double> scores(candidates.size(), 0.0);
    for (std::size_t row = 0; row < candidates.size(); ++row) {
        if (candidates[row].size() != 7 || !candidates[row].allFinite())
            return {};
        for (Eigen::Index joint = 0; joint < seed.size(); ++joint) {
            const double delta =
                std::remainder(candidates[row][joint] - seed[joint], periods[joint]);
            scores[row] += delta * delta;
        }
        if (!std::isfinite(scores[row]))
            return {};
    }
    return scores;
}

} // namespace holistic_motion::robotics::fep_internal
