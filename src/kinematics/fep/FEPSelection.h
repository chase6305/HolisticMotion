#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <Eigen/Core>

namespace holistic_motion::robotics::fep_internal {

using CoordinatePeriods = std::array<double, 7>;
inline constexpr CoordinatePeriods kUnitCoordinatePeriods{
    2.0 * 3.14159265358979323846, 2.0 * 3.14159265358979323846,
    2.0 * 3.14159265358979323846, 2.0 * 3.14159265358979323846,
    2.0 * 3.14159265358979323846, 2.0 * 3.14159265358979323846,
    2.0 * 3.14159265358979323846};

inline bool ValidCoordinatePeriods(const CoordinatePeriods &periods) noexcept {
    return std::all_of(periods.begin(), periods.end(), [](double value) {
        return std::isfinite(value) && value > 0.0;
    });
}

// Seven-coordinate finite inputs and finite positive model-coordinate periods.
// Empty output also signals malformed inputs or unrepresentable scores.
std::vector<double>
ScoreCandidatesCPU(const std::vector<Eigen::VectorXd> &candidates,
                   const Eigen::VectorXd &seed,
                   const CoordinatePeriods &periods = kUnitCoordinatePeriods);

#ifdef HOLISTICMOTION_HAS_CUDA
bool CudaBackendAvailable() noexcept;
// Seven-coordinate finite inputs. A false result always clears scores, so a
// caller can safely fall back to CPU without consuming stale or partial data.
bool ScoreCandidatesCUDA(const std::vector<Eigen::VectorXd> &candidates,
                         const Eigen::VectorXd &seed, std::vector<double> &scores,
                         const CoordinatePeriods &periods = kUnitCoordinatePeriods);
#endif

} // namespace holistic_motion::robotics::fep_internal
