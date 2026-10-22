#include "../FEPSelection.h"
#include "DeviceBuffer.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <limits>

namespace holistic_motion::robotics::fep_internal {
namespace {
constexpr int kThreads = 128;

__global__ void ScoreKernel(const double *candidates, const double *seed,
                            const double *periods, int count, double *scores) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count)
        return;
    double score = 0.0;
    const std::size_t offset = static_cast<std::size_t>(row) * 7;
#pragma unroll
    for (int joint = 0; joint < 7; ++joint) {
        const double delta =
            remainder(candidates[offset + joint] - seed[joint], periods[joint]);
        score += delta * delta;
    }
    scores[row] = score;
}

} // namespace

bool CudaBackendAvailable() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool ScoreCandidatesCUDA(const std::vector<Eigen::VectorXd> &candidates,
                         const Eigen::VectorXd &seed, std::vector<double> &scores,
                         const CoordinatePeriods &periods) {
    scores.clear();
    if (candidates.empty() || seed.size() != 7 || !seed.allFinite() ||
        !ValidCoordinatePeriods(periods) ||
        candidates.size() > static_cast<std::size_t>((std::numeric_limits<int>::max)() -
                                                     (kThreads - 1)) ||
        candidates.size() >
            ((std::numeric_limits<std::size_t>::max)() / sizeof(double) - 14) / 8 ||
        !CudaBackendAvailable())
        return false;
    std::vector<double> packed(candidates.size() * 7);
    for (std::size_t row = 0; row < candidates.size(); ++row) {
        if (candidates[row].size() != 7 || !candidates[row].allFinite())
            return false;
        for (int joint = 0; joint < 7; ++joint)
            packed[row * 7 + joint] = candidates[row][joint];
    }
    const std::size_t candidate_bytes = packed.size() * sizeof(double);
    const std::size_t score_bytes = candidates.size() * sizeof(double);
    std::array<double, 14> seed_and_periods;
    std::copy_n(seed.data(), 7, seed_and_periods.begin());
    std::copy(periods.begin(), periods.end(), seed_and_periods.begin() + 7);
    // Finish host allocations before acquiring device buffers. A host
    // allocation failure must not strand successful cudaMalloc allocations.
    scores.resize(candidates.size());
    DeviceBuffer buffer(candidates.size() * 8 + 14);
    if (!buffer) {
        scores.clear();
        return false;
    }
    double *const device_candidates = buffer.Data();
    double *const device_seed = device_candidates + packed.size();
    double *const device_periods = device_seed + 7;
    double *const device_scores = device_periods + 7;
    bool success = cudaMemcpy(device_candidates, packed.data(), candidate_bytes,
                              cudaMemcpyHostToDevice) == cudaSuccess &&
                   cudaMemcpy(device_seed, seed_and_periods.data(),
                              seed_and_periods.size() * sizeof(double),
                              cudaMemcpyHostToDevice) == cudaSuccess;
    if (success) {
        const int blocks =
            (static_cast<int>(candidates.size()) + kThreads - 1) / kThreads;
        ScoreKernel<<<blocks, kThreads>>>(
            device_candidates, device_seed, device_periods,
            static_cast<int>(candidates.size()), device_scores);
        success = cudaGetLastError() == cudaSuccess &&
                  cudaMemcpy(scores.data(), device_scores, score_bytes,
                             cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (success)
        success = std::all_of(scores.begin(), scores.end(),
                              [](double value) { return std::isfinite(value); });
    if (!success)
        scores.clear();
    return success;
}

} // namespace holistic_motion::robotics::fep_internal
