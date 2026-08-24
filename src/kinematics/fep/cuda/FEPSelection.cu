#include "../FEPSelection.h"

#include <cuda_runtime.h>

namespace holistic_motion::robotics::fep_internal {
namespace {

__global__ void ScoreKernel(const double* candidates,
                            const double* seed,
                            int count,
                            double* scores) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    double score = 0.0;
#pragma unroll
    for (int joint = 0; joint < 7; ++joint) {
        const double delta = remainder(
                candidates[row * 7 + joint] - seed[joint], 2.0 * M_PI);
        score += delta * delta;
    }
    scores[row] = score;
}

}  // namespace

bool CudaBackendAvailable() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool ScoreCandidatesCUDA(const std::vector<Eigen::VectorXd>& candidates,
                         const Eigen::VectorXd& seed,
                         std::vector<double>& scores) {
    if (candidates.empty() || seed.size() != 7 || !CudaBackendAvailable())
        return false;
    std::vector<double> packed(candidates.size() * 7);
    for (std::size_t row = 0; row < candidates.size(); ++row) {
        if (candidates[row].size() != 7) return false;
        for (int joint = 0; joint < 7; ++joint)
            packed[row * 7 + joint] = candidates[row][joint];
    }
    double *device_candidates = nullptr, *device_seed = nullptr,
           *device_scores = nullptr;
    const std::size_t candidate_bytes = packed.size() * sizeof(double);
    const std::size_t score_bytes = candidates.size() * sizeof(double);
    if (cudaMalloc(&device_candidates, candidate_bytes) != cudaSuccess ||
        cudaMalloc(&device_seed, 7 * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&device_scores, score_bytes) != cudaSuccess) {
        cudaFree(device_candidates);
        cudaFree(device_seed);
        cudaFree(device_scores);
        return false;
    }
    bool success =
            cudaMemcpy(device_candidates, packed.data(), candidate_bytes,
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(device_seed, seed.data(), 7 * sizeof(double),
                       cudaMemcpyHostToDevice) == cudaSuccess;
    if (success) {
        const int threads = 128;
        const int blocks = (static_cast<int>(candidates.size()) + threads - 1) /
                           threads;
        ScoreKernel<<<blocks, threads>>>(device_candidates, device_seed,
                                         static_cast<int>(candidates.size()),
                                         device_scores);
        scores.resize(candidates.size());
        success = cudaGetLastError() == cudaSuccess &&
                  cudaMemcpy(scores.data(), device_scores, score_bytes,
                             cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(device_candidates);
    cudaFree(device_seed);
    cudaFree(device_scores);
    return success;
}

}  // namespace holistic_motion::robotics::fep_internal
