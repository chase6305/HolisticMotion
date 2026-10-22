#include <array>
#include <limits>
#include <vector>

#include "FEPBatchCUDA.h"
#include "FEPSelection.h"
#include "cuda/DeviceBuffer.h"

using namespace holistic_motion::robotics::fep_internal;

namespace {
bool fail_allocation = false;
int fail_copy = 0;
int copy_calls = 0;
int live_allocations = 0;
int allocation_calls = 0;
} // namespace

// Linux link wrapping injects errors at the public CUDA runtime boundary.
// This exercises cleanup after successful allocations and after kernel launch
// without exhausting the device or changing another process's memory limits.
extern "C" cudaError_t __real_cudaMalloc(void **pointer, std::size_t bytes);
extern "C" cudaError_t __real_cudaFree(void *pointer);
extern "C" cudaError_t __real_cudaMemcpy(void *target, const void *source,
                                         std::size_t bytes, cudaMemcpyKind kind);

extern "C" cudaError_t __wrap_cudaMalloc(void **pointer, std::size_t bytes) {
    ++allocation_calls;
    if (fail_allocation)
        return cudaErrorMemoryAllocation;
    const auto status = __real_cudaMalloc(pointer, bytes);
    if (status == cudaSuccess)
        ++live_allocations;
    return status;
}

extern "C" cudaError_t __wrap_cudaFree(void *pointer) {
    const auto status = __real_cudaFree(pointer);
    if (pointer && status == cudaSuccess)
        --live_allocations;
    return status;
}

extern "C" cudaError_t __wrap_cudaMemcpy(void *target, const void *source,
                                         std::size_t bytes, cudaMemcpyKind kind) {
    ++copy_calls;
    if (copy_calls == fail_copy)
        return cudaErrorInvalidValue;
    return __real_cudaMemcpy(target, source, bytes, kind);
}

int main() {
    // Invalid sizes never enter CUDA, including builds tested without a GPU.
    {
        DeviceBuffer empty(0);
        DeviceBuffer overflow((std::numeric_limits<std::size_t>::max)());
        if (empty || overflow || allocation_calls != 0)
            return 1;
    }
    fail_allocation = true;
    {
        DeviceBuffer unavailable(1);
        if (unavailable || live_allocations != 0)
            return 2;
    }
    fail_allocation = false;
    if (!CudaBackendAvailable())
        return 0;

    std::array<double, 7> joints{};
    std::array<double, 8 * 16> origins{};
    std::array<double, 7 * 4> axes{};
    std::array<double, 16> tcp{}, output{};
    for (std::size_t node = 0; node < 8; ++node)
        for (std::size_t index = 0; index < 4; ++index)
            origins[node * 16 + index * 5] = 1.0;
    for (std::size_t joint = 0; joint < 7; ++joint) {
        axes[joint * 4 + 2] = 1.0;
        axes[joint * 4 + 3] = 1.0;
    }
    for (std::size_t index = 0; index < 4; ++index)
        tcp[index * 5] = 1.0;
    const auto forward = [&] {
        return ForwardBatchCUDA(joints.data(), 1, origins.data(), axes.data(),
                                tcp.data(), output.data());
    };
    const Eigen::VectorXd seed = Eigen::VectorXd::Zero(7);
    const std::vector<Eigen::VectorXd> candidates{seed};
    std::vector<double> scores;
    const auto score = [&] { return ScoreCandidatesCUDA(candidates, seed, scores); };

    // Warm up outside failure injection, then reject every transfer stage:
    // input uploads and the output download after a launched kernel.
    if (!forward() || !score() || live_allocations != 0)
        return 3;
    for (int cycle = 0; cycle < 8; ++cycle) {
        fail_allocation = true;
        scores = {42.0};
        if (forward() || score() || !scores.empty() || live_allocations != 0)
            return 4;
        fail_allocation = false;
        for (int transfer = 1; transfer <= 3; ++transfer) {
            fail_copy = transfer;
            copy_calls = 0;
            if (forward() || live_allocations != 0)
                return 5;
            copy_calls = 0;
            scores = {42.0};
            if (score() || !scores.empty() || live_allocations != 0)
                return 6;
        }
        fail_copy = 0;
        if (!forward() || !score() || scores.size() != 1 || scores[0] != 0.0 ||
            live_allocations != 0 || output != tcp)
            return 7;
    }
    return 0;
}
