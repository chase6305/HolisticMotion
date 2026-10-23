#pragma once

#include <cstddef>
#include <limits>

#include <cuda_runtime.h>

namespace holistic_motion::robotics::fep_internal {

// Request-local storage shared by the CUDA kernels' input/output slices.
// One allocation avoids independent allocator synchronization for each slice;
// ownership stays with this call, including all error/exception exits.
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t elements) noexcept {
        if (elements == 0 ||
            elements > (std::numeric_limits<std::size_t>::max)() / sizeof(double))
            return;
        if (cudaMalloc(&data_, elements * sizeof(double)) != cudaSuccess)
            data_ = nullptr;
    }

    ~DeviceBuffer() {
        if (data_)
            cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    explicit operator bool() const noexcept { return data_ != nullptr; }
    double *Data() const noexcept { return data_; }

  private:
    double *data_{nullptr};
};

} // namespace holistic_motion::robotics::fep_internal
