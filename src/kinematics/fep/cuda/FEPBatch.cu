#include "../FEPBatchCUDA.h"
#include "DeviceBuffer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace holistic_motion::robotics::fep_internal {
namespace {

__device__ void Identity(double *matrix) {
    for (int i = 0; i < 16; ++i)
        matrix[i] = (i % 5 == 0) ? 1.0 : 0.0;
}

__device__ void Multiply(const double *lhs, const double *rhs, double *out) {
    double temporary[16];
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) {
            double value = 0.0;
            for (int k = 0; k < 4; ++k)
                value += lhs[row * 4 + k] * rhs[k * 4 + col];
            temporary[row * 4 + col] = value;
        }
    for (int i = 0; i < 16; ++i)
        out[i] = temporary[i];
}

__device__ void Rotation(const double *axis, double angle, double *out) {
    Identity(out);
    const double norm = axis[3];
    if (norm < 1e-15)
        return;
    const double x = axis[0], y = axis[1], z = axis[2];
    // The chain coordinate multiplies the raw axis, matching scalar and CPU
    // batch FK even when a caller supplies a non-unit revolute axis.
    angle *= norm;
    const double c = cos(angle), s = sin(angle), v = 1.0 - c;
    out[0] = x * x * v + c;
    out[1] = x * y * v - z * s;
    out[2] = x * z * v + y * s;
    out[4] = y * x * v + z * s;
    out[5] = y * y * v + c;
    out[6] = y * z * v - x * s;
    out[8] = z * x * v - y * s;
    out[9] = z * y * v + x * s;
    out[10] = z * z * v + c;
}

__global__ void ForwardKernel(const double *joints, int count, const double *origins,
                              const double *axes, const double *tcp, double *output) {
    const std::size_t pose =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pose >= static_cast<std::size_t>(count))
        return;
    double transform[16], motion[16];
    Identity(transform);
    for (int joint = 0; joint < 7; ++joint) {
        Multiply(transform, origins + joint * 16, transform);
        Rotation(axes + joint * 4, joints[pose * 7 + joint], motion);
        Multiply(transform, motion, transform);
    }
    Multiply(transform, origins + 7 * 16, transform);
    Multiply(transform, tcp, output + pose * 16);
}

} // namespace

bool ForwardBatchCUDA(const double *joints, int count, const double *origins,
                      const double *axes, const double *tcp, double *output) {
    if (count < 0)
        return false;
    if (count == 0)
        return true;
    constexpr std::size_t model_elements = 8 * 16 + 7 * 4 + 16;
    const auto rows = static_cast<std::size_t>(count);
    if (rows >
        ((std::numeric_limits<std::size_t>::max)() / sizeof(double) - model_elements) /
            23)
        return false;
    const std::size_t joint_elements = rows * 7;
    const std::size_t joint_bytes = joint_elements * sizeof(double);
    const std::size_t output_bytes = rows * 16 * sizeof(double);
    DeviceBuffer buffer(rows * 23 + model_elements);
    if (!buffer)
        return false;
    double *const d_joints = buffer.Data();
    double *const d_origins = d_joints + joint_elements;
    double *const d_axes = d_origins + 8 * 16;
    double *const d_tcp = d_axes + 7 * 4;
    double *const d_output = d_tcp + 16;
    std::array<double, model_elements> model;
    std::copy_n(origins, 8 * 16, model.begin());
    std::copy_n(axes, 7 * 4, model.begin() + 8 * 16);
    std::copy_n(tcp, 16, model.end() - 16);
    bool ok = cudaMemcpy(d_joints, joints, joint_bytes, cudaMemcpyHostToDevice) ==
                  cudaSuccess &&
              cudaMemcpy(d_origins, model.data(), model_elements * sizeof(double),
                         cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        const int threads = 128, blocks = (count - 1) / threads + 1;
        ForwardKernel<<<blocks, threads>>>(d_joints, count, d_origins, d_axes, d_tcp,
                                           d_output);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost) ==
                 cudaSuccess;
    }
    return ok;
}

} // namespace holistic_motion::robotics::fep_internal
