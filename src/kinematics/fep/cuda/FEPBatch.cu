#include "../FEPBatchCUDA.h"

#include <cuda_runtime.h>
#include <cmath>

namespace holistic_motion::robotics::fep_internal {
namespace {

__device__ void Identity(double* matrix) {
    for (int i = 0; i < 16; ++i) matrix[i] = (i % 5 == 0) ? 1.0 : 0.0;
}

__device__ void Multiply(const double* lhs, const double* rhs, double* out) {
    double temporary[16];
    for (int row = 0; row < 4; ++row)
        for (int col = 0; col < 4; ++col) {
            double value = 0.0;
            for (int k = 0; k < 4; ++k)
                value += lhs[row * 4 + k] * rhs[k * 4 + col];
            temporary[row * 4 + col] = value;
        }
    for (int i = 0; i < 16; ++i) out[i] = temporary[i];
}

__device__ void Rotation(const double* raw_axis, double angle, double* out) {
    Identity(out);
    const double norm = sqrt(raw_axis[0] * raw_axis[0] +
                             raw_axis[1] * raw_axis[1] +
                             raw_axis[2] * raw_axis[2]);
    if (norm < 1e-15) return;
    const double x = raw_axis[0] / norm, y = raw_axis[1] / norm,
                 z = raw_axis[2] / norm;
    const double c = cos(angle), s = sin(angle), v = 1.0 - c;
    out[0] = x * x * v + c; out[1] = x * y * v - z * s;
    out[2] = x * z * v + y * s;
    out[4] = y * x * v + z * s; out[5] = y * y * v + c;
    out[6] = y * z * v - x * s;
    out[8] = z * x * v - y * s; out[9] = z * y * v + x * s;
    out[10] = z * z * v + c;
}

__global__ void ForwardKernel(const double* joints,
                              int count,
                              const double* origins,
                              const double* axes,
                              const double* tcp,
                              double* output) {
    const int pose = blockIdx.x * blockDim.x + threadIdx.x;
    if (pose >= count) return;
    double transform[16], motion[16];
    Identity(transform);
    for (int joint = 0; joint < 7; ++joint) {
        Multiply(transform, origins + joint * 16, transform);
        Rotation(axes + joint * 3, joints[pose * 7 + joint], motion);
        Multiply(transform, motion, transform);
    }
    Multiply(transform, origins + 7 * 16, transform);
    Multiply(transform, tcp, output + pose * 16);
}

}  // namespace

bool ForwardBatchCUDA(const double* joints,
                      int count,
                      const double* origins,
                      const double* axes,
                      const double* tcp,
                      double* output) {
    if (count <= 0) return true;
    double *d_joints = nullptr, *d_origins = nullptr, *d_axes = nullptr,
           *d_tcp = nullptr, *d_output = nullptr;
    const std::size_t joint_bytes = static_cast<std::size_t>(count) * 7 * sizeof(double);
    const std::size_t output_bytes = static_cast<std::size_t>(count) * 16 * sizeof(double);
    bool ok = cudaMalloc(&d_joints, joint_bytes) == cudaSuccess &&
              cudaMalloc(&d_origins, 8 * 16 * sizeof(double)) == cudaSuccess &&
              cudaMalloc(&d_axes, 7 * 3 * sizeof(double)) == cudaSuccess &&
              cudaMalloc(&d_tcp, 16 * sizeof(double)) == cudaSuccess &&
              cudaMalloc(&d_output, output_bytes) == cudaSuccess;
    if (ok) ok = cudaMemcpy(d_joints, joints, joint_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(d_origins, origins, 8 * 16 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(d_axes, axes, 7 * 3 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess &&
                 cudaMemcpy(d_tcp, tcp, 16 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        const int threads = 128, blocks = (count + threads - 1) / threads;
        ForwardKernel<<<blocks, threads>>>(d_joints, count, d_origins, d_axes,
                                           d_tcp, d_output);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaMemcpy(output, d_output, output_bytes,
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(d_joints); cudaFree(d_origins); cudaFree(d_axes);
    cudaFree(d_tcp); cudaFree(d_output);
    return ok;
}

}  // namespace holistic_motion::robotics::fep_internal
