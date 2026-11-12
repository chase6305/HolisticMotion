#pragma once

namespace holistic_motion::robotics::fep_internal {

#ifdef HOLISTICMOTION_HAS_CUDA
// Joints are column-major (count x 7); output contains row-major 4x4 poses.
// Seven [unit_x, unit_y, unit_z, raw_axis_norm] blocks, prepared on the host
// with the same norm as scalar/CPU FK; the device must not recompute that scale.
bool ForwardBatchCUDA(const double* joints,
                      int count,
                      const double* origins,
                      const double* axes,
                      const double* tcp,
                      double* output);
#endif

}  // namespace holistic_motion::robotics::fep_internal
