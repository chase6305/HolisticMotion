#pragma once

namespace holistic_motion::robotics::fep_internal {

#ifdef HOLISTICMOTION_HAS_CUDA
bool ForwardBatchCUDA(const double* joints,
                      int count,
                      const double* origins,
                      const double* axes,
                      const double* tcp,
                      double* output);
#endif

}  // namespace holistic_motion::robotics::fep_internal
