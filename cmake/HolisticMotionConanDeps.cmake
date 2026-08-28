# Extra dependency wiring used by Conan's generated CMake targets. The native
# install config performs the same lookup in HolisticMotionConfig.cmake.
if(NOT TARGET CUDA::cudart)
    find_package(CUDAToolkit REQUIRED)
endif()
target_link_libraries(HolisticMotion::holistic_motion INTERFACE CUDA::cudart)
