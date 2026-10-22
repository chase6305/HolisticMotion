# Extra dependency wiring used by Conan's generated CMake targets. The native
# install config performs the same lookup in HolisticMotionConfig.cmake.
if(HolisticMotion_FIND_COMPONENTS AND
   NOT "core" IN_LIST HolisticMotion_FIND_COMPONENTS)
    return()
endif()
find_package(CUDAToolkit @HOLISTICMOTION_NATIVE_CUDA_VERSION@ EXACT REQUIRED)
get_target_property(_HolisticMotion_core_links HolisticMotion::holistic_motion
    INTERFACE_LINK_LIBRARIES)
if(NOT "CUDA::cudart" IN_LIST _HolisticMotion_core_links)
    target_link_libraries(HolisticMotion::holistic_motion INTERFACE CUDA::cudart)
endif()
unset(_HolisticMotion_core_links)
