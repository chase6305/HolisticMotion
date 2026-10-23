# Exercise CMakeDeps' package-level build modules on every Conan consumer run.
# Discovering collision must not require CUDA, even in a CUDA-enabled package.
if(HOLISTICMOTION_CONSUMER_TEST_COLLISION)
    set(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit TRUE)
    find_package(HolisticMotion REQUIRED CONFIG COMPONENTS collision)
    if(TARGET CUDA::cudart)
        message(FATAL_ERROR "Collision-only discovery loaded the CUDA runtime")
    endif()
    unset(CMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit)
endif()

# A preceding collision-only lookup must not suppress the later core lookup.
# Repeated lookups must not append the CUDA runtime more than once.
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS core)
find_package(HolisticMotion REQUIRED CONFIG COMPONENTS core)
if(HOLISTICMOTION_CONSUMER_TEST_CUDA)
    if(NOT TARGET CUDA::cudart)
        message(FATAL_ERROR "Core discovery did not load the CUDA runtime")
    endif()
    get_target_property(_consumer_core_links HolisticMotion::holistic_motion
        INTERFACE_LINK_LIBRARIES)
    set(_consumer_cuda_count 0)
    foreach(_consumer_link IN LISTS _consumer_core_links)
        if(_consumer_link STREQUAL "CUDA::cudart")
            math(EXPR _consumer_cuda_count "${_consumer_cuda_count} + 1")
        endif()
    endforeach()
    if(NOT _consumer_cuda_count EQUAL 1)
        message(FATAL_ERROR "Repeated core discovery duplicated CUDA runtime links")
    endif()
    unset(_consumer_core_links)
    unset(_consumer_cuda_count)
    unset(_consumer_link)
endif()
