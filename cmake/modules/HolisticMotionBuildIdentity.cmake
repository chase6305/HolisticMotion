function(holistic_motion_configure_build_identity)
    # Native components exchange C++ values through pybind11. Keep their source
    # and toolchain cohort explicit, including private shared-data layouts.
    file(GLOB_RECURSE identity_inputs CONFIGURE_DEPENDS
        RELATIVE "${PROJECT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.h"
        "${PROJECT_SOURCE_DIR}/src/*.hpp"
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
        "${PROJECT_SOURCE_DIR}/src/*.cu"
        "${PROJECT_SOURCE_DIR}/bindings/*.h"
        "${PROJECT_SOURCE_DIR}/bindings/*.hpp"
        "${PROJECT_SOURCE_DIR}/bindings/*.cpp"
        "${PROJECT_SOURCE_DIR}/cmake/*.cmake"
        "${PROJECT_SOURCE_DIR}/cmake/*.in")
    list(APPEND identity_inputs CMakeLists.txt)
    list(SORT identity_inputs)
    set(binding_identity "")
    if(HOLISTICMOTION_BUILD_PYTHON)
        set(binding_version "${pybind11_VERSION}")
        if(NOT binding_version)
            set(binding_version "${pybind11_VERSION_STRING}")
        endif()
        string(CONCAT binding_identity
            "${HOLISTICMOTION_PYTHON_NATIVE_VERSION};"
            "${HOLISTICMOTION_PYTHON_NATIVE_ABI};${binding_version}")
    endif()
    string(CONCAT identity "${PROJECT_VERSION};${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR};"
        "${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_CXX_STANDARD};"
        "${CMAKE_CXX_FLAGS};${CMAKE_CXX_FLAGS_DEBUG};${CMAKE_CXX_FLAGS_RELEASE};"
        "${CMAKE_CXX_FLAGS_RELWITHDEBINFO};${CMAKE_CXX_FLAGS_MINSIZEREL};"
        "${BUILD_SHARED_LIBS};${HOLISTICMOTION_ENABLE_CUDA};"
        "${HOLISTICMOTION_ENABLE_COLLISION};${HOLISTICMOTION_ENABLE_SANITIZERS};"
        "${CMAKE_INTERPROCEDURAL_OPTIMIZATION};${HOLISTICMOTION_CUDA_ARCHITECTURES};"
        "${HOLISTICMOTION_NATIVE_CUDA_VERSION};${Eigen3_VERSION};${fmt_VERSION};"
        "${spdlog_VERSION};${urdfdom_VERSION};${pinocchio_VERSION_STRING};${coal_VERSION};"
        "${HOLISTICMOTION_BUILD_PYTHON};${binding_identity}")
    foreach(input IN LISTS identity_inputs)
        set(path "${PROJECT_SOURCE_DIR}/${input}")
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${path}")
        file(SHA256 "${path}" digest)
        string(APPEND identity "\n${input}:${digest}")
    endforeach()
    string(SHA256 HOLISTICMOTION_NATIVE_BUILD_HASH "${identity}")
    configure_file("${PROJECT_SOURCE_DIR}/cmake/BuildIdentityConfig.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/native_identity/BuildIdentityConfig.h" @ONLY)
endfunction()

function(holistic_motion_use_build_identity target)
    target_include_directories(${target} PRIVATE
        "${CMAKE_CURRENT_BINARY_DIR}/native_identity")
    target_compile_definitions(${target} PRIVATE
        HOLISTICMOTION_BUILD_CONFIGURATION="$<CONFIG>")
endfunction()
