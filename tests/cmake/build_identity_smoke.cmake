cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED REPOSITORY_ROOT OR NOT DEFINED WORK_DIRECTORY)
    message(FATAL_ERROR "repository and work directories are required")
endif()
file(REMOVE_RECURSE "${WORK_DIRECTORY}")
set(first "${WORK_DIRECTORY}/first")
set(relocated "${WORK_DIRECTORY}/relocated")
foreach(source IN ITEMS "${first}" "${relocated}")
    file(MAKE_DIRECTORY "${source}/cmake" "${source}/src")
    file(COPY "${REPOSITORY_ROOT}/cmake/BuildIdentityConfig.h.in"
        DESTINATION "${source}/cmake")
    file(WRITE "${source}/src/private.cpp" "struct Data { double value; };\n")
    file(WRITE "${source}/CMakeLists.txt"
        "cmake_minimum_required(VERSION 3.22)\n"
        "project(IdentityFixture VERSION 0.1.0 LANGUAGES NONE)\n"
        "include(\"${REPOSITORY_ROOT}/cmake/modules/HolisticMotionBuildIdentity.cmake\")\n"
        "holistic_motion_configure_build_identity()\n")
endforeach()

function(read_identity build output)
    file(READ "${build}/native_identity/BuildIdentityConfig.h" header)
    string(REGEX MATCH "\"([0-9a-f]+):\"" match "${header}")
    string(LENGTH "${CMAKE_MATCH_1}" size)
    if(NOT size EQUAL 64)
        message(FATAL_ERROR "invalid native build identity: ${header}")
    endif()
    set(${output} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

function(configure source build output)
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${source}" -B "${build}" ${ARGN}
        RESULT_VARIABLE status OUTPUT_VARIABLE log ERROR_VARIABLE errors)
    if(NOT status EQUAL 0)
        message(FATAL_ERROR "identity fixture failed: ${log}${errors}")
    endif()
    read_identity("${build}" identity)
    set(${output} "${identity}" PARENT_SCOPE)
endfunction()

configure("${first}" "${WORK_DIRECTORY}/first-build" original)
configure("${relocated}" "${WORK_DIRECTORY}/relocated-build" moved)
if(NOT original STREQUAL moved)
    message(FATAL_ERROR "relocating native sources changed their identity")
endif()
configure("${first}" "${WORK_DIRECTORY}/first-build" repeated)
if(NOT original STREQUAL repeated)
    message(FATAL_ERROR "identical reconfiguration changed native identity")
endif()

# Changing a private shared-data layout must update the generated identity on
# the next build, even without an edit to a public header or CMakeLists.txt.
execute_process(COMMAND "${CMAKE_COMMAND}" -E sleep 1)
file(WRITE "${first}/src/private.cpp" "struct Data { double value, extra; };\n")
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${WORK_DIRECTORY}/first-build"
    RESULT_VARIABLE status OUTPUT_VARIABLE log ERROR_VARIABLE errors)
if(NOT status EQUAL 0)
    message(FATAL_ERROR "identity regeneration failed: ${log}${errors}")
endif()
read_identity("${WORK_DIRECTORY}/first-build" changed)
if(original STREQUAL changed)
    message(FATAL_ERROR "private native source changes did not refresh identity")
endif()
configure("${relocated}" "${WORK_DIRECTORY}/changed-flags" flags
    "-DCMAKE_CXX_FLAGS=-DABI_VARIANT=1")
if(original STREQUAL flags)
    message(FATAL_ERROR "compiler flag changes did not refresh native identity")
endif()

configure("${relocated}" "${WORK_DIRECTORY}/python-build" python
    "-DHOLISTICMOTION_BUILD_PYTHON=ON"
    "-DHOLISTICMOTION_PYTHON_NATIVE_VERSION=3.13.9"
    "-DHOLISTICMOTION_PYTHON_NATIVE_ABI=cpython"
    "-Dpybind11_VERSION=3.0.1")
configure("${relocated}" "${WORK_DIRECTORY}/python-conan-version" conan_binding
    "-DHOLISTICMOTION_BUILD_PYTHON=ON"
    "-DHOLISTICMOTION_PYTHON_NATIVE_VERSION=3.13.9"
    "-DHOLISTICMOTION_PYTHON_NATIVE_ABI=cpython"
    "-Dpybind11_VERSION_STRING=3.0.1")
if(NOT python STREQUAL conan_binding)
    message(FATAL_ERROR "equivalent pybind11 version variables changed identity")
endif()
foreach(variant IN ITEMS abi python_version binding_version)
    set(abi cpython)
    set(python_version 3.13.9)
    set(binding_version 3.0.1)
    if(variant STREQUAL "abi")
        set(abi cpython-t)
    elseif(variant STREQUAL "python_version")
        set(python_version 3.12.14)
    else()
        set(binding_version 2.13.6)
    endif()
    configure("${relocated}" "${WORK_DIRECTORY}/python-${variant}" changed_binding
        "-DHOLISTICMOTION_BUILD_PYTHON=ON"
        "-DHOLISTICMOTION_PYTHON_NATIVE_VERSION=${python_version}"
        "-DHOLISTICMOTION_PYTHON_NATIVE_ABI=${abi}"
        "-Dpybind11_VERSION=${binding_version}")
    if(python STREQUAL changed_binding)
        message(FATAL_ERROR "${variant} changes did not refresh native identity")
    endif()
endforeach()
