if(COMPILER_CHECK_CHILD)
    include("${CHECK_MODULE}")
    holistic_motion_check_conan_compiler()
    return()
endif()

function(check_case name compiler expected_version detected_id detected_version should_fail)
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DCOMPILER_CHECK_CHILD=ON"
            "-DCHECK_MODULE=${CMAKE_CURRENT_LIST_DIR}/../../cmake/modules/HolisticMotionCompilerCheck.cmake"
            "-DHOLISTICMOTION_CONAN_COMPILER=${compiler}"
            "-DHOLISTICMOTION_CONAN_COMPILER_VERSION=${expected_version}"
            "-DCMAKE_CXX_COMPILER_ID=${detected_id}"
            "-DCMAKE_CXX_COMPILER_VERSION=${detected_version}"
            "-DMSVC_VERSION=${ARGN}"
            -P "${CMAKE_CURRENT_LIST_FILE}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(should_fail)
        if(result EQUAL 0 OR NOT error MATCHES "Conan compiler mismatch")
            message(FATAL_ERROR "${name}: mismatch was not diagnosed: ${output}${error}")
        endif()
    elseif(NOT result EQUAL 0)
        message(FATAL_ERROR "${name}: compatible compiler was rejected: ${output}${error}")
    endif()
endfunction()

check_case(gcc_match gcc 12 GNU 12.3.0 FALSE)
check_case(gcc_major_mismatch gcc 12 GNU 11.4.0 TRUE)
check_case(gcc_prefix_mismatch gcc 1 GNU 12.3.0 TRUE)
check_case(family_mismatch gcc 12 Clang 12.0.1 TRUE)
check_case(clang_match clang 18 Clang 18.1.8 FALSE)
check_case(apple_clang_match apple-clang 13.1 AppleClang 13.1.6 FALSE)
check_case(apple_clang_minor_mismatch apple-clang 13.1 AppleClang 13.0.0 TRUE)
check_case(apple_clang_family_mismatch clang 13 AppleClang 13.1.6 TRUE)
check_case(msvc_match msvc 193 MSVC 19.38.33130.0 FALSE 1938)
check_case(msvc_mismatch msvc 193 MSVC 19.40.33811.0 TRUE 1940)
check_case(standalone "" "" GNU 11.4.0 FALSE)
check_case(unknown_family intel-cc 2024 IntelLLVM 2024.0.0 FALSE)
