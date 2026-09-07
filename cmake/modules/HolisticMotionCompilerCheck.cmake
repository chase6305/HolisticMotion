function(holistic_motion_check_conan_compiler)
    # Standalone CMake builds and older toolchains carry no Conan metadata.
    if(NOT HOLISTICMOTION_CONAN_COMPILER OR NOT HOLISTICMOTION_CONAN_COMPILER_VERSION)
        return()
    endif()

    if(HOLISTICMOTION_CONAN_COMPILER STREQUAL "gcc")
        set(expected_id GNU)
    elseif(HOLISTICMOTION_CONAN_COMPILER STREQUAL "clang")
        set(expected_id Clang)
    elseif(HOLISTICMOTION_CONAN_COMPILER STREQUAL "apple-clang")
        set(expected_id AppleClang)
    elseif(HOLISTICMOTION_CONAN_COMPILER STREQUAL "msvc")
        set(expected_id MSVC)
    else()
        # Other compiler families have different Conan version conventions.
        return()
    endif()

    set(matches TRUE)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL expected_id)
        set(matches FALSE)
    elseif(expected_id STREQUAL "MSVC")
        # Conan 193/194 identifies MSVC toolsets reporting _MSC_VER 193x/194x.
        string(SUBSTRING "${MSVC_VERSION}" 0 3 detected_version)
        if(NOT detected_version STREQUAL HOLISTICMOTION_CONAN_COMPILER_VERSION)
            set(matches FALSE)
        endif()
    else()
        # Compare only the components specified by the profile: GCC 12 accepts
        # 12.3.0, while Apple Clang 13.1 must not silently select 13.0.
        string(REPLACE "." ";" expected_parts "${HOLISTICMOTION_CONAN_COMPILER_VERSION}")
        string(REPLACE "." ";" detected_parts "${CMAKE_CXX_COMPILER_VERSION}")
        list(LENGTH expected_parts expected_count)
        list(LENGTH detected_parts detected_count)
        if(detected_count LESS expected_count)
            set(matches FALSE)
        else()
            math(EXPR last_part "${expected_count} - 1")
            foreach(index RANGE ${last_part})
                list(GET expected_parts ${index} expected_part)
                list(GET detected_parts ${index} detected_part)
                if(NOT expected_part STREQUAL detected_part)
                    set(matches FALSE)
                endif()
            endforeach()
        endif()
    endif()

    if(NOT matches)
        message(FATAL_ERROR
            "Conan compiler mismatch: profile requests "
            "${HOLISTICMOTION_CONAN_COMPILER} ${HOLISTICMOTION_CONAN_COMPILER_VERSION}, "
            "but CMake selected ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} "
            "(${CMAKE_CXX_COMPILER}). Set tools.build:compiler_executables in the "
            "selected Conan profile, or select settings matching the intended compiler, "
            "then configure a fresh build directory.")
    endif()
endfunction()
