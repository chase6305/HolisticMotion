option(HOLISTICMOTION_ENABLE_SANITIZERS
       "Enable ASan, UBSan, and Eigen assertions for C++ CPU development" OFF)

if(HOLISTICMOTION_ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        message(FATAL_ERROR "Sanitizers require GCC or Clang")
    endif()
    if(HOLISTICMOTION_BUILD_PYTHON OR HOLISTICMOTION_ENABLE_CUDA)
        message(FATAL_ERROR
            "Sanitizer builds currently require HOLISTICMOTION_BUILD_PYTHON=OFF "
            "and HOLISTICMOTION_ENABLE_CUDA=OFF. Use a separate C++ CPU build directory.")
    endif()
    # Directory-local instrumentation covers the library and its test drivers.
    # These flags are intentionally not exported as consumer usage requirements;
    # sanitizer builds are for local checks, not distributable release packages.
    add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all
                        -fno-omit-frame-pointer -O1 -g1 -UNDEBUG)
    add_link_options(-fsanitize=address,undefined)
endif()
