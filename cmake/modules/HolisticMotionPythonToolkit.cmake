# This install path deliberately runs before any C++ compiler/dependency lookup.
# It supports independent Python Pinocchio runtimes without a Conan installation.
set(HOLISTICMOTION_PYTHON_TOOLKIT True)
set(HOLISTICMOTION_PYTHON_COLLISION False)
set(HOLISTICMOTION_PYTHON_CUDA False)
set(HOLISTICMOTION_PYTHON_PINOCCHIO_VERSION "")
set(HOLISTICMOTION_PYTHON_NATIVE_VERSION "")
set(HOLISTICMOTION_PYTHON_NATIVE_ABI "")
configure_file(${PROJECT_SOURCE_DIR}/cmake/python_build_config.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/_holistic_motion_build_config.py @ONLY)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/python/holistic_motion/
    DESTINATION holistic_motion FILES_MATCHING PATTERN "*.py")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/_holistic_motion_build_config.py
    DESTINATION holistic_motion RENAME _build_config.py)
