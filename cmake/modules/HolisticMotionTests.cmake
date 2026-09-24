function(holistic_motion_add_smoke_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ${ARGN})
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
    string(REGEX REPLACE "_test$" "" test_name "${target}")
    add_test(NAME ${test_name} COMMAND ${target})
endfunction()

function(holistic_motion_register_tests)
    find_package(Threads REQUIRED)
    holistic_motion_add_smoke_test(
        logging_smoke_test
        tests/cpp/utility/logging_smoke.cpp
        holistic_motion Threads::Threads)
    set_tests_properties(logging_smoke PROPERTIES TIMEOUT 10)
    add_test(NAME compiler_check_smoke COMMAND ${CMAKE_COMMAND}
        -P ${PROJECT_SOURCE_DIR}/tests/cmake/compiler_check_smoke.cmake)
    holistic_motion_add_smoke_test(
        robot_model_smoke_test
        tests/cpp/robot/robot_model_smoke.cpp
        holistic_motion)
    set_tests_properties(robot_model_smoke PROPERTIES TIMEOUT 10)
    holistic_motion_add_smoke_test(
        trajectory_smoke_test
        tests/cpp/trajectory/trajectory_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        path_bezier_smoke_test
        tests/cpp/trajectory/path_bezier_smoke.cpp
        holistic_motion)
    set_tests_properties(path_bezier_smoke PROPERTIES TIMEOUT 10)
    holistic_motion_add_smoke_test(
        pspline_smoke_test
        tests/cpp/trajectory/pspline_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        double_s_profile_smoke_test
        tests/cpp/trajectory/double_s_profile_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        trapezoidal_profile_smoke_test
        tests/cpp/trajectory/trapezoidal_profile_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        analytic_kinematics_smoke_test
        tests/cpp/kinematics/analytic_kinematics_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        numerical_kinematics_smoke_test
        tests/cpp/kinematics/numerical_kinematics_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        ik_limit_smoke_test
        tests/cpp/kinematics/ik_limit_smoke.cpp
        holistic_motion)
    set_tests_properties(ik_limit_smoke PROPERTIES TIMEOUT 10)
    holistic_motion_add_smoke_test(
        kinematic_model_smoke_test
        tests/cpp/kinematics/kinematic_model_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        srs_null_space_smoke_test
        tests/cpp/kinematics/srs_null_space_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        damped_ik_workspace_smoke_test
        tests/cpp/kinematics/damped_ik_workspace_smoke.cpp
        Eigen3::Eigen)
    target_include_directories(damped_ik_workspace_smoke_test PRIVATE
        ${PROJECT_SOURCE_DIR}/src/kinematics)
    holistic_motion_add_smoke_test(
        fep_batch_smoke_test
        tests/cpp/kinematics/fep_batch_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        sampling_planner_smoke_test
        tests/cpp/planning/sampling_planner_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        joint_space_metric_smoke_test
        tests/cpp/planning/joint_space_metric_smoke.cpp
        Eigen3::Eigen)
    target_include_directories(joint_space_metric_smoke_test PRIVATE
        ${PROJECT_SOURCE_DIR}/src/planning)
    holistic_motion_add_smoke_test(
        joint_space_index_smoke_test
        tests/cpp/planning/joint_space_index_smoke.cpp
        Eigen3::Eigen)
    target_include_directories(joint_space_index_smoke_test PRIVATE
        ${PROJECT_SOURCE_DIR}/src/planning)
    set_tests_properties(joint_space_index_smoke PROPERTIES TIMEOUT 30)
    holistic_motion_add_smoke_test(
        path_geometry_workspace_smoke_test
        tests/cpp/planning/path_geometry_workspace_smoke.cpp
        Eigen3::Eigen)
    target_include_directories(path_geometry_workspace_smoke_test PRIVATE
        ${PROJECT_SOURCE_DIR}/src/planning)
    if(HOLISTICMOTION_ENABLE_COLLISION)
        holistic_motion_add_smoke_test(
            collision_smoke_test
            tests/cpp/collision/collision_smoke.cpp
            holistic_motion_collision)
    endif()
endfunction()
