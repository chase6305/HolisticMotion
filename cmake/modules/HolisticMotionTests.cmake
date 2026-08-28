function(holistic_motion_add_smoke_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE ${ARGN})
    target_compile_options(${target} PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)
    string(REGEX REPLACE "_test$" "" test_name "${target}")
    add_test(NAME ${test_name} COMMAND ${target})
endfunction()

function(holistic_motion_register_tests)
    enable_testing()
    holistic_motion_add_smoke_test(
        trajectory_smoke_test
        tests/cpp/trajectory/trajectory_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        analytic_kinematics_smoke_test
        tests/cpp/kinematics/analytic_kinematics_smoke.cpp
        holistic_motion)
    holistic_motion_add_smoke_test(
        sampling_planner_smoke_test
        tests/cpp/planning/sampling_planner_smoke.cpp
        holistic_motion)
    if(HOLISTICMOTION_ENABLE_COLLISION)
        holistic_motion_add_smoke_test(
            collision_smoke_test
            tests/cpp/collision/collision_smoke.cpp
            holistic_motion_collision)
    endif()
endfunction()
