cmake_minimum_required(VERSION 3.22)
project(HolisticMotionConsumer LANGUAGES CXX)

find_package(HolisticMotion REQUIRED CONFIG)

add_executable(trajectory_consumer src/trajectory_consumer.cpp)
target_compile_features(trajectory_consumer PRIVATE cxx_std_17)
target_link_libraries(trajectory_consumer PRIVATE
    HolisticMotion::holistic_motion)

if(HOLISTICMOTION_CONSUMER_TEST_COLLISION)
    add_executable(collision_consumer src/collision_consumer.cpp)
    target_compile_features(collision_consumer PRIVATE cxx_std_17)
    target_link_libraries(collision_consumer PRIVATE
        HolisticMotion::holistic_motion)
endif()
