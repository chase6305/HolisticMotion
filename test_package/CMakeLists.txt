cmake_minimum_required(VERSION 3.22)
project(HolisticMotionConsumer LANGUAGES CXX)

set(HOLISTICMOTION_CONSUMER_COMPONENTS "" CACHE STRING
    "Optional explicit installed-SDK component selection")
if(HOLISTICMOTION_CONSUMER_TEST_DISCOVERY)
    include(cmake/VerifyComponentDiscovery.cmake)
endif()
if(HOLISTICMOTION_CONSUMER_COMPONENTS)
    find_package(HolisticMotion REQUIRED CONFIG
        COMPONENTS ${HOLISTICMOTION_CONSUMER_COMPONENTS})
else()
    find_package(HolisticMotion REQUIRED CONFIG)
endif()

if(HOLISTICMOTION_CONSUMER_TEST_PYTHON)
    find_package(Python3 ${HOLISTICMOTION_CONSUMER_PYTHON_VERSION} EXACT REQUIRED
        COMPONENTS Interpreter)
    file(WRITE "${CMAKE_BINARY_DIR}/python_executable.txt" "${Python3_EXECUTABLE}\n")
endif()

if(NOT HOLISTICMOTION_CONSUMER_COMPONENTS OR
   "core" IN_LIST HOLISTICMOTION_CONSUMER_COMPONENTS)
    add_executable(trajectory_consumer src/trajectory_consumer.cpp)
    target_compile_features(trajectory_consumer PRIVATE cxx_std_17)
    target_link_libraries(trajectory_consumer PRIVATE
        HolisticMotion::holistic_motion)

    add_executable(coordinate_consumer src/coordinate_consumer.cpp)
    target_compile_features(coordinate_consumer PRIVATE cxx_std_17)
    target_link_libraries(coordinate_consumer PRIVATE
        HolisticMotion::holistic_motion)

    add_executable(fep_consumer src/fep_consumer.cpp)
    target_compile_features(fep_consumer PRIVATE cxx_std_17)
    target_link_libraries(fep_consumer PRIVATE HolisticMotion::holistic_motion)
endif()

if(HOLISTICMOTION_CONSUMER_TEST_COLLISION)
    add_executable(collision_consumer src/collision_consumer.cpp)
    target_compile_features(collision_consumer PRIVATE cxx_std_17)
    target_link_libraries(collision_consumer PRIVATE
        HolisticMotion::collision)
endif()
