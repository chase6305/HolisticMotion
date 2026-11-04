cmake_minimum_required(VERSION 3.22)
project(HolisticMotionConsumer LANGUAGES CXX)

# A lower project default verifies the targets export their C++17 requirement.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
   "model" IN_LIST HOLISTICMOTION_CONSUMER_COMPONENTS OR
   "core" IN_LIST HOLISTICMOTION_CONSUMER_COMPONENTS)
    add_executable(model_consumer src/model_consumer.cpp)
    target_link_libraries(model_consumer PRIVATE HolisticMotion::model)
endif()

if(NOT HOLISTICMOTION_CONSUMER_COMPONENTS OR
   "core" IN_LIST HOLISTICMOTION_CONSUMER_COMPONENTS)
    add_executable(trajectory_consumer src/trajectory_consumer.cpp)
    target_link_libraries(trajectory_consumer PRIVATE
        HolisticMotion::holistic_motion)

    add_executable(coordinate_consumer src/coordinate_consumer.cpp)
    target_link_libraries(coordinate_consumer PRIVATE
        holistic-motion::holistic-motion)

    add_executable(fep_consumer src/fep_consumer.cpp)
    target_link_libraries(fep_consumer PRIVATE HolisticMotion::holistic_motion)
endif()

if(HOLISTICMOTION_CONSUMER_TEST_COLLISION)
    add_executable(collision_consumer src/collision_consumer.cpp)
    target_link_libraries(collision_consumer PRIVATE
        HolisticMotion::collision)
endif()
