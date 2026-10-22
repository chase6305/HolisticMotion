#pragma once

// Apply an instantiation rule to one private fixed-size Eigen type batch.

#define HOLISTIC_MOTION_TRAJECTORY_TYPES_1_TO_8(rule, group)                           \
    rule(group, R1d) rule(group, R2d) rule(group, R3d) rule(group, R4d)                \
        rule(group, R5d) rule(group, R6d) rule(group, R7d) rule(group, R8d)            \
            rule(group, SE3d)

#define HOLISTIC_MOTION_TRAJECTORY_TYPES_9_TO_16(rule, group)                          \
    rule(group, R9d) rule(group, R10d) rule(group, R11d) rule(group, R12d)             \
        rule(group, R13d) rule(group, R14d) rule(group, R15d) rule(group, R16d)

#define HOLISTIC_MOTION_TRAJECTORY_TYPES_17_TO_24(rule, group)                         \
    rule(group, R17d) rule(group, R18d) rule(group, R19d) rule(group, R20d)            \
        rule(group, R21d) rule(group, R22d) rule(group, R23d) rule(group, R24d)

#define HOLISTIC_MOTION_TRAJECTORY_TYPES_25_TO_32(rule, group)                         \
    rule(group, R25d) rule(group, R26d) rule(group, R27d) rule(group, R28d)            \
        rule(group, R29d) rule(group, R30d) rule(group, R31d) rule(group, R32d)

#define HOLISTIC_MOTION_INSTANTIATE_TRAJECTORY_CLASS(group, type)                      \
    template class group<type>;
