#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_UTILS_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_UTILS_H_
#pragma once
#include "holistic_motion/manif/constants.h"
namespace holistic_motion {
namespace robotics {

/// @brief Euler angles
/// https://en.wikipedia.org/wiki/Euler_angles
/// Proper Euler angles (z-x-z, x-y-x, y-z-y, z-y-z, x-z-x, y-x-y)
/// Tait–Bryan angles   (x-y-z, y-z-x, z-x-y, x-z-y, z-y-x, y-x-z)
enum class EulerAnglesType {
    XYZ,
    xyz,
    ZYX,
    zyx,
    ZYZ,
    zyz,
    ZXZ,
    zxz,
    YXY,
    yxy,
    YZY,
    yzy,
    XZX,
    xzx,
    XYX,
    xyx,
    rota_vec
};

/**
 * @brief Wrap an angle in -pi,pi.
 * @param[in] angle The angle to be wrapped in radians
 * @return The wrapped angle.
 */
template <typename T>
T Pi2Pi(T angle) {
    while (angle > T(HOLISTIC_MOTION_PI)) angle -= T(2. * HOLISTIC_MOTION_PI);
    while (angle <= T(-HOLISTIC_MOTION_PI)) angle += T(2. * HOLISTIC_MOTION_PI);

    return angle;
}

/**
 * @brief Conversion to radians
 * @param[in] deg angle in degrees
 * @return angle in radians
 */
template <typename T>
constexpr T ToRad(const T deg) {
    return deg * Constants<T>::to_rad;
}

/**
 * @brief Conversion to degrees
 * @param[in] rad angle in radians
 * @return angle in degrees
 */
template <typename T>
constexpr T ToDeg(const T rad) {
    return rad * Constants<T>::to_deg;
}

/**
 * @brief Degree 2 polynomial approximation of 1/sqrt(x) (reciprocal sqrt).
 * @param[in] x
 * @return ~1/sqrt(x)
 */
template <typename T>
constexpr T ApproxSqrtInv(const T x) {
    return (T(15) / T(8)) - (T(5) / T(4)) * x + (T(3) / T(8)) * x * x;
}

template <typename T>
Eigen::AngleAxis<T> RotX(const T rad) {
    return Eigen::AngleAxis<T>(rad, Eigen::Matrix<T, 3, 1>::UnitZ());
}

template <typename T>
Eigen::AngleAxis<T> RotY(const T rad) {
    return Eigen::AngleAxis<T>(rad, Eigen::Matrix<T, 3, 1>::UnitY());
}

template <typename T>
Eigen::AngleAxis<T> RotZ(const T rad) {
    return Eigen::AngleAxis<T>(rad, Eigen::Matrix<T, 3, 1>::UnitZ());
}

}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_UTILS_H_ */
