// Keep Eigen's allocation guard active even when the test is built in Release.
#ifdef NDEBUG
#undef NDEBUG
#endif
#define EIGEN_RUNTIME_NO_MALLOC

#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "JointSpaceTopology.h"

using holistic_motion::robotics::planning::detail::JointSpaceMetric;
using holistic_motion::robotics::planning::detail::kPi;

int main() {
    const Eigen::VectorXd unit = Eigen::VectorXd::Ones(1);
    const Eigen::VectorXd origin = Eigen::VectorXd::Zero(1);
    Eigen::VectorXd query(1);
    const std::vector<bool> circle{true};
    const JointSpaceMetric circular_metric(unit, circle);
    const auto same_squared_distance = [&](double value) {
        query[0] = value;
        const double arc = std::remainder(value, 2.0 * kPi);
        const double expected = arc * arc;
        const double actual = circular_metric.SquaredDistance(origin, query);
        return (std::isnan(expected) && std::isnan(actual)) ||
               actual == expected;
    };
    Eigen::internal::set_is_malloc_allowed(false);
    for (const double value :
         {0.0, -0.0, kPi, -kPi, 3.0 * kPi, -3.0 * kPi, std::nextafter(kPi, 0.0),
          std::nextafter(kPi, 4.0), std::nextafter(-kPi, 0.0),
          std::nextafter(-kPi, -4.0), std::numeric_limits<double>::max(),
          std::numeric_limits<double>::infinity(),
          std::numeric_limits<double>::quiet_NaN()}) {
        if (!same_squared_distance(value))
            return 2;
    }
    std::mt19937_64 generator(214);
    std::uniform_real_distribution<double> angle(-4.0 * kPi, 4.0 * kPi);
    for (int sample = 0; sample < 10000; ++sample) {
        if (!same_squared_distance(angle(generator)))
            return 2;
    }
    Eigen::internal::set_is_malloc_allowed(true);

    const Eigen::VectorXd weights =
        (Eigen::Vector3d() << 4.0, 0.25, 2.0).finished();
    const Eigen::VectorXd first =
        (Eigen::Vector3d() << kPi - 0.1, -0.4, 0.5).finished();
    const Eigen::VectorXd second =
        (Eigen::Vector3d() << -kPi + 0.2, 0.4, 1.0).finished();
    Eigen::VectorXd shifted = second;
    shifted[0] += 4.0 * kPi;
    const std::vector<bool> bounded(3, false);
    const std::vector<bool> periodic{true, false, false};
    const JointSpaceMetric bounded_metric(weights, bounded);
    const JointSpaceMetric periodic_metric(weights, periodic);

    // The seam is 0.3 radians apart on S1, but 2*pi - 0.3 in a bounded joint.
    // Guard all query variants against regression to a temporary VectorXd.
    Eigen::internal::set_is_malloc_allowed(false);
    const double seam = periodic_metric.SquaredDistance(first, second);
    const double reverse = periodic_metric.SquaredDistance(second, first);
    const double equivalent = periodic_metric.SquaredDistance(first, shifted);
    const double euclidean = bounded_metric.SquaredDistance(first, second);
    const double zero = periodic_metric.SquaredDistance(first, first);
    Eigen::internal::set_is_malloc_allowed(true);
    const double expected_seam =
        4.0 * 0.3 * 0.3 + 0.25 * 0.8 * 0.8 + 2.0 * 0.5 * 0.5;
    const double expected_bounded = 4.0 * std::pow(2.0 * kPi - 0.3, 2) + 0.66;
    if (std::abs(seam - expected_seam) > 1e-12 ||
        std::abs(reverse - seam) > 1e-12 ||
        std::abs(equivalent - seam) > 1e-12 ||
        std::abs(euclidean - expected_bounded) > 1e-12 || zero != 0.0) {
        std::cerr << "weighted joint-space metric changed\n";
        return 1;
    }
    // Exercise masks across packed-word boundaries and compare shortest arcs
    // independently, including exact seams and angles several turns apart.
    for (Eigen::Index dof : {7, 14, 63, 64, 65, 129}) {
        const Eigen::VectorXd axis_weights =
            Eigen::VectorXd::LinSpaced(dof, 0.125, 2.0);
        std::vector<bool> topology(static_cast<std::size_t>(dof));
        for (Eigen::Index axis = 0; axis < dof; ++axis)
            topology[static_cast<std::size_t>(axis)] = axis % 3 != 1;
        const JointSpaceMetric mixed_metric(axis_weights, topology);
        Eigen::VectorXd from(dof), to(dof);
        for (int sample = 0; sample < 128; ++sample) {
            double expected = 0.0;
            for (Eigen::Index axis = 0; axis < dof; ++axis) {
                from[axis] = angle(generator);
                to[axis] = angle(generator);
                if (sample % 8 == 0) {
                    from[axis] = 0.0;
                    to[axis] = (axis % 2 ? -1.0 : 1.0) * kPi;
                }
                double delta = to[axis] - from[axis];
                if (topology[static_cast<std::size_t>(axis)])
                    delta = std::remainder(delta, 2.0 * kPi);
                expected += axis_weights[axis] * (delta * delta);
            }
            Eigen::internal::set_is_malloc_allowed(false);
            const double actual = mixed_metric.SquaredDistance(from, to);
            Eigen::internal::set_is_malloc_allowed(true);
            if (actual != expected) {
                std::cerr << "mixed metric changed at dimension " << dof
                          << '\n';
                return 3;
            }
        }
    }
    return 0;
}
