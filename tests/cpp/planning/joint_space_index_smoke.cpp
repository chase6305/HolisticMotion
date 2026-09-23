#ifdef NDEBUG
#undef NDEBUG
#endif
#define EIGEN_RUNTIME_NO_MALLOC

#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "JointSpaceIndex.h"

using holistic_motion::robotics::planning::detail::JointSpaceIndex;
using holistic_motion::robotics::planning::detail::JointSpaceMetric;
using holistic_motion::robotics::planning::detail::kPi;

namespace {

bool Check(JointSpaceIndex &index, const std::vector<Eigen::VectorXd> &states,
           const Eigen::VectorXd &query, const JointSpaceMetric &metric,
           double radius) {
    const auto state_at = [&](std::size_t i) -> const Eigen::VectorXd & {
        return states[i];
    };
    std::size_t best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    std::vector<std::size_t> near;
    for (std::size_t i = 0; i < states.size(); ++i) {
        const double distance = metric.SquaredDistance(states[i], query);
        if (distance < best_distance) {
            best_distance = distance;
            best = i;
        }
        if (distance <= radius * radius &&
            (std::isfinite(distance) || std::isinf(radius)))
            near.push_back(i);
    }
    if (index.Nearest(states.size(), state_at, query, metric) != best ||
        index.Near(states.size(), state_at, query, radius, metric) != near) {
        std::cerr << "index disagrees with linear scan at " << states.size()
                  << " states, radius " << radius << '\n';
        return false;
    }
    // Repeated nearest queries reuse the index without Eigen allocations.
    Eigen::internal::set_is_malloc_allowed(false);
    const auto repeated = index.Nearest(states.size(), state_at, query, metric);
    Eigen::internal::set_is_malloc_allowed(true);
    return repeated == best;
}

} // namespace

int main() {
    std::mt19937_64 generator(20260922);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (Eigen::Index dof : {1, 2, 7, 14}) {
        for (int topology : {0, 1, 2}) {
            Eigen::VectorXd weights(dof);
            std::vector<bool> continuous(static_cast<std::size_t>(dof));
            for (Eigen::Index axis = 0; axis < dof; ++axis) {
                weights[axis] = std::pow(10.0, 4.0 * uniform(generator));
                continuous[static_cast<std::size_t>(axis)] =
                    topology == 2 || (topology == 1 && axis % 2 == 0);
            }
            const JointSpaceMetric metric(weights, continuous);
            JointSpaceIndex index;
            std::vector<Eigen::VectorXd> states;
            for (std::size_t count = 1; count <= 2048; ++count) {
                Eigen::VectorXd state(dof);
                for (Eigen::Index axis = 0; axis < dof; ++axis)
                    state[axis] = 4.0 * kPi * uniform(generator);
                if (count % 17 == 0)
                    state = states.front();
                states.push_back(state);
                if (count > 140 && count % 31 != 0 && count != 2048)
                    continue;
                Eigen::VectorXd query(dof);
                for (Eigen::Index axis = 0; axis < dof; ++axis)
                    query[axis] = 4.0 * kPi * uniform(generator);
                const double radius =
                    std::sqrt(metric.SquaredDistance(states[count / 2], query));
                for (double boundary :
                     {0.0, radius, std::nextafter(radius, 0.0),
                      std::nextafter(
                          radius, std::numeric_limits<double>::infinity())}) {
                    if (!Check(index, states, query, metric, boundary))
                        return 1;
                }
                if (!Check(index, states, states.front(), metric, 0.0))
                    return 1;
            }
            // The planner swaps its two search trees, including their indices.
            JointSpaceIndex moved;
            std::swap(index, moved);
            if (!Check(moved, states, states.back(), metric,
                       std::numeric_limits<double>::infinity()))
                return 1;
        }
    }

    // Every point is equidistant, crossing leaves and the unindexed append
    // buffer. The earliest insertion must win, including at the periodic seam.
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(2);
    const std::vector<bool> continuous{true, false};
    JointSpaceMetric metric(weights, continuous);
    JointSpaceIndex ties;
    std::vector<Eigen::VectorXd> states;
    for (int i = 0; i < 513; ++i)
        states.push_back((Eigen::Vector2d() << (i % 2 ? -kPi : kPi),
                          (i % 4 < 2 ? -1.0 : 1.0))
                             .finished());
    const Eigen::VectorXd query = (Eigen::Vector2d() << kPi, 0.0).finished();
    if (!Check(ties, states, query, metric, 1.0))
        return 1;
    for (double weight : {1e-300, 1e300}) {
        weights.setConstant(weight);
        JointSpaceIndex extreme;
        for (double radius :
             {1e-200, 1e200, std::numeric_limits<double>::infinity()})
            if (!Check(extreme, states, query, metric, radius))
                return 1;
    }
    return 0;
}
