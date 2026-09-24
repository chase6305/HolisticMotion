#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "JointSpaceTopology.h"

namespace holistic_motion::robotics::planning::detail {

// An append-only index of stable state indices, never pointers into the tree.
// Split only bounded coordinates: their ordinary distance is a lower bound on
// the full weighted metric even when other coordinates live on circles. This
// avoids introducing a second implementation of periodic distance semantics.
// Small trees and spaces containing only periodic coordinates use linear scans.
class JointSpaceIndex {
  public:
    template <class StateAt>
    std::size_t Nearest(std::size_t count, const StateAt &state_at,
                        const Eigen::VectorXd &query,
                        const JointSpaceMetric &metric) {
        Refresh(count, state_at, metric);
        std::size_t best = 0;
        double distance = std::numeric_limits<double>::infinity();
        if (nodes_.empty()) {
            for (std::size_t i = 0; i < count; ++i) {
                const double candidate =
                    metric.SquaredDistance(state_at(i), query);
                if (candidate < distance) {
                    distance = candidate;
                    best = i;
                }
            }
            return best;
        }
        // Recent states often lie near the current target. Search the append
        // buffer first to tighten the bound used by the balanced tree.
        for (std::size_t i = indexed_count_; i < count; ++i)
            Consider(i, state_at, query, metric, best, distance);
        if (!nodes_.empty())
            SearchNearest(0, state_at, query, metric, best, distance);
        return best;
    }

    template <class StateAt>
    std::vector<std::size_t> Near(std::size_t count, const StateAt &state_at,
                                  const Eigen::VectorXd &query, double radius,
                                  const JointSpaceMetric &metric) {
        std::vector<std::size_t> result;
        const double squared_radius = radius * radius;
        const auto collect = [&](std::size_t index) {
            const double distance =
                metric.SquaredDistance(state_at(index), query);
            if (distance <= squared_radius &&
                (std::isfinite(distance) || std::isinf(radius)))
                result.push_back(index);
        };
        // Low-dimensional Euclidean radius searches return many neighbors.
        // Their cheap linear metric avoids sorting a large indexed result.
        if (!metric.HasContinuous() && metric.Dimension() <= 3) {
            for (std::size_t i = 0; i < count; ++i)
                collect(i);
            return result;
        }
        Refresh(count, state_at, metric);
        if (!nodes_.empty())
            SearchNear(0, query, squared_radius, metric, collect);
        for (std::size_t i = indexed_count_; i < count; ++i)
            collect(i);
        // RRT* visits candidates in insertion order. Keep parent selection,
        // rewiring, and all equal-distance ties identical to the linear scan.
        if (!nodes_.empty())
            std::sort(result.begin(), result.end());
        return result;
    }

  private:
    struct Node {
        std::size_t begin;
        std::size_t end;
        std::size_t right{0};
        Eigen::Index axis{-1};
        double split{0.0};
    };

    static constexpr std::size_t kMinimumSize = 128;
    static constexpr std::size_t kLeafSize = 16;
    std::vector<Node> nodes_;
    std::vector<std::size_t> order_;
    std::vector<Eigen::Index> bounded_axes_;
    std::size_t indexed_count_{0};
    bool initialized_{false};

    template <class StateAt>
    void Refresh(std::size_t count, const StateAt &state_at,
                 const JointSpaceMetric &metric) {
        // In high dimensions the split-plane bounds reject too few leaves to
        // repay traversal and rebuilding. Keep the contiguous insertion-order
        // scan there; benchmarks cover both sides of this dispatch boundary.
        if (count < kMinimumSize || metric.Dimension() > 8)
            return;
        if (!initialized_) {
            for (Eigen::Index axis = 0; axis < metric.Dimension(); ++axis)
                if (!metric.IsContinuous(axis))
                    bounded_axes_.push_back(axis);
            initialized_ = true;
        }
        if (bounded_axes_.empty() ||
            count - indexed_count_ <
                std::max<std::size_t>(64, indexed_count_ / 8))
            return;
        order_.resize(count);
        std::iota(order_.begin(), order_.end(), 0);
        nodes_.clear();
        nodes_.reserve(4 * (count / kLeafSize));
        Build(0, count, state_at, metric);
        indexed_count_ = count;
    }

    template <class StateAt>
    std::size_t Build(std::size_t begin, std::size_t end,
                      const StateAt &state_at, const JointSpaceMetric &metric) {
        const std::size_t node = nodes_.size();
        nodes_.push_back({begin, end});
        if (end - begin <= kLeafSize)
            return node;

        Eigen::Index selected = bounded_axes_.front();
        double widest = -1.0;
        for (Eigen::Index axis : bounded_axes_) {
            double lower = state_at(order_[begin])[axis];
            double upper = lower;
            for (std::size_t i = begin + 1; i < end; ++i) {
                const double value = state_at(order_[i])[axis];
                lower = std::min(lower, value);
                upper = std::max(upper, value);
            }
            const double width =
                metric.SquaredAxisDistance(upper - lower, axis);
            if (width > widest) {
                widest = width;
                selected = axis;
            }
        }
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(order_.begin() + begin, order_.begin() + middle,
                         order_.begin() + end,
                         [&](std::size_t first, std::size_t second) {
                             const double a = state_at(first)[selected];
                             const double b = state_at(second)[selected];
                             return a < b || (a == b && first < second);
                         });
        nodes_[node].axis = selected;
        nodes_[node].split = state_at(order_[middle])[selected];
        Build(begin, middle, state_at, metric);
        const std::size_t right = Build(middle, end, state_at, metric);
        nodes_[node].right = right;
        return node;
    }

    template <class StateAt>
    static void Consider(std::size_t index, const StateAt &state_at,
                         const Eigen::VectorXd &query,
                         const JointSpaceMetric &metric, std::size_t &best,
                         double &best_distance) {
        const double distance = metric.SquaredDistance(state_at(index), query);
        if (distance < best_distance ||
            (distance == best_distance && index < best)) {
            best_distance = distance;
            best = index;
        }
    }

    template <class StateAt>
    void SearchNearest(std::size_t index, const StateAt &state_at,
                       const Eigen::VectorXd &query,
                       const JointSpaceMetric &metric, std::size_t &best,
                       double &best_distance) const {
        const auto &node = nodes_[index];
        if (node.axis < 0) {
            for (std::size_t i = node.begin; i < node.end; ++i)
                Consider(order_[i], state_at, query, metric, best,
                         best_distance);
            return;
        }
        const double delta = query[node.axis] - node.split;
        const std::size_t close = delta < 0.0 ? index + 1 : node.right;
        const std::size_t far = delta < 0.0 ? node.right : index + 1;
        SearchNearest(close, state_at, query, metric, best, best_distance);
        if (!(metric.SquaredAxisDistance(delta, node.axis) > best_distance))
            SearchNearest(far, state_at, query, metric, best, best_distance);
    }

    template <class Collect>
    void SearchNear(std::size_t index, const Eigen::VectorXd &query,
                    double squared_radius, const JointSpaceMetric &metric,
                    const Collect &collect) const {
        const auto &node = nodes_[index];
        if (node.axis < 0) {
            for (std::size_t i = node.begin; i < node.end; ++i)
                collect(order_[i]);
            return;
        }
        const double delta = query[node.axis] - node.split;
        const std::size_t close = delta < 0.0 ? index + 1 : node.right;
        const std::size_t far = delta < 0.0 ? node.right : index + 1;
        SearchNear(close, query, squared_radius, metric, collect);
        if (!(metric.SquaredAxisDistance(delta, node.axis) > squared_radius))
            SearchNear(far, query, squared_radius, metric, collect);
    }
};

} // namespace holistic_motion::robotics::planning::detail
