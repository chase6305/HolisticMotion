#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace holistic_motion::robotics::detail {

// A balanced, immutable partition of candidate coordinates. Only the retained
// flags and subtree counts change during the deterministic greedy pass. Query
// pruning follows coordinate differences directly, without rounded q +/- eps
// boxes or a nontransitive approximate ordering relation.
class RetainedSolutionIndex {
  public:
    explicit RetainedSolutionIndex(const std::vector<Eigen::VectorXd> &points)
        : points_(points), nodes_(points.size()) {
        std::vector<std::size_t> order(points.size());
        std::iota(order.begin(), order.end(), 0);
        root_ = Build(order, 0, order.size(), kNone);
    }

    bool HasNeighbor(const Eigen::VectorXd &point, double tolerance) const {
        return HasNeighbor(root_, point, tolerance);
    }

    void Retain(std::size_t point) {
        nodes_[point].retained = true;
        for (std::size_t node = point; node != kNone;
             node = nodes_[node].parent)
            ++nodes_[node].count;
    }

  private:
    static constexpr std::size_t kNone =
        std::numeric_limits<std::size_t>::max();
    struct Node {
        std::size_t left{kNone};
        std::size_t right{kNone};
        std::size_t parent{kNone};
        Eigen::Index axis{0};
        int count{0};
        bool retained{false};
    };

    std::size_t Build(std::vector<std::size_t> &order, std::size_t begin,
                      std::size_t end, std::size_t parent) {
        if (begin == end)
            return kNone;
        Eigen::Index axis = 0;
        long double widest = -1.0L;
        for (Eigen::Index joint = 0; joint < points_.front().size(); ++joint) {
            double lower = points_[order[begin]][joint];
            double upper = lower;
            for (std::size_t item = begin + 1; item < end; ++item) {
                lower = std::min(lower, points_[order[item]][joint]);
                upper = std::max(upper, points_[order[item]][joint]);
            }
            const long double width = static_cast<long double>(upper) - lower;
            if (width > widest) {
                widest = width;
                axis = joint;
            }
        }
        const std::size_t middle = begin + (end - begin) / 2;
        std::nth_element(order.begin() + begin, order.begin() + middle,
                         order.begin() + end,
                         [&](std::size_t first, std::size_t second) {
                             const double lhs = points_[first][axis];
                             const double rhs = points_[second][axis];
                             return lhs == rhs ? first < second : lhs < rhs;
                         });
        const std::size_t index = order[middle];
        auto &node = nodes_[index];
        node.axis = axis;
        node.parent = parent;
        node.left = Build(order, begin, middle, index);
        node.right = Build(order, middle + 1, end, index);
        return index;
    }

    bool HasNeighbor(std::size_t index, const Eigen::VectorXd &point,
                     double tolerance) const {
        if (index == kNone || nodes_[index].count == 0)
            return false;
        const auto &node = nodes_[index];
        const auto &pivot = points_[index];
        if (node.retained) {
            bool near = true;
            for (Eigen::Index joint = 0; joint < point.size(); ++joint) {
                if (std::abs(point[joint] - pivot[joint]) >= tolerance) {
                    near = false;
                    break;
                }
            }
            if (near)
                return true;
        }
        const double delta = point[node.axis] - pivot[node.axis];
        const std::size_t near = delta < 0.0 ? node.left : node.right;
        const std::size_t far = delta < 0.0 ? node.right : node.left;
        return HasNeighbor(near, point, tolerance) ||
               (std::abs(delta) < tolerance &&
                HasNeighbor(far, point, tolerance));
    }

    const std::vector<Eigen::VectorXd> &points_;
    std::vector<Node> nodes_;
    std::size_t root_{kNone};
};

} // namespace holistic_motion::robotics::detail
