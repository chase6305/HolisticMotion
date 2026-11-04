#include "holistic_motion/manif/Rn.h"
#include "holistic_motion/manif/SE3.h"
#include "holistic_motion/manif/algorithms/average.h"

#include <array>
#include <list>
#include <vector>

using namespace holistic_motion::robotics;

namespace {
template <class Group> bool CheckMeans(const std::vector<Group> &points, const Group &expected) {
    const std::array<Group, 4> results{AverageBiinvariant(points), Average(points),
                                       AverageFrechetLeft(points), AverageFrechetRight(points)};
    for (const auto &result : results)
        if (!result.Coeffs().allFinite() || !result.IsApprox(expected, 1e-10))
            return false;
    return true;
}
} // namespace

int main() {
    using Point = Rn<double, 2>;
    std::vector<Point> points(3);
    points[0].Coeffs() << 1, -2;
    points[1].Coeffs() << 3, 4;
    points[2].Coeffs() << -1, 1;
    Point expected;
    expected.Coeffs() << 1, 1;
    if (!CheckMeans(points, expected))
        return 1;
    const std::list<Point> linked(points.begin(), points.end());
    if (!Average(linked).IsApprox(expected, 1e-14) ||
        !Average(points, 100.0).IsApprox(points.front(), 0.0))
        return 2;

    for (double offset : {0.0, 2.9}) {
        std::vector<SO3<double>> rotations;
        for (double angle : {-.1, .1, .3})
            rotations.emplace_back(Eigen::AngleAxisd(offset + angle, Eigen::Vector3d::UnitZ()));
        if (!CheckMeans(rotations,
                        SO3<double>(Eigen::AngleAxisd(offset + .1, Eigen::Vector3d::UnitZ()))))
            return 3;
    }
    std::vector<SE3<double>> poses;
    const SO3<double> rotation(.1, -.2, .3);
    for (const auto &point : points)
        poses.emplace_back(Eigen::Vector3d(point.Coeffs()[0], point.Coeffs()[1], .5), rotation);
    if (!CheckMeans(poses, SE3<double>(Eigen::Vector3d(1, 1, .5), rotation)))
        return 4;

    const SE3d center(Eigen::Vector3d(.3, -.2, .1), rotation);
    std::vector<SE3d> symmetric;
    // Each pure translation/rotation pair balances at center; different axes
    // give a point set whose motions do not commute.
    for (int axis = 0; axis < SE3d::DoF; ++axis) {
        SE3Tangentd offset = SE3Tangentd::ZeroHelper();
        offset.Coeffs()[axis] = .2;
        symmetric.push_back(center.Rplus(offset));
        symmetric.push_back(center.Rplus(-offset));
    }
    const std::array<SE3d, 4> symmetric_means{
        AverageBiinvariant(symmetric, 1e-24, 50), Average(symmetric, 1e-24, 50),
        AverageFrechetLeft(symmetric, 1e-24, 50), AverageFrechetRight(symmetric, 1e-24, 50)};
    for (const auto &mean : symmetric_means)
        if (!mean.IsApprox(center, 1e-10))
            return 7;

    const std::vector<Point> single{points.front()}, empty;
    if (!CheckMeans(single, points.front()))
        return 5;
    for (int kind = 0; kind < 4; ++kind) {
        try {
            if (kind == 0)
                AverageBiinvariant(empty);
            if (kind == 1)
                Average(empty);
            if (kind == 2)
                AverageFrechetLeft(empty);
            if (kind == 3)
                AverageFrechetRight(empty);
            return 6;
        } catch (const std::exception &) {
        }
    }
    return 0;
}
