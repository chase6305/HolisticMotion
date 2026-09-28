#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "holistic_motion/trajectory/TrajectoryDoubleS.h"
#include "holistic_motion/trajectory/TrajectoryTrapezoidal.h"

using namespace holistic_motion::robotics;

void CheckHalfTurnBlends(double scale, double offset) {
    const std::array<Eigen::Vector3d, 4> translations{
        Eigen::Vector3d(-155.94625516899751, 110.77719396050814, -23.446442903611267),
        Eigen::Vector3d(-252.31962303244603, 49.191887809643575, -9.5382466032571536),
        Eigen::Vector3d(-305.13427523070965, -9.1072179506613349, -7.3344952083385095),
        Eigen::Vector3d(-341.76599222217521, -84.456947826497185, -2.9312500968858846)};
    std::vector<SE3d> poses;
    for (std::size_t i = 0; i < translations.size(); ++i)
        poses.emplace_back(scale * translations[i],
                           SO3d(0, 0, i % 2 ? 0 : std::acos(-1.0) + offset));
    auto path = std::make_shared<PathBezierCurve<SE3d>>(poses, 5, true, 5 * scale);
    if (!path->IsValid())
        throw std::runtime_error("half-turn blend is invalid");
    const auto segments = path->GetPathSegments();
    for (std::size_t i = 1; i < segments.size(); ++i) {
        const auto &left = segments[i - 1];
        const auto &right = segments[i];
        const auto incoming =
            left->GetTangent(left->GetStartParameter() + left->GetLength());
        const auto outgoing = right->GetTangent(right->GetStartParameter());
        if (!incoming.Coeffs().allFinite() || !outgoing.Coeffs().allFinite() ||
            (incoming - outgoing).Coeffs().norm() > 1e-10)
            throw std::runtime_error("half-turn blend has a discontinuous tangent");
    }
    const Eigen::VectorXd velocity = Eigen::VectorXd::Constant(6, 1000 * scale);
    auto constraints =
        std::make_shared<TrajectoryConstraints>(velocity, 2 * velocity, 5 * velocity);
    TrajectoryDoubleS<SE3d> double_s(path, constraints);
    TrajectoryTrapezoidal<SE3d> trapezoidal(path, constraints);
    for (TrajectoryBase<SE3d> *trajectory :
         std::array<TrajectoryBase<SE3d> *, 2>{&double_s, &trapezoidal}) {
        if (!trajectory->IsValid())
            throw std::runtime_error("half-turn trajectory is invalid");
        for (double slowdown : {1.0, 1.7}) {
            if (slowdown != 1.0 &&
                !trajectory->SetMinimumDuration(trajectory->GetDuration() * slowdown))
                throw std::runtime_error("cannot slow half-turn trajectory");
            const auto report = trajectory->GetConstraintReport(257);
            if (!report.within_limits || !report.velocity_continuous ||
                (trajectory == &double_s && !report.acceleration_continuous))
                throw std::runtime_error(
                    "half-turn trajectory lost continuity or limits");
        }
    }
}

int main() {
    try {
        for (double scale : {0.01, 1.0, 100.0})
            for (double offset : {-1e-9, 1e-9})
                CheckHalfTurnBlends(scale, offset);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
