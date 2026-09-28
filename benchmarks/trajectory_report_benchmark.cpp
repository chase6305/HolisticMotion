#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

using namespace holistic_motion::robotics;

template <typename Group> void Measure(std::size_t waypoint_count) {
    constexpr int DoF = Group::DoF;
    constexpr bool is_pose = std::is_same_v<Group, SE3d>;
    std::vector<Group> waypoints(waypoint_count);
    for (std::size_t i = 0; i < waypoint_count; ++i) {
        if constexpr (is_pose) {
            waypoints[i] = SE3d(
                Eigen::Vector3d(0.3 * std::sin(0.4 * i), 0.3 * std::sin(0.4 * i + 0.3),
                                0.3 * std::sin(0.4 * i + 0.6)),
                SO3d(0.03 * std::sin(0.2 * i), 0.02 * std::cos(0.4 * i), 0.01 * i));
        } else {
            for (int joint = 0; joint < DoF; ++joint)
                waypoints[i].Coeffs()[joint] = 0.3 * std::sin(0.4 * i + 0.3 * joint);
        }
    }
    auto path = std::make_shared<PathBezierCurve<Group>>(waypoints, 5, false, 0.005);
    const Eigen::VectorXd limits = Eigen::VectorXd::Ones(DoF);
    auto constraints = std::make_shared<TrajectoryConstraints>(limits, limits, limits);
    TrajectoryDoubleS<Group> trajectory(path, constraints);
    if (!trajectory.IsValid())
        throw std::runtime_error("invalid benchmark trajectory");

    // Even the two-sample report also evaluates every internal phase endpoint.
    for (std::size_t samples : {2, 2001, 20001}) {
        std::vector<double> elapsed;
        double checksum = 0.0;
        for (int repeat = 0; repeat < 22; ++repeat) {
            const auto begin = std::chrono::steady_clock::now();
            const auto report = trajectory.GetConstraintReport(samples);
            const double microseconds = std::chrono::duration<double, std::micro>(
                                            std::chrono::steady_clock::now() - begin)
                                            .count();
            checksum = report.peak_velocity.sum() + report.peak_acceleration.sum() +
                       report.peak_jerk.sum();
            if (!std::isfinite(checksum) || !report.within_limits ||
                !report.velocity_continuous || !report.acceleration_continuous)
                throw std::runtime_error("invalid benchmark report");
            if (repeat > 0)
                elapsed.push_back(microseconds);
        }
        std::sort(elapsed.begin(), elapsed.end());
        std::cout << (is_pose ? "SE3" : "Rn") << ',' << DoF << ',' << waypoint_count
                  << ',' << samples << ',' << elapsed[elapsed.size() / 2] << ','
                  << checksum << '\n';
    }
}

int main() {
    holistic_motion::utility::SetVerbosityLevel(
        holistic_motion::utility::VerbosityLevel::Warning);
    std::cout << std::setprecision(17)
              << "group,dof,waypoints,samples,median_us,checksum\n";
    for (std::size_t count : {4, 64}) {
        Measure<Rn<double, 2>>(count);
        Measure<Rn<double, 7>>(count);
        Measure<Rn<double, 20>>(count);
        Measure<Rn<double, 32>>(count);
        Measure<SE3d>(count);
    }
}
