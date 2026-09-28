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
    const auto report = trajectory.GetConstraintReport(2001);
    constexpr int samples = 20000;
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<double> elapsed;
        double checksum = 0.0;
        for (int repeat = 0; repeat < 8; ++repeat) {
            checksum = 0.0;
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < samples; ++i) {
                const double time = trajectory.GetDuration() * i / (samples - 1.0);
                if (mode == 0) {
                    const auto state = trajectory.GetState(time);
                    checksum +=
                        state.position.Coeffs().sum() + state.velocity.Coeffs().sum() +
                        state.acceleration.Coeffs().sum() + state.jerk.Coeffs().sum();
                } else {
                    checksum += trajectory.GetPosition(time).Coeffs().sum() +
                                trajectory.GetVelocity(time).Coeffs().sum() +
                                trajectory.GetAcceleration(time).Coeffs().sum() +
                                trajectory.GetJerk(time).Coeffs().sum();
                }
            }
            const double ns = std::chrono::duration<double, std::nano>(
                                  std::chrono::steady_clock::now() - begin)
                                  .count() /
                              samples;
            if (repeat > 0)
                elapsed.push_back(ns);
        }
        std::sort(elapsed.begin(), elapsed.end());
        std::cout << (is_pose ? "SE3" : "Rn") << ',' << DoF << ',' << waypoint_count
                  << ',' << (mode ? "separate" : "state") << ',' << std::setprecision(3)
                  << elapsed[elapsed.size() / 2] << ',' << std::setprecision(12)
                  << trajectory.GetDuration() << ',' << checksum << ','
                  << report.within_limits << ',' << report.velocity_continuous << ','
                  << report.acceleration_continuous << '\n';
    }
}

int main() {
    std::cout << std::fixed
              << "group,dof,waypoints,query,median_ns,duration,checksum,within_limits,"
                 "velocity_continuous,acceleration_continuous\n";
    for (std::size_t count : {4, 64}) {
        Measure<Rn<double, 2>>(count);
        Measure<Rn<double, 7>>(count);
        Measure<Rn<double, 20>>(count);
        Measure<Rn<double, 32>>(count);
        Measure<SE3d>(count);
    }
}
