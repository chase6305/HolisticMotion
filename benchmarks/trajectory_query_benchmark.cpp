#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

using namespace holistic_motion::robotics;

template <int DoF>
void Measure(std::size_t waypoint_count) {
    using Group = Rn<double, DoF>;
    std::vector<Group> waypoints(waypoint_count);
    for (std::size_t i = 0; i < waypoint_count; ++i) {
        for (int joint = 0; joint < DoF; ++joint) {
            waypoints[i].Coeffs()[joint] =
                0.3 * std::sin(0.4 * i + 0.3 * joint);
        }
    }
    auto path =
        std::make_shared<PathBezierCurve<Group>>(waypoints, 5, false, 0.005);
    const Eigen::VectorXd limits = Eigen::VectorXd::Ones(DoF);
    auto constraints =
        std::make_shared<TrajectoryConstraints>(limits, limits, limits);
    TrajectoryDoubleS<Group> trajectory(path, constraints);
    if (!trajectory.IsValid())
        throw std::runtime_error("invalid benchmark trajectory");
    constexpr int samples = 20000;
    for (int mode = 0; mode < 2; ++mode) {
        std::vector<double> elapsed;
        double checksum = 0.0;
        for (int repeat = 0; repeat < 8; ++repeat) {
            checksum = 0.0;
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < samples; ++i) {
                const double time =
                    trajectory.GetDuration() * i / (samples - 1.0);
                if (mode == 0) {
                    const auto state = trajectory.GetState(time);
                    checksum += state.position.Coeffs().sum() +
                                state.velocity.Coeffs().sum() +
                                state.acceleration.Coeffs().sum() +
                                state.jerk.Coeffs().sum();
                } else {
                    checksum +=
                        trajectory.GetPosition(time).Coeffs().sum() +
                        trajectory.GetVelocity(time).Coeffs().sum() +
                        trajectory.GetAcceleration(time).Coeffs().sum() +
                        trajectory.GetJerk(time).Coeffs().sum();
                }
            }
            const double ns = std::chrono::duration<double, std::nano>(
                                  std::chrono::steady_clock::now() - begin)
                                  .count() /
                              samples;
            if (repeat > 0) elapsed.push_back(ns);
        }
        std::sort(elapsed.begin(), elapsed.end());
        std::cout << DoF << ',' << waypoint_count << ','
                  << (mode ? "separate" : "state") << ','
                  << std::setprecision(3) << elapsed[elapsed.size() / 2] << ','
                  << std::setprecision(12) << trajectory.GetDuration() << ','
                  << checksum << '\n';
    }
}

int main() {
    std::cout << std::fixed
              << "dof,waypoints,query,median_ns,duration,checksum\n";
    for (std::size_t count : {4, 64}) {
        Measure<2>(count);
        Measure<7>(count);
    }
}
