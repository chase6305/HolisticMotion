#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "holistic_motion/trajectory/TrajectoryDoubleS.h"
#include "holistic_motion/trajectory/TrajectoryTrapezium.h"

using namespace holistic_motion::robotics;

template <int DoF> void Measure(std::size_t count, double blend, bool trapezoidal) {
    using Group = Rn<double, DoF>;
    std::vector<Group> points(count);
    for (std::size_t i = 0; i < count; ++i) {
        points[i].Coeffs()[0] = 0.1 * i;
        for (int joint = 1; joint < DoF; ++joint)
            points[i].Coeffs()[joint] = 0.2 * std::sin(0.7 * i + 0.3 * joint);
    }
    auto path = std::make_shared<PathBezierCurve<Group>>(points, 5, false, blend);
    const Eigen::VectorXd limits = Eigen::VectorXd::Ones(DoF);
    auto constraints = std::make_shared<TrajectoryConstraints>(limits, limits, limits);
    std::vector<double> timings;
    double duration = 0.0, checksum = 0.0;
    for (int repeat = -1; repeat < 21; ++repeat) {
        std::unique_ptr<TrajectoryBase<Group>> trajectory;
        const auto start = std::chrono::steady_clock::now();
        if (trapezoidal)
            trajectory =
                std::make_unique<TrajectoryTrapezium<Group>>(path, constraints);
        else
            trajectory = std::make_unique<TrajectoryDoubleS<Group>>(path, constraints);
        const double elapsed = std::chrono::duration<double, std::micro>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        if (!trajectory->IsValid())
            throw std::runtime_error("invalid benchmark trajectory");
        if (repeat >= 0 && trajectory->GetDuration() != duration)
            throw std::runtime_error("unstable trajectory duration");
        duration = trajectory->GetDuration();
        checksum = 0.0;
        for (int sample = 0; sample <= 100; ++sample) {
            const auto state = trajectory->GetState(duration * (sample / 100.0));
            checksum += state.position.Coeffs().sum() + state.velocity.Coeffs().sum() +
                        state.acceleration.Coeffs().sum() + state.jerk.Coeffs().sum();
        }
        if (repeat >= 0)
            timings.push_back(elapsed);
    }
    std::sort(timings.begin(), timings.end());
    std::cout << DoF << ',' << count << ',' << blend << ','
              << (trapezoidal ? "trapezoidal" : "double_s") << ','
              << timings[timings.size() / 2] << ',' << duration << ',' << checksum
              << '\n';
}

int main() {
    holistic_motion::utility::SetVerbosityLevel(
        holistic_motion::utility::VerbosityLevel::Warning);
    std::cout << std::setprecision(17)
              << "dof,waypoints,blend,profile,median_us,duration,checksum\n";
    for (std::size_t count : {4, 64})
        for (double blend : {0.0, 0.005})
            for (bool trapezoidal : {false, true}) {
                Measure<2>(count, blend, trapezoidal);
                Measure<7>(count, blend, trapezoidal);
                Measure<20>(count, blend, trapezoidal);
                Measure<32>(count, blend, trapezoidal);
            }
}
