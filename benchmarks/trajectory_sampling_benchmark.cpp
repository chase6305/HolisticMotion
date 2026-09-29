#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

using namespace holistic_motion::robotics;
// Quadratic paths are not accepted by the stock DoubleS constructor. Exercise
// their report fallback with an explicit linear path-parameter clock instead.
template <class G> struct QuadraticProbe : TrajectoryBase<G> {
    explicit QuadraticProbe(const std::shared_ptr<PathBase<G>> &path) {
        this->path_ = path;
        this->dof_ = G::DoF;
        this->valid_ = path->IsValid();
        this->trajectory_pspline_ = std::make_shared<PSpline>();
        this->trajectory_pspline_->PushBack(
            std::make_shared<Polynomial>(
                Eigen::Vector4d(0., path->GetLength(), 0., 0.)),
            1.);
        this->max_velocity_ = this->max_acceleration_ = this->max_jerk_ =
            Eigen::VectorXd::Constant(G::DoF, 1e12);
    }
};

template <typename Group>
void Measure(std::size_t waypoint_count, int degree, double blend) {
    constexpr int DoF = Group::DoF;
    constexpr bool is_pose = std::is_same_v<Group, SE3d>;
    std::vector<Group> waypoints(waypoint_count);
    for (std::size_t i = 0; i < waypoint_count; ++i) {
        if constexpr (is_pose) {
            waypoints[i] = SE3d(Eigen::Vector3d(0.3 * std::sin(0.4 * i),
                                                0.3 * std::sin(0.4 * i + 0.3),
                                                0.3 * std::sin(0.4 * i + 0.6)),
                                SO3d(0.03 * std::sin(0.2 * i),
                                     0.02 * std::cos(0.4 * i), 0.01 * i));
        } else {
            for (int joint = 0; joint < DoF; ++joint)
                waypoints[i].Coeffs()[joint] =
                    0.3 * std::sin(0.4 * i + 0.3 * joint);
        }
    }
    auto path = std::make_shared<PathBezierCurve<Group>>(waypoints, degree,
                                                         false, blend);
    const Eigen::VectorXd limits = Eigen::VectorXd::Ones(DoF);
    auto constraints =
        std::make_shared<TrajectoryConstraints>(limits, limits, limits);
    std::unique_ptr<TrajectoryBase<Group>> owner;
    if (degree == 2)
        owner = std::make_unique<QuadraticProbe<Group>>(path);
    else
        owner = std::make_unique<TrajectoryDoubleS<Group>>(path, constraints);
    const auto &trajectory = *owner;
    if (!trajectory.IsValid())
        throw std::runtime_error("invalid benchmark trajectory");

    // Even the two-sample report also evaluates every internal phase endpoint.
    for (std::size_t samples : {2, 2001, 20001}) {
        std::vector<double> elapsed;
        elapsed.reserve(21);
        double checksum = 0.0;
        double previous_checksum = 0.0;
        for (int repeat = 0; repeat < 22; ++repeat) {
            const auto begin = std::chrono::steady_clock::now();
            const auto report = trajectory.GetConstraintReport(samples);
            const double microseconds =
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - begin)
                    .count();
            checksum = report.peak_velocity.sum() +
                       report.peak_acceleration.sum() + report.peak_jerk.sum();
            if (!std::isfinite(checksum) || !report.within_limits ||
                !report.velocity_continuous ||
                (degree == 5 && !report.acceleration_continuous))
                throw std::runtime_error("invalid benchmark report");
            if (repeat > 0) {
                if (checksum != previous_checksum)
                    throw std::runtime_error("unstable benchmark report");
                elapsed.push_back(microseconds);
            }
            previous_checksum = checksum;
        }
        std::sort(elapsed.begin(), elapsed.end());
        std::cout << (is_pose ? "SE3" : "Rn") << ',' << DoF << ','
                  << waypoint_count << ',' << degree << ',' << blend << ','
                  << (degree == 2 ? "linear_path" : "double_s") << ','
                  << samples << ',' << elapsed[elapsed.size() / 2] << ','
                  << checksum << '\n';
    }
}

template <class Group> void RunGroup() {
    for (const auto config :
         {std::pair<int, double>(2, 0.005), std::pair<int, double>(5, 0.0),
          std::pair<int, double>(5, 0.005)})
        for (std::size_t count : {4, 64})
            Measure<Group>(count, config.first, config.second);
}

template <unsigned D = 1> void RunDimensions() {
    RunGroup<Rn<double, D>>();
    if constexpr (D < 32)
        RunDimensions<D + 1>();
}

int main(int argc, char **argv) {
    bool all_dimensions = false;
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::cout
            << "Usage: trajectory_sampling_benchmark [--all-dimensions]\n"
               "Default: R2, R7, R20, R32 and SE3; --all-dimensions: R1-R32 "
               "and SE3.\n"
               "Emit CSV for quadratic, unblended and quintic report "
               "sampling.\n"
               "Quadratic paths use a synthetic linear clock, not Double-S.\n";
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--all-dimensions")
        all_dimensions = true;
    else if (argc != 1) {
        std::cerr
            << "Usage: trajectory_sampling_benchmark [--all-dimensions]\n";
        return 2;
    }
    holistic_motion::utility::SetVerbosityLevel(
        holistic_motion::utility::VerbosityLevel::Error);
    std::cout << std::setprecision(17)
              << "group,dof,waypoints,degree,blend,clock,samples,median_us,"
                 "checksum\n";
    try {
        if (all_dimensions)
            RunDimensions();
        else {
            RunGroup<R2d>();
            RunGroup<R7d>();
            RunGroup<R20d>();
            RunGroup<R32d>();
        }
        RunGroup<SE3d>();
    } catch (const std::exception &error) {
        std::cerr << "Benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
