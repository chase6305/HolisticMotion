#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "holistic_motion/trajectory/TrajectoryTrapezium.h"

using namespace holistic_motion::robotics;

// Expose only the preliminary sampler; full construction is timed separately.
// Their ratio is diagnostic, not an instrumented share of the constructor.
template <int N> struct Probe : TrajectoryTrapezium<Rn<double, N>> {
    Probe() : TrajectoryTrapezium<Rn<double, N>>(nullptr, nullptr) { this->dof_ = N; }
    using TrajectoryTrapezium<Rn<double, N>>::_ComputeSegmentMaxSVel;
};
template <int N> void Measure(int count, double scale) {
    using G = Rn<double, N>;
    std::vector<G> points(count);
    for (int i = 0; i < count; ++i) {
        points[i].Coeffs()[0] = .1 * i * scale;
        for (int j = 1; j < N; ++j)
            points[i].Coeffs()[j] = .2 * std::sin(.7 * i + .3 * j) * scale;
    }
    auto path = std::make_shared<PathBezierCurve<G>>(points, 5, false, .005 * scale);
    auto segments = path->GetPathSegments();
    Eigen::VectorXd limits = Eigen::VectorXd::Ones(N) * scale;
    auto constraints = std::make_shared<TrajectoryConstraints>(limits, limits, limits);
    Probe<N> probe;
    std::vector<double> cap, full;
    double checksum = 0., duration = 0.;
    int curves = 0;
    for (const auto &s : segments)
        curves += s->GetPathSegType() != PathSegType::LinearSeg;
    double previous_checksum = 0.0, previous_duration = 0.0;
    for (int repeat = 0; repeat < 22; ++repeat) {
        auto start = std::chrono::steady_clock::now();
        checksum = 0.;
        for (const auto &s : segments)
            if (s->GetPathSegType() != PathSegType::LinearSeg)
                checksum += probe._ComputeSegmentMaxSVel(s, limits, limits, limits);
        double cap_us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - start)
                            .count();
        start = std::chrono::steady_clock::now();
        TrajectoryTrapezium<G> t(path, constraints);
        double full_us = std::chrono::duration<double, std::micro>(
                             std::chrono::steady_clock::now() - start)
                             .count();
        if (!t.IsValid())
            throw std::runtime_error("invalid trajectory");
        duration = t.GetDuration();
        if (!std::isfinite(checksum) || (repeat && (checksum != previous_checksum ||
                                                    duration != previous_duration)))
            throw std::runtime_error("unstable benchmark result");
        previous_checksum = checksum;
        previous_duration = duration;
        if (repeat) {
            cap.push_back(cap_us);
            full.push_back(full_us);
        }
    }
    std::sort(cap.begin(), cap.end());
    std::sort(full.begin(), full.end());
    std::cout << N << ',' << count << ',' << scale << ',' << curves << ',' << cap[10]
              << ',' << full[10] << ',' << checksum << ',' << duration << '\n';
}
int main() {
    holistic_motion::utility::SetVerbosityLevel(
        holistic_motion::utility::VerbosityLevel::Error);
    std::cout
        << std::setprecision(17)
        << "dof,waypoints,scale,curves,cap_us,construction_us,cap_checksum,duration\n";
    Measure<2>(64, 1.);
    Measure<7>(64, 1.);
    Measure<20>(64, 1.);
    Measure<32>(64, 1.);
    Measure<2>(4, 1e4);
    Measure<7>(4, 1e4);
    Measure<20>(4, 1e4);
    Measure<32>(4, 1e4);
}
