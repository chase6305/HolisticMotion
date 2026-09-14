#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "holistic_motion/trajectory/PathBezierCurve.h"

using namespace holistic_motion::robotics;

template <int DoF> void Measure(std::size_t count) {
    using Group = Rn<double, DoF>;
    std::vector<Group> points(count);
    for (std::size_t index = 0; index < count; ++index)
        for (int joint = 0; joint < DoF; ++joint)
            points[index].Coeffs()[joint] = 0.3 * std::sin(0.04 * index + 0.3 * joint);
    std::vector<double> elapsed;
    double checksum = 0.0;
    for (int repeat = 0; repeat < 31; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        PathBezierCurve<Group> path(points, 5, false, 0.0);
        if (!path.IsValid())
            throw std::runtime_error("invalid benchmark path");
        const auto end = std::chrono::steady_clock::now();
        checksum += path.GetLength();
        if (repeat > 0)
            elapsed.push_back(
                std::chrono::duration<double, std::micro>(end - begin).count());
    }
    std::sort(elapsed.begin(), elapsed.end());
    std::cout << DoF << ',' << count << ',' << std::setprecision(3)
              << elapsed[elapsed.size() / 2] << ',' << std::setprecision(12) << checksum
              << '\n';
}

int main() {
    holistic_motion::utility::SetVerbosityLevel(
        holistic_motion::utility::VerbosityLevel::Warning);
    std::cout << std::fixed << "dof,waypoints,median_us,checksum\n";
    for (std::size_t count : {32, 512, 4096}) {
        Measure<2>(count);
        Measure<7>(count);
    }
}
