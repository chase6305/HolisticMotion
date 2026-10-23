#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include "../src/planning/JointPositionContext.h"
#include "AllocationCounter.h"

using holistic_motion::robotics::JointCoordinate;
using holistic_motion::robotics::JointPositionMap;
using holistic_motion::robotics::planning::detail::JointPositionContext;
using Callback = std::function<double(const Eigen::VectorXd &)>;

#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
Callback
CopyCallback(const Callback &callback) {
    return callback;
}

template <class F> void Measure(const char *operation, int dof, F call) {
    hm_benchmark::StartCounting();
    double checksum = 0;
    for (int query = 0; query < 128; ++query)
        checksum += call();
    const double allocations = double(hm_benchmark::StopCounting()) / 128;
    std::vector<double> times;
    constexpr int queries = 32768;
    for (int sample = 0; sample < 8; ++sample) {
        const auto start = std::chrono::steady_clock::now();
        for (int query = 0; query < queries; ++query)
            checksum += call();
        if (sample)
            times.push_back(std::chrono::duration<double, std::nano>(
                                std::chrono::steady_clock::now() - start)
                                .count() /
                            queries);
    }
    std::sort(times.begin(), times.end());
    std::cout << operation << ',' << dof << ',' << times[times.size() / 2] << ','
              << allocations << ',' << hm_benchmark::allocation_kind << ',' << checksum
              << '\n';
}

int main() {
    std::cout << std::setprecision(17)
              << "operation,dof,median_ns,allocations,allocation_kind,checksum\n";
    for (int dof : {2, 7, 64, 256}) {
        std::vector<JointCoordinate> joints;
        for (int index = 0; index < dof; ++index)
            joints.push_back({"robot_model_joint_number_" + std::to_string(index),
                              index, 1, index, 1});
        JointPositionMap mapping(dof, dof, std::move(joints));
        const Eigen::VectorXd reference = Eigen::VectorXd::Zero(dof);
        const Callback legacy = [mapping, reference](const Eigen::VectorXd &positions) {
            return mapping.ConfigurationWithJointPositions(reference, positions).sum();
        };
        const auto prepared =
            std::make_shared<const JointPositionContext>(mapping, reference);
        const Callback snapshot = [prepared](const Eigen::VectorXd &positions) {
            return prepared->Configuration(positions).sum();
        };
        Measure("legacy_snapshot", dof,
                [&] { return double(bool(CopyCallback(legacy))); });
        Measure("prepared_snapshot", dof,
                [&] { return double(bool(CopyCallback(snapshot))); });
        Measure("legacy_query", dof, [&] { return legacy(reference); });
        Measure("prepared_query", dof, [&] { return snapshot(reference); });
    }
}
