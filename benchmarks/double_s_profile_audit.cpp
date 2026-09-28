#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <list>
#include <random>
#include <string_view>
#include <vector>

#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

using namespace holistic_motion::robotics;

namespace {
struct ProfileProbe : TrajectoryDoubleS<Rn<double, 2>> {
    using TrajectoryDoubleS::_ComputeDoubleSProfile;
};

bool ParseInteger(std::string_view text, std::uint64_t &value) {
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size();
}

struct Failure {
    std::uint64_t index;
    const char *reason;
    double scale, length, v0, v1, acceleration, jerk;
};

const char *CheckProfile(const std::list<TrajectorySeg> &phases, double scale,
                         double length, double initial_velocity, double final_velocity,
                         double acceleration, double jerk) {
    if (phases.size() != 8 || !std::isfinite(phases.back().timestamp) ||
        phases.back().timestamp <= 0.0)
        return "invalid_duration_or_layout";
    const auto &first = phases.front();
    const auto &last = phases.back();
    if (!std::isfinite(last.pos) || !std::isfinite(last.vel) ||
        !std::isfinite(last.acc) || !std::isfinite(last.jerk))
        return "nonfinite_endpoint";
    if (first.timestamp != 0.0 || first.pos != 0.0 ||
        std::abs((first.vel - initial_velocity) / scale) > 1e-10 ||
        std::abs(first.acc / scale) > 1e-10 * acceleration)
        return "initial_boundary";
    if (std::abs(phases.back().pos / scale - length) > 1e-9 * length ||
        std::abs((phases.back().vel - final_velocity) / scale) > 1e-10 ||
        std::abs(phases.back().acc / scale) > 1e-10 * acceleration)
        return "endpoint";
    for (auto it = phases.begin(), next = std::next(it); next != phases.end();
         ++it, ++next) {
        const long double dt = next->timestamp - it->timestamp;
        const long double q = it->pos / scale, v = it->vel / scale;
        const long double a = it->acc / scale, j = it->jerk / scale;
        // Subtracting stored timestamps quantizes a short phase. Its jerk
        // amplifies that clock uncertainty in the integrated acceleration.
        const long double clock_error =
            8 * std::numeric_limits<double>::epsilon() *
            std::max(std::abs(it->timestamp), std::abs(next->timestamp));
        if (!std::isfinite(dt) || dt < 0 || !std::isfinite(q) || !std::isfinite(v) ||
            !std::isfinite(a) || !std::isfinite(j))
            return "nonfinite_or_backwards_phase";
        if (std::abs(q + v * dt + a * dt * dt / 2 + j * dt * dt * dt / 6 -
                     next->pos / scale) > 1e-9L * length ||
            std::abs(v + a * dt + j * dt * dt / 2 - next->vel / scale) > 1e-10L ||
            std::abs(a + j * dt - next->acc / scale) >
                1e-10L * acceleration + std::abs(j) * clock_error)
            return "phase_integration";
        if (std::abs(a) > acceleration * (1 + 1e-10) ||
            std::abs(j) > jerk * (1 + 1e-10) || v < -1e-10 || v > 1 + 1e-10)
            return "derivative_limit";
        if (j != 0.0L) {
            const long double stationary = -a / j;
            if (stationary > 0 && stationary < dt) {
                const long double velocity =
                    v + a * stationary + j * stationary * stationary / 2;
                if (velocity < -1e-10L || velocity > 1 + 1e-10L)
                    return "interior_velocity_limit";
            }
        }
    }
    return nullptr;
}
} // namespace

int main(int argc, char **argv) {
    std::uint64_t cases = 2000000, seed = 20261010;
    if (argc > 3 || (argc > 1 && !ParseInteger(argv[1], cases)) ||
        (argc > 2 && !ParseInteger(argv[2], seed)) || cases == 0) {
        std::cerr
            << "Usage: double_s_profile_audit [positive_cases] [nonnegative_seed]\n";
        return 2;
    }
    holistic_motion::utility::SetVerbosityLevel(
        holistic_motion::utility::VerbosityLevel::Error);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::uint64_t passed = 0, rejected = 0, failed = 0;
    std::uint64_t start_reductions = 0, end_reductions = 0;
    std::vector<Failure> failures;
    std::size_t rejected_examples = 0, failed_examples = 0;
    const auto started = std::chrono::steady_clock::now();
    for (std::uint64_t trial = 0; trial < cases; ++trial) {
        const double scale = std::pow(10.0, -200 + 400 * uniform(rng));
        const double length = std::pow(10.0, -15 + 18 * uniform(rng));
        double v0 = uniform(rng), v1 = uniform(rng);
        const double acceleration = std::pow(10.0, -3 + 6 * uniform(rng));
        const double jerk = std::pow(10.0, -3 + 6 * uniform(rng));
        if (trial % 4 == 0)
            v0 = v1 = 0.0;
        else if (trial % 4 == 1)
            v1 = v0;
        double first = v0 * scale, last = v1 * scale;
        std::list<TrajectorySeg> phases;
        const bool success = ProfileProbe::_ComputeDoubleSProfile(
            0.0, length * scale, first, last, scale, acceleration * scale, jerk * scale,
            0.0, phases, 0, false);
        const char *reason = "rejected";
        if (phases.empty()) {
            ++rejected;
        } else {
            // A populated false result requests upstream start-speed
            // backtracking. Audit its adjusted boundary conditions explicitly.
            start_reductions += !success;
            end_reductions += last != v1 * scale;
            reason =
                CheckProfile(phases, scale, length, first, last, acceleration, jerk);
            if (reason)
                ++failed;
            else
                ++passed;
        }
        // Rejections must not crowd out examples of invalid accepted
        // profiles. Retain the first ten of each category in trial order.
        auto &examples = phases.empty() ? rejected_examples : failed_examples;
        if (reason && examples < 10) {
            failures.push_back(
                {trial, reason, scale, length, v0, v1, acceleration, jerk});
            ++examples;
        }
    }
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    std::cout << std::setprecision(17) << "{\n  \"seed\": " << seed
              << ",\n  \"cases\": " << cases << ",\n  \"passed\": " << passed
              << ",\n  \"rejected\": " << rejected << ",\n  \"failed\": " << failed
              << ",\n  \"start_speed_reductions\": " << start_reductions
              << ",\n  \"end_speed_reductions\": " << end_reductions
              << ",\n  \"elapsed_seconds\": " << elapsed
              << ",\n  \"first_failures\": [";
    for (std::size_t i = 0; i < failures.size(); ++i) {
        const auto &failure = failures[i];
        std::cout << (i ? "," : "") << "\n    {\"case_index\": " << failure.index
                  << ", \"kind\": \"" << failure.reason
                  << "\", \"scale\": " << failure.scale
                  << ", \"length\": " << failure.length << ", \"v0\": " << failure.v0
                  << ", \"v1\": " << failure.v1
                  << ", \"max_acceleration\": " << failure.acceleration
                  << ", \"max_jerk\": " << failure.jerk << "}";
    }
    std::cout << "\n  ]\n}\n";
    return rejected || failed ? 1 : 0;
}
