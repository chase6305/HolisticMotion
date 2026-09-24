#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "holistic_motion/trajectory/TrajectoryDoubleS.h"
#include "jerk_integration_checks.h"
#include "segment_sampling_checks.h"

using namespace holistic_motion::robotics;

struct ProfileProbe : TrajectoryDoubleS<Rn<double, 2>> {
    ProfileProbe() : TrajectoryDoubleS(nullptr, nullptr) { this->dof_ = 2; }
    using TrajectoryDoubleS<Rn<double, 2>>::_ComputeSegmentMaxSVel;
    using TrajectoryDoubleS<Rn<double, 2>>::_ComputeDoubleSProfile;
    using TrajectoryDoubleS<Rn<double, 2>>::_ComputeNextTrajStep;
    using TrajectoryDoubleS<Rn<double, 2>>::_AlignProfileAfter;
    using TrajectoryDoubleS<Rn<double, 2>>::_ReverseWithMaxJerk;
};

void CheckProfileAlignment() {
    // A zero-length bridge needs no motion, including when both sides stop.
    std::list<TrajectorySeg> stopped{TrajectorySeg(2, 8.0, 1.0, 0.0, 0.0, 5.0),
                                     TrajectorySeg(2, 8.0, 1.0, 0.0, 0.0, -5.0),
                                     TrajectorySeg(2, 9.0, 2.0, 0.0, 0.0, 0.0)};
    if (!ProfileProbe::_AlignProfileAfter(TrajectorySeg(1, 3.0, 1.0, 0.0, 0.0, 0.0),
                                          stopped) ||
        stopped.front().timestamp != 3.0 || stopped.back().timestamp != 4.0 ||
        std::next(stopped.begin())->timestamp != 3.0 || stopped.size() != 3)
        throw std::runtime_error("stationary join must retain finite phase times");

    // Large original timestamps must not erase a small new start time.
    std::list<TrajectorySeg> shifted{
        TrajectorySeg(2, 0x1p54, 0.0, 1.0, 0.0, 0.0),
        TrajectorySeg(2, 0x1p54 + 4.0, 4.0, 1.0, 0.0, 0.0)};
    if (!ProfileProbe::_AlignProfileAfter(TrajectorySeg(1, 1.0, 0.0, 1.0, 0.0, 0.0),
                                          shifted) ||
        shifted.front().timestamp != 1.0 || shifted.back().timestamp != 5.0)
        throw std::runtime_error("alignment lost the new start time");

    // A length-two bridge at speed 0.5 takes four seconds.
    std::list<TrajectorySeg> moving{TrajectorySeg(2, 10.0, 3.0, 0.5, 0.0, 0.0),
                                    TrajectorySeg(2, 12.0, 4.0, 0.5, 0.0, 0.0)};
    if (!ProfileProbe::_AlignProfileAfter(TrajectorySeg(1, 7.0, 1.0, 0.5, 0.0, 0.0),
                                          moving) ||
        moving.front().timestamp != 11.0 || moving.back().timestamp != 13.0 ||
        moving.front().pos != 3.0 || moving.front().vel != 0.5 ||
        moving.back().pos != 4.0 || moving.back().seg_no != 2)
        throw std::runtime_error("alignment must include constant-speed travel");
}

void CheckFailedAlignmentPreservesTimestamps() {
    const auto reject = [](const TrajectorySeg &previous,
                           std::list<TrajectorySeg> phases) {
        const auto original = phases;
        if (ProfileProbe::_AlignProfileAfter(previous, phases) ||
            phases.size() != original.size())
            throw std::runtime_error("invalid alignment must fail");
        auto before = original.begin();
        for (const auto &phase : phases) {
            if (phase.timestamp != before->timestamp)
                throw std::runtime_error("failed alignment changed timestamps");
            ++before;
        }
    };
    const std::list<TrajectorySeg> phases{TrajectorySeg(2, 0.0, 1.0, 0.0, 0.0, 5.0),
                                          TrajectorySeg(2, 1.0, 2.0, 0.0, 0.0, 0.0)};
    reject(TrajectorySeg(1, 0.0, 0.0, 0.0, 0.0, 0.0), phases);
    reject(TrajectorySeg(1, 0.0, 2.0, 1.0, 0.0, 0.0), phases);
    reject(TrajectorySeg(1, 1e20, 1.0, 0.0, 0.0, 0.0), phases);
    reject(TrajectorySeg(1, 1e20, 0.0, 1.0, 0.0, 0.0), phases);
    reject(TrajectorySeg(1, 1e308, 1.0, 0.0, 0.0, 0.0),
           {TrajectorySeg(2, 0.0, 1.0, 0.0, 0.0, 0.0),
            TrajectorySeg(2, 1e307, 2.0, 0.0, 0.0, 0.0),
            TrajectorySeg(2, 1e308, 3.0, 0.0, 0.0, 0.0)});
}

void CheckFailedProfilesAreEmpty() {
    for (const auto &input : {std::pair<double, double>{1e20, 1.0},
                              {0.0, 1e20},
                              {std::numeric_limits<double>::max(), 1e308}}) {
        double start_velocity = 2.0;
        double end_velocity = 0.0;
        std::list<TrajectorySeg> phases{TrajectorySeg(9, 0, 0, 0, 0, 0)};
        const bool success = ProfileProbe::_ComputeDoubleSProfile(
            0.0, input.second, start_velocity, end_velocity, 1.0, 2.0, 5.0, input.first,
            phases, 0, false);
        if (success || !phases.empty() || start_velocity != 2.0 || end_velocity != 0.0)
            throw std::runtime_error(
                "unrepresentable Double-S phases must fail without changing "
                "speeds");
    }
}

void CheckReducedStartProfileIsRetained() {
    double start_velocity = 1.0;
    double end_velocity = 0.0;
    std::list<TrajectorySeg> phases;
    const bool success = ProfileProbe::_ComputeDoubleSProfile(
        0.0, 0.01, start_velocity, end_velocity, 1.0, 2.0, 5.0, 0.0, phases, 0, false);
    if (success || phases.size() != 8 || start_velocity >= 1.0 ||
        phases.front().vel != start_velocity || phases.back().timestamp <= 0.0)
        throw std::runtime_error("start-speed reduction must retain a valid profile");
    bool has_zero_phase = false;
    for (auto previous = phases.begin(), next = std::next(previous);
         next != phases.end(); ++previous, ++next) {
        if (next->timestamp == previous->timestamp) {
            has_zero_phase = true;
            if (next->pos != previous->pos || next->vel != previous->vel ||
                next->acc != previous->acc)
                throw std::runtime_error("zero-duration phase changes state");
        }
    }
    if (!has_zero_phase)
        throw std::runtime_error("expected a retained zero-duration phase");
}

void CheckLongFiniteProfile(double maximum_acceleration = 2e-20) {
    double start_velocity = 0.0;
    double end_velocity = 0.0;
    std::list<TrajectorySeg> phases;
    // The time cube overflows, but all actual integrated states are finite.
    const bool success = ProfileProbe::_ComputeDoubleSProfile(
        0.0, 1e201, start_velocity, end_velocity, 1e90, maximum_acceleration, 1e-130,
        0.0, phases, 0, false);
    if (!success || phases.size() != 8 || start_velocity != 0.0 ||
        end_velocity != 0.0 || std::abs(phases.back().pos / 1e201 - 1.0) > 1e-14 ||
        std::abs(phases.back().timestamp / 1.2e111 - 1.0) > 1e-14 ||
        std::abs(phases.back().vel) > 1e-14 * 1e90 ||
        std::abs(phases.back().acc) > 1e-14 * 1e-20)
        throw std::runtime_error("long finite Double-S profile was lost");
    for (auto previous = phases.begin(), next = std::next(previous);
         next != phases.end(); ++previous, ++next) {
        const long double time = next->timestamp - previous->timestamp;
        const long double position =
            previous->pos + static_cast<long double>(previous->vel) * time +
            previous->acc * time * time / 2 + previous->jerk * time * time * time / 6;
        const long double velocity =
            previous->vel + previous->acc * time + previous->jerk * time * time / 2;
        const long double acceleration = previous->acc + previous->jerk * time;
        if (!std::isfinite(next->pos) || !std::isfinite(next->vel) ||
            !std::isfinite(next->acc) || time < 0.0 ||
            std::abs(position - next->pos) > 1e-13L * 1e201L ||
            std::abs(velocity - next->vel) > 1e-13L * 1e90L ||
            std::abs(acceleration - next->acc) > 1e-13L * 1e-20L)
            throw std::runtime_error("long Double-S phase integration disagrees");
    }
}

void CheckRoundedPlateauProfile() {
    // A triangular/saturated boundary can leave a plateau a few ulps long.
    // Preserve its eight-state layout without inventing motion at equal times.
    for (double t0 : {0.0, 100.0, 1000.0}) {
        double v0 = 0.1664535670022241;
        double v1 = 0.16427253809057177;
        std::list<TrajectorySeg> phases;
        if (!ProfileProbe::_ComputeDoubleSProfile(
                46.99991360518943, 58.464560053823675, v0, v1, 0.3600080633994369,
                0.28474435856838226, 0.4142290961672831, t0, phases, 0, false) ||
            phases.size() != 8 || phases.front().timestamp != t0 ||
            std::abs(phases.back().pos - 58.464560053823675) > 1e-11 ||
            std::abs(phases.back().vel - v1) > 1e-12 ||
            std::abs(phases.back().acc) > 1e-12) {
            throw std::runtime_error("rounding plateau lost a valid profile");
        }
        for (auto it = phases.begin(), next = std::next(it); next != phases.end();
             ++it, ++next) {
            const double dt = next->timestamp - it->timestamp;
            const auto integrated =
                ProfileProbe::_ComputeNextTrajStep(*it, dt, next->jerk, next->seg_no);
            if (!std::isfinite(dt) || dt < 0.0 ||
                std::abs(integrated.pos - next->pos) > 1e-11 ||
                std::abs(integrated.vel - next->vel) > 1e-12 ||
                std::abs(integrated.acc - next->acc) > 1e-12 ||
                (dt == 0.0 && (it->pos != next->pos || it->vel != next->vel ||
                               it->acc != next->acc))) {
                throw std::runtime_error("rounded plateau changed phase motion");
            }
        }
    }
    // Backtracking changes a boundary speed by a few ulps; dividing that
    // uncertainty by a low acceleration amplifies the plateau-time residual.
    double v0 = 0.2949410453414922;
    double v1 = 0.26517024601420963;
    std::list<TrajectorySeg> phases;
    if (!ProfileProbe::_ComputeDoubleSProfile(
            71.78902926256153, 103.01715811090766, v0, v1, 0.29925176944259313,
            0.14509684178434776, 0.6177274774712997, 1000.0, phases, 0, false) ||
        phases.size() != 8 ||
        std::abs(phases.back().pos - 103.01715811090766) > 1e-11 ||
        std::abs(phases.back().vel - v1) > 1e-12 ||
        std::abs(phases.back().acc) > 1e-12) {
        throw std::runtime_error("backtracked roundoff plateau lost its profile");
    }
}

void CheckReverseStopsOnEmptyProfile() {
    double start_velocity = 0.0;
    double end_velocity = 0.0;
    std::list<TrajectorySeg> phases;
    if (!ProfileProbe::_ComputeDoubleSProfile(0.0, 1.0, start_velocity, end_velocity,
                                              1.0, 2.0, 5.0, 0.0, phases, 0, false))
        throw std::runtime_error("missing valid backtracking prefix");
    const auto prefix_size = phases.size();
    // A malformed later segment cannot be repaired by reducing earlier speeds.
    // Previously recursion continued with an empty replacement profile.
    for (int i = 0; i < 8; ++i)
        phases.emplace_back(1, 10.0 + i, 2.0 + i, 0.0, 0.0, 0.0);
    ProfileProbe probe;
    if (probe._ReverseWithMaxJerk(phases) || phases.size() != prefix_size)
        throw std::runtime_error("backtracking must stop on an empty profile");
}

void CheckRecursiveProfileAlignment() {
    double start_velocity = 0.0;
    double end_velocity = 1.0;
    std::list<TrajectorySeg> phases;
    if (!ProfileProbe::_ComputeDoubleSProfile(0.0, 1.0, start_velocity, end_velocity,
                                              1.0, 2.0, 5.0, 0.0, phases, 0, false))
        throw std::runtime_error("missing backtracking prefix");
    const double original_end_time = phases.back().timestamp;
    start_velocity = 1.0;
    end_velocity = 0.0;
    std::list<TrajectorySeg> suffix;
    ProfileProbe::_ComputeDoubleSProfile(1.0, 1.01, start_velocity, end_velocity, 1.0,
                                         2.0, 5.0, original_end_time, suffix, 1, false);
    if (suffix.size() != 8 || start_velocity >= 1.0)
        throw std::runtime_error("missing speed reduction for backtracking");
    // Request a start speed that now requires slowing the preceding segment.
    suffix.front().vel = 1.0;
    phases.splice(phases.end(), suffix);
    ProfileProbe probe;
    if (!probe._ReverseWithMaxJerk(phases) || phases.size() != 16)
        throw std::runtime_error("recursive profile alignment failed");
    const auto join = std::next(phases.begin(), 8);
    const auto previous = std::prev(join);
    if (previous->timestamp == original_end_time ||
        join->timestamp < previous->timestamp ||
        std::abs(join->timestamp - previous->timestamp) > 1e-12 ||
        std::abs(join->pos - previous->pos) > 1e-12 ||
        std::abs(join->vel - previous->vel) > 1e-12 ||
        std::abs(phases.back().pos - 1.01) > 1e-12)
        throw std::runtime_error("backtracking did not preserve the join state");
}

void CheckLongZeroJerkStep() {
    const TrajectorySeg initial(0, 0.0, 3.0, 1e-110, 2e-220, 0.0);
    const auto next = ProfileProbe::_ComputeNextTrajStep(initial, 1e110, 5.0, 9);
    if (!std::isfinite(next.pos) || std::abs(next.pos - 5.0) > 1e-14 ||
        std::abs(next.vel / 1e-110 - 3.0) > 1e-14 || next.acc != initial.acc ||
        next.timestamp != 1e110 || next.jerk != 5.0 || next.seg_no != 9) {
        throw std::runtime_error(
            "zero jerk must not multiply an overflowing time cube");
    }
}

void CheckProfile(double length, double start_velocity, double end_velocity) {
    constexpr double max_velocity = 1.0;
    constexpr double max_acceleration = 2.0;
    constexpr double max_jerk = 5.0;
    std::list<TrajectorySeg> phases;
    // False with a populated profile requests propagation of a reduced start
    // speed to earlier segments; that profile must still be physically valid.
    ProfileProbe::_ComputeDoubleSProfile(0.0, length, start_velocity, end_velocity,
                                         max_velocity, max_acceleration, max_jerk, 0.0,
                                         phases, 0, false);
    if (phases.size() != 8)
        throw std::runtime_error("missing scalar profile");
    if (std::abs(phases.front().vel - start_velocity) > 1e-12 ||
        std::abs(phases.back().vel - end_velocity) > 1e-12 ||
        std::abs(phases.back().pos - length) > 1e-12 ||
        std::abs(phases.back().acc) > 1e-12) {
        std::cerr << std::setprecision(17) << "length=" << length
                  << " requested velocities=" << start_velocity << ',' << end_velocity
                  << " final=" << phases.back().pos << ',' << phases.back().vel << ','
                  << phases.back().acc << '\n';
        throw std::runtime_error("profile does not reach its reported boundary state");
    }
    auto previous = phases.begin();
    for (auto next = std::next(previous); next != phases.end(); ++next, ++previous) {
        const double dt = next->timestamp - previous->timestamp;
        if (!std::isfinite(dt) || dt < 0.0) {
            std::cerr << std::setprecision(17) << "length=" << length
                      << " velocities=" << start_velocity << ',' << end_velocity
                      << " phase duration=" << dt << '\n';
            throw std::runtime_error("negative or non-finite phase duration");
        }
        for (int i = 0; i <= 10; ++i) {
            const double t = dt * i / 10.0;
            const double velocity =
                previous->vel + previous->acc * t + 0.5 * previous->jerk * t * t;
            const double acceleration = previous->acc + previous->jerk * t;
            if (velocity < -1e-12 || velocity > max_velocity + 1e-12 ||
                std::abs(acceleration) > max_acceleration + 1e-12 ||
                std::abs(previous->jerk) > max_jerk + 1e-12)
                throw std::runtime_error("profile exceeds scalar derivative limits");
        }
    }
}

void CheckRecordedTrajectory() {
    using Group = Rn<double, 7>;
    // Seed 6305, case 193: a short linear segment between blends formerly
    // produced negative jerk-phase durations despite a valid public result.
    std::vector<Group> waypoints(11);
    waypoints[0].Coeffs() << -0.002171395323362891, 0.00804854373029958,
        0.025822718106025568, 0.0023208272218802656, -0.005397152283271643,
        -0.0023435228659251556, 0.00009586039454327439;
    waypoints[1].Coeffs() << -0.01417718822017883, -0.00044309652527620147,
        0.01502908726684972, 0.013666901480263005, -0.009141765979553981,
        -0.004215933130450189, 0.005463640333508305;
    waypoints[2].Coeffs() << 0.0050125620390217875, -0.0013930300449973639,
        0.014425016735930583, 0.003451234963263281, -0.005031287754535817,
        -0.01484565724388185, 0.0036797465000371647;
    waypoints[3].Coeffs() << -0.00126087809403372, -0.004010379983360239,
        0.003750733841427557, 0.0025677005718353304, -0.011633445272416712,
        -0.008424355435071992, 0.019864160822495612;
    waypoints[4].Coeffs() << 0.008070420511644338, 0.0044144726690540015,
        -0.00013182958319285162, -0.007331886258874754, -0.02464370366328976,
        -0.00036081370489878493, 0.020873494558539853;
    waypoints[5].Coeffs() << 0.019720933031016928, 0.012287341736599896,
        -0.00386857977046215, -0.00381524212574081, -0.02701150701810295,
        0.006132913384766676, 0.023520619959974235;
    waypoints[6].Coeffs() << 0.013884361170483528, 0.006351191996898709,
        -0.02673612752494162, -0.006495602176049036, -0.03884128002226908,
        0.008280709885430629, 0.02294732860633861;
    waypoints[7].Coeffs() << 0.02551820329160329, 0.007026022747390286,
        -0.025223340060211274, -0.006326139467063624, -0.03503489093680072,
        0.005781310738947729, 0.01577551271024059;
    waypoints[8].Coeffs() << 0.0362592301843452, 0.01672431172266324,
        -0.02752970321388042, 0.005733826080510009, -0.033591058092469886,
        0.007449826951927124, 0.03355763039387799;
    waypoints[9].Coeffs() << 0.0261654730038672, 0.017518639453919245,
        -0.03665525631060239, -0.0031232067715201237, 0.00015517122868762695,
        0.004157178851229362, 0.04284570498137257;
    waypoints[10].Coeffs() << 0.0395306212054862, 0.011049690401317903,
        -0.029432063211148385, -0.016762146300160437, 0.014455010561377685,
        0.0013984632772585312, 0.0571457837114584;
    Eigen::VectorXd velocity(7);
    velocity << 1.0353271102391466, 0.37567792743672634, 0.5076329189162431,
        1.2350379806062537, 0.5675547567696388, 0.7432224702986426, 1.1358735339543438;
    Eigen::VectorXd acceleration(7);
    acceleration << 1.251746271133849, 2.3871232784418948, 1.2372371789057308,
        2.132741613793207, 1.60003136089148, 1.0322981507307176, 2.0739098491318524;
    Eigen::VectorXd jerk(7);
    jerk << 4.0818444643774106, 3.717837575388474, 3.1371072423687307,
        3.7470392460507815, 5.585418503615241, 2.128931426673633, 3.439093942523664;
    auto path = std::make_shared<PathBezierCurve<Group>>(waypoints, 5, false, 0.01);
    auto constraints =
        std::make_shared<TrajectoryConstraints>(velocity, acceleration, jerk);
    TrajectoryDoubleS<Group> trajectory(path, constraints);
    if (!trajectory.IsValid())
        throw std::runtime_error("recorded trajectory is invalid");
    const auto phases = trajectory.GetWaypointSegments();
    auto previous = phases.begin();
    for (auto next = std::next(previous); next != phases.end(); ++next, ++previous) {
        const double t = next->timestamp - previous->timestamp;
        if (!std::isfinite(t) || t < 0.0)
            throw std::runtime_error("recorded trajectory contains backwards time");
        const double position = previous->pos + previous->vel * t +
                                0.5 * previous->acc * t * t +
                                previous->jerk * t * t * t / 6;
        const double speed =
            previous->vel + previous->acc * t + 0.5 * previous->jerk * t * t;
        const double acc = previous->acc + previous->jerk * t;
        if (std::abs(position - next->pos) > 1e-11 ||
            std::abs(speed - next->vel) > 1e-11 || std::abs(acc - next->acc) > 1e-11)
            throw std::runtime_error("recorded phase boundaries are discontinuous");
    }
    if ((trajectory.GetPosition(trajectory.GetDuration()) - waypoints.back())
            .Coeffs()
            .norm() > 1e-7)
        throw std::runtime_error("recorded trajectory misses its final waypoint");
}

void CheckRoundoffEquivalentSpeedCaps() {
    for (double scale : {1e-8, 1.0, 1e8}) {
        {
            const double eps = std::numeric_limits<double>::epsilon();
            double initial = scale * (1.0 + 64.0 * eps);
            const double retained = initial;
            double terminal = scale * (1.0 + 192.0 * eps);
            std::list<TrajectorySeg> phases;
            if (!ProfileProbe::_ComputeDoubleSProfile(
                    0.0, 3.0 * scale, initial, terminal, scale, scale, scale,
                    0.0, phases, 1, false) || initial != retained ||
                terminal != retained)
                throw std::runtime_error("speed-cap budget accumulated across endpoints");
        }
        for (double ulps : {64.0, 512.0}) {
            const double original = scale *
                (1.0 + ulps * std::numeric_limits<double>::epsilon());
            double initial = original;
            double terminal = 0.4 * scale;
            std::list<TrajectorySeg> phases;
            const bool success = ProfileProbe::_ComputeDoubleSProfile(
                0.0, 3.0 * scale, initial, terminal, scale, scale, scale,
                0.0, phases, 1, false);
            if (phases.size() != 8 || success != (ulps == 64.0) ||
                initial != (success ? original : scale) || terminal != 0.4 * scale)
                throw std::runtime_error("speed-cap roundoff changed endpoint semantics");
            auto previous = phases.begin();
            for (auto next = std::next(previous); next != phases.end(); ++previous, ++next) {
                if (next->timestamp < previous->timestamp ||
                    !std::isfinite(next->pos) || !std::isfinite(next->vel) ||
                    !std::isfinite(next->acc))
                    throw std::runtime_error("speed-cap adjustment produced invalid phases");
            }
            if (std::abs(phases.back().pos / scale - 3.0) > 1e-12 ||
                std::abs(phases.back().vel / scale - 0.4) > 1e-13)
                throw std::runtime_error("speed-cap adjustment lost integrated endpoints");
        }
    }
}

void CheckTriangularPlateauCanBeRebased() {
    double start_velocity = 0.010460993369887253;
    double end_velocity = 0.0005738784102910987;
    std::list<TrajectorySeg> phases;
    // A triangular deceleration needs start-speed backtracking. Cancellation
    // used to leave a one-ulp plateau that collapsed after timestamp rebasing.
    ProfileProbe::_ComputeDoubleSProfile(
        0.004866954021559577, 0.006014272160646773,
        start_velocity, end_velocity, 0.13629581798020268,
        1.519583360954827, 0.20274587983978792, 0.6899296743090442,
        phases, 2, false);
    if (phases.size() != 8)
        throw std::runtime_error("triangular deceleration profile missing");
    const TrajectorySeg previous(0, 0.74964750099387845,
                                 0.0023503957270758049,
                                 0.0062706691450572782, 0.0, 0.0);
    if (!ProfileProbe::_AlignProfileAfter(previous, phases))
        throw std::runtime_error("roundoff plateau must not prevent rebasing");
    auto first = phases.begin();
    for (auto next = std::next(first); next != phases.end(); ++first, ++next) {
        if (next->timestamp < first->timestamp ||
            (next->timestamp == first->timestamp &&
             (next->pos != first->pos || next->vel != first->vel ||
              next->acc != first->acc)))
            throw std::runtime_error("rebased plateau changed state at zero time");
    }
}

int main() {
    try {
        CheckRoundoffEquivalentSpeedCaps();
        CheckTriangularPlateauCanBeRebased();
        ProfileProbe probe;
        sampling_checks::CheckSegmentSampling(
            [&](const auto &segment, const auto &v, const auto &a, const auto &j) {
                return probe._ComputeSegmentMaxSVel(segment, v, a, j);
            });
        CheckJerkIntegrationRange(ProfileProbe::_ComputeNextTrajStep);
        CheckQuadraticIntegrationRange(ProfileProbe::_ComputeNextTrajStep);
        CheckProfileAlignment();
        CheckFailedAlignmentPreservesTimestamps();
        CheckFailedProfilesAreEmpty();
        CheckReducedStartProfileIsRetained();
        CheckLongFiniteProfile();
        CheckRoundedPlateauProfile();
        CheckLongFiniteProfile(1e-20);
        CheckReverseStopsOnEmptyProfile();
        CheckRecursiveProfileAlignment();
        CheckLongZeroJerkStep();
        CheckRecordedTrajectory();
        for (double length : {1e-5, 1e-7, 1e-9, 0.01, 1.0}) {
            CheckProfile(length, 0.03, 0.04);
            CheckProfile(length, 0.04, 0.03);
            CheckProfile(length, 0.03, 0.03);
            CheckProfile(length, 0.0, 0.0);
        }
        std::mt19937 generator(193);
        std::uniform_real_distribution<double> exponent(-9.0, 0.0);
        std::uniform_real_distribution<double> velocity(0.0, 1.0);
        for (int trial = 0; trial < 1000; ++trial)
            CheckProfile(std::pow(10.0, exponent(generator)), velocity(generator),
                         velocity(generator));
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
