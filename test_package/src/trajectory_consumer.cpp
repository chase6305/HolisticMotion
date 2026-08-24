#include <Eigen/Core>

#include <memory>
#include <vector>

#include "holistic_motion/trajectory/PathBezierCurve.h"
#include "holistic_motion/trajectory/TrajectoryDoubleS.h"

int main() {
    using namespace holistic_motion::robotics;
    using Group = Rn<double, 7>;
    const Eigen::VectorXd limits = Eigen::VectorXd::Ones(7);
    auto constraints = std::make_shared<TrajectoryConstraints>(
            limits, limits, limits);
    std::vector<Group> waypoints(3);
    for (auto& waypoint : waypoints) waypoint.Coeffs().setZero();
    waypoints[1].Coeffs()[0] = 0.2;
    waypoints[2].Coeffs()[1] = 0.3;
    auto path = std::make_shared<PathBezierCurve<Group>>(
            waypoints, 5, false, 0.005);
    TrajectoryDoubleS<Group> trajectory(path, constraints);
    if (!trajectory.IsValid()) return 1;
    const auto report = trajectory.GetConstraintReport(101);
    return report.within_limits && report.velocity_continuous &&
                           report.acceleration_continuous
                   ? 0
                   : 2;
}
