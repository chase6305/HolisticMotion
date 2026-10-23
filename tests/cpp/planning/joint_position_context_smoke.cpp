#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <vector>

#include "JointPositionContext.h"

using holistic_motion::robotics::JointPositionMap;
using holistic_motion::robotics::planning::detail::JointPositionContext;

int main() {
    JointPositionMap mapping(10, 8, {{"slide", 9, 1, 7, 1}, {"turn", 7, 2, 6, 1}});
    Eigen::VectorXd reference(10);
    reference << 1, 2, 3, 0, 0, 0, 1, 1, 0, 0;
    const Eigen::VectorXd original = reference;
    auto context = std::make_shared<const JointPositionContext>(mapping, reference);
    mapping = JointPositionMap(0, 0, {});
    reference.setConstant(std::numeric_limits<double>::quiet_NaN());
    if (context->GetMapping().GetJoints()[0].name != "slide")
        return 1;
    std::vector<std::future<bool>> workers;
    for (int worker = 0; worker < 8; ++worker)
        workers.push_back(std::async(std::launch::async, [context, original, worker] {
            for (int sample = 0; sample < 1024; ++sample) {
                Eigen::VectorXd positions(2);
                positions << 0.01 * worker, 0.001 * sample;
                auto packed = context->Configuration(positions);
                if (!packed.head(7).isApprox(original.head(7), 0.0) ||
                    !context->GetMapping().JointPositions(packed).isApprox(positions,
                                                                           1e-12))
                    return false;
                packed.setZero();
                const auto independent = context->Configuration(positions);
                if (!independent.head(7).isApprox(original.head(7), 0.0))
                    return false;
            }
            return true;
        }));
    for (auto &worker : workers)
        if (!worker.get())
            return 2;
    const auto gradient = context->JointVelocities(Eigen::VectorXd::LinSpaced(8, 0, 7));
    if (gradient[0] != 7 || gradient[1] != 6)
        return 3;
    for (int invalid = 0; invalid < 2; ++invalid) {
        try {
            Eigen::VectorXd bad = Eigen::VectorXd::Zero(invalid ? 10 : 9);
            if (invalid)
                bad[0] = std::numeric_limits<double>::infinity();
            JointPositionContext rejected(context->GetMapping(), bad);
            return 4;
        } catch (const std::invalid_argument &) {
        }
    }
    try {
        context->Configuration(Eigen::VectorXd::Zero(3));
        return 5;
    } catch (const std::invalid_argument &) {
    }
}
