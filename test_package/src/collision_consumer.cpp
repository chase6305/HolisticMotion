#include <exception>

#include "holistic_motion/collision/CollisionModel.h"

int main() {
    try {
        holistic_motion::robotics::collision::CollisionModel model(
                "/definitely/not/a/robot.urdf");
    } catch (const std::exception&) {
        // Construction reached the Pinocchio-backed implementation and the
        // deliberately missing caller-provided URDF was rejected.
        return 0;
    }
    return 1;
}
