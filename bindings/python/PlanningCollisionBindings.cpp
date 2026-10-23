#include "Bindings.h"
#include "PlanningCallbacks.h"
#include "RealInputs.h"

#ifdef HOLISTICMOTION_HAS_COLLISION
#include <algorithm>
#include <cmath>
#include <memory>
#include <set>
#include <string>

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "../../src/planning/JointPositionContext.h"
#include "holistic_motion/collision/CollisionModel.h"
#include "holistic_motion/collision/SphereCollisionModel.h"
#include "holistic_motion/planning/PathOptimizer.h"
#include "holistic_motion/planning/SamplingPlanner.h"
#endif

namespace holistic_motion::python {

#ifdef HOLISTICMOTION_HAS_COLLISION
namespace {

using robotics::planning::detail::JointPositionContext;

class SphereDistanceCache {
  public:
    explicit SphereDistanceCache(
        robotics::collision::SphereCollisionModel &collision_model)
        : collision_model_(collision_model) {}

    const robotics::collision::SphereDistanceResult &
    CostDistance(const Eigen::VectorXd &state) {
        // A failed new query must not pair the new state with the preceding
        // distance and leave it eligible for the next validation call.
        reusable_ = false;
        state_ = state;
        distance_ = collision_model_.MinimumDistance(state);
        pair_revision_ = collision_model_.GetCollisionPairRevision();
        reusable_ = true;
        return distance_;
    }

    double ValidationDistance(const Eigen::VectorXd &state) {
        if (reusable_ && Matches(state)) {
            reusable_ = false;
            return distance_.distance;
        }
        reusable_ = false;
        return collision_model_.MinimumDistance(state).distance;
    }

  private:
    bool Matches(const Eigen::VectorXd &state) const {
        return state_.size() == state.size() &&
               pair_revision_ == collision_model_.GetCollisionPairRevision() &&
               (state_.array() == state.array()).all();
    }

    robotics::collision::SphereCollisionModel &collision_model_;
    Eigen::VectorXd state_;
    robotics::collision::SphereDistanceResult distance_;
    std::size_t pair_revision_{0};
    bool reusable_{false};
};

template <class Planner>
void ConfigureContinuousJoints(Planner &planner,
                               const robotics::JointPositionMap &mapping) {
    std::vector<std::size_t> continuous;
    for (std::size_t i = 0; i < mapping.GetJoints().size(); ++i)
        if (mapping.GetJoints()[i].nq == 2)
            continuous.push_back(i);
    planner.SetContinuousJoints(continuous);
}

void ValidateContext(const Eigen::VectorXd &context, int nq) {
    if (context.size() != nq || !context.allFinite())
        throw std::invalid_argument("context_configuration must be finite and "
                                    "match collision model nq");
}

template <class Model>
void ValidateMappedContext(const Model &model,
                           const robotics::JointPositionMap &mapping,
                           const Eigen::VectorXd &context) {
    // Selected scalar coordinates are replaced by every query. Validate the
    // preserved manifold blocks without requiring a particular active pose.
    model.ValidateConfiguration(mapping.ConfigurationWithJointPositions(
        context,
        Eigen::VectorXd::Zero(static_cast<Eigen::Index>(mapping.GetJoints().size()))));
}

template <class Model>
std::shared_ptr<const JointPositionContext>
PrepareJointContext(const Model &model, const std::vector<std::string> &names,
                    Eigen::VectorXd context) {
    auto mapping = model.CreateJointPositionMap(names);
    ValidateMappedContext(model, mapping, context);
    return std::make_shared<const JointPositionContext>(std::move(mapping),
                                                        std::move(context));
}

void ValidateMargins(double margin, double clearance = 0.0) {
    if (!std::isfinite(margin) || margin < 0.0 || !std::isfinite(clearance) ||
        clearance < 0.0 || (clearance > 0.0 && clearance < margin))
        throw std::invalid_argument(
            "security_margin must be non-negative; positive clearance must be "
            "at least security_margin");
}

void ValidateClearancePairs(double clearance, std::size_t pair_count,
                            const char *model_kind) {
    if (clearance > 0.0 && pair_count == 0)
        throw std::invalid_argument(std::string("positive clearance requires active ") +
                                    model_kind + " collision pairs");
}

robotics::planning::SamplingPlanner
MakeSphereJointPlanner(robotics::collision::SphereCollisionModel &model,
                       const std::vector<std::string> &names,
                       const Eigen::VectorXd &context, double margin) {
    ValidateContext(context, model.GetConfigurationSize());
    ValidateMargins(margin);
    const auto selection = PrepareJointContext(model, names, context);
    robotics::planning::SamplingPlanner result(
        robotics::planning::JointSpace(model.GetJointLowerLimits(names),
                                       model.GetJointUpperLimits(names), names),
        [&model, selection, margin](const Eigen::VectorXd &positions) {
            return !model.InCollision(selection->Configuration(positions), margin,
                                      true);
        });
    ConfigureContinuousJoints(result, selection->GetMapping());
    return result;
}

robotics::planning::PathOptimizer
MakeSphereJointOptimizer(robotics::collision::SphereCollisionModel &model,
                         const std::vector<std::string> &names,
                         const Eigen::VectorXd &context, double margin,
                         double clearance) {
    ValidateContext(context, model.GetConfigurationSize());
    ValidateMargins(margin, clearance);
    const auto selection = PrepareJointContext(model, names, context);
    ValidateClearancePairs(clearance, model.GetCollisionPairCount(), "sphere");
    auto cache = clearance > 0.0 ? std::make_shared<SphereDistanceCache>(model)
                                 : std::shared_ptr<SphereDistanceCache>{};
    robotics::planning::PathOptimizer result(
        robotics::planning::JointSpace(model.GetJointLowerLimits(names),
                                       model.GetJointUpperLimits(names), names),
        [&model, selection, margin, clearance,
         cache](const Eigen::VectorXd &positions) {
            if (model.GetCollisionPairCount() == 0)
                return true;
            const auto full = selection->Configuration(positions);
            return clearance > 0.0 ? cache->ValidationDistance(full) > margin
                                   : !model.InCollision(full, margin, true);
        });
    ConfigureContinuousJoints(result, selection->GetMapping());
    if (clearance > 0.0) {
        result.SetStateCost(
            [&model, selection, clearance, cache](const Eigen::VectorXd &positions) {
                if (model.GetCollisionPairCount() == 0)
                    return 0.0;
                const auto full = selection->Configuration(positions);
                const double deficit =
                    std::max(0.0, clearance - cache->CostDistance(full).distance);
                return deficit * deficit;
            });
        result.SetStateCostGradient([&model, selection,
                                     clearance](const Eigen::VectorXd &positions)
                                        -> Eigen::VectorXd {
            if (model.GetCollisionPairCount() == 0)
                return Eigen::VectorXd::Zero(positions.size());
            const auto full = selection->Configuration(positions);
            const auto query = model.MinimumDistanceWithGradient(full);
            const double deficit =
                std::max(0.0, clearance - query.distance_result.distance);
            if (deficit == 0.0)
                return Eigen::VectorXd::Zero(positions.size());
            return (-2.0 * deficit * selection->JointVelocities(query.gradient)).eval();
        });
    }
    return result;
}

} // namespace
#endif

void BindPlanningCollision(pybind11::module_ &module) {
#ifdef HOLISTICMOTION_HAS_COLLISION
    namespace py = pybind11;
    using robotics::planning::PathOptimizer;
    using robotics::planning::SamplingPlanner;
    auto optimizer =
        py::reinterpret_borrow<py::class_<PathOptimizer>>(module.attr("PathOptimizer"));
    auto planner = py::reinterpret_borrow<py::class_<SamplingPlanner>>(
        module.attr("SamplingPlanner"));
    optimizer.def_static(
        "from_sphere_collision_model",
        [](const py::object &lower_input, const py::object &upper_input,
           robotics::collision::SphereCollisionModel &collision_model,
           const py::object &margin_input, const py::object &clearance_input) {
            Eigen::VectorXd lower =
                CheckedRealEigen<Eigen::VectorXd>(lower_input, "lower_limits");
            Eigen::VectorXd upper =
                CheckedRealEigen<Eigen::VectorXd>(upper_input, "upper_limits");
            const double security_margin =
                CheckedRealEigen<double>(margin_input, "security_margin");
            const double clearance =
                CheckedRealEigen<double>(clearance_input, "clearance");
            if (!std::isfinite(security_margin) || security_margin < 0.0 ||
                !std::isfinite(clearance) || clearance < 0.0 ||
                (clearance > 0.0 && clearance < security_margin))
                throw std::invalid_argument(
                    "clearance must be finite and at least security_margin");
            if (collision_model.GetConfigurationSize() !=
                collision_model.GetVelocitySize())
                throw std::invalid_argument(
                    "non-Euclidean configuration coordinates require "
                    "from_sphere_collision_joints");
            if (lower.size() != collision_model.GetConfigurationSize())
                throw std::invalid_argument(
                    "full-space limits must match sphere collision model nq");
            ValidateClearancePairs(clearance, collision_model.GetCollisionPairCount(),
                                   "sphere");
            auto cache = clearance > 0.0
                             ? std::make_shared<SphereDistanceCache>(collision_model)
                             : std::shared_ptr<SphereDistanceCache>{};
            PathOptimizer result(
                std::move(lower), std::move(upper),
                [&collision_model, cache, security_margin,
                 clearance](const Eigen::VectorXd &state) {
                    if (collision_model.GetCollisionPairCount() == 0)
                        return true;
                    if (clearance > 0.0)
                        return cache->ValidationDistance(state) > security_margin;
                    return !collision_model.InCollision(state, security_margin, true);
                });
            if (clearance > 0.0) {
                result.SetStateCost([&collision_model, cache,
                                     clearance](const Eigen::VectorXd &state) {
                    if (collision_model.GetCollisionPairCount() == 0)
                        return 0.0;
                    const double deficit =
                        std::max(0.0, clearance - cache->CostDistance(state).distance);
                    return deficit * deficit;
                });
                if (collision_model.GetConfigurationSize() ==
                    collision_model.GetVelocitySize()) {
                    result.SetStateCostGradient(
                        [&collision_model,
                         clearance](const Eigen::VectorXd &state) -> Eigen::VectorXd {
                            if (collision_model.GetCollisionPairCount() == 0)
                                return Eigen::VectorXd::Zero(state.size());
                            const auto query =
                                collision_model.MinimumDistanceWithGradient(state);
                            const double deficit = std::max(
                                0.0, clearance - query.distance_result.distance);
                            if (deficit == 0.0)
                                return Eigen::VectorXd::Zero(state.size());
                            return (-2.0 * deficit * query.gradient).eval();
                        });
                }
            }
            return RetainPlanningResource(
                std::move(result),
                py::cast(&collision_model, py::return_value_policy::reference));
        },
        py::arg("lower_limits"), py::arg("upper_limits"), py::arg("collision_model"),
        py::arg("security_margin") = 0.0, py::arg("clearance") = 0.0);
    optimizer.def_static(
        "from_collision_joints",
        [](robotics::collision::CollisionModel &collision_model,
           std::vector<std::string> joint_names, const py::object &context_input,
           const py::object &margin_input, const py::object &clearance_input) {
            Eigen::VectorXd context_configuration = CheckedRealEigen<Eigen::VectorXd>(
                context_input, "context_configuration");
            const double security_margin =
                CheckedRealEigen<double>(margin_input, "security_margin");
            const double clearance =
                CheckedRealEigen<double>(clearance_input, "clearance");
            if (!std::isfinite(security_margin) || security_margin < 0.0 ||
                !std::isfinite(clearance) || clearance < 0.0 ||
                (clearance > 0.0 && clearance < security_margin))
                throw std::invalid_argument(
                    "clearance must be finite and at least security_margin");
            if (context_configuration.size() !=
                    collision_model.GetConfigurationSize() ||
                !context_configuration.allFinite())
                throw std::invalid_argument("context_configuration must be finite "
                                            "and match collision model nq");
            Eigen::VectorXd lower = collision_model.GetJointLowerLimits(joint_names);
            Eigen::VectorXd upper = collision_model.GetJointUpperLimits(joint_names);
            const auto selection = PrepareJointContext(
                collision_model, joint_names, std::move(context_configuration));
            ValidateClearancePairs(clearance, collision_model.GetCollisionPairCount(),
                                   "mesh");
            PathOptimizer result(
                robotics::planning::JointSpace(std::move(lower), std::move(upper),
                                               joint_names),
                [&collision_model, selection,
                 security_margin](const Eigen::VectorXd &active) {
                    const Eigen::VectorXd full = selection->Configuration(active);
                    return security_margin > 0.0
                               ? !collision_model.IsWithinDistance(
                                     full, security_margin, true)
                               : !collision_model.InCollision(full, true);
                });
            if (clearance > 0.0) {
                result.SetStateCost([&collision_model, selection,
                                     clearance](const Eigen::VectorXd &active) {
                    if (collision_model.GetCollisionPairCount() == 0)
                        return 0.0;
                    const Eigen::VectorXd full = selection->Configuration(active);
                    const double deficit = std::max(
                        0.0,
                        clearance - collision_model.MinimumDistance(full).distance);
                    return deficit * deficit;
                });
            }
            ConfigureContinuousJoints(result, selection->GetMapping());
            return RetainPlanningResource(
                std::move(result),
                py::cast(&collision_model, py::return_value_policy::reference));
        },
        py::arg("collision_model"), py::arg("joint_names"),
        py::arg("context_configuration"), py::arg("security_margin") = 0.0,
        py::arg("clearance") = 0.0);
    planner.def_static(
        "from_sphere_collision_model",
        [](const py::object &lower_input, const py::object &upper_input,
           robotics::collision::SphereCollisionModel &collision_model,
           const py::object &margin_input) {
            Eigen::VectorXd lower =
                CheckedRealEigen<Eigen::VectorXd>(lower_input, "lower_limits");
            Eigen::VectorXd upper =
                CheckedRealEigen<Eigen::VectorXd>(upper_input, "upper_limits");
            const double security_margin =
                CheckedRealEigen<double>(margin_input, "security_margin");
            if (!std::isfinite(security_margin) || security_margin < 0.0) {
                throw std::invalid_argument(
                    "security_margin must be finite and non-negative");
            }
            if (collision_model.GetConfigurationSize() !=
                collision_model.GetVelocitySize())
                throw std::invalid_argument(
                    "non-Euclidean configuration coordinates require "
                    "from_sphere_collision_joints");
            if (lower.size() != collision_model.GetConfigurationSize()) {
                throw std::invalid_argument(
                    "full-space limits must match sphere collision model nq");
            }
            return RetainPlanningResource(
                SamplingPlanner(
                    std::move(lower), std::move(upper),
                    [&collision_model, security_margin](const Eigen::VectorXd &state) {
                        return !collision_model.InCollision(state, security_margin,
                                                            true);
                    }),
                py::cast(&collision_model, py::return_value_policy::reference));
        },
        py::arg("lower_limits"), py::arg("upper_limits"), py::arg("collision_model"),
        py::arg("security_margin") = 0.0);
    planner.def_static(
        "from_collision_model",
        [](const py::object &lower_input, const py::object &upper_input,
           robotics::collision::CollisionModel &collision_model,
           const py::object &indices_input, const py::object &context_input) {
            std::vector<std::size_t> active_indices =
                CheckedRealEigen<std::vector<std::size_t>>(indices_input,
                                                           "active_indices");
            Eigen::VectorXd context_configuration = CheckedRealEigen<Eigen::VectorXd>(
                context_input, "context_configuration");
            Eigen::VectorXd lower =
                CheckedRealEigen<Eigen::VectorXd>(lower_input, "lower_limits");
            Eigen::VectorXd upper =
                CheckedRealEigen<Eigen::VectorXd>(upper_input, "upper_limits");
            if (active_indices.empty()) {
                if (collision_model.GetConfigurationSize() !=
                    collision_model.GetVelocitySize())
                    throw std::invalid_argument(
                        "non-Euclidean configuration coordinates require "
                        "from_collision_joints");
                if (lower.size() != collision_model.GetConfigurationSize()) {
                    throw std::invalid_argument(
                        "full-space limits must match collision model nq");
                }
                return RetainPlanningResource(
                    SamplingPlanner(std::move(lower), std::move(upper),
                                    [&collision_model](const Eigen::VectorXd &state) {
                                        return !collision_model.InCollision(state,
                                                                            true);
                                    }),
                    py::cast(&collision_model, py::return_value_policy::reference));
            }
            if (context_configuration.size() !=
                collision_model.GetConfigurationSize()) {
                throw std::invalid_argument(
                    "context_configuration must match collision model nq");
            }
            if (!context_configuration.allFinite()) {
                throw std::invalid_argument(
                    "context_configuration must contain finite values");
            }
            if (lower.size() != static_cast<Eigen::Index>(active_indices.size())) {
                throw std::invalid_argument(
                    "active limits and active_indices must have equal size");
            }
            collision_model.ValidateConfiguration(context_configuration);
            const auto coordinates = collision_model.GetJointCoordinates();
            std::set<std::size_t> unique_indices;
            std::vector<robotics::JointCoordinate> selected_coordinates;
            selected_coordinates.reserve(active_indices.size());
            for (std::size_t index : active_indices) {
                if (index >= static_cast<std::size_t>(context_configuration.size())) {
                    throw std::out_of_range("active configuration index");
                }
                const auto scalar = std::find_if(
                    coordinates.begin(), coordinates.end(), [index](const auto &joint) {
                        return joint.nq == 1 && joint.nv == 1 &&
                               joint.idx_q == static_cast<int>(index);
                    });
                if (scalar == coordinates.end())
                    throw std::invalid_argument(
                        "active configuration indices must select scalar "
                        "joints; use from_collision_joints");
                if (!unique_indices.insert(index).second) {
                    throw std::invalid_argument("active_indices must be unique");
                }
                selected_coordinates.push_back(*scalar);
            }
            const auto selection = std::make_shared<const JointPositionContext>(
                robotics::JointPositionMap(collision_model.GetConfigurationSize(),
                                           collision_model.GetVelocitySize(),
                                           std::move(selected_coordinates)),
                std::move(context_configuration));
            return RetainPlanningResource(
                SamplingPlanner(
                    std::move(lower), std::move(upper),
                    [&collision_model, selection](const Eigen::VectorXd &active) {
                        return !collision_model.InCollision(
                            selection->Configuration(active), true);
                    }),
                py::cast(&collision_model, py::return_value_policy::reference));
        },
        py::arg("lower_limits"), py::arg("upper_limits"), py::arg("collision_model"),
        py::arg("active_indices") = std::vector<std::size_t>{},
        py::arg("context_configuration") = Eigen::VectorXd{});
    planner.def_static(
        "from_collision_joints",
        [](robotics::collision::CollisionModel &collision_model,
           std::vector<std::string> joint_names, const py::object &context_input,
           const py::object &margin_input) {
            Eigen::VectorXd context_configuration = CheckedRealEigen<Eigen::VectorXd>(
                context_input, "context_configuration");
            const double security_margin =
                CheckedRealEigen<double>(margin_input, "security_margin");
            if (!std::isfinite(security_margin) || security_margin < 0.0) {
                throw std::invalid_argument(
                    "security_margin must be finite and non-negative");
            }
            if (context_configuration.size() !=
                    collision_model.GetConfigurationSize() ||
                !context_configuration.allFinite()) {
                throw std::invalid_argument("context_configuration must be finite "
                                            "and match collision model nq");
            }
            Eigen::VectorXd lower = collision_model.GetJointLowerLimits(joint_names);
            Eigen::VectorXd upper = collision_model.GetJointUpperLimits(joint_names);
            const auto selection = PrepareJointContext(
                collision_model, joint_names, std::move(context_configuration));
            SamplingPlanner result(
                robotics::planning::JointSpace(std::move(lower), std::move(upper),
                                               joint_names),
                [&collision_model, selection,
                 security_margin](const Eigen::VectorXd &active) {
                    Eigen::VectorXd full = selection->Configuration(active);
                    return security_margin > 0.0
                               ? !collision_model.IsWithinDistance(
                                     full, security_margin, true)
                               : !collision_model.InCollision(full, true);
                });
            ConfigureContinuousJoints(result, selection->GetMapping());
            return RetainPlanningResource(
                std::move(result),
                py::cast(&collision_model, py::return_value_policy::reference));
        },
        py::arg("collision_model"), py::arg("joint_names"),
        py::arg("context_configuration"), py::arg("security_margin") = 0.0);
    planner.def_static(
        "from_sphere_collision_joints",
        [](robotics::collision::SphereCollisionModel &model,
           const std::vector<std::string> &names, const py::object &context,
           const py::object &margin) {
            return RetainPlanningResource(
                MakeSphereJointPlanner(
                    model, names,
                    CheckedRealEigen<Eigen::VectorXd>(context, "context_configuration"),
                    CheckedRealEigen<double>(margin, "security_margin")),
                py::cast(&model, py::return_value_policy::reference));
        },
        py::arg("collision_model"), py::arg("joint_names"),
        py::arg("context_configuration"), py::arg("security_margin") = 0.0);
    optimizer.def_static(
        "from_sphere_collision_joints",
        [](robotics::collision::SphereCollisionModel &model,
           const std::vector<std::string> &names, const py::object &context,
           const py::object &margin, const py::object &clearance) {
            return RetainPlanningResource(
                MakeSphereJointOptimizer(
                    model, names,
                    CheckedRealEigen<Eigen::VectorXd>(context, "context_configuration"),
                    CheckedRealEigen<double>(margin, "security_margin"),
                    CheckedRealEigen<double>(clearance, "clearance")),
                py::cast(&model, py::return_value_policy::reference));
        },
        py::arg("collision_model"), py::arg("joint_names"),
        py::arg("context_configuration"), py::arg("security_margin") = 0.0,
        py::arg("clearance") = 0.0);
#else
    (void)module;
#endif
}

} // namespace holistic_motion::python
