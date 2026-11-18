#include "Bindings.h"

#ifdef HOLISTICMOTION_HAS_COLLISION

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "RealInputs.h"

#include "holistic_motion/collision/CollisionModel.h"
#include "holistic_motion/collision/SphereCollisionModel.h"

namespace holistic_motion::python {
namespace {

template <class Class, class Result, class Input>
auto WithRealInput(Result (Class::*method)(const Input &), const char *name) {
    return [method, name](Class &self, const py::object &input) {
        return (self.*method)(CheckedRealEigen<Input>(input, name));
    };
}

template <class Class, class Result, class Input>
auto WithRealInput(Result (Class::*method)(const Input &) const,
                   const char *name) {
    return [method, name](const Class &self, const py::object &input) {
        return (self.*method)(CheckedRealEigen<Input>(input, name));
    };
}

template <class Model> void BindJointCoordinates(py::class_<Model> &binding) {
    binding.def_property_readonly("nq", &Model::GetConfigurationSize)
        .def_property_readonly("nv", &Model::GetVelocitySize)
        .def("validate_configuration",
             WithRealInput(&Model::ValidateConfiguration, "configuration"),
             py::arg("configuration"))
        .def_property_readonly("joint_coordinates", &Model::GetJointCoordinates)
        .def("create_joint_position_map", &Model::CreateJointPositionMap, py::arg("joint_names"))
        .def("neutral_configuration", &Model::NeutralConfiguration)
        .def(
            "configuration_from_joint_positions",
            [](const Model &self, const py::dict &positions) {
                // Keep every key/value alive before scalar conversions can edit
                // the caller's dictionary.
                auto entries = py::reinterpret_steal<py::dict>(PyDict_Copy(positions.ptr()));
                if (!entries)
                    throw py::error_already_set();
                std::map<std::string, double> checked;
                for (const auto &item : entries)
                    checked.emplace(py::cast<std::string>(item.first),
                                    CheckedRealEigen<double>(item.second, "joint_positions"));
                return self.ConfigurationFromJointPositions(checked);
            },
            pybind11::arg("joint_positions"))
        .def(
            "configuration_with_joint_positions",
            [](const Model &self, const py::object &reference,
               const std::vector<std::string> &names, const py::object &positions) {
                const auto configuration =
                    CheckedRealEigen<Eigen::VectorXd>(reference, "reference");
                const auto values = CheckedRealEigen<Eigen::VectorXd>(positions, "joint_positions");
                return self.ConfigurationWithJointPositions(configuration, names, values);
            },
            pybind11::arg("reference"), pybind11::arg("joint_names"),
            pybind11::arg("joint_positions"))
        .def("joint_lower_limits", &Model::GetJointLowerLimits, pybind11::arg("joint_names"))
        .def("joint_upper_limits", &Model::GetJointUpperLimits, pybind11::arg("joint_names"));
}

} // namespace

void BindCollision(pybind11::module_ &module) {
  using robotics::collision::CollisionModel;
  using robotics::collision::CollisionPairInfo;
  using robotics::collision::CollisionReport;
  using robotics::collision::CollisionResult;
  using robotics::collision::CollisionSphere;
  using robotics::collision::SphereCollisionModel;
  using robotics::collision::SphereDistanceGradientResult;
  using robotics::collision::SphereDistanceResult;
  using robotics::collision::SpherePairInfo;

  pybind11::class_<CollisionSphere>(module, "CollisionSphere")
      .def(pybind11::init<>())
      .def(pybind11::init([](std::string name, std::string link,
                             const py::object &center,
                             const py::object &radius) {
               return CollisionSphere{
                   std::move(name), std::move(link),
                   CheckedRealEigen<Eigen::Vector3d>(center, "center"),
                   CheckedRealEigen<double>(radius, "radius")};
           }),
           pybind11::arg("name"), pybind11::arg("link_name"),
           pybind11::arg("center"), pybind11::arg("radius"))
      .def_readwrite("name", &CollisionSphere::name)
      .def_readwrite("link_name", &CollisionSphere::link_name)
      .def_property(
          "center",
          [](const CollisionSphere &self) -> const Eigen::Vector3d & {
              return self.center;
          },
          WriteRealParameter(&CollisionSphere::center, "center"))
      .def_property("radius", ReadParameter(&CollisionSphere::radius),
                    WriteRealParameter(&CollisionSphere::radius, "radius"));
  pybind11::class_<SpherePairInfo>(module, "SpherePairInfo")
      .def_readonly("first", &SpherePairInfo::first)
      .def_readonly("second", &SpherePairInfo::second)
      .def_readonly("first_sphere", &SpherePairInfo::first_sphere)
      .def_readonly("second_sphere", &SpherePairInfo::second_sphere)
      .def_readonly("first_link", &SpherePairInfo::first_link)
      .def_readonly("second_link", &SpherePairInfo::second_link);
  pybind11::class_<SphereDistanceResult>(module, "SphereDistanceResult")
      .def_readonly("pair_index", &SphereDistanceResult::pair_index)
      .def_readonly("first", &SphereDistanceResult::first)
      .def_readonly("second", &SphereDistanceResult::second)
      .def_readonly("distance", &SphereDistanceResult::distance)
      .def_readonly("first_center", &SphereDistanceResult::first_center)
      .def_readonly("second_center", &SphereDistanceResult::second_center)
      .def_readonly("normal", &SphereDistanceResult::normal);
  pybind11::class_<SphereDistanceGradientResult>(module,
                                                 "SphereDistanceGradientResult")
      .def_readonly("distance_result",
                    &SphereDistanceGradientResult::distance_result)
      .def_readonly("gradient", &SphereDistanceGradientResult::gradient);

  pybind11::class_<CollisionPairInfo>(module, "CollisionPairInfo")
      .def_readonly("first_geometry", &CollisionPairInfo::first_geometry)
      .def_readonly("second_geometry", &CollisionPairInfo::second_geometry)
      .def_readonly("first_link", &CollisionPairInfo::first_link)
      .def_readonly("second_link", &CollisionPairInfo::second_link);
  pybind11::class_<CollisionReport>(module, "CollisionReport")
      .def_readonly("in_collision", &CollisionReport::in_collision)
      .def_readonly("has_minimum_distance",
                    &CollisionReport::has_minimum_distance)
      .def_readonly("collisions", &CollisionReport::collisions)
      .def_readonly("minimum_distance", &CollisionReport::minimum_distance)
      .def_readonly("query_time_ms", &CollisionReport::query_time_ms);
  pybind11::class_<CollisionResult>(module, "CollisionResult")
      .def_readonly("pair_index", &CollisionResult::pair_index)
      .def_readonly("first_geometry", &CollisionResult::first_geometry)
      .def_readonly("second_geometry", &CollisionResult::second_geometry)
      .def_readonly("distance", &CollisionResult::distance)
      .def_readonly("nearest_point_first",
                    &CollisionResult::nearest_point_first)
      .def_readonly("nearest_point_second",
                    &CollisionResult::nearest_point_second);
  auto mesh = pybind11::class_<CollisionModel>(module, "CollisionModel");
  BindJointCoordinates(mesh);
  mesh.def(pybind11::init<const std::string &, const std::vector<std::string> &, bool>(),
           pybind11::arg("urdf_path"), pybind11::arg("package_dirs") = std::vector<std::string>{},
           pybind11::arg("exclude_adjacent") = true)
      .def("clone", &CollisionModel::Clone,
           "Deep-copy the current geometry and pairs with independent query "
           "workspaces; the source must have no active edits or queries.")
      .def_property_readonly("geometry_count", &CollisionModel::GetGeometryCount)
      .def_property_readonly("pair_count", &CollisionModel::GetCollisionPairCount)
      .def_property_readonly("geometry_names", &CollisionModel::GetGeometryNames)
      .def_property_readonly("collision_link_names", &CollisionModel::GetCollisionLinkNames)
      .def_property_readonly("collision_pairs", &CollisionModel::GetCollisionPairs)
      .def(
          "add_box_obstacle",
          [](CollisionModel &self, const std::string &name, const py::object &size,
             const py::object &pose) {
              const auto obstacle_size = CheckedRealEigen<Eigen::Vector3d>(size, "size");
              const auto obstacle_pose = CheckedRealEigen<Eigen::Matrix4d>(pose, "pose");
              self.AddBoxObstacle(name, obstacle_size, obstacle_pose);
          },
          pybind11::arg("name"), pybind11::arg("size"), pybind11::arg("pose"))
      .def("remove_obstacle", &CollisionModel::RemoveObstacle, pybind11::arg("name"))
      .def("remove_collision_pair", &CollisionModel::RemoveCollisionPair)
      .def("remove_collision_pairs_by_links", &CollisionModel::RemoveCollisionPairsByLinks,
           pybind11::arg("first_link"), pybind11::arg("second_link"))
      .def("remove_adjacent_pairs", &CollisionModel::RemoveAdjacentCollisionPairs)
      .def("reset_collision_pairs", &CollisionModel::ResetCollisionPairs,
           pybind11::arg("exclude_adjacent") = true)
      .def("clear_collision_pairs", &CollisionModel::ClearCollisionPairs)
      .def("set_collision_groups", &CollisionModel::SetCollisionGroups, pybind11::arg("groups"),
           pybind11::arg("group_pairs"))
      .def(
          "in_collision",
          [](CollisionModel &self, const py::object &configuration, bool stop_at_first) {
              return self.InCollision(CheckedRealEigen<Eigen::VectorXd>(
                                          configuration, "configuration"),
                                      stop_at_first);
          },
          pybind11::arg("configuration"), pybind11::arg("stop_at_first") = true)
      .def(
          "is_within_distance",
          [](CollisionModel &self, const py::object &configuration, const py::object &margin,
             bool stop_at_first) {
              const auto state = CheckedRealEigen<Eigen::VectorXd>(configuration, "configuration");
              const double security_margin = CheckedRealEigen<double>(margin, "security_margin");
              return self.IsWithinDistance(state, security_margin, stop_at_first);
          },
          pybind11::arg("configuration"), pybind11::arg("security_margin"),
          pybind11::arg("stop_at_first") = true)
      .def("collisions", WithRealInput(&CollisionModel::ComputeCollisions, "configuration"),
           pybind11::arg("configuration"))
      .def("minimum_distance", WithRealInput(&CollisionModel::MinimumDistance, "configuration"),
           pybind11::arg("configuration"))
      .def("evaluate", WithRealInput(&CollisionModel::Evaluate, "configuration"),
           pybind11::arg("configuration"));
  auto spheres = pybind11::class_<SphereCollisionModel>(module, "SphereCollisionModel");
  BindJointCoordinates(spheres);
  spheres
      .def(pybind11::init<const std::string &, std::vector<CollisionSphere>, bool>(),
           pybind11::arg("urdf_path"), pybind11::arg("spheres"),
           pybind11::arg("exclude_same_link") = true)
      .def_property_readonly("sphere_count", &SphereCollisionModel::GetSphereCount)
      .def_property_readonly("pair_count", &SphereCollisionModel::GetCollisionPairCount)
      .def_property_readonly("pair_revision", &SphereCollisionModel::GetCollisionPairRevision)
      // Mutable Python spheres must remain independent of model geometry.
      .def_property_readonly("spheres", &SphereCollisionModel::GetSpheres,
                             py::return_value_policy::copy)
      .def_property_readonly("collision_pairs", &SphereCollisionModel::GetCollisionPairs)
      .def("set_collision_groups", &SphereCollisionModel::SetCollisionGroups,
           pybind11::arg("groups"), pybind11::arg("group_pairs"))
      .def("reset_collision_pairs", &SphereCollisionModel::ResetCollisionPairs,
           pybind11::arg("exclude_same_link") = true)
      .def("clear_collision_pairs", &SphereCollisionModel::ClearCollisionPairs)
      .def("world_spheres",
           WithRealInput(&SphereCollisionModel::ComputeWorldSpheres, "configuration"),
           pybind11::arg("configuration"))
      .def(
          "in_collision",
          [](SphereCollisionModel &self, const py::object &configuration, const py::object &margin,
             bool stop_at_first) {
              const auto state = CheckedRealEigen<Eigen::VectorXd>(configuration, "configuration");
              const double security_margin = CheckedRealEigen<double>(margin, "security_margin");
              return self.InCollision(state, security_margin, stop_at_first);
          },
          pybind11::arg("configuration"), pybind11::arg("security_margin") = 0.0,
          pybind11::arg("stop_at_first") = true)
      .def("minimum_distance",
           WithRealInput(&SphereCollisionModel::MinimumDistance, "configuration"),
           pybind11::arg("configuration"))
      .def("minimum_distance_gradient",
           WithRealInput(&SphereCollisionModel::MinimumDistanceGradient, "configuration"),
           pybind11::arg("configuration"))
      .def("minimum_distance_with_gradient",
           WithRealInput(&SphereCollisionModel::MinimumDistanceWithGradient, "configuration"),
           pybind11::arg("configuration"))
      .def(
          "distances",
          [](SphereCollisionModel &self, const py::object &configuration,
             const py::object &maximum) {
              const auto state = CheckedRealEigen<Eigen::VectorXd>(configuration, "configuration");
              const double maximum_distance = CheckedRealEigen<double>(maximum, "maximum_distance");
              return self.ComputeDistances(state, maximum_distance);
          },
          pybind11::arg("configuration"), pybind11::arg("maximum_distance"))
      .def("batch_minimum_distances",
           WithRealInput(&SphereCollisionModel::BatchMinimumDistances, "configurations"),
           pybind11::arg("configurations"));
}

} // namespace holistic_motion::python

#else

namespace holistic_motion::python {
void BindCollision(pybind11::module_ &) {}
} // namespace holistic_motion::python

#endif
