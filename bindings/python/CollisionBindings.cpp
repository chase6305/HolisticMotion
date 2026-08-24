#include "Bindings.h"

#ifdef HOLISTICMOTION_HAS_COLLISION

#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "holistic_motion/collision/CollisionModel.h"

namespace holistic_motion::python {

void BindCollision(pybind11::module_& module) {
    using robotics::collision::CollisionModel;
    using robotics::collision::CollisionPairInfo;
    using robotics::collision::CollisionReport;
    using robotics::collision::CollisionResult;

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
            .def_readonly("minimum_distance",
                          &CollisionReport::minimum_distance)
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
    pybind11::class_<CollisionModel>(module, "CollisionModel")
            .def(pybind11::init<const std::string&,
                                const std::vector<std::string>&, bool>(),
                 pybind11::arg("urdf_path"),
                 pybind11::arg("package_dirs") = std::vector<std::string>{},
                 pybind11::arg("exclude_adjacent") = true)
            .def_property_readonly("nq", &CollisionModel::GetConfigurationSize)
            .def_property_readonly("nv", &CollisionModel::GetVelocitySize)
            .def_property_readonly("geometry_count",
                                   &CollisionModel::GetGeometryCount)
            .def_property_readonly("pair_count",
                                   &CollisionModel::GetCollisionPairCount)
            .def_property_readonly("geometry_names",
                                   &CollisionModel::GetGeometryNames)
            .def_property_readonly("collision_link_names",
                                   &CollisionModel::GetCollisionLinkNames)
            .def_property_readonly("collision_pairs",
                                   &CollisionModel::GetCollisionPairs)
            .def("neutral_configuration",
                 &CollisionModel::NeutralConfiguration)
            .def("configuration_from_joint_positions",
                 &CollisionModel::ConfigurationFromJointPositions,
                 pybind11::arg("joint_positions"))
            .def("remove_collision_pair", &CollisionModel::RemoveCollisionPair)
            .def("remove_collision_pairs_by_links",
                 &CollisionModel::RemoveCollisionPairsByLinks,
                 pybind11::arg("first_link"), pybind11::arg("second_link"))
            .def("remove_adjacent_pairs",
                 &CollisionModel::RemoveAdjacentCollisionPairs)
            .def("reset_collision_pairs", &CollisionModel::ResetCollisionPairs,
                 pybind11::arg("exclude_adjacent") = true)
            .def("clear_collision_pairs", &CollisionModel::ClearCollisionPairs)
            .def("set_collision_groups", &CollisionModel::SetCollisionGroups,
                 pybind11::arg("groups"), pybind11::arg("group_pairs"))
            .def("in_collision", &CollisionModel::InCollision,
                 pybind11::arg("configuration"),
                 pybind11::arg("stop_at_first") = true)
            .def("is_within_distance", &CollisionModel::IsWithinDistance,
                 pybind11::arg("configuration"),
                 pybind11::arg("security_margin"),
                 pybind11::arg("stop_at_first") = true)
            .def("collisions", &CollisionModel::ComputeCollisions,
                 pybind11::arg("configuration"))
            .def("minimum_distance", &CollisionModel::MinimumDistance,
                 pybind11::arg("configuration"))
            .def("evaluate", &CollisionModel::Evaluate,
                 pybind11::arg("configuration"));
}

}  // namespace holistic_motion::python

#else

namespace holistic_motion::python {
void BindCollision(pybind11::module_&) {}
}  // namespace holistic_motion::python

#endif
