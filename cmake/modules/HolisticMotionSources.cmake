set(HOLISTICMOTION_CORE_SOURCES
    src/utility/Logging.cpp
    src/robot/Robot.cpp
    src/robot/Joint.cpp
    src/kinematics/KinematicsBase.cpp
    src/kinematics/NumericalKinematics.cpp
    src/kinematics/OPWKinematics.cpp
    src/kinematics/URKinematics.cpp
    src/kinematics/utility.cpp
    src/kinematics/srs/SRSGeometry.cpp
    src/kinematics/srs/SRSKinematics.cpp
    src/kinematics/srs/SRSNullSpace.cpp
    src/kinematics/fep/FEPBatch.cpp
    src/kinematics/fep/FEPKinematics.cpp
    src/kinematics/fep/FEPSelection.cpp
    src/planning/NullSpacePlanner.cpp
    src/planning/SamplingPlanner.cpp
    src/trajectory/PathBase.cpp
    src/trajectory/PathBezierCurve.cpp
    src/trajectory/PathSegment.cpp
    src/trajectory/TrajectoryBase.cpp
    src/trajectory/TrajectoryDoubleS.cpp
    src/trajectory/TrajectoryTrapezium.cpp
    src/trajectory/Types.cpp)

set(HOLISTICMOTION_CUDA_SOURCES
    src/kinematics/fep/cuda/FEPSelection.cu
    src/kinematics/fep/cuda/FEPBatch.cu)

set(HOLISTICMOTION_COLLISION_SOURCES
    src/collision/CollisionModel.cpp
    src/collision/SphereCollisionModel.cpp)

set(HOLISTICMOTION_PYTHON_BINDING_SOURCES
    bindings/python/module.cpp
    bindings/python/PlanningBindings.cpp
    bindings/python/CollisionBindings.cpp)
