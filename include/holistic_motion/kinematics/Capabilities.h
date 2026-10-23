#pragma once

namespace holistic_motion::robotics {

enum class KinematicsFamily { CUSTOM, NUMERICAL, OPW, UR, SRS, FEP };

/// Algorithm used by the common GetIK/GetNearestIK entry points. Specialized
/// SRS/FEP Solve methods are separate and do not change this default.
enum class IKAlgorithm { CUSTOM, NUMERICAL, ANALYTIC };

/// Value snapshot of the current solver/model capabilities. These describe
/// available operations, not reachability or success for a particular target.
/// Concurrent queries always exclude configuration/model updates and require
/// caller-owned seeds and outputs. CUDA availability is queried separately so
/// inspection never initializes a device runtime.
struct KinematicsCapabilities {
    KinematicsFamily family{KinematicsFamily::CUSTOM};
    IKAlgorithm default_ik_algorithm{IKAlgorithm::CUSTOM};
    int dof{0};
    bool supports_jacobian{false};
    bool supports_redundancy{false};
    /// For SRS, the current model passes the default geometry tolerances;
    /// candidates still require branch, joint-limit and FK validation.
    bool supports_closed_form_ik{false};
    bool supports_batch_fk{false};
    bool cuda_compiled{false};
    /// Custom derived solvers must opt in explicitly. Python solve_ik releases
    /// the GIL when true, so overrides must support that execution context.
    bool supports_concurrent_queries{false};
};

} // namespace holistic_motion::robotics
