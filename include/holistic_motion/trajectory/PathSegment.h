#pragma once

#include "holistic_motion/trajectory/PathBase.h"

namespace holistic_motion {
namespace robotics {

/// \brief Constructing the path segment of the linear
///
/// \TODO:
template <typename LieGroup>
class PathSegLinear : public PathSegmentBase<LieGroup> {
private:
    using Tangent = typename LieGroup::Tangent;

public:
    PathSegLinear(const std::array<LieGroup, 2>& waypoints,
                  const double& s0 = 0.0,
                  const bool& is_cartesian_space = false);
    ~PathSegLinear() { holistic_motion::utility::LogDebug("Destructing PathSegLinear."); };

    std::shared_ptr<PathSegmentBase<LieGroup>> Copy() const {
        return std::make_shared<PathSegLinear<LieGroup>>(*this);
    }

    virtual LieGroup GetConfig(double s) const override;
    virtual Tangent GetTangent(double s) const override;

private:
    PathSegLinear() = default;
};

template <typename LieGroup>
class PathSegBezierCurve2nd : public PathSegmentBase<LieGroup> {
private:
    using Tangent = typename LieGroup::Tangent;

public:
    PathSegBezierCurve2nd(const std::array<LieGroup, 3>& waypoints,
                          const double& sp = 0.0,
                          const bool& is_cartesian_space = false);
    ~PathSegBezierCurve2nd() {
        holistic_motion::utility::LogDebug("Destructing PathSegBezierCurve2nd.");
    };

    std::shared_ptr<PathSegmentBase<LieGroup>> Copy() const {
        return std::make_shared<PathSegBezierCurve2nd<LieGroup>>(*this);
    }

    virtual LieGroup GetConfig(double s) const override;
    virtual Tangent GetTangent(double s) const override;
    virtual Tangent GetCurvature(double s) const override;

protected:
    std::vector<LieGroup>
            control_points_;  ///< control points of Bezier segment

private:
    PathSegBezierCurve2nd() = default;
};

template <typename LieGroup>
class PathSegBezierCurve5th : public PathSegmentBase<LieGroup> {
private:
    using Tangent = typename LieGroup::Tangent;

public:
    PathSegBezierCurve5th(const std::array<LieGroup, 3>& waypoints,
                          const double& sp,
                          const double& tstart_norm = 1.,
                          const double& cstart_norm = 0.,
                          const double& tend_norm = 1.,
                          const double& cend_norm = 0.,
                          const bool& is_cartesian_space = false);
    ~PathSegBezierCurve5th() {
        holistic_motion::utility::LogDebug("Destructing PathSegBezierCurve5th.");
    };

    std::shared_ptr<PathSegmentBase<LieGroup>> Copy() const {
        return std::make_shared<PathSegBezierCurve5th<LieGroup>>(*this);
    }

    virtual LieGroup GetConfig(double s) const override;
    virtual Tangent GetTangent(double s) const override;
    virtual Tangent GetCurvature(double s) const override;
    virtual Tangent GetTorsion(double s) const override;

protected:
    std::vector<LieGroup>
            control_points_;  ///< control points of Bezier segment

private:
    PathSegBezierCurve5th() = default;
};

}  // namespace robotics
}  // namespace holistic_motion
