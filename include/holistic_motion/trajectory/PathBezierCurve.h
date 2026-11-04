#pragma once

#include "holistic_motion/trajectory/PathSegment.h"

namespace holistic_motion {
namespace robotics {

template <typename LieGroup>
class PathBezierCurve : public PathBase<LieGroup> {
private:
    using Tangent = typename LieGroup::Tangent;

public:
  virtual ~PathBezierCurve() = default;

  PathBezierCurve(const std::vector<LieGroup> &waypoints, const int degree = 2,
                  const bool is_cartesian_space = true,
                  const double blend_tolerance = 0.5);

  /// \brief Constructing the path of the 2nd-bezier-curve
  /// Rebuilds from stored waypoints, replacing previous segments.
  void PathBezierCurve2nd();

  /// \brief Constructing the path of the 5th-bezier-curve
  /// Rebuilds from stored waypoints, replacing previous segments.
  void PathBezierCurve5th();

private:
  void Build(int degree);
  PathBezierCurve() = default;
};

}  // namespace robotics
}  // namespace holistic_motion
