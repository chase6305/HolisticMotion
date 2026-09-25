#pragma once

#include <stdexcept>

#include "holistic_motion/trajectory/Types.h"

namespace holistic_motion {
namespace robotics {

template <typename LieGroup> class TrajectoryBase;
template <typename LieGroup> class TrajectoryDoubleS;
template <typename LieGroup> class TrajectoryTrapezium;

template <typename LieGroup>
class PathSegmentBase : public std::enable_shared_from_this<PathSegmentBase<LieGroup>> {
private:
    using Tangent = typename LieGroup::Tangent;
    friend class TrajectoryBase<LieGroup>;
    friend class TrajectoryDoubleS<LieGroup>;
    friend class TrajectoryTrapezium<LieGroup>;

    // Share geometry work for a complete trajectory state. Custom segments
    // retain their virtual scalar-query behavior.
    void ComputeJet(double s, LieGroup &position, Tangent &tangent, Tangent &curvature,
                    Tangent &torsion) const;
    void ComputeDerivatives(double s, Tangent &tangent, Tangent &curvature,
                            Tangent &torsion) const;

public:
    virtual ~PathSegmentBase() {
        holistic_motion::utility::LogDebug("Destructing PathSegmentBase");
    }

    /// \brief judge whether there is a trajectory length to determine its
    /// effectiveness
    bool IsValid() const {
        return std::isfinite(length_) && length_ >= 0.0 && std::isfinite(sp_) &&
               std::isfinite(sp_ + length_);
    }

    double GetStartParameter() const { return sp_; }

    /// \brief get the type of the path segment
    ///
    /// \return PathSegType
    PathSegType GetPathSegType() const { return path_seg_type_; };

    /// \brief get the length of the path segment
    ///
    /// \param target Eigen::VectorXd
    /// \param weights Eigen::VectorXd
    /// \return double
    double GetLength() const { return length_; }

    /// \brief get the config of the s path
    ///
    /// \param s path parameter
    /// \return Eigen::VectorXd
    virtual LieGroup GetConfig(double s) const = 0;

    /// \brief get the tangent of the s path
    ///
    /// \param s path parameter
    /// \return Eigen::VectorXd
    virtual Tangent GetTangent(double s) const = 0;

    /// \brief get the curvature of the s path
    ///
    /// \param s path parameter
    /// \return Eigen::VectorXd
    virtual Tangent GetCurvature(double s) const {
        ValidateQuery(s);
        return Tangent::ZeroHelper();
    };

    /// \brief get the curvature of the s path
    ///
    /// \param s path parameter
    /// \return Eigen::VectorXd
    virtual Tangent GetTorsion(double s) const {
        ValidateQuery(s);
        return Tangent::ZeroHelper();
    };

protected:
    // Native segments use the same query contract as PathBase: reject non-finite
    // parameters and invalid geometry before clamping or accessing control points.
    void ValidateQuery(double s) const {
        if (!std::isfinite(s))
            throw std::invalid_argument("path parameter must be finite");
        if (!IsValid())
            throw std::logic_error("cannot query an invalid path segment");
    }

    PathSegmentBase() = default;
    PathSegType path_seg_type_{PathSegType::LinearSeg};
    std::vector<LieGroup> waypoints_; ///< path blending tolerance
    double length_{0.0};              ///< path blending tolerance
    double sp_{0.0};                  ///< start param of segment
    Tangent tangent_;                 ///< tangent of segment
};

template <typename LieGroup>
class PathBase : public std::enable_shared_from_this<PathBase<LieGroup>> {
private:
    using Tangent = typename LieGroup::Tangent;

public:
    virtual ~PathBase() { holistic_motion::utility::LogDebug("Destructing PathBase"); };

    /// \brief check colinear or overlap
    ///  waypoints need to be 1x7,
    ///  Position and attitude are represented by 4x4 matrix or 1x7 (xyz and
    ///  quaternion), Joints are represented by 1x6(currently commonly used as
    ///  6-axis industrial robotic arms)
    ///
    /// \param waypoints waypoints of the path
    void _CheckPathWaypoints(std::vector<LieGroup> &waypoints);

    /// \brief judge whether the path is valid
    bool IsValid() const { return this->valid_; };

    /// \brief get the path type
    PathType GetType() const { return this->path_type_; }

    /// \brief get the waypoints of the path
    const std::vector<LieGroup> &GetWaypoints() const { return waypoints_; }

    /// \brief get the blend tolerance of the path
    double GetBlendTolerance() const { return this->blend_tolerance_; }

    /// \brief get the length of the path
    ///
    /// \param target Eigen::VectorXd
    /// \param weights Eigen::VectorXd
    /// \return double
    double GetLength() const { return this->length_; }

    /// \brief get all segment of the path
    ///
    /// \param s path parameter
    /// \return std::shared_ptr<PathSegmentBase>
    std::vector<std::shared_ptr<PathSegmentBase<LieGroup>>> GetPathSegments() const {
        return path_segments_;
    }

    /// \brief get number of the path segments
    ///
    /// \return int
    int GetNumOfPathSegments() const { return path_segments_.size(); }

    /// \brief get the segment at a finite path parameter
    ///
    /// \param s path parameter
    /// \return std::shared_ptr<PathSegmentBase>
    /// Finite out-of-range parameters are clamped; internal boundaries select
    /// the following segment. Returns nullptr if the path is invalid.
    /// \throws std::invalid_argument if s is not finite.
    std::shared_ptr<PathSegmentBase<LieGroup>> GetPathSegmentAtS(double s) const;

    /// \brief get the segment of the path by index
    ///
    /// \param index path segment index
    /// \return std::shared_ptr<PathSegmentBase>
    /// Returns nullptr if the path is invalid; valid paths clamp the index.
    std::shared_ptr<PathSegmentBase<LieGroup>>
    GetPathSegmentByIndex(const int &index) const;

    /// \brief get the segment of the path
    ///
    /// \param s path parameter
    /// \return std::shared_ptr<PathSegmentBase>
    // std::shared_ptr<PathSegmentBase> GetPathSegment(double s) const;

    /// \brief get the config of the s path
    ///
    /// \param s path parameter
    /// \return Eigen::VectorXd
    LieGroup GetConfig(double s) const;

    /// \brief get the tangent of the s path
    ///
    /// \param s path parameter
    /// \return Eigen::VectorXd
    Tangent GetTangent(double s) const;

    /// \brief get the curvature of the s path
    ///
    /// \param s path parameter
    /// \return Eigen::VectorXd
    Tangent GetCurvature(double s) const;

    /// \brief get the torsion of the path
    ///
    /// \param s path parameter
    /// \return Eigen::VectorXd
    Tangent GetTorsion(double s) const;

protected:
    PathBase() = default;
    PathType path_type_{PathType::NoBlend};
    bool is_cartesian_space_{false};
    std::vector<LieGroup> waypoints_;      ///< waypoints of the path
    std::vector<LieGroup> control_points_; ///< control points of the path
    std::vector<std::shared_ptr<PathSegmentBase<LieGroup>>> path_segments_;
    double blend_tolerance_{0.0}; ///< path blending tolerance
    bool valid_{false};           ///< if path is valid
    double length_{0.0};          ///< path length
    Eigen::VectorXd weights_;
};

} // namespace robotics
} // namespace holistic_motion
