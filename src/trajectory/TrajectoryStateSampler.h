#pragma once

#include "PathSegmentEvaluation.h"
#include "TrajectoryStateEvaluation.h"
#include "holistic_motion/trajectory/TrajectoryBase.h"

#include <optional>

namespace holistic_motion::robotics::detail {

// A workspace belongs to one sampling call. Retain its current geometry while
// the evaluator borrows control points; custom geometry keeps virtual queries.
template <typename LieGroup> class TrajectoryStateSampler {
public:
    explicit TrajectoryStateSampler(const TrajectoryBase<LieGroup> &trajectory)
        : trajectory_(trajectory) {}

    typename TrajectoryBase<LieGroup>::State GetState(double time) {
        if (!trajectory_.valid_ || !trajectory_.trajectory_pspline_ ||
            !trajectory_.path_)
            throw std::logic_error("cannot query an invalid trajectory");
        if (!std::isfinite(time))
            throw std::invalid_argument("trajectory time must be finite");
        time = clamp(time, 0.0, trajectory_.GetDuration()) /
               trajectory_.time_scale_;
        std::size_t phase;
        std::array<double, 4> jet;
        // The hint preserves snapping and phase ownership and falls back to a
        // full lookup for backward or distant queries.
        if (sampling_phase_ + 1 <
            trajectory_.trajectory_pspline_->GetKnots().size())
            jet = trajectory_.trajectory_pspline_->ComputeJetInPhase(
                time, sampling_phase_, phase);
        else
            jet = trajectory_.trajectory_pspline_->ComputeJetAtS(time, phase);
        sampling_phase_ = phase;
        if (!std::isfinite(jet[0]))
            throw std::runtime_error(
                "trajectory evaluated a non-finite path parameter");
        const auto geometry =
            !trajectory_.phase_path_segments_.empty() &&
                    trajectory_.phase_path_segments_[phase]
                ? trajectory_.phase_path_segments_[phase]
                : trajectory_.path_->GetPathSegmentAtS(jet[0]);
        if (!geometry)
            throw std::logic_error("cannot query an invalid path");
        if (geometry != sampled_segment_) {
            sampler_.reset();
            sampled_segment_ = geometry;
            sampler_.emplace(*geometry);
        }
        typename TrajectoryBase<LieGroup>::State state;
        typename LieGroup::Tangent tangent;
        if (sampler_->ComputeLinearJet(jet[0], state.position, tangent)) {
            ComposeLinearDerivatives(state, jet, tangent,
                                     trajectory_.time_scale_);
        } else {
            typename LieGroup::Tangent curvature, torsion;
            sampler_->ComputeJet(jet[0], state.position, tangent, curvature,
                                 torsion);
            ComposeDerivatives(state, jet, tangent, curvature, torsion,
                               trajectory_.time_scale_);
        }
        return state;
    }

private:
    const TrajectoryBase<LieGroup> &trajectory_;
    std::shared_ptr<PathSegmentBase<LieGroup>> sampled_segment_;
    std::optional<SegmentEvaluationSampler<LieGroup>> sampler_;
    std::size_t sampling_phase_{0};
};

} // namespace holistic_motion::robotics::detail
