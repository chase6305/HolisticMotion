#pragma once
#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <stdexcept>

#include "holistic_motion/trajectory/Types.h"

namespace holistic_motion {
namespace robotics {

class Polynomial : public std::enable_shared_from_this<Polynomial> {
    ///< https://en.wikipedia.org/wiki/Polynomial
   public:
    ///@brief Default construction method
    Polynomial() : data_(Eigen::Vector4d::Zero()), coefficient_count_(0) {}

    Polynomial(const Eigen::Vector4d& data) : data_(data) {
        coefficient_count_ = static_cast<unsigned>(data_.size());
    }

    /// calculate polynomial value at s of given deriv_order
    inline double ComputePolyValueAtS(const double s,
                                      const unsigned order = 0) const {
        if (order >= coefficient_count_) return 0.0;

        double value = 0.0;
        for (int power = static_cast<int>(coefficient_count_) - 1;
             power >= static_cast<int>(order); --power) {
            double coefficient = data_[power];
            for (unsigned derivative = 0; derivative < order; ++derivative) {
                coefficient *= power - static_cast<int>(derivative);
            }
            value = value * s + coefficient;
        }
        return value;
    }

    ///@brief Get degree of polynomial
    inline unsigned GetDegree() const {
        return coefficient_count_ == 0 ? 0 : coefficient_count_ - 1;
    }

   private:
     friend class PSpline;

     std::array<double, 4> ComputeJet(double s) const {
         if (coefficient_count_ == 0)
             return {};
         // Share derivative coefficients while retaining the scalar evaluator's
         // multiplication order, including for subnormal coefficients.
         const double cubic = 0.0 + data_[3];
         const double cubic_derivative = 0.0 + data_[3] * 3.0;
         const double quadratic_derivative = data_[2] * 2.0;
         const double jerk = 0.0 + data_[3] * 3.0 * 2.0;
         return {((cubic * s + data_[2]) * s + data_[1]) * s + data_[0],
                 (cubic_derivative * s + quadratic_derivative) * s + data_[1],
                 jerk * s + quadratic_derivative, jerk};
     }

    Eigen::Vector4d data_;  ///< data in turn: pos, vel, acc, jerk

    unsigned coefficient_count_;
};

class PSpline {
    ///<
    ///< Eilers, Paul & Marx, Brian & Durbán, María. (2015). Twenty years of
    ///< P-splines. SORT (Statistics and Operations Research Transactions). 39.
    ///< 149-186.
   public:
    PSpline() : knots_(std::vector<double>(1, 0)) {}

    bool PushBack(const std::shared_ptr<Polynomial>& polynomial, double t = 1) {
        if (!polynomial || !std::isfinite(t) || t <= 0.0) return false;
        const double next = knots_.back() + t;
        if (!std::isfinite(next) || next <= knots_.back()) return false;
        this->polynomials_.push_back(polynomial);
        this->knots_.push_back(next);
        return true;
    }

    inline double GetLastTimeStamp() const { return knots_.back(); }

    const std::vector<double>& GetKnots() const { return knots_; }

    // calculate the p-form-spline value of given derivative order at s
    double ComputeValueAtS(double s, const unsigned order = 0) const {
        const auto index = LocatePolynomial(s);
        return polynomials_[index]->ComputePolyValueAtS(s, order);
    }

    /// Evaluate position through jerk after locating the spline segment once.
    std::array<double, 4> ComputeJetAtS(double s) const {
        std::size_t index;
        return ComputeJetAtS(s, index);
    }

    /// Also return the active phase index, with the same knot snapping policy.
    std::array<double, 4> ComputeJetAtS(double s, std::size_t& index) const {
        index = LocatePolynomial(s);
        return polynomials_[index]->ComputeJet(s);
    }

    unsigned GetDoF() const { return dof_; };

   private:
    template <typename LieGroup> friend class TrajectoryBase;

    // Diagnostics need the actual one-sided endpoint, without knot snapping
    // or a round trip through the externally scaled trajectory clock.
    std::array<double, 4> ComputePhaseEndpoint(std::size_t phase, bool at_end) const {
        return polynomials_[phase]->ComputeJet(
            at_end ? knots_[phase + 1] - knots_[phase] : 0.0);
    }

    double KnotTolerance(std::size_t index) const {
        // An unrelated long tail must not move earlier queries. Cap snapping
        // so short phases remain queryable, without a floor on the time unit.
        double span = std::numeric_limits<double>::infinity();
        if (index > 0)
            span = knots_[index] - knots_[index - 1];
        if (index + 1 < knots_.size())
            span = std::min(span, knots_[index + 1] - knots_[index]);
        return std::min(64.0 * std::numeric_limits<double>::epsilon() *
                            std::abs(knots_[index]),
                        0.25 * span);
    }

    std::array<double, 4> ComputeJetInPhase(double s, std::size_t phase,
                                          std::size_t& index) const {
        // Construction already walks the phases in order. Reuse that location
        // while retaining the public evaluator's exact knot-snapping behavior.
        const double start = knots_[phase];
        const double end = knots_[phase + 1];
        if (!std::isfinite(s) || s < start || s > end)
            return ComputeJetAtS(s, index);
        if (s == start) {
            index = phase;
            s = 0.0;
        } else if (end - s <= KnotTolerance(phase + 1)) {
            index = std::min(phase + 1, polynomials_.size() - 1);
            s = end - knots_[index];
        } else {
            index = phase;
            s = s - start <= KnotTolerance(phase) ? 0.0 : s - start;
        }
        holistic_motion::utility::LogDebug(
            "[PSpline] index:{}, knots_[index]:{}, local time:{}", index,
            knots_[index], s);
        return polynomials_[index]->ComputeJet(s);
    }

    /// Locate a right-continuous segment and convert s to segment-local time.
    std::size_t LocatePolynomial(double& s) const {
        if (polynomials_.empty()) {
            throw std::logic_error("cannot evaluate an empty PSpline");
        }
        if (!std::isfinite(s)) {
            throw std::invalid_argument("spline time must be finite");
        }
        s = clamp(s, knots_.front(), knots_.back());
        const auto nearest_right =
            std::lower_bound(knots_.begin(), knots_.end(), s);
        const auto right =
            static_cast<std::size_t>(nearest_right - knots_.begin());
        std::size_t index;
        if (*nearest_right - s <= KnotTolerance(right)) {
            s = *nearest_right;
            index = std::min(right, polynomials_.size() - 1);
        } else {
            // A finite, clamped query has a right knot. If it is not snapped
            // to that knot, lower_bound also identifies its containing segment.
            index = right - 1;
            if (s - knots_[index] <= KnotTolerance(index)) {
                s = knots_[index];
            }
        }

        s -= knots_[index];
        holistic_motion::utility::LogDebug(
            "[PSpline] index:{}, knots_[index]:{}, local time:{}", index,
            knots_[index], s);
        return index;
    }

   protected:
    std::vector<double> knots_;
    std::vector<std::shared_ptr<Polynomial>> polynomials_;
    unsigned dof_{1};  ///< DoF = 1
};

}  // namespace robotics
}  // namespace holistic_motion
