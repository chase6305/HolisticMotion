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

    Polynomial(const Eigen::Vector4d &data) : data_(data) {
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
        this->polynomials_.push_back(polynomial);
        this->knots_.push_back(t + knots_.back());
        return true;
    }

    inline double GetLastTimeStamp() const { return knots_.back(); }

    const std::vector<double>& GetKnots() const { return knots_; }

    // calculate the p-form-spline value of given derivative order at s
    double ComputeValueAtS(double s, const unsigned order = 0) const {
        const unsigned index = LocatePolynomial(s);
        return polynomials_[index]->ComputePolyValueAtS(s, order);
    }

    /// Evaluate position through jerk after locating the spline segment once.
    std::array<double, 4> ComputeJetAtS(double s) const {
        const unsigned index = LocatePolynomial(s);
        std::array<double, 4> result{};
        for (unsigned order = 0; order < result.size(); ++order) {
            result[order] =
                    polynomials_[index]->ComputePolyValueAtS(s, order);
        }
        return result;
    }

    unsigned GetDoF() const { return dof_; };

private:
    /// Locate a right-continuous segment and convert s to segment-local time.
    unsigned LocatePolynomial(double& s) const {
        if (polynomials_.empty()) {
            throw std::logic_error("cannot evaluate an empty PSpline");
        }
        const int num = static_cast<int>(knots_.size());
        s = clamp(s, knots_.front(), knots_.back());
        const double knot_tolerance =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, knots_.back());
        const auto nearest_right =
                std::lower_bound(knots_.begin(), knots_.end(), s);
        if (nearest_right != knots_.end() &&
            std::abs(*nearest_right - s) <= knot_tolerance) {
            s = *nearest_right;
        } else if (nearest_right != knots_.begin()) {
            const auto nearest_left = std::prev(nearest_right);
            if (std::abs(*nearest_left - s) <= knot_tolerance) {
                s = *nearest_left;
            }
        }

        holistic_motion::utility::LogDebug(
                "ComputeValueAtS PSpline, s:{}, num: {}, this->knots_[0]:{} "
                ",this->knots_[num-1]:{}",
                s, num, this->knots_[0], this->knots_[num - 1]);

        unsigned index = 0;
        if (s <= knots_.front()) {
            index = 0;
        } else if (s >= knots_[num - 2]) {
            index = num - 2;
        } else {
            index = static_cast<unsigned>(
                    std::upper_bound(knots_.begin(), knots_.end(), s) -
                    knots_.begin() - 1);
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
