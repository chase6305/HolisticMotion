#pragma once

#include <cmath>

namespace holistic_motion::robotics::detail {

// Accumulate products before restoring their common binary scale. Individual
// terms may round to zero even when their combined norm is representable.
class ScaledNormAccumulator {
  public:
    void AddProduct(double first, double second) {
        if (!std::isfinite(first) || !std::isfinite(second)) {
            AddNonFinite(first * second);
            return;
        }
        int first_exponent = 0, second_exponent = 0;
        const double first_mantissa = std::frexp(first, &first_exponent);
        const double second_mantissa = std::frexp(second, &second_exponent);
        Add(first_mantissa * second_mantissa, first_exponent + second_exponent);
    }

    void AddProduct(double first, double second, double third) {
        if (!std::isfinite(first) || !std::isfinite(second) ||
            !std::isfinite(third)) {
            AddNonFinite(first * second * third);
            return;
        }
        int first_exponent = 0, second_exponent = 0, third_exponent = 0;
        const double first_mantissa = std::frexp(first, &first_exponent);
        const double second_mantissa = std::frexp(second, &second_exponent);
        const double third_mantissa = std::frexp(third, &third_exponent);
        Add(first_mantissa * second_mantissa * third_mantissa,
            first_exponent + second_exponent + third_exponent);
    }

    double Value() const { return std::scalbn(norm_, exponent_); }

  private:
    void Add(double mantissa, int exponent) {
        if (mantissa == 0.0)
            return;
        if (norm_ == 0.0) {
            norm_ = std::abs(mantissa);
            exponent_ = exponent;
        } else if (exponent > exponent_) {
            norm_ =
                std::hypot(std::scalbn(norm_, exponent_ - exponent), mantissa);
            exponent_ = exponent;
        } else {
            norm_ =
                std::hypot(norm_, std::scalbn(mantissa, exponent - exponent_));
        }
    }

    void AddNonFinite(double value) {
        norm_ = std::hypot(Value(), value);
        exponent_ = 0;
    }

    double norm_{0.0};
    int exponent_{0};
};

} // namespace holistic_motion::robotics::detail
