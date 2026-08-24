#pragma once

#include <type_traits>

namespace holistic_motion::robotics {

template <typename Enum>
constexpr auto to_underlying_type(Enum value) noexcept {
    return static_cast<std::underlying_type_t<Enum>>(value);
}

template <typename Value>
constexpr Value clamp(Value value, Value lower, Value upper) {
    return value < lower ? lower : (value > upper ? upper : value);
}

}  // namespace holistic_motion::robotics
