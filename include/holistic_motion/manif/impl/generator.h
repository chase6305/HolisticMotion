#ifndef _HOLISTIC_MOTION_HOLISTIC_MOTION_GENERATOR_H_
#define _HOLISTIC_MOTION_HOLISTIC_MOTION_GENERATOR_H_

namespace holistic_motion {
namespace robotics {
namespace internal {

template <typename Derived>
struct GeneratorEvaluator {
    static typename Derived::LieAlg run(const unsigned int) {
        /// @todo print actual Derived type
        static_assert(constexpr_false<Derived>(),
                      "GeneratorEvaluator not overloaded for Derived type!");
    }
};

template <typename Derived>
struct InnerWeightsEvaluator {
    static typename Derived::InnerWeightsMatrix run() {
        using InnerWeightsMatrix = typename Derived::InnerWeightsMatrix;

        auto computeW = []() {
            InnerWeightsMatrix W = InnerWeightsMatrix::Zero();

            for (int r = 0; r < Derived::DoF; ++r)
                for (int c = 0; c < Derived::DoF; ++c)
                    W(r, c) = (Derived::GeneratorHelper(r) *
                               Derived::GeneratorHelper(c).transpose())
                                      .trace();

            return W;
        };

        const static InnerWeightsMatrix W = computeW();

        return W;
    }
};

}  // namespace internal
}  // namespace robotics
}  // namespace holistic_motion

#endif /* _HOLISTIC_MOTION_HOLISTIC_MOTION_GENERATOR_H_ */
