#include "BuildIdentity.h"

#include <cstring>

namespace holistic_motion::utility::detail {

const char *CoreBuildIdentity() noexcept {
    return std::strcmp(ModelBuildIdentity(), HOLISTICMOTION_NATIVE_BUILD_ID) ==
                   0
               ? HOLISTICMOTION_NATIVE_BUILD_ID
               : "";
}

} // namespace holistic_motion::utility::detail
