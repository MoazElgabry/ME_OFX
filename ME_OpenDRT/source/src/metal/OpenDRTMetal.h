#pragma once

#include "../OpenDRTParams.h"

namespace OpenDRTMetal {

bool render(const float* src, float* dst, int width, int height, const OpenDRTParams& params);

}  // namespace OpenDRTMetal
