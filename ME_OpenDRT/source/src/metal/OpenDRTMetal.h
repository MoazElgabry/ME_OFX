#pragma once

#include <cstddef>

#include "../OpenDRTParams.h"

namespace OpenDRTMetal {

bool render(
    const float* src,
    float* dst,
    int width,
    int height,
    size_t srcRowBytes,
    size_t dstRowBytes,
    const OpenDRTParams& params,
    const OpenDRTDerivedParams& derived);

}  // namespace OpenDRTMetal
