#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "OpenDRTParams.h"

struct GLFWwindow;

namespace OpenDRTViewerMetal {

struct ProbeResult {
  bool available = false;
  bool queueReady = false;
  const char* deviceName = "";
};

struct StartupValidationResult {
  bool ready = false;
  std::string reason;
};

struct PresenterInitResult {
  bool ready = false;
  std::string reason;
};

struct MeshCache {
  std::uint64_t builtSerial = 0;
  int pointCount = 0;
  bool available = false;
  void* internal = nullptr;
};

struct DrawSource {
  const float* cpuVerts = nullptr;
  const float* cpuColors = nullptr;
  void* vertsHandle = nullptr;
  void* colorsHandle = nullptr;
  int pointCount = 0;
  bool gpuBacked = false;
};

ProbeResult probe();
StartupValidationResult validateStartup();
PresenterInitResult initializePresenter(GLFWwindow* window);
void shutdownPresenter();

bool buildIdentityMesh(
    MeshCache* cache,
    const OpenDRTParams& params,
    int resolution,
    bool showOverflow,
    bool highlightOverflow,
    std::uint64_t serial,
    std::string* transformBackendLabel,
    float* maxDelta,
    std::string* errorOut);

bool buildInputCloudMesh(
    MeshCache* cache,
    const float* rawPoints,
    size_t rawPointFloatCount,
    bool showOverflow,
    bool highlightOverflow,
    std::uint64_t serial,
    std::string* errorOut);

bool resolveDrawSource(const MeshCache* cache, DrawSource* out);
void releaseCache(MeshCache* cache);

bool renderScene(GLFWwindow* window,
                 const DrawSource& source,
                 const float* mvp,
                 float pointSize,
                 float fillPointSize,
                 float clearR,
                 float clearG,
                 float clearB,
                 float clearA,
                 std::string* errorOut);

}  // namespace OpenDRTViewerMetal
