#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "OpenDRTParams.h"

namespace OpenDRTViewerCuda {

struct ProbeResult {
  bool available = false;
  bool interopReady = false;
  const char* deviceName = "";
  const char* reason = "";
};

struct StartupValidationResult {
  bool ready = false;
  std::string reason;
};

struct IdentityRequest {
  int resolution = 25;
  int showOverflow = 0;
  int highlightOverflow = 1;
  OpenDRTParams params{};
};

struct InputRequest {
  int pointCount = 0;
  int showOverflow = 0;
  int highlightOverflow = 1;
};

struct IdentityCache {
  unsigned int verts = 0;
  unsigned int colors = 0;
  std::uint64_t builtSerial = 0;
  int pointCount = 0;
  bool available = false;
  void* internal = nullptr;
};

struct InputCache {
  unsigned int verts = 0;
  unsigned int colors = 0;
  std::uint64_t builtSerial = 0;
  int pointCount = 0;
  bool available = false;
  void* internal = nullptr;
};

ProbeResult probe();
StartupValidationResult validateStartup();
void releaseIdentityCache(IdentityCache* cache);
void releaseInputCache(InputCache* cache);
bool buildIdentityMesh(IdentityCache* cache,
                       const IdentityRequest& request,
                       std::uint64_t serial,
                       std::string* error);
bool buildInputCloudMesh(InputCache* cache,
                         const InputRequest& request,
                         const std::vector<float>& rawPoints,
                         std::uint64_t serial,
                         std::string* error);

}  // namespace OpenDRTViewerCuda
