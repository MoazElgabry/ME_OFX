#include "OpenDRTViewerCuda.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#if defined(_WIN32)
#include <GL/gl.h>
#else
#include <GL/gl.h>
#endif

#include "OpenDRTProcessor.h"

namespace OpenDRTViewerCuda {
namespace {

struct CudaContext {
  bool initAttempted = false;
  bool ready = false;
  bool interopReady = false;
  int device = -1;
  std::string deviceName;
  std::string reason;
};

struct IdentityKernelUniforms {
  unsigned int resolution = 0;
  unsigned int interiorStep = 0;
  int showOverflow = 0;
  int highlightOverflow = 0;
};

struct InputKernelUniforms {
  int pointCount = 0;
  int showOverflow = 0;
  int highlightOverflow = 0;
};

struct CacheImpl {
  cudaGraphicsResource* vertsResource = nullptr;
  cudaGraphicsResource* colorsResource = nullptr;
  GLuint registeredVerts = 0;
  GLuint registeredColors = 0;
  size_t pointCapacity = 0;
  float* deviceSrc = nullptr;
  float* deviceDst = nullptr;
  size_t devicePointCapacity = 0;
  float* deviceInput = nullptr;
  size_t inputCapacityFloats = 0;
  unsigned int* deviceCounter = nullptr;
};

CudaContext& context() {
  static CudaContext ctx;
  return ctx;
}

const char* errorString(cudaError_t err) {
  return cudaGetErrorString(err);
}

bool ensureContext(std::string* error) {
  static std::once_flag once;
  CudaContext& ctx = context();
  std::call_once(once, []() {
    CudaContext& c = context();
    c.initAttempted = true;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
      c.reason = std::string("cudaGetDeviceCount failed: ") + errorString(err);
      return;
    }
    if (deviceCount <= 0) {
      c.reason = "No CUDA devices found.";
      return;
    }

    unsigned int glCount = 0;
    int glDevices[8] = {};
    err = cudaGLGetDevices(&glCount, glDevices, 8, cudaGLDeviceListAll);
    if (err != cudaSuccess || glCount == 0) {
      c.reason = std::string("CUDA-GL interop probe failed: ") +
                 errorString(err == cudaSuccess ? cudaErrorUnknown : err);
      return;
    }

    c.device = glDevices[0];
    err = cudaSetDevice(c.device);
    if (err != cudaSuccess) {
      c.reason = std::string("cudaSetDevice failed: ") + errorString(err);
      return;
    }
    err = cudaFree(0);
    if (err != cudaSuccess) {
      c.reason = std::string("CUDA warm-up failed: ") + errorString(err);
      return;
    }

    cudaDeviceProp prop{};
    err = cudaGetDeviceProperties(&prop, c.device);
    if (err != cudaSuccess) {
      c.reason = std::string("cudaGetDeviceProperties failed: ") + errorString(err);
      return;
    }

    c.deviceName = prop.name;
    c.interopReady = true;
    c.ready = true;
  });

  if (!ctx.ready && error) *error = ctx.reason;
  return ctx.ready;
}

template <typename CacheT>
CacheImpl* ensureImpl(CacheT* cache) {
  if (!cache) return nullptr;
  if (!cache->internal) cache->internal = new CacheImpl();
  return reinterpret_cast<CacheImpl*>(cache->internal);
}

void releaseImpl(CacheImpl* impl) {
  if (!impl) return;
  if (impl->vertsResource) cudaGraphicsUnregisterResource(impl->vertsResource);
  if (impl->colorsResource) cudaGraphicsUnregisterResource(impl->colorsResource);
  if (impl->deviceSrc) cudaFree(impl->deviceSrc);
  if (impl->deviceDst) cudaFree(impl->deviceDst);
  if (impl->deviceInput) cudaFree(impl->deviceInput);
  if (impl->deviceCounter) cudaFree(impl->deviceCounter);
  delete impl;
}

template <typename CacheT>
void releaseCache(CacheT* cache) {
  if (!cache) return;
  releaseImpl(reinterpret_cast<CacheImpl*>(cache->internal));
  cache->internal = nullptr;
  cache->builtSerial = 0;
  cache->pointCount = 0;
  cache->available = false;
}

bool ensureRegistered(GLuint verts, GLuint colors, size_t pointCapacity, CacheImpl* impl, std::string* error) {
  if (!impl || verts == 0 || colors == 0) {
    if (error) *error = "Missing GL buffers for CUDA interop.";
    return false;
  }
  if (impl->registeredVerts == verts && impl->registeredColors == colors &&
      impl->pointCapacity == pointCapacity && impl->vertsResource && impl->colorsResource) {
    return true;
  }

  if (impl->vertsResource) {
    cudaGraphicsUnregisterResource(impl->vertsResource);
    impl->vertsResource = nullptr;
  }
  if (impl->colorsResource) {
    cudaGraphicsUnregisterResource(impl->colorsResource);
    impl->colorsResource = nullptr;
  }

  cudaError_t err = cudaGraphicsGLRegisterBuffer(&impl->vertsResource, verts, cudaGraphicsRegisterFlagsWriteDiscard);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to register CUDA verts buffer: ") + errorString(err);
    return false;
  }
  err = cudaGraphicsGLRegisterBuffer(&impl->colorsResource, colors, cudaGraphicsRegisterFlagsWriteDiscard);
  if (err != cudaSuccess) {
    cudaGraphicsUnregisterResource(impl->vertsResource);
    impl->vertsResource = nullptr;
    if (error) *error = std::string("Failed to register CUDA colors buffer: ") + errorString(err);
    return false;
  }

  impl->registeredVerts = verts;
  impl->registeredColors = colors;
  impl->pointCapacity = pointCapacity;
  return true;
}

bool ensureIdentityCapacity(CacheImpl* impl, size_t pointCount, std::string* error) {
  if (!impl) return false;
  if (impl->devicePointCapacity >= pointCount && impl->deviceSrc && impl->deviceDst && impl->deviceCounter) {
    return true;
  }
  if (impl->deviceSrc) cudaFree(impl->deviceSrc);
  if (impl->deviceDst) cudaFree(impl->deviceDst);
  if (impl->deviceCounter) cudaFree(impl->deviceCounter);
  impl->deviceSrc = nullptr;
  impl->deviceDst = nullptr;
  impl->deviceCounter = nullptr;
  impl->devicePointCapacity = 0;

  cudaError_t err = cudaMalloc(&impl->deviceSrc, pointCount * 4u * sizeof(float));
  if (err == cudaSuccess) err = cudaMalloc(&impl->deviceDst, pointCount * 4u * sizeof(float));
  if (err == cudaSuccess) err = cudaMalloc(&impl->deviceCounter, sizeof(unsigned int));
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to allocate CUDA identity buffers: ") + errorString(err);
    return false;
  }
  impl->devicePointCapacity = pointCount;
  return true;
}

bool ensureInputCapacity(CacheImpl* impl, size_t floatCount, std::string* error) {
  if (!impl) return false;
  if (impl->inputCapacityFloats >= floatCount && impl->deviceInput) return true;
  if (impl->deviceInput) {
    cudaFree(impl->deviceInput);
    impl->deviceInput = nullptr;
    impl->inputCapacityFloats = 0;
  }
  cudaError_t err = cudaMalloc(&impl->deviceInput, floatCount * sizeof(float));
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to allocate CUDA input buffer: ") + errorString(err);
    return false;
  }
  impl->inputCapacityFloats = floatCount;
  return true;
}

inline __device__ float clamp01(float v) {
  return fminf(fmaxf(v, 0.0f), 1.0f);
}

inline __device__ bool outOfBounds(float r, float g, float b) {
  return r < 0.0f || r > 1.0f || g < 0.0f || g > 1.0f || b < 0.0f || b > 1.0f;
}

inline __device__ void mapDisplayColor(float inR, float inG, float inB, float* outR, float* outG, float* outB) {
  float r = powf(clamp01(inR), 0.90f);
  float g = powf(clamp01(inG), 0.90f);
  float b = powf(clamp01(inB), 0.90f);
  const float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
  r = clamp01(luma + (r - luma));
  g = clamp01(luma + (g - luma));
  b = clamp01(luma + (b - luma));
  *outR = r;
  *outG = g;
  *outB = b;
}

__global__ void identityPackKernel(float* verts,
                                   float* colors,
                                   const float* srcVals,
                                   const float* dstVals,
                                   unsigned int* outCount,
                                   IdentityKernelUniforms u) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int res = u.resolution;
  const unsigned int count = res * res * res;
  if (index >= count) return;

  const unsigned int plane = res * res;
  const unsigned int z = index / plane;
  const unsigned int rem = index - z * plane;
  const unsigned int y = rem / res;
  const unsigned int x = rem - y * res;
  const bool onBoundary =
      (x == 0u || y == 0u || z == 0u || x == (res - 1u) || y == (res - 1u) || z == (res - 1u));
  if (!onBoundary) {
    if ((x % u.interiorStep) != 0u || (y % u.interiorStep) != 0u || (z % u.interiorStep) != 0u) return;
  }

  const unsigned int base = index * 4u;
  const float sr = srcVals[base + 0u];
  const float sg = srcVals[base + 1u];
  const float sb = srcVals[base + 2u];
  const float dr = dstVals[base + 0u];
  const float dg = dstVals[base + 1u];
  const float db = dstVals[base + 2u];
  const bool overflowPoint = outOfBounds(dr, dg, db);
  const float plotR = (u.showOverflow != 0) ? dr : clamp01(dr);
  const float plotG = (u.showOverflow != 0) ? dg : clamp01(dg);
  const float plotB = (u.showOverflow != 0) ? db : clamp01(db);

  float cr = 0.0f;
  float cg = 0.0f;
  float cb = 0.0f;
  if (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint) {
    cr = 1.0f;
    cg = 0.0f;
    cb = 0.0f;
  } else {
    const float mixR = sr * 0.86f + clamp01(dr) * 0.14f;
    const float mixG = sg * 0.86f + clamp01(dg) * 0.14f;
    const float mixB = sb * 0.86f + clamp01(db) * 0.14f;
    mapDisplayColor(mixR, mixG, mixB, &cr, &cg, &cb);
  }

  const unsigned int outIndex = atomicAdd(outCount, 1u);
  const unsigned int outBase = outIndex * 3u;
  verts[outBase + 0u] = plotR * 2.0f - 1.0f;
  verts[outBase + 1u] = plotG * 2.0f - 1.0f;
  verts[outBase + 2u] = plotB * 2.0f - 1.0f;
  colors[outBase + 0u] = cr;
  colors[outBase + 1u] = cg;
  colors[outBase + 2u] = cb;
}

__global__ void inputPackKernel(float* verts,
                                float* colors,
                                const float* rawVals,
                                InputKernelUniforms u) {
  const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int total = (u.pointCount > 0) ? static_cast<unsigned int>(u.pointCount) : 0u;
  if (index >= total) return;

  const unsigned int inBase = index * 6u;
  const float sr = rawVals[inBase + 0u];
  const float sg = rawVals[inBase + 1u];
  const float sb = rawVals[inBase + 2u];
  const float dr = rawVals[inBase + 3u];
  const float dg = rawVals[inBase + 4u];
  const float db = rawVals[inBase + 5u];
  const bool overflowPoint = outOfBounds(dr, dg, db);
  const float plotR = (u.showOverflow != 0) ? dr : clamp01(dr);
  const float plotG = (u.showOverflow != 0) ? dg : clamp01(dg);
  const float plotB = (u.showOverflow != 0) ? db : clamp01(db);

  const unsigned int outBase = index * 3u;
  verts[outBase + 0u] = plotR * 2.0f - 1.0f;
  verts[outBase + 1u] = plotG * 2.0f - 1.0f;
  verts[outBase + 2u] = plotB * 2.0f - 1.0f;

  float cr = 0.0f;
  float cg = 0.0f;
  float cb = 0.0f;
  if (u.showOverflow != 0 && u.highlightOverflow != 0 && overflowPoint) {
    cr = 1.0f;
    cg = 0.0f;
    cb = 0.0f;
  } else {
    mapDisplayColor(clamp01(sr), clamp01(sg), clamp01(sb), &cr, &cg, &cb);
  }
  colors[outBase + 0u] = cr;
  colors[outBase + 1u] = cg;
  colors[outBase + 2u] = cb;
}

size_t expectedIdentityPointCount(int res) {
  const int interiorStep = (res <= 25) ? 2 : (res <= 41 ? 2 : 3);
  size_t count = 0;
  for (int z = 0; z < res; ++z) {
    for (int y = 0; y < res; ++y) {
      for (int x = 0; x < res; ++x) {
        const bool onBoundary = (x == 0 || y == 0 || z == 0 || x == res - 1 || y == res - 1 || z == res - 1);
        if (!onBoundary &&
            (((x % interiorStep) != 0) || ((y % interiorStep) != 0) || ((z % interiorStep) != 0))) {
          continue;
        }
        ++count;
      }
    }
  }
  return count;
}

}  // namespace

ProbeResult probe() {
  ProbeResult result{};
  std::string error;
  result.available = ensureContext(&error);
  CudaContext& ctx = context();
  result.interopReady = ctx.interopReady;
  result.deviceName = ctx.deviceName.c_str();
  result.reason = ctx.reason.c_str();
  return result;
}

StartupValidationResult validateStartup() {
  StartupValidationResult result{};
  std::string error;
  if (!ensureContext(&error)) {
    result.reason = error.empty() ? std::string("CUDA viewer context unavailable.") : error;
    return result;
  }
  result.ready = true;
  return result;
}

void releaseIdentityCache(IdentityCache* cache) {
  releaseCache(cache);
}

void releaseInputCache(InputCache* cache) {
  releaseCache(cache);
}

bool buildIdentityMesh(IdentityCache* cache,
                       const IdentityRequest& request,
                       std::uint64_t serial,
                       std::string* error) {
  if (!cache || cache->verts == 0 || cache->colors == 0) {
    if (error) *error = "CUDA identity cache has no GL buffers.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }

  const int res = std::max(request.resolution, 2);
  const size_t pointCount = static_cast<size_t>(res) * static_cast<size_t>(res) * static_cast<size_t>(res);
  CacheImpl* impl = ensureImpl(cache);
  if (!impl) {
    if (error) *error = "Failed to allocate CUDA identity cache.";
    return false;
  }
  if (!ensureRegistered(cache->verts, cache->colors, pointCount, impl, &localError)) {
    if (error) *error = localError;
    return false;
  }
  if (!ensureIdentityCapacity(impl, pointCount, &localError)) {
    if (error) *error = localError;
    return false;
  }

  std::vector<float> src(pointCount * 4u, 1.0f);
  const float denom = static_cast<float>(res - 1);
  size_t offset = 0;
  for (int z = 0; z < res; ++z) {
    for (int y = 0; y < res; ++y) {
      for (int x = 0; x < res; ++x) {
        src[offset + 0u] = static_cast<float>(x) / denom;
        src[offset + 1u] = static_cast<float>(y) / denom;
        src[offset + 2u] = static_cast<float>(z) / denom;
        src[offset + 3u] = 1.0f;
        offset += 4u;
      }
    }
  }

  cudaError_t err =
      cudaMemcpy(impl->deviceSrc, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to upload CUDA identity lattice: ") + errorString(err);
    return false;
  }

  OpenDRTProcessor processor(request.params);
  const size_t packedRowBytes = pointCount * 4u * sizeof(float);
  if (!processor.renderCUDAHostBuffers(
          impl->deviceSrc,
          impl->deviceDst,
          static_cast<int>(pointCount),
          1,
          packedRowBytes,
          packedRowBytes,
          nullptr)) {
    if (error) *error = "OpenDRT CUDA identity transform failed.";
    return false;
  }

  unsigned int zero = 0u;
  err = cudaMemcpy(impl->deviceCounter, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to reset CUDA identity counter: ") + errorString(err);
    return false;
  }

  std::array<cudaGraphicsResource*, 2> resources = {impl->vertsResource, impl->colorsResource};
  err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to map CUDA identity draw buffers: ") + errorString(err);
    return false;
  }

  float* devVerts = nullptr;
  float* devColors = nullptr;
  size_t vertsBytes = 0;
  size_t colorsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devVerts), &vertsBytes, impl->vertsResource);
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devColors), &colorsBytes, impl->colorsResource);
  }
  if (err != cudaSuccess) {
    cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
    if (error) *error = std::string("Failed to access CUDA identity draw buffers: ") + errorString(err);
    return false;
  }

  IdentityKernelUniforms uniforms{};
  uniforms.resolution = static_cast<unsigned int>(res);
  uniforms.interiorStep = static_cast<unsigned int>((res <= 25) ? 2 : (res <= 41 ? 2 : 3));
  uniforms.showOverflow = request.showOverflow;
  uniforms.highlightOverflow = request.highlightOverflow;

  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((pointCount + threads - 1u) / threads);
  identityPackKernel<<<blocks, threads>>>(devVerts, devColors, impl->deviceSrc, impl->deviceDst, impl->deviceCounter, uniforms);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA identity pack kernel failed: ") + errorString(err);
    return false;
  }

  unsigned int outCount = 0u;
  err = cudaMemcpy(&outCount, impl->deviceCounter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to read CUDA identity point count: ") + errorString(err);
    return false;
  }

  cache->builtSerial = serial;
  cache->pointCount = static_cast<int>(outCount);
  cache->available = (outCount == expectedIdentityPointCount(res));
  if (!cache->available && error) {
    std::ostringstream os;
    os << "Unexpected CUDA identity point count: expected=" << expectedIdentityPointCount(res)
       << " got=" << outCount;
    *error = os.str();
  }
  return cache->available;
}

bool buildInputCloudMesh(InputCache* cache,
                         const InputRequest& request,
                         const std::vector<float>& rawPoints,
                         std::uint64_t serial,
                         std::string* error) {
  if (!cache || cache->verts == 0 || cache->colors == 0) {
    if (error) *error = "CUDA input cache has no GL buffers.";
    return false;
  }
  if (request.pointCount <= 0 || rawPoints.empty()) {
    if (error) *error = "Invalid CUDA input-cloud request.";
    return false;
  }

  std::string localError;
  if (!ensureContext(&localError)) {
    if (error) *error = localError;
    return false;
  }

  const size_t pointCount = static_cast<size_t>(request.pointCount);
  CacheImpl* impl = ensureImpl(cache);
  if (!impl) {
    if (error) *error = "Failed to allocate CUDA input cache.";
    return false;
  }
  if (!ensureRegistered(cache->verts, cache->colors, pointCount, impl, &localError)) {
    if (error) *error = localError;
    return false;
  }
  if (!ensureInputCapacity(impl, rawPoints.size(), &localError)) {
    if (error) *error = localError;
    return false;
  }

  cudaError_t err =
      cudaMemcpy(impl->deviceInput, rawPoints.data(), rawPoints.size() * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to upload CUDA input cloud: ") + errorString(err);
    return false;
  }

  std::array<cudaGraphicsResource*, 2> resources = {impl->vertsResource, impl->colorsResource};
  err = cudaGraphicsMapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("Failed to map CUDA input draw buffers: ") + errorString(err);
    return false;
  }

  float* devVerts = nullptr;
  float* devColors = nullptr;
  size_t vertsBytes = 0;
  size_t colorsBytes = 0;
  err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devVerts), &vertsBytes, impl->vertsResource);
  if (err == cudaSuccess) {
    err = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&devColors), &colorsBytes, impl->colorsResource);
  }
  if (err != cudaSuccess) {
    cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
    if (error) *error = std::string("Failed to access CUDA input draw buffers: ") + errorString(err);
    return false;
  }

  InputKernelUniforms uniforms{};
  uniforms.pointCount = request.pointCount;
  uniforms.showOverflow = request.showOverflow;
  uniforms.highlightOverflow = request.highlightOverflow;
  const unsigned int threads = 256u;
  const unsigned int blocks = static_cast<unsigned int>((pointCount + threads - 1u) / threads);
  inputPackKernel<<<blocks, threads>>>(devVerts, devColors, impl->deviceInput, uniforms);
  err = cudaGetLastError();
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(static_cast<int>(resources.size()), resources.data(), 0);
  if (err != cudaSuccess) {
    if (error) *error = std::string("CUDA input-cloud pack kernel failed: ") + errorString(err);
    return false;
  }

  cache->builtSerial = serial;
  cache->pointCount = request.pointCount;
  cache->available = true;
  return true;
}

}  // namespace OpenDRTViewerCuda
