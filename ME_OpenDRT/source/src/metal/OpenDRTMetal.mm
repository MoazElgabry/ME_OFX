#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <cstddef>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>

#include "OpenDRTMetal.h"

namespace {

static_assert(sizeof(int) == 4, "Metal path requires 32-bit int");
static_assert(sizeof(float) == 4, "Metal path requires 32-bit float");
static_assert(alignof(OpenDRTParams) == 4, "Unexpected OpenDRTParams alignment");

struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  std::mutex initMutex;
  bool initialized = false;
  bool initAttempted = false;
};

struct ThreadBuffers {
  id<MTLBuffer> srcBuffer = nil;
  id<MTLBuffer> dstBuffer = nil;
  size_t bufferBytes = 0;
};

MetalContext& context() {
  static MetalContext ctx;
  return ctx;
}

ThreadBuffers& threadBuffers() {
  thread_local ThreadBuffers buffers;
  return buffers;
}

std::mutex& legacyRenderMutex() {
  static std::mutex m;
  return m;
}

bool envFlagEnabled(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr || v[0] == '\0') return false;
  return !(v[0] == '0' && v[1] == '\0');
}

bool perfLogEnabled() {
  static const bool enabled = envFlagEnabled("ME_OPENDRT_PERF_LOG");
  return enabled;
}

bool debugLogEnabled() {
  static const bool enabled = envFlagEnabled("ME_OPENDRT_DEBUG_LOG");
  return enabled;
}

bool shouldSerializeRender() {
  static const bool enabled = envFlagEnabled("ME_OPENDRT_METAL_SERIALIZE");
  return enabled;
}

bool disableMetal2DCopy() {
  static const bool enabled = envFlagEnabled("ME_OPENDRT_DISABLE_METAL_2D_COPY");
  return enabled;
}

void debugLog(const char* msg) {
  if (!debugLogEnabled()) return;
  std::fprintf(stderr, "[ME_OpenDRT][Metal] %s\n", msg);
}

void perfLogStage(const char* stage, const std::chrono::steady_clock::time_point& start) {
  if (!perfLogEnabled()) return;
  const auto now = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(now - start).count();
  std::fprintf(stderr, "[ME_OpenDRT][PERF][Metal] %s: %.3f ms\n", stage, ms);
}

std::string moduleDirectory() {
  // Resolve bundle-relative paths from the loaded plugin binary location.
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void*>(&context), &info) == 0 || info.dli_fname == nullptr) {
    return std::string();
  }
  std::filesystem::path p(info.dli_fname);
  return p.parent_path().string();
}

std::string metallibPath() {
  const std::filesystem::path macosDir(moduleDirectory());
  if (macosDir.empty()) {
    return std::string();
  }
  return (macosDir.parent_path() / "Resources" / "OpenDRT.metallib").string();
}

bool initialize() {
  auto& ctx = context();
  std::lock_guard<std::mutex> lock(ctx.initMutex);
  if (ctx.initialized) {
    return true;
  }
  if (ctx.initAttempted) {
    return false;
  }
  ctx.initAttempted = true;

  ctx.device = MTLCreateSystemDefaultDevice();
  if (ctx.device == nil) return false;

  ctx.queue = [ctx.device newCommandQueue];
  if (ctx.queue == nil) return false;

  // Metallib is packaged into Contents/Resources during CMake build.
  const std::string libPathStr = metallibPath();
  if (libPathStr.empty()) return false;
  NSString* libPath = [NSString stringWithUTF8String:libPathStr.c_str()];
  if (libPath == nil) return false;
  NSError* error = nil;
  NSURL* libURL = [NSURL fileURLWithPath:libPath];
  id<MTLLibrary> library = [ctx.device newLibraryWithURL:libURL error:&error];
  if (library == nil) {
    if (error != nil) {
      NSLog(@"ME_OpenDRT Metal: failed to load metallib: %@", error.localizedDescription);
    }
    return false;
  }

  id<MTLFunction> function = [library newFunctionWithName:@"OpenDRTKernel"];
  if (function == nil) {
    NSLog(@"ME_OpenDRT Metal: OpenDRTKernel entry not found in metallib");
    return false;
  }

  ctx.pipeline = [ctx.device newComputePipelineStateWithFunction:function error:&error];
  if (ctx.pipeline == nil) {
    if (error != nil) {
      NSLog(@"ME_OpenDRT Metal: failed to create compute pipeline: %@", error.localizedDescription);
    }
    return false;
  }

  ctx.initialized = true;
  return true;
}

}  // namespace

namespace OpenDRTMetal {

static bool renderImpl(
    const float* src,
    float* dst,
    int width,
    int height,
    size_t srcRowBytes,
    size_t dstRowBytes,
    const OpenDRTParams& params,
    const OpenDRTDerivedParams& derived) {
  const auto tStart = std::chrono::steady_clock::now();
  if (src == nullptr || dst == nullptr || width <= 0 || height <= 0) return false;
  if (!initialize()) {
    debugLog("Initialization failed.");
    return false;
  }

  auto& ctx = context();
  auto& buffers = threadBuffers();
  const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u * sizeof(float);
  const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
  const bool packedSrc = (srcRowBytes == packedRowBytes);
  const bool packedDst = (dstRowBytes == packedRowBytes);
  const auto tCopyInStart = std::chrono::steady_clock::now();
  if (buffers.srcBuffer == nil || buffers.dstBuffer == nil || buffers.bufferBytes != bytes) {
    buffers.srcBuffer = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    buffers.dstBuffer = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    buffers.bufferBytes = bytes;
  }
  if (buffers.srcBuffer == nil || buffers.dstBuffer == nil) {
    debugLog("Thread-local Metal buffer allocation failed.");
    return false;
  }
  if (packedSrc) {
    std::memcpy(buffers.srcBuffer.contents, src, bytes);
  } else {
    if (disableMetal2DCopy()) return false;
    char* dstBase = static_cast<char*>(buffers.srcBuffer.contents);
    const char* srcBase = reinterpret_cast<const char*>(src);
    for (int y = 0; y < height; ++y) {
      std::memcpy(dstBase + static_cast<size_t>(y) * packedRowBytes, srcBase + static_cast<size_t>(y) * srcRowBytes, packedRowBytes);
    }
  }
  perfLogStage("Host->Metal staging", tCopyInStart);

  const auto tGpuStart = std::chrono::steady_clock::now();
  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  if (cmd == nil) {
    debugLog("Failed to create command buffer.");
    return false;
  }

  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (enc == nil) {
    debugLog("Failed to create compute encoder.");
    return false;
  }

  [enc setComputePipelineState:ctx.pipeline];
  [enc setBuffer:buffers.srcBuffer offset:0 atIndex:0];
  [enc setBuffer:buffers.dstBuffer offset:0 atIndex:1];
  [enc setBytes:&params length:sizeof(OpenDRTParams) atIndex:2];
  [enc setBytes:&width length:sizeof(int) atIndex:3];
  [enc setBytes:&height length:sizeof(int) atIndex:4];
  [enc setBytes:&derived length:sizeof(OpenDRTDerivedParams) atIndex:5];

  // Mirrors CUDA-style 2D launch: one thread per output pixel.
  auto chooseThreadsPerThreadgroup = [&]() -> MTLSize {
    const char* env = std::getenv("ME_OPENDRT_METAL_BLOCK");
    if (env != nullptr && env[0] != '\0') {
      int bx = 0;
      int by = 0;
      if (std::sscanf(env, "%dx%d", &bx, &by) == 2 && bx > 0 && by > 0) {
        const NSUInteger ux = static_cast<NSUInteger>(bx);
        const NSUInteger uy = static_cast<NSUInteger>(by);
        const NSUInteger maxThreads = ctx.pipeline.maxTotalThreadsPerThreadgroup;
        if (ux * uy <= maxThreads) {
          return MTLSizeMake(ux, uy, 1);
        }
      }
    }
    const NSUInteger maxThreads = ctx.pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger tew = ctx.pipeline.threadExecutionWidth;
    NSUInteger tx = tew > 0 ? tew : 16;
    if (tx > maxThreads) tx = maxThreads;
    NSUInteger ty = maxThreads / tx;
    if (ty == 0) ty = 1;
    if (ty > 16) ty = 16;
    return MTLSizeMake(tx, ty, 1);
  };
  const MTLSize threadsPerThreadgroup = chooseThreadsPerThreadgroup();
  const MTLSize threadsPerGrid = MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1);
  [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
  [enc endEncoding];

  [cmd commit];
  [cmd waitUntilCompleted];

  if (cmd.status != MTLCommandBufferStatusCompleted) {
    if (cmd.error != nil) {
      NSLog(@"ME_OpenDRT Metal: command buffer failed: %@", cmd.error.localizedDescription);
    }
    debugLog("Command buffer failed.");
    return false;
  }

  const auto tCopyOutStart = std::chrono::steady_clock::now();
  if (packedDst) {
    std::memcpy(dst, buffers.dstBuffer.contents, bytes);
  } else {
    if (disableMetal2DCopy()) return false;
    const char* srcBase = static_cast<const char*>(buffers.dstBuffer.contents);
    char* dstBase = reinterpret_cast<char*>(dst);
    for (int y = 0; y < height; ++y) {
      std::memcpy(dstBase + static_cast<size_t>(y) * dstRowBytes, srcBase + static_cast<size_t>(y) * packedRowBytes, packedRowBytes);
    }
  }
  perfLogStage("Metal GPU submit+wait", tGpuStart);
  perfLogStage("Metal->Host copy", tCopyOutStart);
  perfLogStage("Metal total", tStart);
  return true;
}

bool render(
    const float* src,
    float* dst,
    int width,
    int height,
    size_t srcRowBytes,
    size_t dstRowBytes,
    const OpenDRTParams& params,
    const OpenDRTDerivedParams& derived) {
  if (shouldSerializeRender()) {
    std::lock_guard<std::mutex> lock(legacyRenderMutex());
    return renderImpl(src, dst, width, height, srcRowBytes, dstRowBytes, params, derived);
  }
  return renderImpl(src, dst, width, height, srcRowBytes, dstRowBytes, params, derived);
}

}  // namespace OpenDRTMetal
