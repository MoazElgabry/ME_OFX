#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <cstddef>
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
  id<MTLBuffer> srcBuffer = nil;
  id<MTLBuffer> dstBuffer = nil;
  size_t bufferBytes = 0;
  bool initialized = false;
  bool initAttempted = false;
};

MetalContext& context() {
  static MetalContext ctx;
  return ctx;
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
  if (ctx.initialized || ctx.initAttempted) {
    return ctx.device != nil && ctx.queue != nil && ctx.pipeline != nil;
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

bool render(const float* src, float* dst, int width, int height, const OpenDRTParams& params) {
  static std::mutex m;
  // Keep this path serialized for stability while iterating on host integration.
  std::lock_guard<std::mutex> lock(m);

  if (src == nullptr || dst == nullptr || width <= 0 || height <= 0) return false;
  if (!initialize()) return false;

  auto& ctx = context();
  const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u * sizeof(float);
  if (ctx.srcBuffer == nil || ctx.dstBuffer == nil || ctx.bufferBytes != bytes) {
    ctx.srcBuffer = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    ctx.dstBuffer = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    ctx.bufferBytes = bytes;
  }
  if (ctx.srcBuffer == nil || ctx.dstBuffer == nil) {
    return false;
  }
  std::memcpy(ctx.srcBuffer.contents, src, bytes);

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  if (cmd == nil) return false;

  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (enc == nil) return false;

  [enc setComputePipelineState:ctx.pipeline];
  [enc setBuffer:ctx.srcBuffer offset:0 atIndex:0];
  [enc setBuffer:ctx.dstBuffer offset:0 atIndex:1];
  [enc setBytes:&params length:sizeof(OpenDRTParams) atIndex:2];
  [enc setBytes:&width length:sizeof(int) atIndex:3];
  [enc setBytes:&height length:sizeof(int) atIndex:4];

  // Mirrors CUDA-style 2D launch: one thread per output pixel.
  NSUInteger maxThreads = ctx.pipeline.maxTotalThreadsPerThreadgroup;
  NSUInteger side = 1;
  while ((side + 1) * (side + 1) <= maxThreads) {
    ++side;
  }
  side = side > 16 ? 16 : side;
  const MTLSize threadsPerThreadgroup = MTLSizeMake(side, side, 1);
  const MTLSize threadsPerGrid = MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1);
  [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
  [enc endEncoding];

  [cmd commit];
  [cmd waitUntilCompleted];

  if (cmd.status != MTLCommandBufferStatusCompleted) {
    if (cmd.error != nil) {
      NSLog(@"ME_OpenDRT Metal: command buffer failed: %@", cmd.error.localizedDescription);
    }
    return false;
  }

  std::memcpy(dst, ctx.dstBuffer.contents, bytes);
  return true;
}

}  // namespace OpenDRTMetal
