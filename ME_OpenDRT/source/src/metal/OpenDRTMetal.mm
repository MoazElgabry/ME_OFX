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
  bool initialized = false;
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
  if (ctx.initialized) {
    return ctx.device != nil && ctx.queue != nil && ctx.pipeline != nil;
  }
  ctx.initialized = true;

  ctx.device = MTLCreateSystemDefaultDevice();
  if (ctx.device == nil) return false;

  ctx.queue = [ctx.device newCommandQueue];
  if (ctx.queue == nil) return false;

  // Metallib is packaged into Contents/Resources during CMake build.
  NSString* libPath = [NSString stringWithUTF8String:metallibPath().c_str()];
  NSError* error = nil;
  id<MTLLibrary> library = [ctx.device newLibraryWithFile:libPath error:&error];
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

  id<MTLBuffer> srcBuffer = [ctx.device newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> dstBuffer = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  id<MTLBuffer> paramsBuffer = [ctx.device newBufferWithBytes:&params length:sizeof(OpenDRTParams) options:MTLResourceStorageModeShared];
  id<MTLBuffer> widthBuffer = [ctx.device newBufferWithBytes:&width length:sizeof(int) options:MTLResourceStorageModeShared];
  id<MTLBuffer> heightBuffer = [ctx.device newBufferWithBytes:&height length:sizeof(int) options:MTLResourceStorageModeShared];

  if (srcBuffer == nil || dstBuffer == nil || paramsBuffer == nil || widthBuffer == nil || heightBuffer == nil) {
    return false;
  }

  id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
  if (cmd == nil) return false;

  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (enc == nil) return false;

  [enc setComputePipelineState:ctx.pipeline];
  [enc setBuffer:srcBuffer offset:0 atIndex:0];
  [enc setBuffer:dstBuffer offset:0 atIndex:1];
  [enc setBuffer:paramsBuffer offset:0 atIndex:2];
  [enc setBuffer:widthBuffer offset:0 atIndex:3];
  [enc setBuffer:heightBuffer offset:0 atIndex:4];

  // Mirrors CUDA-style 2D launch: one thread per output pixel.
  const MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
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

  std::memcpy(dst, dstBuffer.contents, bytes);
  return true;
}

}  // namespace OpenDRTMetal
