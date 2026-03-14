#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>

#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "OpenDRTProcessor.h"
#include "OpenDRTViewerMetal.h"

namespace {

struct MetalViewerContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  id<MTLComputePipelineState> identityPipeline = nil;
  id<MTLComputePipelineState> inputPipeline = nil;
  id<MTLRenderPipelineState> colorPipeline = nil;
  id<MTLRenderPipelineState> solidPipeline = nil;
  id<MTLDepthStencilState> depthState = nil;
  CAMetalLayer* layer = nil;
  id<MTLBuffer> guideVerts = nil;
  id<MTLBuffer> guideColors = nil;
  id<MTLTexture> depthTexture = nil;
  std::mutex initMutex;
  bool initAttempted = false;
  bool initialized = false;
  bool presenterReady = false;
  CGSize drawableSize = CGSizeZero;
  std::string initError;
};

struct IdentityUniforms {
  uint32_t resolution = 0;
  uint32_t interiorStep = 0;
  uint32_t showOverflow = 0;
  uint32_t highlightOverflow = 0;
};

struct InputUniforms {
  uint32_t pointCount = 0;
  uint32_t showOverflow = 0;
  uint32_t highlightOverflow = 0;
  uint32_t padding = 0;
};

struct SceneUniforms {
  float mvp[16];
  float pointSize = 1.0f;
  float alpha = 1.0f;
  float padding0 = 0.0f;
  float padding1 = 0.0f;
};

struct SolidColorUniforms {
  float color[4];
};

struct CacheImpl {
  id<MTLBuffer> srcBuffer = nil;
  id<MTLBuffer> dstBuffer = nil;
  id<MTLBuffer> inputBuffer = nil;
  id<MTLBuffer> vertsBuffer = nil;
  id<MTLBuffer> colorsBuffer = nil;
  id<MTLBuffer> counterBuffer = nil;
  id<MTLBuffer> uniformBuffer = nil;
  size_t identityPointCapacity = 0;
  size_t inputFloatCapacity = 0;
  size_t meshPointCapacity = 0;
};

MetalViewerContext& context() {
  static MetalViewerContext ctx;
  return ctx;
}

const char* kViewerMetalShaderSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct IdentityUniforms {
  uint resolution;
  uint interiorStep;
  uint showOverflow;
  uint highlightOverflow;
};

struct InputUniforms {
  uint pointCount;
  uint showOverflow;
  uint highlightOverflow;
  uint padding;
};

struct SceneUniforms {
  float4x4 mvp;
  float pointSize;
  float alpha;
  float padding0;
  float padding1;
};

struct SolidColorUniforms {
  float4 color;
};

struct VSOut {
  float4 position [[position]];
  float4 color;
  float pointSize [[point_size]];
};

inline float clamp01(float v) {
  return clamp(v, 0.0f, 1.0f);
}

inline bool outOfBounds(float r, float g, float b) {
  return r < 0.0f || r > 1.0f || g < 0.0f || g > 1.0f || b < 0.0f || b > 1.0f;
}

inline void mapDisplayColor(float inR, float inG, float inB, thread float& outR, thread float& outG, thread float& outB) {
  float r = pow(clamp01(inR), 0.90f);
  float g = pow(clamp01(inG), 0.90f);
  float b = pow(clamp01(inB), 0.90f);
  const float luma = 0.2126f * r + 0.7152f * g + 0.0722f * b;
  r = clamp01(luma + (r - luma));
  g = clamp01(luma + (g - luma));
  b = clamp01(luma + (b - luma));
  outR = r;
  outG = g;
  outB = b;
}

kernel void buildIdentityMeshKernel(
    device const float* srcVals [[buffer(0)]],
    device const float* dstVals [[buffer(1)]],
    device float* vertVals [[buffer(2)]],
    device float* colorVals [[buffer(3)]],
    device atomic_uint* outCount [[buffer(4)]],
    constant IdentityUniforms& u [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  const uint res = u.resolution;
  const uint total = res * res * res;
  if (gid >= total) return;

  const uint plane = res * res;
  const uint z = gid / plane;
  const uint rem = gid - z * plane;
  const uint y = rem / res;
  const uint x = rem - y * res;
  const bool onBoundary = (x == 0u || y == 0u || z == 0u || x == (res - 1u) || y == (res - 1u) || z == (res - 1u));
  if (!onBoundary) {
    if ((x % u.interiorStep) != 0u || (y % u.interiorStep) != 0u || (z % u.interiorStep) != 0u) {
      return;
    }
  }

  const uint srcBase = gid * 4u;
  const float sr = srcVals[srcBase + 0u];
  const float sg = srcVals[srcBase + 1u];
  const float sb = srcVals[srcBase + 2u];
  const float dr = dstVals[srcBase + 0u];
  const float dg = dstVals[srcBase + 1u];
  const float db = dstVals[srcBase + 2u];
  const bool overflow = outOfBounds(dr, dg, db);
  const float plotR = (u.showOverflow != 0u) ? dr : clamp01(dr);
  const float plotG = (u.showOverflow != 0u) ? dg : clamp01(dg);
  const float plotB = (u.showOverflow != 0u) ? db : clamp01(db);

  float cr = 0.0f;
  float cg = 0.0f;
  float cb = 0.0f;
  if (u.showOverflow != 0u && u.highlightOverflow != 0u && overflow) {
    cr = 1.0f;
    cg = 0.0f;
    cb = 0.0f;
  } else {
    const float mixR = sr * 0.86f + clamp01(dr) * 0.14f;
    const float mixG = sg * 0.86f + clamp01(dg) * 0.14f;
    const float mixB = sb * 0.86f + clamp01(db) * 0.14f;
    mapDisplayColor(mixR, mixG, mixB, cr, cg, cb);
  }

  const uint writeIndex = atomic_fetch_add_explicit(outCount, 1u, memory_order_relaxed);
  const uint outBase = writeIndex * 3u;
  vertVals[outBase + 0u] = plotR * 2.0f - 1.0f;
  vertVals[outBase + 1u] = plotG * 2.0f - 1.0f;
  vertVals[outBase + 2u] = plotB * 2.0f - 1.0f;
  colorVals[outBase + 0u] = cr;
  colorVals[outBase + 1u] = cg;
  colorVals[outBase + 2u] = cb;
}

kernel void buildInputCloudMeshKernel(
    device const float* rawVals [[buffer(0)]],
    device float* vertVals [[buffer(1)]],
    device float* colorVals [[buffer(2)]],
    constant InputUniforms& u [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid >= u.pointCount) return;
  const uint inBase = gid * 6u;
  const float sr = rawVals[inBase + 0u];
  const float sg = rawVals[inBase + 1u];
  const float sb = rawVals[inBase + 2u];
  const float dr = rawVals[inBase + 3u];
  const float dg = rawVals[inBase + 4u];
  const float db = rawVals[inBase + 5u];
  const bool overflow = outOfBounds(dr, dg, db);
  const float plotR = (u.showOverflow != 0u) ? dr : clamp01(dr);
  const float plotG = (u.showOverflow != 0u) ? dg : clamp01(dg);
  const float plotB = (u.showOverflow != 0u) ? db : clamp01(db);
  const uint outBase = gid * 3u;
  vertVals[outBase + 0u] = plotR * 2.0f - 1.0f;
  vertVals[outBase + 1u] = plotG * 2.0f - 1.0f;
  vertVals[outBase + 2u] = plotB * 2.0f - 1.0f;

  float cr = 0.0f;
  float cg = 0.0f;
  float cb = 0.0f;
  if (u.showOverflow != 0u && u.highlightOverflow != 0u && overflow) {
    cr = 1.0f;
    cg = 0.0f;
    cb = 0.0f;
  } else {
    mapDisplayColor(clamp01(sr), clamp01(sg), clamp01(sb), cr, cg, cb);
  }
  colorVals[outBase + 0u] = cr;
  colorVals[outBase + 1u] = cg;
  colorVals[outBase + 2u] = cb;
}

vertex VSOut viewerColorVertex(
    const device packed_float3* positions [[buffer(0)]],
    const device packed_float3* colors [[buffer(1)]],
    constant SceneUniforms& scene [[buffer(2)]],
    uint vid [[vertex_id]]) {
  VSOut out;
  const float3 pos = float3(positions[vid]);
  out.position = scene.mvp * float4(pos, 1.0);
  const float3 color = float3(colors[vid]);
  out.color = float4(color, scene.alpha);
  out.pointSize = scene.pointSize;
  return out;
}

vertex VSOut viewerSolidVertex(
    const device packed_float3* positions [[buffer(0)]],
    constant SceneUniforms& scene [[buffer(2)]],
    uint vid [[vertex_id]]) {
  VSOut out;
  const float3 pos = float3(positions[vid]);
  out.position = scene.mvp * float4(pos, 1.0);
  out.color = float4(1.0);
  out.pointSize = scene.pointSize;
  return out;
}

fragment float4 viewerColorFragment(VSOut in [[stage_in]]) {
  return in.color;
}

fragment float4 viewerSolidFragment(VSOut, constant SolidColorUniforms& solid [[buffer(0)]]) {
  return solid.color;
}
)MSL";

void copySceneUniforms(const float* mvp, SceneUniforms* scene, float pointSize, float alpha) {
  if (!mvp || !scene) return;
  std::memcpy(scene->mvp, mvp, sizeof(scene->mvp));
  scene->pointSize = pointSize;
  scene->alpha = alpha;
  scene->padding0 = 0.0f;
  scene->padding1 = 0.0f;
}

bool computeIdentityLattice(id<MTLBuffer> buffer, int res) {
  if (buffer == nil || res <= 1) return false;
  float* vals = reinterpret_cast<float*>([buffer contents]);
  if (!vals) return false;
  const float denom = static_cast<float>(res - 1);
  size_t index = 0u;
  for (int z = 0; z < res; ++z) {
    for (int y = 0; y < res; ++y) {
      for (int x = 0; x < res; ++x) {
        vals[index + 0u] = static_cast<float>(x) / denom;
        vals[index + 1u] = static_cast<float>(y) / denom;
        vals[index + 2u] = static_cast<float>(z) / denom;
        vals[index + 3u] = 1.0f;
        index += 4u;
      }
    }
  }
  return true;
}

bool ensureMeshCapacity(CacheImpl* impl, size_t pointCount, std::string* errorOut) {
  MetalViewerContext& ctx = context();
  if (!impl || ctx.device == nil) return false;
  if (impl->meshPointCapacity < pointCount || impl->vertsBuffer == nil || impl->colorsBuffer == nil) {
    const NSUInteger meshBytes = static_cast<NSUInteger>(pointCount * 3u * sizeof(float));
    impl->vertsBuffer = [ctx.device newBufferWithLength:meshBytes options:MTLResourceStorageModeShared];
    impl->colorsBuffer = [ctx.device newBufferWithLength:meshBytes options:MTLResourceStorageModeShared];
    impl->meshPointCapacity = pointCount;
  }
  if (impl->counterBuffer == nil) {
    impl->counterBuffer = [ctx.device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
  }
  if (impl->uniformBuffer == nil) {
    const NSUInteger uniformBytes = static_cast<NSUInteger>(std::max(sizeof(IdentityUniforms), sizeof(InputUniforms)));
    impl->uniformBuffer = [ctx.device newBufferWithLength:uniformBytes options:MTLResourceStorageModeShared];
  }
  if (impl->vertsBuffer == nil || impl->colorsBuffer == nil || impl->counterBuffer == nil || impl->uniformBuffer == nil) {
    if (errorOut) *errorOut = "Failed to allocate Metal mesh buffers";
    return false;
  }
  return true;
}

bool ensureIdentityCapacity(CacheImpl* impl, size_t pointCount, std::string* errorOut) {
  MetalViewerContext& ctx = context();
  if (!impl || ctx.device == nil) return false;
  if (impl->identityPointCapacity < pointCount || impl->srcBuffer == nil || impl->dstBuffer == nil) {
    const NSUInteger bytes = static_cast<NSUInteger>(pointCount * 4u * sizeof(float));
    impl->srcBuffer = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    impl->dstBuffer = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    impl->identityPointCapacity = pointCount;
  }
  if (impl->srcBuffer == nil || impl->dstBuffer == nil) {
    if (errorOut) *errorOut = "Failed to allocate Metal identity buffers";
    return false;
  }
  return true;
}

bool ensureInputCapacity(CacheImpl* impl, size_t rawPointFloatCount, std::string* errorOut) {
  MetalViewerContext& ctx = context();
  if (!impl || ctx.device == nil) return false;
  if (impl->inputFloatCapacity < rawPointFloatCount || impl->inputBuffer == nil) {
    const NSUInteger bytes = static_cast<NSUInteger>(rawPointFloatCount * sizeof(float));
    impl->inputBuffer = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    impl->inputFloatCapacity = rawPointFloatCount;
  }
  if (impl->inputBuffer == nil) {
    if (errorOut) *errorOut = "Failed to allocate Metal input-cloud buffer";
    return false;
  }
  return true;
}

bool buildGuideBuffers(std::string* errorOut) {
  MetalViewerContext& ctx = context();
  if (ctx.device == nil) return false;
  if (ctx.guideVerts != nil && ctx.guideColors != nil) return true;

  static const float kGuideVerts[] = {
      -1.f,-1.f,-1.f,  1.f,-1.f,-1.f,
       1.f,-1.f,-1.f,  1.f, 1.f,-1.f,
       1.f, 1.f,-1.f, -1.f, 1.f,-1.f,
      -1.f, 1.f,-1.f, -1.f,-1.f,-1.f,
      -1.f,-1.f, 1.f,  1.f,-1.f, 1.f,
       1.f,-1.f, 1.f,  1.f, 1.f, 1.f,
       1.f, 1.f, 1.f, -1.f, 1.f, 1.f,
      -1.f, 1.f, 1.f, -1.f,-1.f, 1.f,
      -1.f,-1.f,-1.f, -1.f,-1.f, 1.f,
       1.f,-1.f,-1.f,  1.f,-1.f, 1.f,
       1.f, 1.f,-1.f,  1.f, 1.f, 1.f,
      -1.f, 1.f,-1.f, -1.f, 1.f, 1.f,
      -1.f,-1.f,-1.f, 1.35f,-1.f,-1.f,
      -1.f,-1.f,-1.f, -1.f,1.35f,-1.f,
      -1.f,-1.f,-1.f, -1.f,-1.f,1.35f,
      -1.f,-1.f,-1.f, 1.f,1.f,1.f
  };
  static const float kGuideColors[] = {
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      0.97f,0.97f,0.97f, 0.97f,0.97f,0.97f,
      1.0f,0.32f,0.32f, 1.0f,0.32f,0.32f,
      0.35f,1.0f,0.35f, 0.35f,1.0f,0.35f,
      0.35f,0.60f,1.0f, 0.35f,0.60f,1.0f,
      1.0f,1.0f,1.0f, 1.0f,1.0f,1.0f
  };

  ctx.guideVerts = [ctx.device newBufferWithBytes:kGuideVerts
                                           length:sizeof(kGuideVerts)
                                          options:MTLResourceStorageModeShared];
  ctx.guideColors = [ctx.device newBufferWithBytes:kGuideColors
                                            length:sizeof(kGuideColors)
                                           options:MTLResourceStorageModeShared];
  if (ctx.guideVerts == nil || ctx.guideColors == nil) {
    if (errorOut) *errorOut = "Failed to allocate Metal guide buffers";
    return false;
  }
  return true;
}

bool ensureBaseContext(std::string* errorOut) {
  MetalViewerContext& ctx = context();
  std::lock_guard<std::mutex> lock(ctx.initMutex);
  if (ctx.initialized) return true;

  @autoreleasepool {
    ctx.device = MTLCreateSystemDefaultDevice();
    if (ctx.device == nil) {
      ctx.initError = "Metal device unavailable";
      if (errorOut) *errorOut = ctx.initError;
      return false;
    }
    ctx.queue = [ctx.device newCommandQueue];
    if (ctx.queue == nil) {
      ctx.initError = "Metal command queue unavailable";
      if (errorOut) *errorOut = ctx.initError;
      return false;
    }

    NSError* error = nil;
    NSString* shaderSource = [NSString stringWithUTF8String:kViewerMetalShaderSrc];
    ctx.library = [ctx.device newLibraryWithSource:shaderSource options:nil error:&error];
    if (ctx.library == nil) {
      ctx.initError = error ? [[error localizedDescription] UTF8String] : "Metal shader library compile failed";
      if (errorOut) *errorOut = ctx.initError;
      return false;
    }

    id<MTLFunction> identityFn = [ctx.library newFunctionWithName:@"buildIdentityMeshKernel"];
    id<MTLFunction> inputFn = [ctx.library newFunctionWithName:@"buildInputCloudMeshKernel"];
    id<MTLFunction> colorVertexFn = [ctx.library newFunctionWithName:@"viewerColorVertex"];
    id<MTLFunction> colorFragmentFn = [ctx.library newFunctionWithName:@"viewerColorFragment"];
    id<MTLFunction> solidVertexFn = [ctx.library newFunctionWithName:@"viewerSolidVertex"];
    id<MTLFunction> solidFragmentFn = [ctx.library newFunctionWithName:@"viewerSolidFragment"];
    if (identityFn == nil || inputFn == nil || colorVertexFn == nil || colorFragmentFn == nil ||
        solidVertexFn == nil || solidFragmentFn == nil) {
      ctx.initError = "Metal viewer shader functions unavailable";
      if (errorOut) *errorOut = ctx.initError;
      return false;
    }

    ctx.identityPipeline = [ctx.device newComputePipelineStateWithFunction:identityFn error:&error];
    if (ctx.identityPipeline == nil) {
      ctx.initError = error ? [[error localizedDescription] UTF8String] : "Metal identity pipeline creation failed";
      if (errorOut) *errorOut = ctx.initError;
      return false;
    }
    error = nil;
    ctx.inputPipeline = [ctx.device newComputePipelineStateWithFunction:inputFn error:&error];
    if (ctx.inputPipeline == nil) {
      ctx.initError = error ? [[error localizedDescription] UTF8String] : "Metal input pipeline creation failed";
      if (errorOut) *errorOut = ctx.initError;
      return false;
    }

    auto configureBlend = [](MTLRenderPipelineColorAttachmentDescriptor* attachment) {
      attachment.pixelFormat = MTLPixelFormatBGRA8Unorm;
      attachment.blendingEnabled = YES;
      attachment.rgbBlendOperation = MTLBlendOperationAdd;
      attachment.alphaBlendOperation = MTLBlendOperationAdd;
      attachment.sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
      attachment.sourceAlphaBlendFactor = MTLBlendFactorSourceAlpha;
      attachment.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
      attachment.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    };

    MTLRenderPipelineDescriptor* colorDesc = [[MTLRenderPipelineDescriptor alloc] init];
    colorDesc.vertexFunction = colorVertexFn;
    colorDesc.fragmentFunction = colorFragmentFn;
    configureBlend(colorDesc.colorAttachments[0]);
    colorDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    ctx.colorPipeline = [ctx.device newRenderPipelineStateWithDescriptor:colorDesc error:&error];
    if (ctx.colorPipeline == nil) {
      ctx.initError = error ? [[error localizedDescription] UTF8String] : "Metal color render pipeline creation failed";
      if (errorOut) *errorOut = ctx.initError;
      return false;
    }

    error = nil;
    MTLRenderPipelineDescriptor* solidDesc = [[MTLRenderPipelineDescriptor alloc] init];
    solidDesc.vertexFunction = solidVertexFn;
    solidDesc.fragmentFunction = solidFragmentFn;
    configureBlend(solidDesc.colorAttachments[0]);
    solidDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
    ctx.solidPipeline = [ctx.device newRenderPipelineStateWithDescriptor:solidDesc error:&error];
    if (ctx.solidPipeline == nil) {
      ctx.initError = error ? [[error localizedDescription] UTF8String] : "Metal solid render pipeline creation failed";
      if (errorOut) *errorOut = ctx.initError;
      return false;
    }

    MTLDepthStencilDescriptor* depthDesc = [[MTLDepthStencilDescriptor alloc] init];
    depthDesc.depthCompareFunction = MTLCompareFunctionLessEqual;
    depthDesc.depthWriteEnabled = YES;
    ctx.depthState = [ctx.device newDepthStencilStateWithDescriptor:depthDesc];
    if (ctx.depthState == nil) {
      ctx.initError = "Metal depth state creation failed";
      if (errorOut) *errorOut = ctx.initError;
      return false;
    }

    if (!buildGuideBuffers(errorOut)) {
      ctx.initError = errorOut ? *errorOut : std::string("Metal guide buffer creation failed");
      return false;
    }
  }

  ctx.initialized = true;
  ctx.initAttempted = true;
  ctx.initError.clear();
  return true;
}

bool ensurePresenter(GLFWwindow* window, std::string* errorOut) {
  if (!window) {
    if (errorOut) *errorOut = "GLFW window unavailable for Metal presenter";
    return false;
  }
  if (!ensureBaseContext(errorOut)) return false;

  MetalViewerContext& ctx = context();
  @autoreleasepool {
    NSWindow* cocoaWindow = glfwGetCocoaWindow(window);
    NSView* contentView = cocoaWindow ? [cocoaWindow contentView] : nil;
    if (contentView == nil) {
      if (errorOut) *errorOut = "Cocoa content view unavailable";
      return false;
    }

    if (ctx.layer == nil) {
      ctx.layer = [[CAMetalLayer alloc] init];
    }
    ctx.layer.device = ctx.device;
    ctx.layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    ctx.layer.framebufferOnly = NO;
    ctx.layer.opaque = YES;
    ctx.layer.needsDisplayOnBoundsChange = YES;
    const CGFloat scale = cocoaWindow ? [cocoaWindow backingScaleFactor] : 1.0;
    ctx.layer.contentsScale = scale;
    ctx.layer.frame = [contentView bounds];
    ctx.layer.drawableSize = CGSizeMake([contentView bounds].size.width * scale,
                                        [contentView bounds].size.height * scale);
    [contentView setWantsLayer:YES];
    [contentView setLayer:ctx.layer];
  }

  ctx.presenterReady = true;
  return true;
}

bool ensureDepthTextureForDrawable(id<CAMetalDrawable> drawable, std::string* errorOut) {
  MetalViewerContext& ctx = context();
  if (drawable == nil || drawable.texture == nil) {
    if (errorOut) *errorOut = "Metal drawable texture unavailable";
    return false;
  }
  const NSUInteger width = drawable.texture.width;
  const NSUInteger height = drawable.texture.height;
  if (ctx.depthTexture != nil && ctx.drawableSize.width == width && ctx.drawableSize.height == height) {
    return true;
  }

  MTLTextureDescriptor* depthDesc =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  depthDesc.storageMode = MTLStorageModePrivate;
  depthDesc.usage = MTLTextureUsageRenderTarget;
  ctx.depthTexture = [ctx.device newTextureWithDescriptor:depthDesc];
  if (ctx.depthTexture == nil) {
    if (errorOut) *errorOut = "Failed to allocate Metal depth texture";
    return false;
  }
  ctx.drawableSize = CGSizeMake(width, height);
  return true;
}

bool finishCommandBuffer(id<MTLCommandBuffer> commandBuffer, std::string* errorOut) {
  if (commandBuffer == nil) {
    if (errorOut) *errorOut = "Metal command buffer unavailable";
    return false;
  }
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
    if (errorOut) {
      NSError* error = commandBuffer.error;
      *errorOut = error ? [[error localizedDescription] UTF8String] : "Metal command buffer failed";
    }
    return false;
  }
  return true;
}

template <typename CacheT>
CacheImpl* ensureImpl(CacheT* cache) {
  if (!cache) return nullptr;
  if (!cache->internal) cache->internal = new CacheImpl();
  return reinterpret_cast<CacheImpl*>(cache->internal);
}

void releaseCacheImpl(CacheImpl* impl) {
  delete impl;
}

}  // namespace

namespace OpenDRTViewerMetal {

ProbeResult probe() {
  ProbeResult result{};
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil) return result;
    result.available = true;
    result.deviceName = [[device name] UTF8String];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    result.queueReady = (queue != nil);
  }
  return result;
}

StartupValidationResult validateStartup() {
  StartupValidationResult result{};
  std::string error;
  result.ready = ensureBaseContext(&error);
  if (!result.ready) {
    result.reason = error.empty() ? std::string("Metal viewer context unavailable") : error;
  }
  return result;
}

PresenterInitResult initializePresenter(GLFWwindow* window) {
  PresenterInitResult result{};
  std::string error;
  result.ready = ensurePresenter(window, &error);
  if (!result.ready) result.reason = error;
  return result;
}

void shutdownPresenter() {
  MetalViewerContext& ctx = context();
  ctx.depthTexture = nil;
  ctx.layer = nil;
  ctx.guideVerts = nil;
  ctx.guideColors = nil;
  ctx.colorPipeline = nil;
  ctx.solidPipeline = nil;
  ctx.depthState = nil;
  ctx.presenterReady = false;
  ctx.drawableSize = CGSizeZero;
}

bool buildIdentityMesh(
    MeshCache* cache,
    const OpenDRTParams& params,
    int resolution,
    bool showOverflow,
    bool highlightOverflow,
    std::uint64_t serial,
    std::string* transformBackendLabel,
    float* maxDelta,
    std::string* errorOut) {
  if (!cache || resolution <= 1) return false;
  std::string initError;
  if (!ensureBaseContext(&initError)) {
    if (errorOut) *errorOut = initError;
    return false;
  }

  const int res = resolution <= 25 ? 25 : (resolution <= 41 ? 41 : 57);
  const size_t pointCount = static_cast<size_t>(res) * static_cast<size_t>(res) * static_cast<size_t>(res);
  const int interiorStep = (res <= 25) ? 2 : (res <= 41 ? 2 : 3);
  CacheImpl* impl = ensureImpl(cache);
  if (!impl) return false;
  if (!ensureIdentityCapacity(impl, pointCount, errorOut) || !ensureMeshCapacity(impl, pointCount, errorOut)) return false;
  if (!computeIdentityLattice(impl->srcBuffer, res)) {
    if (errorOut) *errorOut = "Failed to build Metal identity lattice";
    return false;
  }

  OpenDRTProcessor proc(params);
  const size_t rowBytes = pointCount * 4u * sizeof(float);
  const bool ok = proc.renderMetalHostBuffers(
      (__bridge const void*)impl->srcBuffer,
      (__bridge void*)impl->dstBuffer,
      static_cast<int>(pointCount),
      1,
      rowBytes,
      rowBytes,
      0,
      0,
      (__bridge void*)context().queue);
  if (!ok) {
    if (errorOut) *errorOut = "OpenDRT Metal host render failed";
    return false;
  }

  if (transformBackendLabel) *transformBackendLabel = proc.lastBackendLabel();
  if (maxDelta) {
    const float* src = reinterpret_cast<const float*>([impl->srcBuffer contents]);
    const float* dst = reinterpret_cast<const float*>([impl->dstBuffer contents]);
    float localMax = 0.0f;
    for (size_t i = 0; i < pointCount; ++i) {
      const size_t si = i * 4u;
      localMax = std::max(localMax, std::fabs(dst[si + 0] - src[si + 0]));
      localMax = std::max(localMax, std::fabs(dst[si + 1] - src[si + 1]));
      localMax = std::max(localMax, std::fabs(dst[si + 2] - src[si + 2]));
    }
    *maxDelta = localMax;
  }

  IdentityUniforms uniforms{};
  uniforms.resolution = static_cast<uint32_t>(res);
  uniforms.interiorStep = static_cast<uint32_t>(interiorStep);
  uniforms.showOverflow = showOverflow ? 1u : 0u;
  uniforms.highlightOverflow = highlightOverflow ? 1u : 0u;
  std::memcpy([impl->uniformBuffer contents], &uniforms, sizeof(uniforms));
  std::memset([impl->counterBuffer contents], 0, sizeof(uint32_t));

  MetalViewerContext& ctx = context();
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (errorOut) *errorOut = "Metal identity encoder unavailable";
      return false;
    }
    [encoder setComputePipelineState:ctx.identityPipeline];
    [encoder setBuffer:impl->srcBuffer offset:0 atIndex:0];
    [encoder setBuffer:impl->dstBuffer offset:0 atIndex:1];
    [encoder setBuffer:impl->vertsBuffer offset:0 atIndex:2];
    [encoder setBuffer:impl->colorsBuffer offset:0 atIndex:3];
    [encoder setBuffer:impl->counterBuffer offset:0 atIndex:4];
    [encoder setBuffer:impl->uniformBuffer offset:0 atIndex:5];
    const NSUInteger threadWidth = std::max<NSUInteger>(1u, ctx.identityPipeline.threadExecutionWidth);
    [encoder dispatchThreads:MTLSizeMake(pointCount, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
    [encoder endEncoding];
    if (!finishCommandBuffer(commandBuffer, errorOut)) return false;
  }

  const uint32_t writtenPoints = *reinterpret_cast<const uint32_t*>([impl->counterBuffer contents]);
  cache->builtSerial = serial;
  cache->pointCount = static_cast<int>(writtenPoints);
  cache->available = (writtenPoints > 0u);
  return true;
}

bool buildInputCloudMesh(
    MeshCache* cache,
    const float* rawPoints,
    size_t rawPointFloatCount,
    bool showOverflow,
    bool highlightOverflow,
    std::uint64_t serial,
    std::string* errorOut) {
  if (!cache || !rawPoints) return false;
  if (rawPointFloatCount == 0u || (rawPointFloatCount % 6u) != 0u) return false;
  std::string initError;
  if (!ensureBaseContext(&initError)) {
    if (errorOut) *errorOut = initError;
    return false;
  }

  const size_t pointCount = rawPointFloatCount / 6u;
  CacheImpl* impl = ensureImpl(cache);
  if (!impl) return false;
  if (!ensureInputCapacity(impl, rawPointFloatCount, errorOut) || !ensureMeshCapacity(impl, pointCount, errorOut)) return false;
  std::memcpy([impl->inputBuffer contents], rawPoints, rawPointFloatCount * sizeof(float));
  InputUniforms uniforms{};
  uniforms.pointCount = static_cast<uint32_t>(pointCount);
  uniforms.showOverflow = showOverflow ? 1u : 0u;
  uniforms.highlightOverflow = highlightOverflow ? 1u : 0u;
  std::memcpy([impl->uniformBuffer contents], &uniforms, sizeof(uniforms));

  MetalViewerContext& ctx = context();
  @autoreleasepool {
    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
      if (errorOut) *errorOut = "Metal input encoder unavailable";
      return false;
    }
    [encoder setComputePipelineState:ctx.inputPipeline];
    [encoder setBuffer:impl->inputBuffer offset:0 atIndex:0];
    [encoder setBuffer:impl->vertsBuffer offset:0 atIndex:1];
    [encoder setBuffer:impl->colorsBuffer offset:0 atIndex:2];
    [encoder setBuffer:impl->uniformBuffer offset:0 atIndex:3];
    const NSUInteger threadWidth = std::max<NSUInteger>(1u, ctx.inputPipeline.threadExecutionWidth);
    [encoder dispatchThreads:MTLSizeMake(pointCount, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threadWidth, 1, 1)];
    [encoder endEncoding];
    if (!finishCommandBuffer(commandBuffer, errorOut)) return false;
  }

  cache->builtSerial = serial;
  cache->pointCount = static_cast<int>(pointCount);
  cache->available = (pointCount > 0u);
  return true;
}

bool resolveDrawSource(const MeshCache* cache, DrawSource* out) {
  if (!cache || !out || !cache->available || !cache->internal || cache->pointCount <= 0) return false;
  const CacheImpl* impl = reinterpret_cast<const CacheImpl*>(cache->internal);
  if (impl->vertsBuffer == nil || impl->colorsBuffer == nil) return false;
  out->cpuVerts = nullptr;
  out->cpuColors = nullptr;
  out->vertsHandle = (__bridge void*)impl->vertsBuffer;
  out->colorsHandle = (__bridge void*)impl->colorsBuffer;
  out->pointCount = cache->pointCount;
  out->gpuBacked = true;
  return true;
}

void releaseCache(MeshCache* cache) {
  if (!cache) return;
  releaseCacheImpl(reinterpret_cast<CacheImpl*>(cache->internal));
  cache->internal = nullptr;
  cache->builtSerial = 0;
  cache->pointCount = 0;
  cache->available = false;
}

bool renderScene(GLFWwindow* window,
                 const DrawSource& source,
                 const float* mvp,
                 float pointSize,
                 float fillPointSize,
                 float clearR,
                 float clearG,
                 float clearB,
                 float clearA,
                 std::string* errorOut) {
  if (!mvp) return false;
  if (!ensurePresenter(window, errorOut)) return false;
  MetalViewerContext& ctx = context();

  @autoreleasepool {
    NSWindow* cocoaWindow = glfwGetCocoaWindow(window);
    NSView* contentView = cocoaWindow ? [cocoaWindow contentView] : nil;
    if (contentView == nil) {
      if (errorOut) *errorOut = "Cocoa content view unavailable";
      return false;
    }
    const CGFloat scale = [cocoaWindow backingScaleFactor];
    const NSSize bounds = [contentView bounds].size;
    ctx.layer.contentsScale = scale;
    ctx.layer.drawableSize = CGSizeMake(bounds.width * scale, bounds.height * scale);

    id<CAMetalDrawable> drawable = [ctx.layer nextDrawable];
    if (drawable == nil) {
      if (errorOut) *errorOut = "Metal drawable unavailable";
      return false;
    }
    if (!ensureDepthTextureForDrawable(drawable, errorOut)) return false;

    MTLRenderPassDescriptor* renderPass = [MTLRenderPassDescriptor renderPassDescriptor];
    renderPass.colorAttachments[0].texture = drawable.texture;
    renderPass.colorAttachments[0].loadAction = MTLLoadActionClear;
    renderPass.colorAttachments[0].storeAction = MTLStoreActionStore;
    renderPass.colorAttachments[0].clearColor = MTLClearColorMake(clearR, clearG, clearB, clearA);
    renderPass.depthAttachment.texture = ctx.depthTexture;
    renderPass.depthAttachment.loadAction = MTLLoadActionClear;
    renderPass.depthAttachment.storeAction = MTLStoreActionDontCare;
    renderPass.depthAttachment.clearDepth = 1.0;

    id<MTLCommandBuffer> commandBuffer = [ctx.queue commandBuffer];
    id<MTLRenderCommandEncoder> encoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPass];
    if (encoder == nil) {
      if (errorOut) *errorOut = "Metal render encoder unavailable";
      return false;
    }
    [encoder setDepthStencilState:ctx.depthState];
    [encoder setCullMode:MTLCullModeNone];

    SceneUniforms scene{};
    copySceneUniforms(mvp, &scene, 1.0f, 0.55f);
    [encoder setRenderPipelineState:ctx.colorPipeline];
    [encoder setVertexBuffer:ctx.guideVerts offset:0 atIndex:0];
    [encoder setVertexBuffer:ctx.guideColors offset:0 atIndex:1];
    [encoder setVertexBytes:&scene length:sizeof(scene) atIndex:2];
    [encoder drawPrimitives:MTLPrimitiveTypeLine vertexStart:0 vertexCount:24];
    scene.alpha = 0.90f;
    [encoder setVertexBytes:&scene length:sizeof(scene) atIndex:2];
    [encoder drawPrimitives:MTLPrimitiveTypeLine vertexStart:24 vertexCount:6];
    scene.alpha = 0.38f;
    [encoder setVertexBytes:&scene length:sizeof(scene) atIndex:2];
    [encoder drawPrimitives:MTLPrimitiveTypeLine vertexStart:30 vertexCount:2];

    if (source.pointCount > 0) {
      id<MTLBuffer> vertsBuffer = nil;
      id<MTLBuffer> colorsBuffer = nil;
      if (source.gpuBacked) {
        vertsBuffer = (__bridge id<MTLBuffer>)source.vertsHandle;
        colorsBuffer = (__bridge id<MTLBuffer>)source.colorsHandle;
      } else if (source.cpuVerts && source.cpuColors) {
        vertsBuffer = [ctx.device newBufferWithBytes:source.cpuVerts
                                              length:static_cast<NSUInteger>(source.pointCount) * 3u * sizeof(float)
                                             options:MTLResourceStorageModeShared];
        colorsBuffer = [ctx.device newBufferWithBytes:source.cpuColors
                                               length:static_cast<NSUInteger>(source.pointCount) * 3u * sizeof(float)
                                              options:MTLResourceStorageModeShared];
      }

      if (vertsBuffer != nil && colorsBuffer != nil) {
        copySceneUniforms(mvp, &scene, pointSize, 1.0f);
        [encoder setRenderPipelineState:ctx.colorPipeline];
        [encoder setVertexBuffer:vertsBuffer offset:0 atIndex:0];
        [encoder setVertexBuffer:colorsBuffer offset:0 atIndex:1];
        [encoder setVertexBytes:&scene length:sizeof(scene) atIndex:2];
        [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:source.pointCount];

        SolidColorUniforms solid{};
        solid.color[0] = 0.95f;
        solid.color[1] = 0.96f;
        solid.color[2] = 1.0f;
        solid.color[3] = 0.05f;
        copySceneUniforms(mvp, &scene, fillPointSize, 1.0f);
        [encoder setDepthStencilState:nil];
        [encoder setRenderPipelineState:ctx.solidPipeline];
        [encoder setVertexBuffer:vertsBuffer offset:0 atIndex:0];
        [encoder setVertexBytes:&scene length:sizeof(scene) atIndex:2];
        [encoder setFragmentBytes:&solid length:sizeof(solid) atIndex:0];
        [encoder drawPrimitives:MTLPrimitiveTypePoint vertexStart:0 vertexCount:source.pointCount];
      }
    }

    [encoder endEncoding];
    [commandBuffer presentDrawable:drawable];
    if (!finishCommandBuffer(commandBuffer, errorOut)) return false;
  }

  return true;
}

}  // namespace OpenDRTViewerMetal
