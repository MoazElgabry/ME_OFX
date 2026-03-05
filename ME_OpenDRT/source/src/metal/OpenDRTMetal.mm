#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <dlfcn.h>
#include <cstddef>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <mutex>
#include <string>
#include <atomic>
#include <sys/stat.h>

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
  std::atomic<int> hostAsyncErrorCount{0};
  std::atomic<bool> hostTemporarilyDisabled{false};
  bool initialized = false;
  bool initAttempted = false;
};

struct ThreadBuffers {
  id<MTLBuffer> srcBuffer = nil;
  id<MTLBuffer> dstBuffer = nil;
  size_t bufferBytes = 0;
  id<MTLBuffer> hostTempSrcBuffer = nil;
  size_t hostTempSrcBytes = 0;
  id<MTLBuffer> hostTempDstBuffer = nil;
  size_t hostTempDstBytes = 0;
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

bool forceHostMetalWait() {
  static const bool enabled = envFlagEnabled("ME_OPENDRT_HOST_METAL_FORCE_WAIT");
  return enabled;
}

bool disableMetal2DCopy() {
  static const bool enabled = envFlagEnabled("ME_OPENDRT_DISABLE_METAL_2D_COPY");
  return enabled;
}

bool metalDiagEnabled() {
  static const bool enabled = envFlagEnabled("ME_OPENDRT_METAL_DIAG");
  return enabled;
}

enum class HostOffsetMode {
  Auto,
  Zero,
  Origin,
  FallbackAmbiguous
};

HostOffsetMode hostOffsetMode() {
  static const HostOffsetMode mode = []() {
    const char* v = std::getenv("ME_OPENDRT_HOST_METAL_OFFSET_MODE");
    if (v != nullptr && v[0] != '\0') {
      std::string s(v);
      for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
      if (s == "ZERO") return HostOffsetMode::Zero;
      if (s == "ORIGIN") return HostOffsetMode::Origin;
      if (s == "FALLBACK" || s == "SAFE") return HostOffsetMode::FallbackAmbiguous;
    }
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    // Apple Silicon safety default:
    // if host offset interpretation is ambiguous, do not guess; fall back out of host-metal path.
    return HostOffsetMode::FallbackAmbiguous;
#else
    return HostOffsetMode::Auto;
#endif
  }();
  return mode;
}

bool useHostIntermediateDst() {
  static const bool enabled = []() {
    const char* v = std::getenv("ME_OPENDRT_HOST_METAL_INTERMEDIATE_DST");
    if (v != nullptr && v[0] != '\0') {
      return !(v[0] == '0' && v[1] == '\0');
    }
#if defined(__APPLE__) && (defined(__aarch64__) || defined(__arm64__))
    // On Apple Silicon host-Metal, writing to a temp buffer then blitting to host dst
    // reduces visible partial-write artifacts without forcing full command completion waits.
    return true;
#else
    return false;
#endif
  }();
  return enabled;
}

void debugLog(const char* msg) {
  if (!debugLogEnabled()) return;
  std::fprintf(stderr, "[ME_OpenDRT][Metal] %s\n", msg);
  static std::mutex logMutex;
  static bool pathInit = false;
  static std::string logPath;
  if (!pathInit) {
    pathInit = true;
    const char* home = std::getenv("HOME");
    if (home != nullptr && home[0] != '\0') {
      const std::string logsDir = std::string(home) + "/Library/Logs";
      (void)::mkdir(logsDir.c_str(), 0755);
      logPath = logsDir + "/ME_OpenDRT.log";
    }
  }
  if (!logPath.empty()) {
    std::lock_guard<std::mutex> lock(logMutex);
    FILE* f = std::fopen(logPath.c_str(), "a");
    if (f != nullptr) {
      std::fprintf(f, "[ME_OpenDRT][Metal] %s\n", msg);
      std::fclose(f);
    }
  }
}

void diagLog(const std::string& msg) {
  if (!metalDiagEnabled()) return;
  std::fprintf(stderr, "[ME_OpenDRT][MetalDiag] %s\n", msg.c_str());
  static std::mutex logMutex;
  static bool pathInit = false;
  static std::string logPath;
  if (!pathInit) {
    pathInit = true;
    const char* home = std::getenv("HOME");
    if (home != nullptr && home[0] != '\0') {
      const std::string logsDir = std::string(home) + "/Library/Logs";
      (void)::mkdir(logsDir.c_str(), 0755);
      logPath = logsDir + "/ME_OpenDRT.log";
    }
  }
  if (!logPath.empty()) {
    std::lock_guard<std::mutex> lock(logMutex);
    FILE* f = std::fopen(logPath.c_str(), "a");
    if (f != nullptr) {
      std::fprintf(f, "[ME_OpenDRT][MetalDiag] %s\n", msg.c_str());
      std::fclose(f);
    }
  }
}

const char* hostOffsetModeName(HostOffsetMode mode) {
  switch (mode) {
    case HostOffsetMode::Zero:
      return "ZERO";
    case HostOffsetMode::Origin:
      return "ORIGIN";
    case HostOffsetMode::FallbackAmbiguous:
      return "FALLBACK";
    case HostOffsetMode::Auto:
    default:
      return "AUTO";
  }
}

void perfLogStage(const char* stage, const std::chrono::steady_clock::time_point& start) {
  if (!perfLogEnabled()) return;
  const auto now = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(now - start).count();
  std::fprintf(stderr, "[ME_OpenDRT][PERF][Metal] %s: %.3f ms\n", stage, ms);
  static std::mutex logMutex;
  static bool pathInit = false;
  static std::string logPath;
  if (!pathInit) {
    pathInit = true;
    const char* home = std::getenv("HOME");
    if (home != nullptr && home[0] != '\0') {
      const std::string logsDir = std::string(home) + "/Library/Logs";
      (void)::mkdir(logsDir.c_str(), 0755);
      logPath = logsDir + "/ME_OpenDRT.log";
    }
  }
  if (!logPath.empty()) {
    std::lock_guard<std::mutex> lock(logMutex);
    FILE* f = std::fopen(logPath.c_str(), "a");
    if (f != nullptr) {
      std::fprintf(f, "[ME_OpenDRT][PERF][Metal] %s: %.3f ms\n", stage, ms);
      std::fclose(f);
    }
  }
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

bool initializePipelineForDevice(id<MTLDevice> device, id<MTLCommandQueue> defaultQueue) {
  auto& ctx = context();
  if (device == nil || defaultQueue == nil) return false;
  ctx.device = device;
  ctx.queue = defaultQueue;
  if (ctx.queue == nil) return false;

  // Metallib is packaged into Contents/Resources during CMake build.
  const std::string libPathStr = metallibPath();
  if (libPathStr.empty()) return false;
  if (debugLogEnabled()) {
    NSOperatingSystemVersion osv = [[NSProcessInfo processInfo] operatingSystemVersion];
    NSLog(@"ME_OpenDRT Metal: runtime macOS %ld.%ld.%ld, metallib path: %s",
          (long)osv.majorVersion,
          (long)osv.minorVersion,
          (long)osv.patchVersion,
          libPathStr.c_str());
  }
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
    NSArray<NSString*>* names = [library functionNames];
    if (names != nil && names.count > 0) {
      NSMutableString* joined = [NSMutableString string];
      for (NSUInteger i = 0; i < names.count; ++i) {
        if (i > 0) [joined appendString:@", "];
        [joined appendString:names[i]];
      }
      NSLog(@"ME_OpenDRT Metal: available metallib functions: %@", joined);
    } else {
      NSLog(@"ME_OpenDRT Metal: metallib contains no discoverable functions");
    }
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
  ctx.initAttempted = true;
  return true;
}

bool initialize(id<MTLCommandQueue> preferredQueue = nil) {
  auto& ctx = context();
  std::lock_guard<std::mutex> lock(ctx.initMutex);

  id<MTLDevice> desiredDevice = preferredQueue != nil ? preferredQueue.device : MTLCreateSystemDefaultDevice();
  id<MTLCommandQueue> desiredQueue = preferredQueue != nil ? preferredQueue : (desiredDevice != nil ? [desiredDevice newCommandQueue] : nil);
  if (desiredDevice == nil || desiredQueue == nil) return false;

  if (ctx.initialized) {
    if (ctx.device == desiredDevice) {
      // Keep shared queue stable for internal path if already set; only initialize it once.
      if (ctx.queue == nil) ctx.queue = desiredQueue;
      return true;
    }
    // Host queue/device changed (or host uses different device): rebuild pipeline safely.
    ctx.pipeline = nil;
    ctx.queue = nil;
    ctx.device = nil;
    ctx.initialized = false;
    ctx.initAttempted = false;
  } else if (ctx.initAttempted) {
    // Prior init attempt failed and no valid initialized context exists.
    return false;
  }

  return initializePipelineForDevice(desiredDevice, desiredQueue);
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
  const int packedRowFloats = width * 4;
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
  [enc setBytes:&packedRowFloats length:sizeof(int) atIndex:6];
  [enc setBytes:&packedRowFloats length:sizeof(int) atIndex:7];

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

void resetHostMetalFailureState() {
  auto& ctx = context();
  ctx.hostAsyncErrorCount.store(0, std::memory_order_relaxed);
  ctx.hostTemporarilyDisabled.store(false, std::memory_order_relaxed);
}

bool renderHost(
    const void* srcMetalBuffer,
    void* dstMetalBuffer,
    int width,
    int height,
    size_t srcRowBytes,
    size_t dstRowBytes,
    int originX,
    int originY,
    const OpenDRTParams& params,
    const OpenDRTDerivedParams& derived,
    void* metalCommandQueue) {
  const auto tStart = std::chrono::steady_clock::now();
  if (srcMetalBuffer == nullptr || dstMetalBuffer == nullptr || metalCommandQueue == nullptr || width <= 0 || height <= 0) {
    return false;
  }

  id<MTLCommandQueue> hostQueue = (id<MTLCommandQueue>)metalCommandQueue;
  if (hostQueue == nil) return false;
  auto& ctx = context();
  if (ctx.hostTemporarilyDisabled.load(std::memory_order_relaxed)) {
    debugLog("Host Metal path temporarily disabled due to prior async errors.");
    return false;
  }
  if (!initialize(hostQueue)) {
    debugLog("Host Metal initialization failed.");
    return false;
  }

  id<MTLBuffer> srcBuffer = (id<MTLBuffer>)srcMetalBuffer;
  id<MTLBuffer> dstBuffer = (id<MTLBuffer>)dstMetalBuffer;
  if (srcBuffer == nil || dstBuffer == nil) return false;

  const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
  if (srcRowBytes == 0) srcRowBytes = packedRowBytes;
  if (dstRowBytes == 0) dstRowBytes = packedRowBytes;
  if (srcRowBytes < packedRowBytes || dstRowBytes < packedRowBytes) return false;
  if ((srcRowBytes % sizeof(float)) != 0 || (dstRowBytes % sizeof(float)) != 0) return false;

  const size_t requiredSrcBytes = srcRowBytes * static_cast<size_t>(height);
  const size_t requiredDstBytes = dstRowBytes * static_cast<size_t>(height);
  const size_t pixelBytes = 4u * sizeof(float);
  const bool originValid = (originX >= 0 && originY >= 0);
  const bool hasOriginOffset = originValid && (originX > 0 || originY > 0);
  size_t srcOffsetBytes = 0;
  size_t dstOffsetBytes = 0;
  size_t srcCandidateOffset = 0;
  size_t dstCandidateOffset = 0;
  bool canUseOffset = false;
  const bool canUseZero = (srcBuffer.length >= requiredSrcBytes && dstBuffer.length >= requiredDstBytes);
  if (hasOriginOffset) {
    const size_t ox = static_cast<size_t>(originX);
    const size_t oy = static_cast<size_t>(originY);
    srcCandidateOffset = oy * srcRowBytes + ox * pixelBytes;
    dstCandidateOffset = oy * dstRowBytes + ox * pixelBytes;
    canUseOffset =
        (srcBuffer.length >= srcCandidateOffset + requiredSrcBytes && dstBuffer.length >= dstCandidateOffset + requiredDstBytes);
  }

  const bool ambiguousOffset = hasOriginOffset && canUseZero && canUseOffset;
  const HostOffsetMode offsetMode = hostOffsetMode();
  {
    static std::atomic<unsigned long long> seq{0};
    const unsigned long long n = ++seq;
    if (metalDiagEnabled() && (n <= 32ull || (n % 120ull) == 0ull || ambiguousOffset)) {
      std::ostringstream oss;
      oss << "host-call#" << n
          << " size=" << width << "x" << height
          << " rowBytes=" << srcRowBytes << "/" << dstRowBytes
          << " origin=(" << originX << "," << originY << ")"
          << " mode=" << hostOffsetModeName(offsetMode)
          << " hasOrigin=" << (hasOriginOffset ? 1 : 0)
          << " canZero=" << (canUseZero ? 1 : 0)
          << " canOffset=" << (canUseOffset ? 1 : 0)
          << " ambiguous=" << (ambiguousOffset ? 1 : 0)
          << " srcLen=" << static_cast<size_t>(srcBuffer.length)
          << " dstLen=" << static_cast<size_t>(dstBuffer.length)
          << " forceWait=" << (forceHostMetalWait() ? 1 : 0)
          << " intermDst=" << (useHostIntermediateDst() ? 1 : 0);
      diagLog(oss.str());
    }
  }
  switch (offsetMode) {
    case HostOffsetMode::Zero:
      if (!canUseZero) return false;
      break;
    case HostOffsetMode::Origin:
      if (!canUseOffset) return false;
      srcOffsetBytes = srcCandidateOffset;
      dstOffsetBytes = dstCandidateOffset;
      break;
    case HostOffsetMode::FallbackAmbiguous:
      if (ambiguousOffset) {
        if (debugLogEnabled()) {
          std::fprintf(
              stderr,
              "[ME_OpenDRT][Metal] Host offset ambiguous (origin=%d,%d rowBytes=%zu/%zu len=%zu/%zu). Falling back.\n",
              originX,
              originY,
              srcRowBytes,
              dstRowBytes,
              static_cast<size_t>(srcBuffer.length),
              static_cast<size_t>(dstBuffer.length));
        }
        return false;
      }
      if (canUseOffset) {
        srcOffsetBytes = srcCandidateOffset;
        dstOffsetBytes = dstCandidateOffset;
      } else if (!canUseZero) {
        return false;
      }
      break;
    case HostOffsetMode::Auto:
    default:
      if (canUseOffset && (!canUseZero || hasOriginOffset)) {
        srcOffsetBytes = srcCandidateOffset;
        dstOffsetBytes = dstCandidateOffset;
      } else if (!canUseZero) {
        return false;
      }
      if (ambiguousOffset && debugLogEnabled()) {
        std::fprintf(
            stderr,
            "[ME_OpenDRT][Metal] Host offset ambiguous -> AUTO chose ORIGIN (origin=%d,%d).\n",
            originX,
            originY);
      }
      break;
  }

  if (metalDiagEnabled()) {
    std::ostringstream oss;
    oss << "offset-select srcOff=" << srcOffsetBytes << " dstOff=" << dstOffsetBytes
        << " required=" << requiredSrcBytes << "/" << requiredDstBytes;
    diagLog(oss.str());
  }

  if (srcBuffer.length < srcOffsetBytes + requiredSrcBytes || dstBuffer.length < dstOffsetBytes + requiredDstBytes) {
    return false;
  }

  id<MTLBuffer> kernelSrcBuffer = srcBuffer;
  size_t kernelSrcOffset = srcOffsetBytes;
  id<MTLBuffer> kernelDstBuffer = dstBuffer;
  size_t kernelDstOffset = dstOffsetBytes;

  const bool sameBuffer = (srcBuffer == dstBuffer);
  const size_t srcBegin = srcOffsetBytes;
  const size_t srcEnd = srcOffsetBytes + requiredSrcBytes;
  const size_t dstBegin = dstOffsetBytes;
  const size_t dstEnd = dstOffsetBytes + requiredDstBytes;
  const bool overlappingRanges = sameBuffer && (srcBegin < dstEnd) && (dstBegin < srcEnd);
  const bool needsAliasProtection = overlappingRanges && (srcOffsetBytes != dstOffsetBytes || srcRowBytes != dstRowBytes);
  if (needsAliasProtection) {
    auto& buffers = threadBuffers();
    if (buffers.hostTempSrcBuffer == nil || buffers.hostTempSrcBytes < requiredSrcBytes) {
      buffers.hostTempSrcBuffer = [ctx.device newBufferWithLength:requiredSrcBytes options:MTLResourceStorageModeShared];
      buffers.hostTempSrcBytes = (buffers.hostTempSrcBuffer != nil) ? requiredSrcBytes : 0;
    }
    if (buffers.hostTempSrcBuffer == nil) {
      debugLog("Failed to allocate host temp src buffer for alias protection.");
      return false;
    }
    kernelSrcBuffer = buffers.hostTempSrcBuffer;
    kernelSrcOffset = 0;
    diagLog("alias-protect enabled: src/dst overlap on same host buffer.");
  }
  if (!forceHostMetalWait() && useHostIntermediateDst()) {
    auto& buffers = threadBuffers();
    if (buffers.hostTempDstBuffer == nil || buffers.hostTempDstBytes < requiredDstBytes) {
      buffers.hostTempDstBuffer = [ctx.device newBufferWithLength:requiredDstBytes options:MTLResourceStorageModeShared];
      buffers.hostTempDstBytes = (buffers.hostTempDstBuffer != nil) ? requiredDstBytes : 0;
    }
    if (buffers.hostTempDstBuffer == nil) {
      debugLog("Failed to allocate host temp dst buffer.");
      return false;
    }
    kernelDstBuffer = buffers.hostTempDstBuffer;
    kernelDstOffset = 0;
  }

  const int srcRowFloats = static_cast<int>(srcRowBytes / sizeof(float));
  const int dstRowFloats = static_cast<int>(dstRowBytes / sizeof(float));

  id<MTLCommandBuffer> cmd = [hostQueue commandBuffer];
  if (cmd == nil) {
    debugLog("Failed to create host Metal command buffer.");
    return false;
  }
  id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  if (enc == nil) {
    debugLog("Failed to create host Metal compute encoder.");
    return false;
  }

  if (needsAliasProtection) {
    id<MTLBlitCommandEncoder> preBlit = [cmd blitCommandEncoder];
    if (preBlit == nil) {
      debugLog("Failed to create precompute alias-protect blit encoder.");
      return false;
    }
    [preBlit copyFromBuffer:srcBuffer sourceOffset:srcOffsetBytes toBuffer:kernelSrcBuffer destinationOffset:0 size:requiredSrcBytes];
    [preBlit endEncoding];
  }

  [enc setComputePipelineState:ctx.pipeline];
  [enc setBuffer:kernelSrcBuffer offset:kernelSrcOffset atIndex:0];
  [enc setBuffer:kernelDstBuffer offset:kernelDstOffset atIndex:1];
  [enc setBytes:&params length:sizeof(OpenDRTParams) atIndex:2];
  [enc setBytes:&width length:sizeof(int) atIndex:3];
  [enc setBytes:&height length:sizeof(int) atIndex:4];
  [enc setBytes:&derived length:sizeof(OpenDRTDerivedParams) atIndex:5];
  [enc setBytes:&srcRowFloats length:sizeof(int) atIndex:6];
  [enc setBytes:&dstRowFloats length:sizeof(int) atIndex:7];

  const NSUInteger maxThreads = ctx.pipeline.maxTotalThreadsPerThreadgroup;
  const NSUInteger tew = ctx.pipeline.threadExecutionWidth;
  NSUInteger tx = tew > 0 ? tew : 16;
  if (tx > maxThreads) tx = maxThreads;
  NSUInteger ty = maxThreads / tx;
  if (ty == 0) ty = 1;
  if (ty > 16) ty = 16;
  const MTLSize threadsPerThreadgroup = MTLSizeMake(tx, ty, 1);
  const MTLSize threadsPerGrid = MTLSizeMake(static_cast<NSUInteger>(width), static_cast<NSUInteger>(height), 1);
  [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
  [enc endEncoding];
  if (kernelDstBuffer != dstBuffer) {
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (blit == nil) {
      debugLog("Failed to create host Metal blit encoder.");
      return false;
    }
    [blit copyFromBuffer:kernelDstBuffer
            sourceOffset:0
                toBuffer:dstBuffer
       destinationOffset:dstOffsetBytes
                    size:requiredDstBytes];
    [blit endEncoding];
  }
  if (!forceHostMetalWait()) {
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      if (cb.status != MTLCommandBufferStatusCompleted) {
        context().hostAsyncErrorCount.fetch_add(1, std::memory_order_relaxed);
        if (debugLogEnabled()) {
          if (cb.error != nil) {
            NSLog(@"ME_OpenDRT Metal host async failure: %@", cb.error.localizedDescription);
          } else {
            NSLog(@"ME_OpenDRT Metal host async failure with unknown error");
          }
        }
        if (metalDiagEnabled()) {
          std::ostringstream oss;
          oss << "async-complete status=" << static_cast<int>(cb.status)
              << " errors=" << context().hostAsyncErrorCount.load(std::memory_order_relaxed);
          diagLog(oss.str());
        }
      } else {
        context().hostAsyncErrorCount.store(0, std::memory_order_relaxed);
        if (metalDiagEnabled()) {
          diagLog("async-complete status=COMPLETED errors=0");
        }
      }
    }];
  }
  [cmd commit];

  if (forceHostMetalWait()) {
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      if (cmd.error != nil) {
        NSLog(@"ME_OpenDRT Metal host: command buffer failed: %@", cmd.error.localizedDescription);
      }
      debugLog("Host Metal command buffer failed.");
      return false;
    }
    ctx.hostAsyncErrorCount.store(0, std::memory_order_relaxed);
    if (metalDiagEnabled()) {
      diagLog("sync-complete status=COMPLETED errors=0");
    }
  }

  if (!forceHostMetalWait()) {
    const int errors = ctx.hostAsyncErrorCount.load(std::memory_order_relaxed);
    if (errors >= 3) {
      ctx.hostTemporarilyDisabled.store(true, std::memory_order_relaxed);
      debugLog("Host Metal path auto-disabled after repeated async errors.");
      return false;
    }
  }

  perfLogStage("Metal host total", tStart);
  return true;
}

}  // namespace OpenDRTMetal
