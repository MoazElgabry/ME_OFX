#pragma once

#include <cstddef>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include "OpenDRTParams.h"

#if defined(_WIN32)
#include <cuda_runtime.h>
extern "C" void launchOpenDRTKernel(
    const float* src,
    float* dst,
    int width,
    int height,
    const OpenDRTParams* p,
    const OpenDRTDerivedParams* d,
    cudaStream_t stream);
#endif

#if defined(__APPLE__)
#include "metal/OpenDRTMetal.h"
#endif

class OpenDRTProcessor {
 public:
  explicit OpenDRTProcessor(const OpenDRTParams& params) : params_(params) { initRuntimeFlags(); }
  void setParams(const OpenDRTParams& params) { params_ = params; }
  ~OpenDRTProcessor() {
#if defined(_WIN32)
    releaseCudaStream();
    releaseCudaBuffers();
#endif
  }

  bool render(const float* src, float* dst, int width, int height, bool preferCuda, bool hostSupportsOpenCL) {
    const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
    return renderWithLayout(src, dst, width, height, packedRowBytes, packedRowBytes, preferCuda, hostSupportsOpenCL);
  }

  bool renderWithLayout(
      const float* src,
      float* dst,
      int width,
      int height,
      size_t srcRowBytes,
      size_t dstRowBytes,
      bool preferCuda,
      bool hostSupportsOpenCL) {
    computeDerivedParams();
    // Platform-first dispatch: Metal on macOS, CUDA on Windows, then fallbacks.
#if defined(__APPLE__)
    if (renderMetal(src, dst, width, height, srcRowBytes, dstRowBytes)) {
      return true;
    }
    debugLog("Metal path failed, falling back.");
#endif
#if defined(_WIN32)
    if (preferCuda && cudaAvailableCached()) {
      if (renderCUDA(src, dst, width, height, srcRowBytes, dstRowBytes)) {
        return true;
      }
      debugLog("CUDA path failed, falling back.");
    }
#endif
    if (hostSupportsOpenCL && renderOpenCL(src, dst, width, height)) {
      return true;
    }
    if (hostSupportsOpenCL) {
      debugLog("OpenCL path failed, falling back to CPU.");
    }
    return renderCPU(src, dst, width, height);
  }

#if defined(_WIN32)
  bool renderCUDA(const float* src, float* dst, int width, int height, size_t srcRowBytes, size_t dstRowBytes) {
    std::lock_guard<std::mutex> lock(cudaMutex_);
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u * sizeof(float);
    const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
    const bool packedSrc = (srcRowBytes == packedRowBytes);
    const bool packedDst = (dstRowBytes == packedRowBytes);

    if (!ensureCudaDevice()) {
      return false;
    }

    if (!ensureCudaBuffers(bytes)) {
      return false;
    }
    if (!cudaLegacySyncEnabled_ && !ensureCudaStream()) {
      debugLog("CUDA stream init failed, using legacy sync path.");
      return renderCUDALegacy(src, dst, width, height, bytes, srcRowBytes, dstRowBytes);
    }

    if (cudaLegacySyncEnabled_) {
      return renderCUDALegacy(src, dst, width, height, bytes, srcRowBytes, dstRowBytes);
    }

    const auto t0 = std::chrono::steady_clock::now();
    const auto tH2D = std::chrono::steady_clock::now();

    if (!cudaDisable2DCopyEnabled_ && !packedSrc) {
      if (cudaMemcpy2DAsync(
              cudaSrc_,
              packedRowBytes,
              src,
              srcRowBytes,
              packedRowBytes,
              static_cast<size_t>(height),
              cudaMemcpyHostToDevice,
              cudaStream_) != cudaSuccess) {
        return false;
      }
    } else {
      if (!packedSrc) return false;
      if (cudaMemcpyAsync(cudaSrc_, src, bytes, cudaMemcpyHostToDevice, cudaStream_) != cudaSuccess) {
        return false;
      }
    }
    perfLogStage("CUDA H2D", tH2D);

    // Kernel reads flat RGBA float pixels and resolved scalar params.
    const auto tKernel = std::chrono::steady_clock::now();
    launchOpenDRTKernel(cudaSrc_, cudaDst_, width, height, &params_, &derived_, cudaStream_);

    const cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
      return false;
    }
    perfLogStage("CUDA kernel launch", tKernel);

    const auto tD2H = std::chrono::steady_clock::now();
    if (!cudaDisable2DCopyEnabled_ && !packedDst) {
      if (cudaMemcpy2DAsync(
              dst,
              dstRowBytes,
              cudaDst_,
              packedRowBytes,
              packedRowBytes,
              static_cast<size_t>(height),
              cudaMemcpyDeviceToHost,
              cudaStream_) != cudaSuccess) {
        return false;
      }
    } else {
      if (!packedDst) return false;
      if (cudaMemcpyAsync(dst, cudaDst_, bytes, cudaMemcpyDeviceToHost, cudaStream_) != cudaSuccess) {
        return false;
      }
    }
    perfLogStage("CUDA D2H", tD2H);

    if (cudaStreamSynchronize(cudaStream_) != cudaSuccess) {
      return false;
    }

    perfLogStage("CUDA render", t0);

    return true;
  }
#endif

#if defined(__APPLE__)
  bool renderMetal(
      const float* src,
      float* dst,
      int width,
      int height,
      size_t srcRowBytes,
      size_t dstRowBytes) {
    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = OpenDRTMetal::render(src, dst, width, height, srcRowBytes, dstRowBytes, params_, derived_);
    perfLogStage("Metal render", t0);
    return ok;
  }
#endif

  bool renderOpenCL(const float* src, float* dst, int width, int height) {
    // OpenCL path is reserved for future parity work.
    (void)src;
    (void)dst;
    (void)width;
    (void)height;
    return false;
  }

  bool renderCPU(const float* src, float* dst, int width, int height) {
    // Safety fallback used when no GPU path is available.
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u * sizeof(float);
    std::memcpy(dst, src, bytes);
    return true;
  }

 private:
  bool envFlagEnabled(const char* name) const {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return false;
    return !(v[0] == '0' && v[1] == '\0');
  }

  void debugLog(const char* msg) const {
    if (!debugLogEnabled_) return;
    std::fprintf(stderr, "[ME_OpenDRT] %s\n", msg);
  }

  void perfLogStage(const char* label, const std::chrono::steady_clock::time_point& start) const {
    if (!perfLogEnabled_) return;
    const auto now = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - start).count();
    std::fprintf(stderr, "[ME_OpenDRT][PERF] %s: %.3f ms\n", label, ms);
  }

  void initRuntimeFlags() {
    debugLogEnabled_ = envFlagEnabled("ME_OPENDRT_DEBUG_LOG");
    perfLogEnabled_ = envFlagEnabled("ME_OPENDRT_PERF_LOG");
#if defined(_WIN32)
    cudaLegacySyncEnabled_ = envFlagEnabled("ME_OPENDRT_CUDA_LEGACY_SYNC");
    cudaDisable2DCopyEnabled_ = envFlagEnabled("ME_OPENDRT_DISABLE_CUDA_2D_COPY");
#endif
    disableDerivedEnabled_ = envFlagEnabled("ME_OPENDRT_DISABLE_DERIVED");
  }

  void computeDerivedParams() {
    if (disableDerivedEnabled_) {
      derived_.enabled = 0;
      return;
    }
    derived_.enabled = 1;
    const float ts_x1 = std::pow(2.0f, 6.0f * params_.tn_sh + 4.0f);
    const float ts_y1 = params_.tn_Lp / 100.0f;
    const float ts_x0 = 0.18f + params_.tn_off;
    const float ts_y0 = params_.tn_Lg / 100.0f * (1.0f + params_.tn_gb * std::log2(ts_y1));
    const float ts_s0 = params_.tn_toe == 0.0f ? ts_y0 : (ts_y0 + std::sqrt(ts_y0 * (4.0f * params_.tn_toe + ts_y0))) / 2.0f;
    const float ts_p = params_.tn_con / (1.0f + static_cast<float>(params_.tn_su) * 0.05f);
    const float ts_s10 = ts_x0 * (std::pow(ts_s0, -1.0f / params_.tn_con) - 1.0f);
    const float ts_m1 = ts_y1 / std::pow(ts_x1 / (ts_x1 + ts_s10), params_.tn_con);
    const float ts_m2 = params_.tn_toe == 0.0f ? ts_m1 : (ts_m1 + std::sqrt(ts_m1 * (4.0f * params_.tn_toe + ts_m1))) / 2.0f;
    const float ts_s = ts_x0 * (std::pow(ts_s0 / ts_m2, -1.0f / params_.tn_con) - 1.0f);
    const float ts_dsc = params_.eotf == 4 ? 0.01f : params_.eotf == 5 ? 0.1f : 100.0f / params_.tn_Lp;
    const float pt_cmp_Lf = params_.pt_hdr * std::fmin(1.0f, (params_.tn_Lp - 100.0f) / 900.0f);
    const float s_Lp100 = ts_x0 * (std::pow((params_.tn_Lg / 100.0f), -1.0f / params_.tn_con) - 1.0f);
    const float ts_s1 = ts_s * pt_cmp_Lf + s_Lp100 * (1.0f - pt_cmp_Lf);

    derived_.ts_x1 = ts_x1;
    derived_.ts_y1 = ts_y1;
    derived_.ts_x0 = ts_x0;
    derived_.ts_y0 = ts_y0;
    derived_.ts_s0 = ts_s0;
    derived_.ts_p = ts_p;
    derived_.ts_s10 = ts_s10;
    derived_.ts_m1 = ts_m1;
    derived_.ts_m2 = ts_m2;
    derived_.ts_s = ts_s;
    derived_.ts_dsc = ts_dsc;
    derived_.pt_cmp_Lf = pt_cmp_Lf;
    derived_.s_Lp100 = s_Lp100;
    derived_.ts_s1 = ts_s1;
  }

#if defined(_WIN32)
  bool ensureCudaStream() {
    if (cudaStream_ != nullptr) {
      return true;
    }
    return cudaStreamCreate(&cudaStream_) == cudaSuccess;
  }

  void releaseCudaStream() {
    if (cudaStream_ != nullptr) {
      cudaStreamDestroy(cudaStream_);
      cudaStream_ = nullptr;
    }
  }

  bool renderCUDALegacy(
      const float* src,
      float* dst,
      int width,
      int height,
      size_t bytes,
      size_t srcRowBytes,
      size_t dstRowBytes) {
    const auto t0 = std::chrono::steady_clock::now();
    const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
    const bool packedSrc = (srcRowBytes == packedRowBytes);
    const bool packedDst = (dstRowBytes == packedRowBytes);
    if (!cudaDisable2DCopyEnabled_ && !packedSrc) {
      if (cudaMemcpy2D(
              cudaSrc_,
              packedRowBytes,
              src,
              srcRowBytes,
              packedRowBytes,
              static_cast<size_t>(height),
              cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
      }
    } else {
      if (!packedSrc) return false;
      if (cudaMemcpy(cudaSrc_, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        return false;
      }
    }

    launchOpenDRTKernel(cudaSrc_, cudaDst_, width, height, &params_, &derived_, nullptr);

    const cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
      return false;
    }

    const cudaError_t syncStatus = cudaDeviceSynchronize();
    if (syncStatus != cudaSuccess) {
      return false;
    }

    if (!cudaDisable2DCopyEnabled_ && !packedDst) {
      if (cudaMemcpy2D(
              dst,
              dstRowBytes,
              cudaDst_,
              packedRowBytes,
              packedRowBytes,
              static_cast<size_t>(height),
              cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
      }
    } else {
      if (!packedDst) return false;
      if (cudaMemcpy(dst, cudaDst_, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
      }
    }

    perfLogStage("CUDA render legacy", t0);
    return true;
  }

  bool cudaAvailableCached() {
    if (cudaAvailabilityKnown_) {
      return cudaAvailability_;
    }
    cudaAvailability_ = queryCudaAvailable();
    cudaAvailabilityKnown_ = true;
    return cudaAvailability_;
  }

  bool queryCudaAvailable() const {
    int count = 0;
    const cudaError_t st = cudaGetDeviceCount(&count);
    return st == cudaSuccess && count > 0;
  }

  bool ensureCudaDevice() {
    if (cudaDeviceReady_) {
      return true;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
      return false;
    }
    cudaDeviceReady_ = true;
    return true;
  }

  bool ensureCudaBuffers(size_t bytes) {
    if (cudaSrc_ != nullptr && cudaDst_ != nullptr && cudaBytes_ == bytes) {
      return true;
    }
    releaseCudaBuffers();
    if (cudaMalloc(&cudaSrc_, bytes) != cudaSuccess) {
      cudaSrc_ = nullptr;
      return false;
    }
    if (cudaMalloc(&cudaDst_, bytes) != cudaSuccess) {
      cudaFree(cudaSrc_);
      cudaSrc_ = nullptr;
      cudaDst_ = nullptr;
      return false;
    }
    cudaBytes_ = bytes;
    return true;
  }

  void releaseCudaBuffers() {
    if (cudaSrc_ != nullptr) {
      cudaFree(cudaSrc_);
      cudaSrc_ = nullptr;
    }
    if (cudaDst_ != nullptr) {
      cudaFree(cudaDst_);
      cudaDst_ = nullptr;
    }
    cudaBytes_ = 0;
  }
#endif

  OpenDRTParams params_;
  OpenDRTDerivedParams derived_{};
  bool debugLogEnabled_ = false;
  bool perfLogEnabled_ = false;
  bool disableDerivedEnabled_ = false;
#if defined(_WIN32)
  bool cudaLegacySyncEnabled_ = false;
  bool cudaDisable2DCopyEnabled_ = false;
  bool cudaAvailability_ = false;
  bool cudaAvailabilityKnown_ = false;
  bool cudaDeviceReady_ = false;
  float* cudaSrc_ = nullptr;
  float* cudaDst_ = nullptr;
  cudaStream_t cudaStream_ = nullptr;
  size_t cudaBytes_ = 0;
  std::mutex cudaMutex_;
#endif
};
