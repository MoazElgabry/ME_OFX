#pragma once

#include <cstddef>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include "OpenDRTParams.h"

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
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
    releaseOpenCL();
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
    (void)hostSupportsOpenCL;
    // Derived values are computed once per frame on host and consumed by all backends.
    // This is a parity + performance guardrail (avoid per-pixel recompute drift).
    computeDerivedParams();
    // Backend dispatch policy:
    // - macOS: Metal first, then CPU fallback.
    // - Windows: CUDA first (unless forced OpenCL), then OpenCL, then CPU fallback.
#if defined(__APPLE__)
    if (renderMetal(src, dst, width, height, srcRowBytes, dstRowBytes)) {
      return true;
    }
    debugLog("Metal path failed, falling back.");
#endif
#if defined(_WIN32)
    const bool forceOpenCL = openclForceEnabled_;
    if (!forceOpenCL && preferCuda && cudaAvailableCached()) {
      if (renderCUDA(src, dst, width, height, srcRowBytes, dstRowBytes)) return true;
      debugLog("CUDA path failed, falling back.");
    }
    if (!openclDisableEnabled_ && openclAvailableCached()) {
      if (renderOpenCL(src, dst, width, height, srcRowBytes, dstRowBytes)) return true;
      debugLog("OpenCL path failed, falling back to CPU.");
    }
#endif
    // Last-resort correctness fallback; should never crash the host.
    return renderCPU(src, dst, width, height);
  }

#if defined(_WIN32)
  // CUDA path is the primary Windows GPU backend.
  // It keeps existing async + 2D copy optimizations and can fall back to legacy sync via env flag.
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
  // Metal path remains unchanged and is the primary backend on macOS.
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

  // OpenCL path for non-CUDA systems (primarily AMD/Intel GPUs on Windows).
  // Uses persistent runtime objects and buffers to avoid per-frame setup overhead.
  bool renderOpenCL(
      const float* src,
      float* dst,
      int width,
      int height,
      size_t srcRowBytes,
      size_t dstRowBytes) {
#if !defined(_WIN32)
    (void)src;
    (void)dst;
    (void)width;
    (void)height;
    (void)srcRowBytes;
    (void)dstRowBytes;
    return false;
#else
    std::lock_guard<std::mutex> lock(openclMutex_);
    if (!initializeOpenCLRuntime()) return false;

    const size_t packedRowBytes = static_cast<size_t>(width) * 4u * sizeof(float);
    const size_t bytes = packedRowBytes * static_cast<size_t>(height);
    const bool packedSrc = (srcRowBytes == packedRowBytes);
    const bool packedDst = (dstRowBytes == packedRowBytes);
    if (!ensureOpenCLBuffers(bytes)) return false;

    const auto t0 = std::chrono::steady_clock::now();
    const auto tH2D = std::chrono::steady_clock::now();
    if (!openclDisable2DCopyEnabled_ && !packedSrc) {
      const size_t origin[3] = {0, 0, 0};
      const size_t region[3] = {packedRowBytes, static_cast<size_t>(height), 1};
      if (clEnqueueWriteBufferRect(
              clQueue_,
              clSrc_,
              CL_FALSE,
              origin,
              origin,
              region,
              packedRowBytes,
              0,
              srcRowBytes,
              0,
              src,
              0,
              nullptr,
              nullptr) != CL_SUCCESS) {
        return false;
      }
    } else {
      if (!packedSrc) return false;
      if (clEnqueueWriteBuffer(clQueue_, clSrc_, CL_FALSE, 0, bytes, src, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
      }
    }
    if (clEnqueueWriteBuffer(clQueue_, clParams_, CL_FALSE, 0, sizeof(OpenDRTParams), &params_, 0, nullptr, nullptr) != CL_SUCCESS) {
      return false;
    }
    if (clEnqueueWriteBuffer(clQueue_, clDerived_, CL_FALSE, 0, sizeof(OpenDRTDerivedParams), &derived_, 0, nullptr, nullptr) != CL_SUCCESS) {
      return false;
    }
    perfLogStage("OpenCL H2D", tH2D);

    const auto tKernel = std::chrono::steady_clock::now();
    if (clSetKernelArg(clKernel_, 0, sizeof(cl_mem), &clSrc_) != CL_SUCCESS) return false;
    if (clSetKernelArg(clKernel_, 1, sizeof(cl_mem), &clDst_) != CL_SUCCESS) return false;
    if (clSetKernelArg(clKernel_, 2, sizeof(int), &width) != CL_SUCCESS) return false;
    if (clSetKernelArg(clKernel_, 3, sizeof(int), &height) != CL_SUCCESS) return false;
    if (clSetKernelArg(clKernel_, 4, sizeof(cl_mem), &clParams_) != CL_SUCCESS) return false;
    if (clSetKernelArg(clKernel_, 5, sizeof(cl_mem), &clDerived_) != CL_SUCCESS) return false;

    const size_t global[2] = {static_cast<size_t>(width), static_cast<size_t>(height)};
    if (clEnqueueNDRangeKernel(clQueue_, clKernel_, 2, nullptr, global, nullptr, 0, nullptr, nullptr) != CL_SUCCESS) {
      return false;
    }
    perfLogStage("OpenCL kernel", tKernel);

    const auto tD2H = std::chrono::steady_clock::now();
    if (!openclDisable2DCopyEnabled_ && !packedDst) {
      const size_t origin[3] = {0, 0, 0};
      const size_t region[3] = {packedRowBytes, static_cast<size_t>(height), 1};
      if (clEnqueueReadBufferRect(
              clQueue_,
              clDst_,
              CL_FALSE,
              origin,
              origin,
              region,
              packedRowBytes,
              0,
              dstRowBytes,
              0,
              dst,
              0,
              nullptr,
              nullptr) != CL_SUCCESS) {
        return false;
      }
    } else {
      if (!packedDst) return false;
      if (clEnqueueReadBuffer(clQueue_, clDst_, CL_FALSE, 0, bytes, dst, 0, nullptr, nullptr) != CL_SUCCESS) {
        return false;
      }
    }
    perfLogStage("OpenCL D2H", tD2H);

    if (clFinish(clQueue_) != CL_SUCCESS) return false;
    perfLogStage("OpenCL render", t0);
    return true;
#endif
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
    // Runtime switches for triage/perf tuning without rebuilding.
    cudaLegacySyncEnabled_ = envFlagEnabled("ME_OPENDRT_CUDA_LEGACY_SYNC");
    cudaDisable2DCopyEnabled_ = envFlagEnabled("ME_OPENDRT_DISABLE_CUDA_2D_COPY");
    openclForceEnabled_ = envFlagEnabled("ME_OPENDRT_FORCE_OPENCL");
    openclDisableEnabled_ = envFlagEnabled("ME_OPENDRT_DISABLE_OPENCL");
    openclDisable2DCopyEnabled_ = envFlagEnabled("ME_OPENDRT_OPENCL_DISABLE_2D_COPY");
#endif
    disableDerivedEnabled_ = envFlagEnabled("ME_OPENDRT_DISABLE_DERIVED");
  }

  // Computes frame-constant terms for tonescale/purity logic once per frame.
  // Backends consume this struct to preserve parity and reduce GPU workload.
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
  // ----- OpenCL runtime lifecycle -----
  // Lazy-init the runtime once, cache availability, and reuse buffers per frame size.
  bool openclAvailableCached() {
    if (openclAvailabilityKnown_) return openclAvailability_;
    openclAvailability_ = initializeOpenCLRuntime();
    openclAvailabilityKnown_ = true;
    return openclAvailability_;
  }

  std::string openclKernelPath() const {
    HMODULE self = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCSTR>(&launchOpenDRTKernel), &self)) {
      return std::string();
    }
    char modulePath[MAX_PATH] = {0};
    if (GetModuleFileNameA(self, modulePath, MAX_PATH) == 0) return std::string();
    std::string p(modulePath);
    const size_t pos = p.find_last_of("\\/");
    if (pos == std::string::npos) return "OpenDRT.cl";
    return p.substr(0, pos + 1) + "OpenDRT.cl";
  }

  bool initializeOpenCLRuntime() {
    if (clInitFailed_) return false;
    if (clKernel_ != nullptr) return true;

    cl_int err = CL_SUCCESS;
    cl_uint numPlatforms = 0;
    if (clGetPlatformIDs(0, nullptr, &numPlatforms) != CL_SUCCESS || numPlatforms == 0) {
      clInitFailed_ = true;
      return false;
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    if (clGetPlatformIDs(numPlatforms, platforms.data(), nullptr) != CL_SUCCESS) {
      clInitFailed_ = true;
      return false;
    }

    cl_device_id chosenDevice = nullptr;
    cl_platform_id chosenPlatform = nullptr;
    for (cl_platform_id pid : platforms) {
      cl_uint numDevices = 0;
      if (clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices) != CL_SUCCESS || numDevices == 0) {
        continue;
      }
      std::vector<cl_device_id> devices(numDevices);
      if (clGetDeviceIDs(pid, CL_DEVICE_TYPE_GPU, numDevices, devices.data(), nullptr) != CL_SUCCESS) continue;
      chosenPlatform = pid;
      chosenDevice = devices[0];
      break;
    }
    if (chosenDevice == nullptr) {
      clInitFailed_ = true;
      return false;
    }

    clContext_ = clCreateContext(nullptr, 1, &chosenDevice, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || clContext_ == nullptr) {
      clInitFailed_ = true;
      releaseOpenCL();
      return false;
    }
    clDevice_ = chosenDevice;
    clPlatform_ = chosenPlatform;
    clQueue_ = clCreateCommandQueue(clContext_, clDevice_, 0, &err);
    if (err != CL_SUCCESS || clQueue_ == nullptr) {
      clInitFailed_ = true;
      releaseOpenCL();
      return false;
    }

    const std::string path = openclKernelPath();
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
      debugLog("OpenCL kernel source not found next to plugin binary.");
      clInitFailed_ = true;
      releaseOpenCL();
      return false;
    }
    const std::string source((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    const char* src = source.c_str();
    const size_t len = source.size();
    clProgram_ = clCreateProgramWithSource(clContext_, 1, &src, &len, &err);
    if (err != CL_SUCCESS || clProgram_ == nullptr) {
      clInitFailed_ = true;
      releaseOpenCL();
      return false;
    }

    err = clBuildProgram(clProgram_, 1, &clDevice_, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t logSize = 0;
      clGetProgramBuildInfo(clProgram_, clDevice_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
      if (logSize > 1) {
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(clProgram_, clDevice_, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        if (debugLogEnabled_) std::fprintf(stderr, "[ME_OpenDRT] OpenCL build log:\n%s\n", log.data());
      }
      clInitFailed_ = true;
      releaseOpenCL();
      return false;
    }

    clKernel_ = clCreateKernel(clProgram_, "OpenDRTKernel", &err);
    if (err != CL_SUCCESS || clKernel_ == nullptr) {
      clInitFailed_ = true;
      releaseOpenCL();
      return false;
    }
    return true;
  }

  bool ensureOpenCLBuffers(size_t bytes) {
    if (clSrc_ != nullptr && clDst_ != nullptr && clParams_ != nullptr && clDerived_ != nullptr && clBytes_ == bytes) {
      return true;
    }
    releaseOpenCLBuffers();
    cl_int err = CL_SUCCESS;
    clSrc_ = clCreateBuffer(clContext_, CL_MEM_READ_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS || clSrc_ == nullptr) return false;
    clDst_ = clCreateBuffer(clContext_, CL_MEM_WRITE_ONLY, bytes, nullptr, &err);
    if (err != CL_SUCCESS || clDst_ == nullptr) return false;
    clParams_ = clCreateBuffer(clContext_, CL_MEM_READ_ONLY, sizeof(OpenDRTParams), nullptr, &err);
    if (err != CL_SUCCESS || clParams_ == nullptr) return false;
    clDerived_ = clCreateBuffer(clContext_, CL_MEM_READ_ONLY, sizeof(OpenDRTDerivedParams), nullptr, &err);
    if (err != CL_SUCCESS || clDerived_ == nullptr) return false;
    clBytes_ = bytes;
    return true;
  }

  void releaseOpenCLBuffers() {
    if (clDerived_ != nullptr) { clReleaseMemObject(clDerived_); clDerived_ = nullptr; }
    if (clParams_ != nullptr) { clReleaseMemObject(clParams_); clParams_ = nullptr; }
    if (clDst_ != nullptr) { clReleaseMemObject(clDst_); clDst_ = nullptr; }
    if (clSrc_ != nullptr) { clReleaseMemObject(clSrc_); clSrc_ = nullptr; }
    clBytes_ = 0;
  }

  void releaseOpenCL() {
    releaseOpenCLBuffers();
    if (clKernel_ != nullptr) { clReleaseKernel(clKernel_); clKernel_ = nullptr; }
    if (clProgram_ != nullptr) { clReleaseProgram(clProgram_); clProgram_ = nullptr; }
    if (clQueue_ != nullptr) { clReleaseCommandQueue(clQueue_); clQueue_ = nullptr; }
    if (clContext_ != nullptr) { clReleaseContext(clContext_); clContext_ = nullptr; }
    clDevice_ = nullptr;
    clPlatform_ = nullptr;
  }

  // ----- CUDA runtime lifecycle -----
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
  // Generic runtime flags shared across backends.
  bool debugLogEnabled_ = false;
  bool perfLogEnabled_ = false;
  bool disableDerivedEnabled_ = false;
#if defined(_WIN32)
  // CUDA feature flags and cached device/runtime state.
  bool cudaLegacySyncEnabled_ = false;
  bool cudaDisable2DCopyEnabled_ = false;
  // OpenCL feature flags and cached runtime state.
  bool openclForceEnabled_ = false;
  bool openclDisableEnabled_ = false;
  bool openclDisable2DCopyEnabled_ = false;
  bool openclAvailability_ = false;
  bool openclAvailabilityKnown_ = false;
  bool clInitFailed_ = false;
  bool cudaAvailability_ = false;
  bool cudaAvailabilityKnown_ = false;
  bool cudaDeviceReady_ = false;
  float* cudaSrc_ = nullptr;
  float* cudaDst_ = nullptr;
  cudaStream_t cudaStream_ = nullptr;
  size_t cudaBytes_ = 0;
  std::mutex cudaMutex_;
  cl_platform_id clPlatform_ = nullptr;
  cl_device_id clDevice_ = nullptr;
  cl_context clContext_ = nullptr;
  cl_command_queue clQueue_ = nullptr;
  cl_program clProgram_ = nullptr;
  cl_kernel clKernel_ = nullptr;
  cl_mem clSrc_ = nullptr;
  cl_mem clDst_ = nullptr;
  cl_mem clParams_ = nullptr;
  cl_mem clDerived_ = nullptr;
  size_t clBytes_ = 0;
  std::mutex openclMutex_;
#endif
};
