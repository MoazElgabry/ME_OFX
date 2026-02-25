#pragma once

#include <cstddef>
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
    cudaStream_t stream);
#endif

#if defined(__APPLE__)
#include "metal/OpenDRTMetal.h"
#endif

class OpenDRTProcessor {
 public:
  explicit OpenDRTProcessor(const OpenDRTParams& params) : params_(params) {}
  ~OpenDRTProcessor() {
#if defined(_WIN32)
    releaseCudaBuffers();
#endif
  }

  bool render(const float* src, float* dst, int width, int height, bool preferCuda, bool hostSupportsOpenCL) {
    // Platform-first dispatch: Metal on macOS, CUDA on Windows, then fallbacks.
#if defined(__APPLE__)
    if (renderMetal(src, dst, width, height)) {
      return true;
    }
#endif
#if defined(_WIN32)
    if (preferCuda && cudaAvailable() && renderCUDA(src, dst, width, height)) {
      return true;
    }
#endif
    if (hostSupportsOpenCL && renderOpenCL(src, dst, width, height)) {
      return true;
    }
    return renderCPU(src, dst, width, height);
  }

#if defined(_WIN32)
  bool renderCUDA(const float* src, float* dst, int width, int height) {
    std::lock_guard<std::mutex> lock(cudaMutex_);
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u * sizeof(float);

    if (cudaSetDevice(0) != cudaSuccess) {
      return false;
    }

    if (!ensureCudaBuffers(bytes)) {
      return false;
    }

    if (cudaMemcpy(cudaSrc_, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      return false;
    }

    // Kernel reads flat RGBA float pixels and resolved scalar params.
    launchOpenDRTKernel(cudaSrc_, cudaDst_, width, height, &params_, nullptr);

    const cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
      return false;
    }

    const cudaError_t syncStatus = cudaDeviceSynchronize();
    if (syncStatus != cudaSuccess) {
      return false;
    }

    if (cudaMemcpy(dst, cudaDst_, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }

    return true;
  }
#endif

#if defined(__APPLE__)
  bool renderMetal(const float* src, float* dst, int width, int height) {
    return OpenDRTMetal::render(src, dst, width, height, params_);
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
#if defined(_WIN32)
  bool cudaAvailable() const {
    int count = 0;
    const cudaError_t st = cudaGetDeviceCount(&count);
    return st == cudaSuccess && count > 0;
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
#if defined(_WIN32)
  float* cudaSrc_ = nullptr;
  float* cudaDst_ = nullptr;
  size_t cudaBytes_ = 0;
  std::mutex cudaMutex_;
#endif
};
