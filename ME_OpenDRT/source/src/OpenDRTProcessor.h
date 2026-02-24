#pragma once

#include <cstddef>
#include <cstring>

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

  bool render(const float* src, float* dst, int width, int height, bool preferCuda, bool hostSupportsOpenCL) {
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
    const size_t bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u * sizeof(float);

    if (cudaSetDevice(0) != cudaSuccess) {
      return false;
    }

    float* dSrc = nullptr;
    float* dDst = nullptr;

    if (cudaMalloc(&dSrc, bytes) != cudaSuccess) return false;
    if (cudaMalloc(&dDst, bytes) != cudaSuccess) {
      cudaFree(dSrc);
      return false;
    }

    if (cudaMemcpy(dSrc, src, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(dSrc);
      cudaFree(dDst);
      return false;
    }

    launchOpenDRTKernel(dSrc, dDst, width, height, &params_, nullptr);

    const cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
      cudaFree(dSrc);
      cudaFree(dDst);
      return false;
    }

    const cudaError_t syncStatus = cudaDeviceSynchronize();
    if (syncStatus != cudaSuccess) {
      cudaFree(dSrc);
      cudaFree(dDst);
      return false;
    }

    if (cudaMemcpy(dst, dDst, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
      cudaFree(dSrc);
      cudaFree(dDst);
      return false;
    }

    cudaFree(dSrc);
    cudaFree(dDst);
    return true;
  }
#endif

#if defined(__APPLE__)
  bool renderMetal(const float* src, float* dst, int width, int height) {
    return OpenDRTMetal::render(src, dst, width, height, params_);
  }
#endif

  bool renderOpenCL(const float* src, float* dst, int width, int height) {
    (void)src;
    (void)dst;
    (void)width;
    (void)height;
    return false;
  }

  bool renderCPU(const float* src, float* dst, int width, int height) {
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
#endif

  OpenDRTParams params_;
};
