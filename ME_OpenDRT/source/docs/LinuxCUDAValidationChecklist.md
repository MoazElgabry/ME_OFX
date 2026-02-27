# Linux CUDA Manual Validation Checklist

## Environment
- Linux distro + version:
- Resolve/Fusion host + version:
- NVIDIA driver version:
- CUDA toolkit/runtime version:
- GPU model:

## Build Artifact
- Confirm bundle contains `Contents/Linux-x86-64/ME_OpenDRT.ofx`.
- Confirm artifact flavor under test:
  - `opencl-cpu`
  - `cuda-opencl-cpu`

## Backend Routing Checks
- Default run (`ME_OPENDRT_RENDER_MODE` unset): expect CUDA first when available.
- Force internal CUDA: `ME_OPENDRT_RENDER_MODE=INTERNAL`.
- Force host CUDA: `ME_OPENDRT_RENDER_MODE=HOST` on hosts exposing CUDA render resources.
- Force OpenCL for fallback test: `ME_OPENDRT_FORCE_OPENCL=1`.
- Disable OpenCL for CPU fallback test: `ME_OPENDRT_DISABLE_OPENCL=1`.

## Host-CUDA Scenarios
- Host CUDA available (`args.isEnabledCudaRender=1`, valid stream/device pointers):
  - verify host-CUDA path is used
  - verify no crash under timeline playback/scrubbing
- Host CUDA unavailable:
  - verify internal CUDA path is used when CUDA runtime is available

## Failure/Fallback Safety
- Inject CUDA failure (invalid runtime/toolkit mismatch or forced failure build) and verify:
  - OpenCL fallback is attempted
  - CPU fallback is reached if OpenCL unavailable/disabled
  - host process does not crash

## Performance Capture
- Enable logs: `ME_OPENDRT_PERF_LOG=1` and `ME_OPENDRT_DEBUG_LOG=1`.
- Host CUDA profiling pass: `ME_OPENDRT_HOST_CUDA_FORCE_SYNC=1`.
- Capture playback FPS and per-stage logs for:
  - CUDA host
  - CUDA internal
  - OpenCL baseline

## Image Parity Matrix
- Compare CUDA vs OpenCL vs CPU on:
  - Standard
  - Umbra
  - Dystopic
  - Base
- Stress controls:
  - high HDR values
  - clamp on/off
  - display gamut variants
  - EOTF variants
- Record max/mean absolute differences and visible deltas.

## Stability
- Repeated apply/remove in node graph (20+ cycles).
- Long playback (5+ minutes) with parameter scrubbing.
- Multi-instance timeline/node usage.

## Release Gate
- Pass only if:
  - no host crashes
  - fallback path works deterministically
  - parity is within agreed tolerance
  - CUDA paths outperform Linux OpenCL baseline in target scenes