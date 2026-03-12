# ME_OpenDRT OFX

ME_OpenDRT is an OFX plugin port of OpenDRT v1.1.0 for modern color-managed finishing workflows.

This project exists to make OpenDRT easier to use directly inside OFX hosts, with a familiar preset-driven UI and fast GPU paths where available.

Current plugin UI version: `v1.2.0`

## Why This Plugin Exists
- Bring OpenDRT into OFX hosts without rebuilding node graphs from scratch.
- Keep a parity-first implementation so the same settings produce consistent results.
- Provide practical performance backends (CUDA, Metal, OpenCL, CPU fallback).
- Preserve a preset workflow suitable for real grading sessions.

## Who It Is For
- Colorists and finishing artists who want OpenDRT in an OFX workflow.
- Developers/TDs who need a maintainable, cross-platform OpenDRT implementation.

## What You Get
- OpenDRT controls exposed as OFX parameters.
- Preset-driven workflow (built-in + user presets).
- Backend fallback strategy for stability:
  - Windows/Linux CUDA-enabled builds: `CUDA -> OpenCL -> CPU`
  - macOS: `Metal -> CPU`

## Host and Platform Support
- Windows (x86_64): CUDA + OpenCL + CPU fallback
- macOS (arm64 + x86_64): Metal + CPU fallback
- Linux (x86_64): CUDA/OpenCL/CPU fallback chain (depends on build variant)

## Installation (End Users)
1. Download the latest portable build artifact for your platform from the project releases/workflows.
2. Copy `ME_OpenDRT.ofx.bundle` to your OFX plugin directory:
   - Windows: `C:\Program Files\Common Files\OFX\Plugins\`
   - macOS: `/Library/OFX/Plugins/`
   - Linux: `/usr/OFX/Plugins/`
3. Restart your host application.

Expected binary locations inside the bundle:
- Windows: `Contents/Win64/ME_OpenDRT.ofx`
- macOS: `Contents/MacOS/ME_OpenDRT.ofx`
- Linux: `Contents/Linux-x86-64/ME_OpenDRT.ofx`

## Quick Usage
1. Add `ME_OpenDRT` to your node/clip.
2. Start from a built-in look preset.
3. Adjust tonescale/purity/hue/brilliance sections as needed.
4. Save your look as a user preset for reuse.

## Debug and Performance Environment Variables
- `ME_OPENDRT_PERF_LOG=1`
- `ME_OPENDRT_DEBUG_LOG=1`
- `ME_OPENDRT_RENDER_MODE=HOST|AUTO|INTERNAL`
- `ME_OPENDRT_METAL_RENDER_MODE=HOST|AUTO|INTERNAL`
- `ME_OPENDRT_HOST_CUDA_FORCE_SYNC=1`
- `ME_OPENDRT_HOST_METAL_FORCE_WAIT=1`
- `ME_OPENDRT_FORCE_OPENCL=1`
- `ME_OPENDRT_DISABLE_OPENCL=1`

## Build From Source (Developers)

### Windows
```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

### Linux
```bash
cmake -S . -B build-linux -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux -j
```

### macOS
```bash
cmake -S . -B build-mac -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-mac --config Release
```

## Project Status
Active development, parity-first and stability-focused.

Current focus areas:
- Host GPU path robustness
- Cross-platform packaging and CI reliability
- Preset UX and interoperability improvements

## Contributing
Issues and PRs are welcome. Include:
- Host app/version
- OS and GPU details
- Repro steps
- Logs (`ME_OPENDRT_DEBUG_LOG=1` / `ME_OPENDRT_PERF_LOG=1`) when relevant

## Credits
- OpenDRT by Jed Smith: https://github.com/jedypod/open-display-transform
- OFX port and ongoing development: ME_OpenDRT contributors

## License
GNU GPL v3
