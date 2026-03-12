# ME_OpenDRT Cube Viewer Stub

This is a transport stub for phase-1 research.

It does not render 3D yet. It only listens on the same IPC endpoint the OFX plugin uses and prints incoming JSON messages.

## Build (optional)
Enable from top-level CMake with:
- `-DME_OPENDRT_BUILD_CUBE_VIEWER_STUB=ON`

Output binary name:
- `ME_OpenDRT_CubeViewer`

## Runtime
Set optional env overrides:
- `ME_OPENDRT_CUBE_VIEWER_PIPE`

Then run the stub and use the OFX plugin buttons:
- `Open 3D Cube Viewer`
- `Close 3D Cube Viewer`

This validates:
- non-blocking launch behavior
- message schema flow (`hello`, `open_session`, `params_snapshot`, `params_delta`, `close_session`)
