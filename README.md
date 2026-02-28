# ME_OpenDRT OFX

ME_OpenDRT OFX is a DaVinci Resolve OFX port of **OpenDRT v1.1.0**.
This OFX aims to enable the full potential of OpenDRT by giving colorists the flexibility to take control of how the DRT interacts with their look-development process
and design the DRT aspects according to their projects and stories. I've built a better-organized UI to navigate the vast options OpenDrt offers, as well as a preset system
that makes it easy to store and share configurations to give the colorist both flexibility and speed. 

Current plugin in this repo:
- **ME_OpenDRT OFX v1.1-rc1** (A direct port from OpenDRT v1.1.0)

## Upstream Project and License
This project is based on OpenDRT and is distributed under **GNU GPL v3**.
- Upstream OpenDRT project: https://github.com/jedypod/open-display-transform

## Current Platform Status
- Windows (x86_64): CUDA + OpenCL + CPU, including host-CUDA path.
- macOS (arm64 + x86_64): Metal + CPU, including host-Metal path.
- Linux (x86_64): CUDA/OpenCL/CPU fallback chain (build depends on toolkit availability and workflow variant).

Automated build available:
- **macOS Apple Silicon (arm64)** via GitHub Actions
- **macOS Intel (x86_64)** via GitHub Actions

## Installation
### Installer (recommended)
Use the generated installer from this project.
https://github.com/MoazElgabry/ME_OFX/blob/main/ME_OpenDRT/ME_OpenDRT_OFX_v1.0_Installer.exe

## Installation (End Users)
1. Download the latest portable build for your platform from the latest release or [here](https://github.com/MoazElgabry/ME_OFX/tree/main/ME_OpenDRT).
2. Copy `ME_OpenDRT.ofx.bundle` to your OFX plugin directory:
   - Windows: `C:\Program Files\Common Files\OFX\Plugins\`
   - macOS: `/Library/OFX/Plugins/`
   - Linux: `/usr/OFX/Plugins/`
3. Restart resolve.


## GitHub Actions 
This repo includes compiling workflows accessible through GitHub Actions
There is a dedicated workflow for each platform and a combined one to build the OFX for all platforms supported.
To build an artifact:
1. Go to `Actions` -> `Build ME_OpenDRT macOS (arm64 + x86_64)`. (choose the correct one for your plaatform)
2. Click `Run workflow`.
3. Download the artifact.

Source code can be found in:
- `ME_OpenDRT/source/`


Architecture mapping:
- Apple Silicon (M1/M2/M3/M4): `ME_OpenDRT-macos-arm64`
- Intel Mac: `ME_OpenDRT-macos-x86_64`


## macOS Gatekeeper (How to bypass security blocking and get the OFX to load)
`ME_OpenDRT.ofx.bundle` is currently unsigned/not notarized, so macOS Gatekeeper may block it by default.

Use one of these methods:

### Method 1 (recommended): Terminal fix
Run these commands in order:

```bash
sudo chmod -R 755 /Library/OFX/Plugins/ME_OpenDRT.ofx.bundle
sudo chown -R root:wheel /Library/OFX/Plugins/ME_OpenDRT.ofx.bundle
sudo xattr -dr com.apple.quarantine /Library/OFX/Plugins/ME_OpenDRT.ofx.bundle
sudo codesign --force --deep --sign - /Library/OFX/Plugins/ME_OpenDRT.ofx.bundle
```

Then relaunch DaVinci Resolve.

### Method 2: macOS UI flow (no Terminal)
1. Install/copy the plugin fresh to `/Library/OFX/Plugins/ME_OpenDRT.ofx.bundle`.
2. Launch Resolve. When macOS shows the verification warning, click `Done`.
3. Open `System Settings` -> `Privacy & Security`, scroll down, and click `Allow Anyway` for ME_OpenDRT.
4. In Resolve, go to `Preferences` -> `Video Plugins`, find `ME_OpenDRT`, enable/check it, save, and quit Resolve.
5. Relaunch Resolve, then click `Open Anyway` when prompted and authenticate with your Mac password.

Note:
- If the Mac account has no password, macOS may not show the required `Open Anyway` flow correctly.
