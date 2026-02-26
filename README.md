# ME_OpenDRT OFX

ME_OpenDRT OFX is a DaVinci Resolve OFX port of **OpenDRT v1.1.0**.

Current plugin in this repo:
- **ME_OpenDRT OFX v1.0** (based on OpenDRT 1.1.0)


## Upstream Project and License
This project is based on OpenDRT and is distributed under **GNU GPL v3**.

- Upstream OpenDRT project: https://github.com/jedypod/open-display-transform

## Versioning
Two version identifiers are used:
- **OpenDRT version**: the upstream algorithm version (currently `1.1.0`)
- **OFX version**: the port/release version for this plugin (currently `v1.0`)

## Platform Support
Current build target:
- **Windows (x64)**

Automated build available:
- **macOS Apple Silicon (arm64)** via GitHub Actions
- **macOS Intel (x86_64)** via GitHub Actions

## Installation
### Installer (recommended)
Use the generated installer from this project.
https://github.com/MoazElgabry/ME_OFX/blob/main/ME_OpenDRT/ME_OpenDRT_OFX_v1.0_Installer.exe
### Manual / Portable
Copy this bundle:
- `ME_OpenDRT.ofx.bundle`

To OFX plugins path:
- `C:\Program Files\Common Files\OFX\Plugins\`

Final expected path:
- `C:\Program Files\Common Files\OFX\Plugins\ME_OpenDRT.ofx.bundle\Contents\Win64\ME_OpenDRT.ofx`

After install, restart DaVinci Resolve.

## Uninstall
Remove:
- `C:\Program Files\Common Files\OFX\Plugins\ME_OpenDRT.ofx.bundle`

Then restart Resolve.

## GitHub Actions macOS Build
This repo includes a workflow at:
- `.github/workflows/build-me_opendrt-macos.yml`

Source used by CI is in:
- `ME_OpenDRT/source/`

To build macOS artifact:
1. Push this repository to GitHub.
2. Go to `Actions` -> `Build ME_OpenDRT macOS (arm64 + x86_64)`.
3. Click `Run workflow`.
4. Download the artifact matching your Mac architecture.

The artifact includes:
- `ME_OpenDRT_macOS_arm64_portable.zip`
- `ME_OpenDRT_macOS_x86_64_portable.zip`
- built bundle `ME_OpenDRT.ofx.bundle` (with `.metallib` in `Contents/Resources`)

Architecture mapping:
- Apple Silicon (M1/M2/M3/M4): `ME_OpenDRT-macos-arm64`
- Intel Mac: `ME_OpenDRT-macos-x86_64`

macOS install path:
- `/Library/OFX/Plugins/ME_OpenDRT.ofx.bundle`

## macOS Gatekeeper (Unsigned Plugin)
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
