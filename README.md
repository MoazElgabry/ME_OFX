# ME_OFX

ME_OFX is a DaVinci Resolve OFX port of **OpenDRT v1.1.0**.

Current plugin in this repo:
- **ME_OpenDRT OFX v1.0** (based on OpenDRT 1.1.0)

## Credits
- Original OpenDRT: **Jed Smith**
- OFX port: **Moaz ELgabry**

Website: https://moazelgabry.com  
GitHub: https://github.com/moazelgabry

## Upstream Project and License
This project is based on OpenDRT and is distributed under **GNU GPL v3**.

- Upstream OpenDRT project: https://github.com/jedypod/open-display-transform
- License in this repo: [LICENSE](./LICENSE)

If you distribute binaries, make sure corresponding source is also made available under GPLv3 terms.

## Versioning
Two version identifiers are used:
- **OpenDRT version**: the upstream algorithm version (currently `1.1.0`)
- **OFX version**: the port/release version for this plugin (currently `v1.0`)

## Platform Support
Current build target:
- **Windows (x64)**

Automated build available:
- **macOS Apple Silicon (arm64)** via GitHub Actions

## Installation
### Installer (recommended)
Use the generated installer from this project.

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
2. Go to `Actions` -> `Build ME_OpenDRT macOS (arm64)`.
3. Click `Run workflow`.
4. Download artifact `ME_OpenDRT-macos-arm64`.

The artifact includes:
- `ME_OpenDRT_macOS_arm64_portable.zip`
- built bundle `ME_OpenDRT.ofx.bundle` (with `.metallib` in `Contents/Resources`)
