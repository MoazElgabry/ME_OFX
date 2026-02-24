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
