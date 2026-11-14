# ![Alt text](nodemap/icons/nodemap.png) Nodemap

**A minimap for Blender’s Node Editor.**
See your entire node tree at a glance and navigate it fast.

[![Blender 4.4+](https://img.shields.io/badge/Blender-4.4%2B-E87D0D?logo=blender&logoColor=white)](https://www.blender.org/)
[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fn4dirp%2FNodemap%2Fmain%2Fnodemap%2Fblender_manifest.toml&query=%24.version&label=version&color=blue)](CHANGELOG.md)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
[![Blender Extensions](https://img.shields.io/badge/Blender_Extensions-Nodemap-E87D0D?logo=blender&logoColor=white)](https://extensions.blender.org/add-ons/nodemap/)


## Features
- See your entire node tree at a glance
- Jump to any part of the graph instantly
- Find and select nodes by type
- Frame nodes or selections in one click
- Customize the minimap to fit your workflow

Works in all node editors: Shader, Geometry Nodes, Compositor and Texture nodes.

## Requirements

- Blender **4.4 or newer**

## Installation

### From Blender Extensions (Recommended)

Install from the [Blender Extensions](https://extensions.blender.org/add-ons/nodemap/) by drag the **Install** button into Blender, or search for **Nodemap** under `Edit > Preferences > Get Extensions`. This also enables automatic updates.

### Build from source (Latest Development Version)

Requires Blender on your PATH (or use the full path to the executable):

```bash
git clone https://github.com/n4dirp/Nodemap.git
cd "Nodemap/nodemap"
blender --command extension build
```

Drag the generated `nodemap-<version>.zip` into Blender.

## Usage

- **Toggle the minimap:** click the minimap icon at the right end of the Node Editor header, or press `Ctrl+M`. The minimap is hidden by default in new editors.
- **Quick options:** the *Nodemap Options* popover next to the header icon. Right-click a minimap button to toggle its option directly.
- **Advanced options:** `Edit > Preferences > Extensions > Nodemap`. Save and restore your setups as presets.

## Shortcuts
### Node Editor

| Shortcut | Action |
| -------- | ------ |
| `Ctrl+M` | Toggle minimap |

### Minimap — Keyboard

| Shortcut   | Action                              |
| ---------- | ----------------------------------- |
| `Home`     | Frame all nodes                     |
| `End`      | Frame current view                  |
| `Numpad .` | Frame selected nodes                |
| `T`        | Toggle node-type list               |
| `Shift+A`  | Expand / collapse list groups       |
| `Ctrl+F`   | Reveal list and focus the filter    |

### Minimap — Mouse

| Input          | Action                            |
| -------------- | --------------------------------- |
| Left click     | Pan the view to the clicked point |
| Right click    | Select nodes and frame them       |
| Left drag      | Center Pan*                       |
| Right drag     | Frame Region*                     |
| `Ctrl+Drag`    | Pan the view                      |
| `Shift+Drag`   | Frame a region (minimap)          |
| `Alt+Drag`     | Frame a region (editor)           |
| Middle drag    | Pan the minimap itself            |
| Scroll         | Zoom minimap / Node Editor        |
| `Alt+Scroll`   | Temporarily swap the zoom target  |

\* Left- and right-drag actions are defaults and can be reassigned in the preferences.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) or the [version history](https://extensions.blender.org/add-ons/nodemap/#versions) on Blender Extensions.

## Support & Contributing

- Found a bug or have a feature request? [Open an issue](https://github.com/n4dirp/Nodemap/issues) — please include your Blender version, OS, and steps to reproduce.
- Pull requests are welcome.
- Enjoying Nodemap? A rating on [Blender Extensions](https://extensions.blender.org/add-ons/nodemap/) helps a lot.

## License

[GPL-3.0-or-later](LICENSE)
