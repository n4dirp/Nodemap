# Nodemap

A minimap for Blender’s Node Editor, giving you a clear overview and faster way to navigate your node trees.

- See your entire node tree at a glance
- Jump to any part of the graph instantly
- Find and select nodes by type
- Frame nodes or selections in one click
- Customize the minimap to fit your workflow


## Location

**How do I toggle the minimap?** Click the minimap icon at the right end of the Node Editor header, or press `Ctrl+M`.

**Where are the settings?** Basic options are in the Nodemap Options popover next to the header icon; advanced options are in `Edit > Preferences > Extensions > Nodemap`.


## Shortcuts

| Input                      | Shortcut             | Action                       |
| -------------------------- | -------------------- | ---------------------------- |
| Keyboard                   | `Ctrl+M`             | Toggle minimap               |
| Keyboard · Minimap focused | `Home`               | Frame all nodes              |
| Keyboard · Minimap focused | `Shift+Home`         | Frame current view           |
| Keyboard · Minimap focused | `Numpad .`           | Frame selected nodes         |
| Minimap                    | `Left Click`         | Pan the view                 |
| Minimap                    | `Left Drag`          | Pan the Node Editor          |
| Minimap                    | `Right Click`        | Select nodes and frame them  |
| Minimap                    | `Middle Drag`        | Pan the minimap              |
| Minimap                    | `Scroll`             | Zoom minimap / Node Editor   |
| Minimap                    | `Alt + Scroll`       | Temporarily swap zoom target |
| Minimap                    | `Ctrl + Scroll`      | Pan horizontally             |
| Minimap                    | `Shift + Scroll`     | Pan vertically               |
| Minimap                    | `Drag edge / corner` | Resize minimap               |
| Minimap · Type list        | `Left Click`         | Select all nodes of a type   |
| Minimap · Type list        | `Right Click`        | Select nodes and frame them  |


## Installation

Requires **Blender 5.2+**.

To build from source:

```bash
git clone https://github.com/n4dirp/Nodemap.git
cd "Nodemap/nodemap"
blender --command extension build
```

Drag the generated `.zip` file into Blender to install it.
