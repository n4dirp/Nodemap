# Nodemap

A minimap overlay for the Blender Node Editor that provides an interactive overview of your entire node tree for faster navigation.


## Features

* **Interactive navigation** -- Click or drag inside the minimap to navigate the Node Editor
* **Node overview** -- Displays nodes, connections, labels, and socket indicators
* **Viewport tracking** -- Shows the current editor view and can automatically follow it
* **Frame controls** -- Quickly frame all nodes, the current view, or selected nodes
* **Scroll zoom** -- Zoom the minimap or the Node Editor with the scroll wheel (configurable)
* **Resizable** -- Resize the minimap by dragging its edges or corners
* **Customizable** -- Configure colors, opacity, labels, shortcuts, and interaction behavior
* **Theme-aware** -- Matches your Blender theme while supporting custom colors


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
| Minimap                    | `Left Click`         | Pan / Select / Pan+Select*   |
| Minimap                    | `Left Drag`          | Pan the Node Editor          |
| Minimap                    | `Right Click`        | Pan / Select / Pan+Select*   |
| Minimap                    | `Right Drag`         | Pan the Node Editor          |
| Minimap                    | `Middle Drag`        | Pan the minimap              |
| Minimap                    | `Scroll`             | Zoom minimap / Node Editor*  |
| Minimap                    | `Alt + Scroll`       | Temporarily swap zoom target |
| Minimap                    | `Ctrl + Scroll`      | Pan horizontally             |
| Minimap                    | `Shift + Scroll`     | Pan vertically               |
| Minimap                    | `Drag edge / corner` | Resize minimap               |
| Minimap · Type list        | `Left Click`         | Select all nodes of a type   |
| Minimap · Type list        | `Right Click`        | Select nodes and frame them  |



## Requirements

- Blender 5.1.0+

## Build from Source

```bash
git clone https://github.com/n4dirp/Nodemap.git
cd "Nodemap/nodemap"
blender --command extension build
```
