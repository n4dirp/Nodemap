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


## Shortcuts

### Keyboard

| Shortcut | Action |
| --- | --- |
| `Ctrl+M` | Toggle minimap on/off |

### Keyboard *(when the cursor is over the minimap)*

| Shortcut     | Action               |
| ------------ | -------------------- |
| `Home`       | Frame all nodes      |
| `Shift+Home` | Frame current view   |
| `Numpad .`   | Frame selected nodes |

### Mouse *(inside the minimap)*

| Input                | Action                                         |
| -------------------- | ---------------------------------------------- |
| `Left Click`         | Pan / Select / Pan+Select (configurable)       |
| `Left Drag`          | Pan the Node Editor view                       |
| `Right Click`        | Pan / Select / Pan+Select (configurable)       |
| `Right Drag`         | Pan the Node Editor view                       |
| `Middle Drag`        | Pan the minimap                                |
| `Scroll`             | Zoom the minimap or Node Editor (configurable) |
| `Alt + Scroll`       | Temporarily swap the scroll zoom target        |
| `Ctrl + Scroll`      | Pan the editor horizontally                    |
| `Shift + Scroll`     | Pan the editor vertically                      |
| `Drag edge / corner` | Resize the minimap                             |


## Requirements

- Blender 5.1.0+

## Build from Source

```bash
git clone https://github.com/n4dirp/Nodemap.git
cd "Nodemap/nodemap"
blender --command extension build
```
