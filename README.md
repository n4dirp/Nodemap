# Nodemap

A minimap for Blender’s Node Editor, giving you a clear overview and faster way to navigate your node trees.


## Features
- See your entire node tree at a glance
- Jump to any part of the graph instantly
- Find and select nodes by type
- Frame nodes or selections in one click
- Customize the minimap to fit your workflow


## Installation

```bash
git clone https://github.com/n4dirp/Nodemap.git
cd "Nodemap/nodemap"
blender --command extension build
```

Drag the generated `.zip` file into Blender to install it.


## Location

**How do I toggle the minimap?** 
- Click the minimap icon at the right end of the Node Editor header, or press `Ctrl+M`

**Where are the settings?** 
- Basic options are in the Nodemap Options popover next to the header icon
- Advanced options are in `Edit > Preferences > Extensions > Nodemap`


## Shortcuts

| Input                      | Shortcut             | Action                       |
| -------------------------- | -------------------- | ---------------------------- |
| Node Editor                | `Ctrl+M`             | Toggle minimap               |
| Minimap                    | `Home`               | Frame all nodes              |
| Minimap                    | `Shift+Home`         | Frame current view           |
| Minimap                    | `Numpad .`           | Frame selected nodes         |
| Minimap                    | `T`                  | Toggle node-type list        |
| Minimap                    | `Ctrl+F`             | Reveal list and focus filter |
| Minimap                    | `Left Click`         | Pan the view                 |
| Minimap                    | `Right Click`        | Select nodes and frame them  |
| Minimap                    | `Middle Drag`        | Pan the minimap              |
| Minimap                    | `Middle Scroll`      | Zoom minimap / Node Editor   |
| Minimap                    | `Alt + Scroll`       | Temporarily swap zoom target |
