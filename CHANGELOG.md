# Changelog

## [2.0.0] [Unreleased]

### Added
- Added an interactive type list (T) and a search bar to filter nodes (Ctrl+F)
- Added presets that save and restore all nodemap options
- Added new dock positions, a Floating mode, and drag-and-snap to editor borders
- Added configurable left and right mouse drag actions, including Frame Region in the minimap or editor
- Added a right-click menu on the minimap buttons to toggle their options
- Added a Frame Selected Nodes button to the minimap
- Improved wire visualization with selection highlighting, curved links, and dashes for field/modifier sockets
- Added single-color, linear-gradient, and vignette backgrounds for the minimap
- Added smooth animations for frame actions
- Added Auto Zoom option to keep the minimap zoom stable when the node layout changes
- Added group markers beneath group nodes

### Changed
- Improved performance on large graphs and when several minimaps share the same tree
- Added per-tree minimap views that restore the previous pan and zoom
- Moved all extension options to the addon preferences
- Type list selections now apply on click instead of release
- Frame Selected now adjusts the zoom to fit multiple nodes or a frame
- Frame menu buttons now animate like the minimap buttons when animations are on
- Improved minimap node labels to match the editor titles

### Fixed
- Fixed node selections triggering a full EEVEE material rebuild
- Fixed minimap redraws refreshing all open Node Editors instead of only the interacted one
- Fixed scrollbars showing even when nodes fit within the visible area

## [1.5.0] - 2026-08-14

### Added
- Active View Fill theme option to highlight the active view rect with a customizable color

## [1.4.1] - 2026-08-14

### Added
- Node Borders toggle to show or hide node selection and active borders

### Fixed
- Node borders no longer hidden on small nodes, so selection and active state stay visible regardless of node size
- Node label initials now show every word's initial instead of limiting to the first two

## [1.4.0] - 2026-07-28

### Added
- Toggle Nodemap shortcut (Ctrl+M)
- Frame View shortcut (Shift+Home)

### Changed
- Moved advanced options from popup panel to the addon preferences
- Nodemap overlay is now hidden by default in new editors
- Pan animation is now disabled when the Reduce Motion option is enabled

### Fixed
- Node label initials now display only alphanumeric characters
- Node sizes don't update when expanding node properties
- Fixed the Nodemap toggle button status when the editor is empty
- Fixed map interaction cancellation when the editor overlays are hidden

## [1.3.0] - 2026-07-11

### Added
- Smooth pan animation when paning the view
- Added Frame View button to the minimap

### Fixed
- Node collapse, expand, and resize not updating in the minimap
- Resize handles when it hit the max region width
- Fixed panel margins under different editor layouts

## [1.2.0] - 2026-07-07

### Added
- Frame View operator
- Viewport overlay with customizable color and toggle
- Update Delay setting to control minimap refresh responsiveness

### Fixed
- Performance: Implemented batch GPU shaders and tree fingerprint caching
- Interactive minimap failing to start in existing node editors
- Frame label sizing more uniform over zoom level

## [1.1.1] - 2026-07-04

### Fixed
- Fix node editor display being clipped after using the minimap on systems using OpenGL

## [1.1.0] - 2026-07-04

### Added
- Add operator to frame all nodes
- Add node socket indicators and improve wire positioning
- Add `Follow View` option for automatic panning
- Add custom background color

### Fixed
- Fix active view mapping with Blender UI scale
