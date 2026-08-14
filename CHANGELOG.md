# Changelog

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
