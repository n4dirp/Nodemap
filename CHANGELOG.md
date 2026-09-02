# Changelog

## [2.0.0] [Unreleased]

### Added
- Interactive node-type list with per-type counts
- New Presets menu in the Nodemap popup
- Added group markers beneath group nodes
- Added Smooth animation for frame actions
- Added Frame Selected button in the minimap
- Smooth editor animation when framing selected nodes
- Curved wires in the minimap that match Blender's link curves
- More theme customization options for colors and appearance
- All extension options now available in the addon preferences panel
- Per-tree minimap view: pan/zoom restored on returning to a node-tree
- Minimap dock positions: corners, edges, and a new Floating mode
- Drag handle to reposition the minimap, with snap to editor borders and corners

### Changed
- Frame Selected now adjusts the zoom to fit multiple nodes or a frame
- Renamed the "Smooth Pan" preference to "Animations"; it now also gates the type-list show/hide animation
- Moved advanced options from the popup panel to the addon preferences for a cleaner interface
- Performance improvements when working with large node graphs

### Fixed
- Selecting a node from the type list no longer forces a full EEVEE material rebuild (now uses the native node.select operator like the minimap)
- Minimap redraws now target only the interacted Node Editor instead of refreshing all open Node Editors
- Scrollbars now appear only when nodes actually go out of view


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
