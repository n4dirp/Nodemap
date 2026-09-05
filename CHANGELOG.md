# Changelog

## [2.0.0] [Unreleased]

### Added
- Interactive node-type list (Shortcut: `T`)
- Search bar to filter nodes by name (Shortcut: `Ctrl+F`)
- Support for presets
- **Frame Selected Nodes** button in the minimap
- Improved minimap wire visualization, including selection highlighting, curved links, and dashed lines for field/modifier sockets
- More theme customization options for colors and appearance
- More flexible minimap positioning, with additional dock positions, a new Floating mode, and drag-and-snap support for editor borders and corners.
- Group markers displayed beneath group nodes
- Smooth animations for frame actions

### Changed
- Frame Selected now automatically adjusts the zoom to fit multiple nodes or a frame
- Improved performance when working with large node graphs
- Added per-tree minimap views, restoring the previous pan and zoom when returning to a node tree
- All extension options are now available in the addon preferences

### Fixed
- Selecting a node from the type list no longer triggers a full EEVEE material rebuild
- Minimap redraws now affect only the Node Editor being interacted with instead of refreshing all open Node Editors
- Scrollbars now appear only when nodes extend beyond the visible area

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
