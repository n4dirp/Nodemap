# Changelog

## [1.3.3] - 2026-07-21

### Changed
- Nodemap overlay is now hidden by default

### Fixed
- Fixed node label initials to only show characters that are alphanumeric

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
