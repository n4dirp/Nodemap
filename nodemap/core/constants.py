"""Shared numeric and string constants for the nodemap extension."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Minimap geometry
# ---------------------------------------------------------------------------
MIN_MAP_WIDTH: int = 160
MIN_MAP_HEIGHT: int = 100

# ---------------------------------------------------------------------------
# Interaction handles
# ---------------------------------------------------------------------------
# Grab width of the resize hit areas. Visual only for hit-testing; the gap
# between the minimap frame and its content is CONTENT_PADDING below.
HANDLE_THICKNESS: int = 6

# Half-width of the grab band for resizing the type-list height when the
# list sits on top; the band centers on the strip's bottom edge.
LIST_TOP_RESIZE_HALF_WIDTH: int = 5

# ---------------------------------------------------------------------------
# Content padding
# ---------------------------------------------------------------------------
# Single knob for the gap between the minimap frame and its content (node
# area, type-list zone, edge pills). Scrollbars keep their own
# SCROLLBAR_INSET/SCROLLBAR_THICKNESS metrics. Spacing between the elements
# themselves uses ELEMENT_GAP below.
CONTENT_PADDING: int = 6

# ---------------------------------------------------------------------------
# Element gap
# ---------------------------------------------------------------------------
# Single knob for the space between UI elements (search bar to type list,
# type list to button row). Offsets from the exterior border stay on
# CONTENT_PADDING above.
ELEMENT_GAP: int = 4

# Single knob for the gap between the minimap and the editor region edges.
# Breadcrumb and asset-shelf offsets derive from this plus their fixed heights.
MINIMAP_MARGIN: int = 12

# Heights of the editor rows the minimap must clear: the breadcrumb context
# path and the compositor asset shelf.
CONTEXT_PATH_HEIGHT: int = 29
ASSET_SHELF_HEIGHT: int = 24

# ---------------------------------------------------------------------------
# Dock / snap-to-border
# ---------------------------------------------------------------------------
MAP_SNAP_TOLERANCE: float = 8.0
DOCK_DWELL_MS: float = 180.0

# Corners snap from farther away than plain edges so they win whenever the map
# is near one. The centered edge docks are narrow: the map center must sit
# within this percentage of the safe axis span from the border's midpoint.
MAP_CORNER_SNAP_RADIUS: float = 20.0
MAP_SNAP_CENTER_ZONE_PCT: float = 10.0

# Position categories for the free-drag / docked minimap.
CORNER_POSITIONS: frozenset[str] = frozenset({"TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"})
BORDER_POSITIONS: frozenset[str] = frozenset({"TOP_BORDER", "BOTTOM_BORDER", "LEFT_BORDER", "RIGHT_BORDER"})

# ---------------------------------------------------------------------------
# Framing / zoom
# ---------------------------------------------------------------------------
# Editor marquee/selection framing treats MAX_FRAME_ZOOM as an absolute
# pixels-per-unit cap; minimap zoom uses MAX_MAP_SCALE/MIN_MAP_SCALE below.
MAX_FRAME_ZOOM: float = 20.0
EDITOR_FIT_MARGIN: float = 0.15

# Absolute minimap zoom limits in pixels per tree unit, anchored to a
# default-width (140-unit) node so the usable range does not depend on tree
# size: at the max a node measures 280px, at the min about 3px.
MAX_MAP_SCALE: float = 2.0
MIN_MAP_SCALE: float = 0.02

# Wheel zoom steps the view by this factor per notch.
WHEEL_ZOOM_IN: float = 1.15
WHEEL_ZOOM_OUT: float = 0.85

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------
LABEL_MARGIN_PX: float = 12.0

# ---------------------------------------------------------------------------
# Text outline
# ---------------------------------------------------------------------------
# blf.shadow level 6 is FontShadowType::Outline, a 1px dilation of the glyph alpha.
BLF_SHADOW_OUTLINE: int = 6
TEXT_OUTLINE_ALPHA: float = 0.8
TEXT_OUTLINE_LUMINANCE_THRESHOLD: float = 0.4

# Rec. 709 luminance weights used to pick a contrasting outline color.
TEXT_LUMINANCE_R: float = 0.2126
TEXT_LUMINANCE_G: float = 0.7152
TEXT_LUMINANCE_B: float = 0.0722

# ---------------------------------------------------------------------------
# Type-list zone
# ---------------------------------------------------------------------------
TYPE_LIST_FONT_ID: int = 0
TYPE_LIST_FONT_SIZE: int = 11
TYPE_LIST_MIN_WIDTH: float = 72.0
LIST_PAD_X: float = 6.0
LIST_SWATCH: float = 10.0
LIST_SWATCH_GAP: float = 5.0
LIST_COUNT_GAP: float = 4.0
SCROLLBAR_HIT_PAD: float = 6.0
TYPE_LIST_ANIM_DURATION: float = 0.08
EMPTY_FINGERPRINT: tuple = (0, 0.0, "", 0, 0, 0, 0.0, 0.0, 0, 0)

# Scrollbar appearance
SCROLLBAR_THICKNESS: float = 3.0
SCROLLBAR_THICKNESS_HOVER: float = 6.0
SCROLLBAR_INSET: float = 2.0
SCROLLBAR_MIN_THUMB: float = 9.0
SCROLLBAR_ALPHA: float = 0.2
TYPE_LIST_ANIM_AWAIT_TIMEOUT: float = 1.0
TYPE_LIST_MIN_LABEL_W: float = 32.0
TYPE_LIST_ROW_HEIGHT_OFFSET: float = 8.0

# The type list moves from the left edge to the top edge once the minimap
# height exceeds width multiplied by this ratio (auto position only).
TYPE_LIST_ASPECT_THRESHOLD: float = 1.0

# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------
# Smooth-drag, inertia, and frame/editor animations tick at this rate; the
# pan-speed preference then spreads them over a whole number of frames.
PAN_ANIM_FPS: float = 100.0
PAN_ANIM_INTERVAL: float = 1.0 / PAN_ANIM_FPS

# Fixed animation duration in frames for click-to-pan (fast snap).
PAN_FRAMES: float = 12.0

# Minimum animation duration in frames so close pans stay visible instead of
# collapsing to a single tick.
PAN_MIN_FRAMES: float = PAN_FRAMES * 0.75

# Inertia kicks in when a released drag ends above this velocity A view that has come to a stop
# under SMOOTH_DAMP_STILL sits still instead of wobbling.
INERTIA_MIN_SPEED: float = 2.0
SMOOTH_DAMP_STILL: float = 0.15

# Middle-mouse minimap pan uses a shorter glide than view pans: the release
# velocity is scaled down, needs a higher speed to trigger, and decays
# faster per tick.
INERTIA_PAN_VELOCITY_SCALE: float = 0.6
INERTIA_PAN_MIN_SPEED: float = 3.0
INERTIA_PAN_DECAY: float = 0.85

# ---------------------------------------------------------------------------
# Minimap chrome (header buttons, font)
# ---------------------------------------------------------------------------
FONT_SIZE: int = 11
BUTTON_SIZE: int = 20
# Drawn glyph extent for the 32px button icon PNGs, in ui_scale units.
BUTTON_ICON_SIZE: int = 16
BUTTON_MARGIN: int = 0
BUTTON_HOVER_ALPHA: float = 0.05

# Minimap area kept free of the type list: left placement reserves horizontal
# space, top placement reserves vertical space.
TYPE_LIST_RESERVE_LEFT: int = 2 * BUTTON_SIZE + 4 * CONTENT_PADDING - 1
TYPE_LIST_RESERVE_TOP: int = 2 * BUTTON_SIZE + 3 * CONTENT_PADDING - 1

# ---------------------------------------------------------------------------
# Batch building
# ---------------------------------------------------------------------------
MIN_SOCKET_SCALE: float = 0.1
SCALE_REBUILD_REL: float = 0.05
BATCH_DRIFT_PX: float = 384.0
CULL_MARGIN_PX: float = BATCH_DRIFT_PX + 32.0
# While the user is actively zooming, defer scale-bucket rebuilds and keep
# scaling the existing batches via the content matrix; rebuild once the scale
# holds still for ZOOM_SETTLE_MS. The ratio window bounds how far a deferred
# bake may drift before an immediate rebuild keeps culling/typography fresh.
ZOOM_SETTLE_MS: float = 180.0
ZOOM_DEFER_RATIO_MIN: float = 0.25
ZOOM_DEFER_RATIO_MAX: float = 0.75
# Group node marker strip thickness in tree units: the baked thickness
# scales with the zoom like node geometry, floored to stay visible.
GROUP_MARKER_THICKNESS: float = 10.0
# Gap between the node bottom and the marker strip in tree units, scaling
# with the zoom so the strip sits below the node instead of touching it.
GROUP_MARKER_GAP: float = 4.0

# ---------------------------------------------------------------------------
# Socket indicator
# ---------------------------------------------------------------------------
SOCKET_PILL_SIZE_MULTIPLIER: float = 3.0

# ---------------------------------------------------------------------------
# Node rendering
# ---------------------------------------------------------------------------
NODE_ROUNDNESS_DEFAULT: float = 2.0
MUTE_ALPHA: float = 0.4

# ---------------------------------------------------------------------------
# Zone types
# ---------------------------------------------------------------------------
# Input -> output node bl_idname pairs that delimit a geometry-node zone body.
ZONE_INPUT_BL_IDNAMES: dict[str, str] = {
    "GeometryNodeSimulationInput": "GeometryNodeSimulationOutput",
    "GeometryNodeRepeatInput": "GeometryNodeRepeatOutput",
    "GeometryNodeForeachGeometryElementInput": "GeometryNodeForeachGeometryElementOutput",
    "NodeClosureInput": "NodeClosureOutput",
}

# Zone fill shape (in widget-unit multiples, matching Blender's node_draw.cc).
NODE_ZONE_PADDING: float = 1.0
ZONE_FILLET_RADIUS: float = 0.2
ZONE_FILLET_SEGMENTS: int = 5
ZONE_IO_TRIM: float = 0.75

# sRGB fallback values (Blender 5.3 default theme).
ZONE_THEME_DEFAULTS: dict[str, tuple[float, float, float, float]] = {
    "GeometryNodeSimulationInput": (0.400, 0.255, 0.384, 0.20),
    "GeometryNodeRepeatInput": (0.463, 0.318, 0.184, 0.20),
    "GeometryNodeForeachGeometryElementInput": (0.200, 0.322, 0.498, 0.20),
    "NodeClosureInput": (0.490, 0.490, 0.227, 0.20),
}
