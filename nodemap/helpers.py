"""Shared helper utilities for node minimap."""

import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import blf
import bpy

logger = logging.getLogger(__package__)

LUMINANCE_R: float = 0.299
LUMINANCE_G: float = 0.587
LUMINANCE_B: float = 0.114
OUTLINE_ALPHA: float = 0.8
MAP_PADDING: float = 12.0
MIN_MAP_WIDTH: int = 120
MIN_MAP_HEIGHT: int = 80
MAX_FRAME_ZOOM: float = 20.0
_EDITOR_FIT_MARGIN: float = 0.10
_HANDLE_THICKNESS: int = 6


def redraw_ui(mode: str = "VIEW_3D", area_pointer: int | None = None) -> None:
    """Redraw all areas matching the given mode, or a specific area by pointer."""
    ctx = bpy.context
    if not ctx or not ctx.window_manager:
        return
    for window in ctx.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area_pointer is not None:
                try:
                    if area.as_pointer() != area_pointer:
                        continue
                except ReferenceError:
                    continue
            if mode == "ALL" or area.type == mode:
                area.tag_redraw()
                # logger.info(f"Redrawing area: {area.type}")


def _theme(path: str, default: tuple[float, ...]) -> tuple[float, ...]:
    """Resolve a dotted theme attribute path to a color tuple, falling back to default."""
    prefs = bpy.context.preferences
    if not prefs.themes:
        return default
    value: Any = prefs.themes[0]
    try:
        for part in path.split("."):
            value = getattr(value, part)
        if hasattr(value, "copy"):
            return tuple(value)
        try:
            return tuple(value)
        except TypeError:
            return default
    except AttributeError:
        return default


def _theme_float(path: str, default: float) -> float:
    """Resolve a dotted theme attribute path to a float value, falling back to default."""
    prefs = bpy.context.preferences
    if not prefs.themes:
        return default
    value = prefs.themes[0]
    try:
        for part in path.split("."):
            value = getattr(value, part)
        return float(value)
    except (AttributeError, TypeError, ValueError):
        return default


def _srgb_to_linear(c: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert an sRGB color tuple to linear color space."""

    def _conv(ch: float) -> float:
        return ch / 12.92 if ch <= 0.04045 else ((ch + 0.055) / 1.055) ** 2.4

    return (_conv(c[0]), _conv(c[1]), _conv(c[2]), c[3])


def _rgba(value: tuple[float, ...], alpha: float) -> tuple[float, float, float, float]:
    """Convert a multi-channel tuple to RGBA using the given alpha."""
    return (float(value[0]), float(value[1]), float(value[2]), float(alpha))


def _alpha_mul(color: tuple[float, ...], alpha: float) -> tuple[float, float, float, float]:
    """Return RGBA with the original alpha multiplied by alpha."""
    return (float(color[0]), float(color[1]), float(color[2]), float(color[3] * alpha))


def _get_ui_scale() -> float:
    """Return the Blender UI scale factor from preferences."""
    return float(bpy.context.preferences.system.ui_scale)


def _compute_outline_color(rgb: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Compute black or white outline based on luminance of the given color."""
    luminance = rgb[0] * LUMINANCE_R + rgb[1] * LUMINANCE_G + rgb[2] * LUMINANCE_B
    if luminance > 0.5:
        return (0.0, 0.0, 0.0, OUTLINE_ALPHA)
    return (1.0, 1.0, 1.0, OUTLINE_ALPHA)


def _color_contrast(color: tuple[float, ...], factor: float = 0.85) -> tuple[float, float, float, float]:
    """Darken a color by the given factor to produce a contrast variant."""
    return (float(color[0] * factor), float(color[1] * factor), float(color[2] * factor), 1.0)


_COLOR_TAG_TO_THEME_ATTR: dict[str, str] = {
    "INPUT": "input_node",
    "OUTPUT": "output_node",
    "FILTER": "filter_node",
    "VECTOR": "vector_node",
    "CONVERTER": "converter_node",
    "COLOR": "color_node",
    "GROUP": "group_node",
    "MATTE": "matte_node",
    "DISTORT": "distor_node",
    "PATTERN": "filter_node",
    "TEXTURE": "texture_node",
    "SHADER": "shader_node",
    "SCRIPT": "script_node",
    "GEOMETRY": "geometry_node",
    "ATTRIBUTE": "attribute_node",
    "FRAME": "frame_node",
}


def _get_node_color(node: bpy.types.Node, fallback_color: tuple[float, ...]) -> tuple[float, ...]:
    """Return the node's custom color, theme-mapped color_tag color, or fallback."""
    if getattr(node, "use_custom_color", False):
        return _rgba(node.color, fallback_color[3])
    color_tag = getattr(node, "color_tag", "NONE")
    if color_tag != "NONE":
        theme_attr = _COLOR_TAG_TO_THEME_ATTR.get(color_tag)
        if theme_attr:
            return _theme_rgba(f"node_editor.{theme_attr}", fallback_color)
    return fallback_color


def _get_safe_bounds(
    area: bpy.types.Area,
    region: bpy.types.Region,
) -> tuple[int, int, int, int]:
    """Compute drawable region bounds excluding toolbars, shelves, headers, and UI panels."""
    left = 0
    bottom = 0
    right = region.width
    top = region.height

    for r in area.regions:
        if r.type == "TOOLS":
            left = max(left, r.width)
        elif "ASSET_SHELF" in r.type:
            bottom = max(bottom, r.height)
        elif r.type == "UI":
            right = min(right, region.width - r.width)

    return int(left), int(bottom), int(right), int(top)


def _get_minimap_margins(space, corner: str, ui_scale: float) -> tuple[float, float, float]:
    """Return ``(x_margin, y_margin, margin_bottom)`` based on corner and visible UI elements.

    Adjusts margins when the breadcrumb context path or compositing asset shelf
    occupies space near the minimap's corner.
    ``margin_bottom`` is the additional margin on the edge opposite the header.
    """
    is_compositor = space.node_tree is not None and space.node_tree.type == "COMPOSITING"
    show_asset_shelf = getattr(space, "show_region_asset_shelf", False)
    show_context_path = getattr(space.overlay, "show_context_path", False)

    x_margin = MAP_PADDING * ui_scale
    y_margin = x_margin
    margin_bottom = x_margin

    adjusted = (MAP_PADDING + 25) * ui_scale

    match corner:
        case "TOP_RIGHT" | "TOP_LEFT":
            if show_context_path:
                y_margin = adjusted
            if is_compositor and show_asset_shelf:
                margin_bottom = adjusted

        case "BOTTOM_RIGHT" | "BOTTOM_LEFT":
            if is_compositor and show_asset_shelf:
                y_margin = adjusted
            if show_context_path:
                margin_bottom = adjusted

    return x_margin, y_margin, margin_bottom


@dataclass
class MinimapState:
    rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    tree_bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    margin: float = 10.0
    padding: float = 6.0
    scale: float = 1.0
    hovered_node: str | None = None
    zoom: float = 1.0
    base_zoom: float = 1.0
    pan: list[float] = field(default_factory=lambda: [0.0, 0.0])
    enabled: bool = True
    frame_all_btn: tuple[float, float, float, float] | None = None
    frame_view_btn: tuple[float, float, float, float] | None = None
    frame_selected_btn: tuple[float, float, float, float] | None = None
    list_toggle_btn: tuple[float, float, float, float] | None = None
    hovered_frame_btn: str | None = None
    width_clamped: bool = False
    height_clamped: bool = False
    hovered_handle: str | None = None
    resize_active: str | None = None
    pressed: bool = False
    list_width: float = 0.0
    list_scroll: float = 0.0
    list_scroll_max: float = 0.0
    list_row_h: float = 16.0
    hovered_type_label: str | None = None
    list_row_rects: list = field(default_factory=list)
    list_expanded: set = field(default_factory=set)
    list_node_rects: list = field(default_factory=list)
    list_toggle_rects: dict = field(default_factory=dict)
    hovered_list_node: tuple | None = None
    hovered_list_scrollbar: bool = False
    list_scroll_dragging: bool = False
    list_scrollbar_thumb: tuple[float, float, float, float] | None = None
    list_scrollbar_track: tuple[float, float, float, float] | None = None
    list_zone_rect: tuple[float, float, float, float] | None = None
    list_anim_active: bool = False
    list_anim_from: float = 0.0
    list_anim_target: float = -1.0
    list_anim_start: float = 0.0
    list_anim_duration: float = 0.33
    list_anim_timer: Any = None
    cached_fingerprint: Any = None
    pending_timer: Any = None
    pending_timer_deadline: float = 0.0
    pending_fingerprint: Any = None
    tree_data: dict | None = None
    cached_backdrops_batch: Any = None
    cached_borders_batch: Any = None
    cached_frames_fill_batch: Any = None
    cached_frames_border_batch: Any = None
    cached_text: list | None = None
    cached_wire_batches: list | None = None
    cached_wire_shadow_batch: Any = None
    cached_marker_batches: list | None = None
    cached_socket_batch: Any = None
    cached_socket_ph: float = 2.0
    cached_socket_shadow: list | None = None
    list_cache_key: Any = None
    cached_list_entries: list | None = None
    cached_list_layout: dict | None = None
    cached_list_children: dict = field(default_factory=dict)
    cached_list_swatches_batch: Any = None
    tree_data_version: int = 0
    pos_data_version: int = 0
    batch_cache_key: Any = None
    batch_scale: float = 1.0
    batch_anchor: tuple[float, float] = (0.0, 0.0)
    wire_cache_key: Any = None
    wire_scale: float = 1.0
    pending_settle_flush: bool = False
    last_move_refresh: float = 0.0
    _profiler: Any = field(default=None, repr=False)
    _profiling_active: bool = field(default=False, repr=False)
    _profiling_frame_count: int = field(default=0, repr=False)


_minimap_state: dict[int, MinimapState] = {}
_minimap_window_operators: dict[int, Any] = {}
_registration_state: dict[str, bool] = {"done": False}

# Interactive minimap buttons as (id, show-preference attr, MinimapState attr).
# Order defines the right-edge capsule stack; "LIST" renders standalone.
_MINIMAP_BUTTONS: tuple[tuple[str, str, str], ...] = (
    ("ALL", "show_frame_all_btn", "frame_all_btn"),
    ("VIEW", "show_frame_view_btn", "frame_view_btn"),
    ("SELECTED", "show_frame_selected_btn", "frame_selected_btn"),
    ("LIST", "show_list_toggle_btn", "list_toggle_btn"),
)


def _state(area_ptr: int | None = None) -> MinimapState:
    """Return the minimap state for the given area, initializing defaults if needed."""
    if area_ptr is None:
        try:
            area_ptr = bpy.context.area.as_pointer()
        except (AttributeError, ReferenceError):
            return MinimapState()
    if area_ptr not in _minimap_state:
        state = MinimapState()
        try:
            prefs = bpy.context.preferences.addons.get(__package__)
            if prefs:
                state.enabled = getattr(prefs.preferences.settings, "show_by_default", True)
        except (AttributeError, ReferenceError):
            pass
        _minimap_state[area_ptr] = state
    return _minimap_state[area_ptr]


def _ensure_area_states() -> None:
    """Pre-populate state for all existing NODE_EDITOR areas (called at registration)."""
    wm = bpy.context.window_manager
    if not wm:
        logger.debug("_ensure_area_states: no window_manager")
        return
    count = 0
    for window in wm.windows:
        if not window or not window.screen:
            continue
        for area in window.screen.areas:
            if area.type == "NODE_EDITOR":
                ptr = area.as_pointer()
                _state(ptr)
                count += 1
                win_name = window.screen.name if window.screen else "?"
                logger.debug("_ensure_area_states: created state for area %d (window %s)", ptr, win_name)
    logger.debug("_ensure_area_states: %d NODE_EDITOR areas processed", count)


def _get_node_dims(node: bpy.types.Node, ui_scale: float | None = None) -> tuple[float, float]:
    """Robust extraction of width and height ensuring positive float values."""
    if ui_scale is None:
        ui_scale = _get_ui_scale()
    if getattr(node, "hide", False):
        return 100.0, 30.0
    try:
        dims = node.dimensions
        w = abs(dims[0])
        if w == 0:
            w = abs(node.width)
    except (AttributeError, TypeError, IndexError):
        w = abs(node.width)

    try:
        dims = node.dimensions
        h = abs(dims[1])
        if h == 0:
            h = abs(getattr(node, "height", 30.0))
    except (AttributeError, TypeError, IndexError):
        h = abs(getattr(node, "height", 30.0))

    return max(w / ui_scale, 5.0), max(h / ui_scale, 5.0)


def _get_node_tree_bounds(nodes: bpy.types.Nodes) -> tuple[float, float, float, float]:
    """Compute the bounding box of all nodes in a node tree as (min_x, min_y, max_x, max_y)."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    ui = _get_ui_scale()
    for node in nodes:
        w, h = _get_node_dims(node, ui)
        x, y = node.location_absolute.x, node.location_absolute.y
        min_x = min(min_x, x)
        max_x = max(max_x, x + w)
        min_y = min(min_y, y - h)
        max_y = max(max_y, y)

    if min_x == float("inf"):
        return 0.0, 0.0, 200.0, 200.0
    return min_x, min_y, max_x, max_y


def _expand_bounds_margin(
    bounds: tuple[float, float, float, float], ui_scale: float, mh: float, padding: float
) -> tuple[float, float, float, float]:
    """Expand tree bounds by a small margin so frame labels stay inside the minimap."""
    LABEL_MARGIN_PX = 12 * ui_scale
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    inner_h = max(mh - 2 * padding, 1.0)
    margin = LABEL_MARGIN_PX * bbox_h / inner_h
    return (bounds[0] - margin - 50, bounds[1] - margin, bounds[2] + margin + 100, bounds[3] + margin)


def _get_area_and_region_under_mouse(context, event) -> tuple:
    """Find the area and WINDOW region under the mouse cursor using screen coordinates."""
    window = getattr(context, "window", None)
    if not window:
        return None, None
    mx, my = event.mouse_x, event.mouse_y
    for area in window.screen.areas:
        if area.x <= mx <= area.x + area.width and area.y <= my <= area.y + area.height:
            for region in area.regions:
                if (
                    region.type == "WINDOW"
                    and region.x <= mx <= region.x + region.width
                    and region.y <= my <= region.y + region.height
                ):
                    return area, region
    return None, None


_LIST_CONTENT_GAP = 4.0


def _get_map_content_rect(st: MinimapState) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, width, height)`` of the map content area.

    Subtracts the type-list zone plus a margin from the left edge so node
    framing and panning never place tree content behind the list.
    """
    mx, my, mw, mh = st.rect
    pad = st.padding
    left_inset = pad + st.list_width
    if st.list_width > 0:
        left_inset += _LIST_CONTENT_GAP * _get_ui_scale()
    return mx + left_inset, my + pad, max(mw - pad - left_inset, 1.0), max(mh - 2 * pad, 1.0)


STATS_FONT_ID = 1
STATS_FONT_SIZE = 8
_TYPE_LIST_MIN_WIDTH = 70.0
_TYPE_LIST_MAX_WIDTH_PCT = 0.35
_LIST_PAD_X = 6.0
_LIST_SWATCH = 5.0
_LIST_SWATCH_GAP = 5.0
_LIST_COUNT_GAP = 8.0
# Extra width around the type-list scrollbar track that still counts as a
# hover/press hit, matching the generous gutter of Blender overlay scrollbars.
_SCROLLBAR_HIT_PAD = 6.0


def _get_type_list_width(
    settings, st: MinimapState, mw: float, ui_scale: float, font_size: int = STATS_FONT_SIZE
) -> float:
    """Measure the type-list zone width for the current tree data (0 when disabled).

    Called before the map transform is computed so node framing can reserve
    the zone; ``st.list_width`` must be assigned from its result.
    """
    if not getattr(settings, "show_type_list", False):
        return 0.0
    tree_data = st.tree_data
    type_stats = tree_data.get("type_stats") if tree_data else None
    if not type_stats:
        return 0.0

    font_id = STATS_FONT_ID
    blf.size(font_id, int(font_size * ui_scale))
    pad_x = _LIST_PAD_X * ui_scale
    swatch = _LIST_SWATCH * ui_scale
    swatch_gap = _LIST_SWATCH_GAP * ui_scale
    count_gap = _LIST_COUNT_GAP * ui_scale
    widest_label = max(blf.dimensions(font_id, label)[0] for label in type_stats)
    widest_count = max(blf.dimensions(font_id, str(count))[0] for count in type_stats.values())
    content_w = pad_x * 2 + swatch + swatch_gap + widest_label + count_gap + widest_count
    return min(max(content_w, _TYPE_LIST_MIN_WIDTH * ui_scale), mw * _TYPE_LIST_MAX_WIDTH_PCT)


_LIST_ANIM_FRAMES: dict[str, int] = {"FAST": 10, "MEDIUM": 20, "SLOW": 30}


def start_list_width_animation(st: MinimapState, settings) -> None:
    """Begin animating the type-list zone width after a toggle-button click.

    An expansion defers the target measurement to the draw step because
    measurable type stats only exist once the pending tree compile lands.
    Skipped when Reduce Motion is enabled so the zone snaps instantly.
    """
    try:
        if bpy.context.preferences.view.use_reduce_motion:
            return
    except AttributeError:
        pass
    st.list_anim_active = True
    st.list_anim_from = st.list_width
    st.list_anim_target = 0.0 if not getattr(settings, "show_type_list", False) else -1.0
    frames = _LIST_ANIM_FRAMES.get(getattr(settings, "pan_speed", "MEDIUM"), 24)
    st.list_anim_duration = frames / 60.0
    st.list_anim_start = time.perf_counter()


def _list_anim_tick(st: MinimapState) -> None:
    st.list_anim_timer = None
    if st.list_anim_active:
        redraw_ui("NODE_EDITOR")


def _schedule_list_anim_redraw(st: MinimapState) -> None:
    """Schedule a one-shot timer tick that forces a redraw while the list animates."""
    if st.list_anim_timer is not None:
        return
    try:
        bpy.app.timers.register(lambda: _list_anim_tick(st), first_interval=1 / 60)
        st.list_anim_timer = True
    except (RuntimeError, ValueError):
        pass


def _compute_base_map_geom(
    st: MinimapState,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Return the scale-independent geometry shared by all map transforms.

    Computes ``(inner_l, inner_b, inner_w, inner_h, bbox_w, bbox_h, base_scale,
    tree_cx, tree_cy)`` from the current state. Pure: no prefs lookups, no
    mutation of *st*.
    """
    bounds = st.tree_bounds
    inner_l, inner_b, inner_w, inner_h = _get_map_content_rect(st)
    bbox_w = max(bounds[2] - bounds[0], 1.0)
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    base_scale = min(inner_w / bbox_w, inner_h / bbox_h)
    tree_cx = (bounds[0] + bounds[2]) / 2
    tree_cy = (bounds[1] + bounds[3]) / 2
    return inner_l, inner_b, inner_w, inner_h, bbox_w, bbox_h, base_scale, tree_cx, tree_cy


def _compute_map_transform(
    st: MinimapState | None = None,
) -> tuple[float, float, float, float, float]:
    """Compute the screen mapping ``(cx, cy, scale, tree_cx, tree_cy)`` for the minimap.

    Pure: no side effects, no preference lookups. Callers that need the
    scale-independent geometry (inner rect, base_scale) can use
    :func:`_compute_base_map_geom` directly.
    """
    if st is None:
        st = _state()
    inner_l, inner_b, inner_w, inner_h, _bw, _bh, base_scale, tree_cx, tree_cy = _compute_base_map_geom(st)
    scale = base_scale * st.zoom
    cx = inner_l + inner_w / 2 + st.pan[0]
    cy = inner_b + inner_h / 2 + st.pan[1]
    return cx, cy, scale, tree_cx, tree_cy


def _get_minimap_transform(
    st: MinimapState | None = None,
    space: Any = None,
    region: Any = None,
    visible: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float, float]:
    """Computes internal transformations representing scale, zoom, and panning inside the minimap."""
    if st is None:
        st = _state()
    base_zoom = st.base_zoom
    zoom = base_zoom

    geom = _compute_base_map_geom(st)
    inner_l, inner_b, inner_w, inner_h, _bw, _bh, base_scale, _tcx, _tcy = geom

    # Dynamic Auto-Zoom if follow_view is active
    addon = bpy.context.preferences.addons.get(__package__)
    if addon and getattr(addon.preferences.settings, "follow_view", False):
        if space is None:
            space = bpy.context.space_data
        if region is None:
            region = bpy.context.region

        if space and space.type == "NODE_EDITOR" and region:
            if visible is None:
                visible = _get_visible_rect(space, region)
            if visible:
                vw = max(visible[2] - visible[0], 1.0)
                vh = max(visible[3] - visible[1], 1.0)

                req_zoom_w = (inner_w / vw) / base_scale
                req_zoom_h = (inner_h / vh) / base_scale
                min_req_zoom = min(req_zoom_w, req_zoom_h)

                # If viewport indicator exceeds bounds, dynamically zoom out to fit it perfectly
                if min_req_zoom < zoom:
                    zoom = min_req_zoom

                st.zoom = zoom
                # Execute clamping passively during draw so panning outside the minimap updates bounds
                _clamp_pan_to_viewport(space, region, st, visible)

    st.zoom = zoom
    scale = base_scale * st.zoom
    cx = inner_l + inner_w / 2 + st.pan[0]
    cy = inner_b + inner_h / 2 + st.pan[1]
    tree_cx = (st.tree_bounds[0] + st.tree_bounds[2]) / 2
    tree_cy = (st.tree_bounds[1] + st.tree_bounds[3]) / 2
    return cx, cy, scale, tree_cx, tree_cy


def _clamp_pan_to_viewport(
    space, region, st: MinimapState, visible: tuple[float, float, float, float] | None = None
) -> None:
    """Clamp *st.pan* so the editor viewport stays inside the minimap (follow mode).

    No-op when the ``follow_view`` preference is off.
    """
    addon = bpy.context.preferences.addons.get(__package__)
    if not addon or not getattr(addon.preferences.settings, "follow_view", False):
        return

    if visible is None:
        visible = _get_visible_rect(space, region)
    if not visible:
        return

    cx, cy, scale, tree_cx, tree_cy = _compute_map_transform(st)
    inner_l, inner_b, inner_w, inner_h = _get_map_content_rect(st)
    inner_r = inner_l + inner_w
    inner_t = inner_b + inner_h

    # Transform viewport corners to minimap pixel space
    vl, vb, vr, vt = visible
    vx = cx + (vl - tree_cx) * scale
    vy = cy + (vb - tree_cy) * scale
    vw = (vr - vl) * scale
    vh = (vt - vb) * scale

    dx = 0.0
    dy = 0.0

    if vw <= inner_w:
        if vx < inner_l:
            dx = inner_l - vx
        elif vx + vw > inner_r:
            dx = inner_r - (vx + vw)
    else:
        if vx < inner_r - vw:
            dx = inner_r - vw - vx
        elif vx > inner_l:
            dx = inner_l - vx

    if vh <= inner_h:
        if vy < inner_b:
            dy = inner_b - vy
        elif vy + vh > inner_t:
            dy = inner_t - (vy + vh)
    else:
        if vy < inner_t - vh:
            dy = inner_t - vh - vy
        elif vy > inner_b:
            dy = inner_b - vy

    if abs(dx) > 0.5:
        st.pan[0] += dx
    if abs(dy) > 0.5:
        st.pan[1] += dy


def _find_node_at(nodes: bpy.types.Nodes, tree_x: float, tree_y: float) -> bpy.types.Node | None:
    """Accurately finds hovered node via true box intersection, favoring top-level over frames."""
    best_node = None
    for node in nodes:
        w, h = _get_node_dims(node)
        x, y = node.location_absolute.x, node.location_absolute.y

        # Checking exact bounds since layout is strictly Y-down
        if x <= tree_x <= x + w and (y - h) <= tree_y <= y:
            if node.type != "FRAME":
                return node
            else:
                best_node = node
    return best_node


def _get_visible_rect(
    space: bpy.types.SpaceNodeEditor, region: bpy.types.Region
) -> tuple[float, float, float, float] | None:
    """Return the visible viewport rectangle in tree coordinates, or None if unavailable.

    Accounts for Blender UI scaling to return unscaled tree coordinates.
    """
    try:
        w, h = region.width, region.height
        vr = region.view2d
        if not vr:
            logger.log(5, "_get_visible_rect: region.view2d unavailable")
            return None

        points = [
            vr.region_to_view(0, 0),
            vr.region_to_view(w, 0),
            vr.region_to_view(0, h),
            vr.region_to_view(w, h),
        ]
        points = [p for p in points if p is not None]
        if not points:
            logger.log(5, "_get_visible_rect: all corners returned None (region %dx%d)", w, h)
            return None

        ui_scale = _get_ui_scale()
        xs = [p[0] / ui_scale for p in points]
        ys = [p[1] / ui_scale for p in points]
        result = (min(xs), min(ys), max(xs), max(ys))
        return result
    except Exception as e:
        logger.log(5, "_get_visible_rect failed: %s", e)
        return None


def _compute_frame_all_targets(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> tuple[float, float, float] | None:
    """Compute target zoom and pan to frame the entire node tree.

    Updates ``st.tree_bounds`` immediately (required for correct targets and
    drawing during animation). Returns ``(zoom, pan_x, pan_y)`` or ``None``
    when data is unavailable.
    """
    st = _state(area_ptr)
    if space is None:
        space = bpy.context.space_data
    if region is None:
        region = bpy.context.region
    if not space or not region:
        return None
    node_tree = space.edit_tree
    if not node_tree:
        return None

    bounds = _get_node_tree_bounds(node_tree.nodes)

    _, _, _, mh = st.rect
    bounds = _expand_bounds_margin(bounds, _get_ui_scale(), mh, st.padding)
    st.tree_bounds = bounds

    addon = bpy.context.preferences.addons.get(__package__)
    follow = addon and getattr(addon.preferences.settings, "follow_view", False)

    if not follow:
        return 1.0, 0.0, 0.0

    visible = _get_visible_rect(space, region)
    if visible:
        c_min_x = min(bounds[0], visible[0])
        c_min_y = min(bounds[1], visible[1])
        c_max_x = max(bounds[2], visible[2])
        c_max_y = max(bounds[3], visible[3])
    else:
        c_min_x, c_min_y, c_max_x, c_max_y = bounds

    _, _, inner_w, inner_h, _, _, base_scale, tree_cx, tree_cy = _compute_base_map_geom(st)

    combined_w = max(c_max_x - c_min_x, 1.0)
    combined_h = max(c_max_y - c_min_y, 1.0)
    zoom = min(inner_w / (base_scale * combined_w), inner_h / (base_scale * combined_h), 1.0)

    combined_cx = (c_min_x + c_max_x) / 2
    combined_cy = (c_min_y + c_max_y) / 2

    pan_x = -(combined_cx - tree_cx) * base_scale * zoom
    pan_y = -(combined_cy - tree_cy) * base_scale * zoom
    return zoom, pan_x, pan_y


def frame_all(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the entire node tree.

    When ``follow_view`` is enabled the editor viewport is included in the
    frame so that clamping cannot clip nodes afterward.
    """
    targets = _compute_frame_all_targets(space, region, area_ptr)
    if targets is None:
        return
    st = _state(area_ptr)
    zoom, pan_x, pan_y = targets
    st.base_zoom = zoom
    st.zoom = zoom
    st.pan = [pan_x, pan_y]
    redraw_ui("NODE_EDITOR")


def _compute_frame_to_bounds_targets(
    target_bounds: tuple[float, float, float, float],
    fill: bool = False,
    area_ptr: int | None = None,
) -> tuple[float, float, float]:
    """Compute target zoom and pan to frame the given bounds without applying them.

    Returns ``(zoom, pan_x, pan_y)``.
    """
    st = _state(area_ptr)

    _, _, inner_w, inner_h, _, _, base_scale, tree_cx, tree_cy = _compute_base_map_geom(st)

    tw = max(target_bounds[2] - target_bounds[0], 1.0)
    th = max(target_bounds[3] - target_bounds[1], 1.0)
    if fill:
        zoom = min(inner_w / (base_scale * tw), inner_h / (base_scale * th))
    else:
        zoom = min(inner_w / (base_scale * tw), inner_h / (base_scale * th), 1.0)

    target_cx = (target_bounds[0] + target_bounds[2]) / 2
    target_cy = (target_bounds[1] + target_bounds[3]) / 2

    pan_x = -(target_cx - tree_cx) * base_scale * zoom
    pan_y = -(target_cy - tree_cy) * base_scale * zoom
    return zoom, pan_x, pan_y


def _frame_to_bounds(
    target_bounds: tuple[float, float, float, float],
    fill: bool = False,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the given bounds in tree coordinates.

    When *fill* is True the bounds are zoomed to entirely fill the minimap
    (one axis may clip); when False the bounds frame within the minimap
    (empty space may remain).
    """
    st = _state(area_ptr)
    zoom, pan_x, pan_y = _compute_frame_to_bounds_targets(target_bounds, fill, area_ptr)
    st.base_zoom = zoom
    st.zoom = zoom
    st.pan = [pan_x, pan_y]
    redraw_ui("NODE_EDITOR")


def _compute_center_pan(tree_x: float, tree_y: float, area_ptr: int | None = None) -> tuple[float, float]:
    """Compute minimap pan values that center the given tree point, keeping zoom."""
    st = _state(area_ptr)
    _, _, scale, tree_cx, tree_cy = _compute_map_transform(st)
    return -(tree_x - tree_cx) * scale, -(tree_y - tree_cy) * scale


def _get_selected_bounds(nodes: Iterable[bpy.types.Node]) -> tuple[float, float, float, float] | None:
    """Return the ``(min_x, min_y, max_x, max_y)`` bounds of the selected nodes, or None."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for node in nodes:
        if not node.select:
            continue
        w, h = _get_node_dims(node)
        x, y = node.location_absolute.x, node.location_absolute.y
        min_x = min(min_x, x)
        max_x = max(max_x, x + w)
        min_y = min(min_y, y - h)
        max_y = max(max_y, y)
    if min_x == float("inf"):
        return None
    return min_x, min_y, max_x, max_y


def _compute_frame_selected_targets(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> tuple[float | None, float, float] | None:
    """Compute target zoom and pan for the selected nodes without applying them.

    Returns ``(zoom, pan_x, pan_y)`` where *zoom* is ``None`` when the current
    zoom should be kept (single regular node selected). Returns ``None`` when
    nothing is selected or data is unavailable.
    """
    st = _state(area_ptr)
    if space is None:
        space = bpy.context.space_data
    if not space or space.type != "NODE_EDITOR":
        return None
    node_tree = space.edit_tree
    if not node_tree:
        return None

    selected = [n for n in node_tree.nodes if n.select]
    if not selected:
        return None

    bounds = _get_selected_bounds(selected)
    if bounds is None:
        return None
    min_x, min_y, max_x, max_y = bounds

    rect = st.rect
    _, _, mw, mh = rect
    st.tree_bounds = _expand_bounds_margin(_get_node_tree_bounds(node_tree.nodes), _get_ui_scale(), mh, st.padding)

    if len(selected) > 1 or selected[0].type == "FRAME":
        zoom, pan_x, pan_y = _compute_frame_to_bounds_targets(
            (min_x, min_y, max_x, max_y), fill=True, area_ptr=area_ptr
        )
        return min(zoom, MAX_FRAME_ZOOM), pan_x, pan_y

    pan_x, pan_y = _compute_center_pan((min_x + max_x) / 2, (min_y + max_y) / 2, area_ptr)
    return None, pan_x, pan_y


def _compute_editor_frame_selected_targets(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
) -> tuple[float, float, float, float] | None:
    """Compute the editor viewport rect that frames the selected nodes.

    Fits multiple selections or a single frame node with a margin; centers a
    single regular node while keeping the current viewport size. Returns
    tree-space ``(left, bottom, right, top)`` or ``None`` when unavailable.
    """
    if space is None:
        space = bpy.context.space_data
    if region is None:
        region = bpy.context.region
    if not space or space.type != "NODE_EDITOR" or not region:
        return None
    node_tree = space.edit_tree
    if not node_tree:
        return None

    visible = _get_visible_rect(space, region)
    if not visible:
        return None

    selected = [n for n in node_tree.nodes if n.select]
    if not selected:
        return None
    bounds = _get_selected_bounds(selected)
    if bounds is None:
        return None
    min_x, min_y, max_x, max_y = bounds
    sel_cx = (min_x + max_x) / 2
    sel_cy = (min_y + max_y) / 2
    vw = visible[2] - visible[0]
    vh = visible[3] - visible[1]

    if len(selected) > 1 or selected[0].type == "FRAME":
        bw = max(max_x - min_x, 1.0)
        bh = max(max_y - min_y, 1.0)
        mx = bw * _EDITOR_FIT_MARGIN
        my = bh * _EDITOR_FIT_MARGIN
        left, bottom, right, top = min_x - mx, min_y - my, max_x + mx, max_y + my

        # Limit zoom-in so tiny selections do not magnify excessively.
        hw = max((right - left) / 2, vw / MAX_FRAME_ZOOM / 2)
        hh = max((top - bottom) / 2, vh / MAX_FRAME_ZOOM / 2)
        cx = (left + right) / 2
        cy = (bottom + top) / 2
        return cx - hw, cy - hh, cx + hw, cy + hh

    return sel_cx - vw / 2, sel_cy - vh / 2, sel_cx + vw / 2, sel_cy + vh / 2


def frame_selected(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the selected node(s).

    Zooms to fit multiple selections or a single frame; centers a single
    regular node without changing the zoom level.
    """
    targets = _compute_frame_selected_targets(space, region, area_ptr)
    if targets is None:
        return
    zoom, pan_x, pan_y = targets
    st = _state(area_ptr)
    if zoom is not None:
        st.base_zoom = zoom
        st.zoom = zoom
    st.pan = [pan_x, pan_y]
    redraw_ui("NODE_EDITOR")


def frame_view(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the current editor viewport."""
    st = _state(area_ptr)
    if space is None:
        space = bpy.context.space_data
    if region is None:
        region = bpy.context.region
    if not space or not region:
        return
    node_tree = space.edit_tree
    if not node_tree:
        return

    visible = _get_visible_rect(space, region)
    if not visible:
        return

    addon = bpy.context.preferences.addons.get(__package__)
    fill = addon and getattr(addon.preferences.settings, "frame_view_fill", False)

    rect = st.rect
    _, _, mw, mh = rect
    st.tree_bounds = _expand_bounds_margin(_get_node_tree_bounds(node_tree.nodes), _get_ui_scale(), mh, st.padding)
    _frame_to_bounds(visible, fill=fill, area_ptr=area_ptr)


def _theme_rgba(path: str, default: tuple[float, ...]) -> tuple[float, ...]:
    """Resolve a dotted theme attribute path to an RGBA tuple, ensuring 4 channels."""
    result = _theme(path, default)
    if len(result) == 3:
        return result + (1.0,)
    return result


def _get_node_editor_theme_colors() -> dict[str, Any]:
    """Fetch theme color palette for the minimap drawing."""
    addon = bpy.context.preferences.addons.get(__package__)
    theme_bg = _theme_rgba("node_editor.node_backdrop", (0.22, 0.22, 0.22, 0.95))
    if addon and getattr(addon.preferences.settings, "custom_bg_color", False):
        bg = tuple(getattr(addon.preferences.settings, "bg_color", (0.22, 0.22, 0.22, 0.85)))
    else:
        bg = theme_bg

    return {
        "bg": bg,
        "bg_border": _theme_rgba("user_interface.wcol_toolbar_item.outline", (1.0, 1.0, 1.0, 0.08)),
        "node": _theme_rgba("user_interface.wcol_regular.inner", (0.25, 0.25, 0.25, 1.0)),
        "node_selected": _theme_rgba("node_editor.node_selected", (0.28, 0.45, 0.7, 1.0)),
        "node_active": _theme_rgba("node_editor.node_active", (1.0, 1.0, 1.0, 1.0)),
        "node_border": _theme_rgba("user_interface.wcol_regular.outline", (1.0, 1.0, 1.0, 0.12)),
        "wire": _theme_rgba("node_editor.wire_inner", (0.45, 0.45, 0.45, 0.5)),
        "indicator": _theme_rgba("view_3d.object_active", (1.0, 0.63, 0.16, 1.0)),
        "node_outline": _theme_rgba("node_editor.node_outline", (1.0, 0.37, 0.34, 0.9)),
        "frame_node": _theme_rgba("node_editor.frame_node", (0.22, 0.22, 0.22, 0.85)),
        "text": _theme_rgba("user_interface.wcol_regular.text_sel", (1.0, 1.0, 1.0, 1.0)),
        "scroll_item": _theme_rgba("user_interface.wcol_scroll.item", (0.35, 0.35, 0.35, 0.75)),
        "panel_roundness": _theme_float("user_interface.panel_roundness", 0.4) * 15,
        "node_roundness": _theme_float("user_interface.wcol_regular.roundness", 0.2) * 10,
    }


_EMPTY_FINGERPRINT = (0, 0.0, "", 0, 0, 0, 0.0, 0.0, 0)


def _get_tree_snapshot(
    node_tree, include_selection: bool = True
) -> tuple[tuple, tuple[float, float, float, float], int]:
    """Return ``(fingerprint, raw bounds, drawable node count)`` in a single RNA pass.

    The fingerprint matches :func:`get_tree_fingerprint`; bounds use the same
    clamped dims as :func:`_get_node_tree_bounds`; the count excludes FRAME
    and REROUTE nodes.
    """
    if not node_tree or not hasattr(node_tree, "nodes") or len(node_tree.nodes) == 0:
        return _EMPTY_FINGERPRINT, (0.0, 0.0, 200.0, 200.0), 0
    nodes = node_tree.nodes
    ui = _get_ui_scale()

    loc_sum = 0.0
    width_sum = 0.0
    height_sum = 0.0
    mute_sum = 0
    hide_sum = 0
    select_sum = 0
    content_count = 0

    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for node in nodes:
        loc = node.location_absolute
        x = loc.x
        y = loc.y
        loc_sum += x + y

        w_rna = abs(node.width)
        width_sum += node.width

        try:
            dims = node.dimensions
            h_abs = abs(dims[1])
            w_dim = abs(dims[0])
        except (AttributeError, TypeError, IndexError):
            h_abs = abs(getattr(node, "height", 30.0))
            w_dim = w_rna
        height_sum += h_abs

        if node.mute:
            mute_sum += 1
        if node.hide:
            hide_sum += 1
            bw, bh = 100.0, 30.0
        else:
            bw = w_dim if w_dim > 0 else w_rna
            bh = h_abs if h_abs > 0 else abs(getattr(node, "height", 30.0))
            bw = max(bw / ui, 5.0)
            bh = max(bh / ui, 5.0)

        if include_selection and node.select:
            select_sum += 1
        if node.type not in ("FRAME", "REROUTE"):
            content_count += 1

        rx = x + bw
        by = y - bh
        if x < min_x:
            min_x = x
        if rx > max_x:
            max_x = rx
        if by < min_y:
            min_y = by
        if y > max_y:
            max_y = y

    links_count = len(node_tree.links) if hasattr(node_tree, "links") else 0
    active_name = ""
    if include_selection:
        active = nodes.active
        if active:
            active_name = active.name

    fingerprint = (
        len(nodes),
        loc_sum,
        active_name,
        select_sum,
        mute_sum,
        hide_sum,
        width_sum,
        height_sum,
        links_count,
    )
    bounds = (min_x, min_y, max_x, max_y) if min_x != float("inf") else (0.0, 0.0, 200.0, 200.0)
    return fingerprint, bounds, content_count


def get_tree_fingerprint(node_tree, include_selection: bool = True) -> tuple:
    """Generate a lightweight fingerprint of the node tree structure and selection states."""
    return _get_tree_snapshot(node_tree, include_selection)[0]


def _get_node_initials(name: str) -> str:
    """Extract uppercase initials from each word of a node label."""
    name = name.strip()
    if not name:
        return "?"
    words = name.split()
    if len(words) >= 2:
        initials = "".join(w[0] for w in words if w[0].isalnum()).upper()
        if initials:
            return initials
    for ch in name:
        if ch.isalnum():
            return ch.upper()
    return name[0].upper()


def _get_node_label_lines(label: str, font_id: int, font_size: int, max_width: float, max_lines: int = 3) -> list[str]:
    """Word-wrap a label into up to max_lines, each fitting within max_width pixels."""
    blf.size(font_id, font_size)
    words = label.split()
    if not words:
        return []
    if blf.dimensions(font_id, label)[0] <= max_width:
        return [label]
    lines = []
    i = 0
    while i < len(words) and len(lines) < max_lines:
        line_words = [words[i]]
        i += 1
        while i < len(words):
            candidate = " ".join(line_words + [words[i]])
            w, _ = blf.dimensions(font_id, candidate)
            if w > max_width:
                break
            line_words.append(words[i])
            i += 1
        lines.append(" ".join(line_words))
    return lines
