"""Shared helper utilities for node minimap."""

import logging
from typing import Any

import blf
import bpy

logger = logging.getLogger(__package__)

LUMINANCE_R: float = 0.299
LUMINANCE_G: float = 0.587
LUMINANCE_B: float = 0.114
OUTLINE_ALPHA: float = 0.8


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
    space: bpy.types.SpaceNodeEditor | None = None,
    corner: str = "TOP_RIGHT",
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
        elif r.type == "HEADER" and getattr(r, "alignment", "") == "BOTTOM":
            bottom = max(bottom, r.height)
        elif r.type == "UI":
            right = min(right, region.width - r.width)

    scale = _get_ui_scale()

    if space and corner in ("TOP_LEFT", "TOP_RIGHT") and getattr(space.overlay, "show_overlays", False):
        top = min(top, region.height - int(8 * scale))

    if space and getattr(space, "show_region_asset_shelf", False):
        bottom = max(bottom, int(10 * scale))

    return int(left), int(bottom), int(right), int(top)


_minimap_state: dict[int, dict] = {}
_minimap_window_operators: dict[int, Any] = {}
_registration_state: dict[str, bool] = {"done": False}

_DEFAULT_STATE: dict = {
    "rect": (0, 0, 0, 0),
    "tree_bounds": (0.0, 0.0, 0.0, 0.0),
    "margin": 10,
    "padding": 6,
    "scale": 1.0,
    "hovered_node": None,
    "zoom": 1.0,
    "base_zoom": 1.0,
    "pan": [0.0, 0.0],
    "enabled": True,
    "frame_all_btn": None,
}


def _state(area_ptr: int | None = None) -> dict:
    """Return the minimap state dict for the given area, initializing defaults if needed."""
    if area_ptr is None:
        try:
            area_ptr = bpy.context.area.as_pointer()
        except (AttributeError, ReferenceError):
            return {}
    if area_ptr not in _minimap_state:
        state = dict(_DEFAULT_STATE)
        try:
            prefs = bpy.context.preferences.addons.get(__package__)
            if prefs:
                state["enabled"] = getattr(prefs.preferences.settings, "show_by_default", True)
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


def _get_node_dims(node: bpy.types.Node) -> tuple[float, float]:
    """Robust extraction of width and height ensuring positive float values."""
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

    ui_scale = _get_ui_scale()
    return max(w / ui_scale, 5.0), max(h / ui_scale, 5.0)


def _get_node_tree_bounds(nodes: bpy.types.Nodes) -> tuple[float, float, float, float]:
    """Compute the bounding box of all nodes in a node tree as (min_x, min_y, max_x, max_y)."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for node in nodes:
        w, h = _get_node_dims(node)
        x, y = node.location_absolute.x, node.location_absolute.y
        min_x = min(min_x, x)
        max_x = max(max_x, x + w)
        min_y = min(min_y, y - h)
        max_y = max(max_y, y)

    if min_x == float("inf"):
        return 0.0, 0.0, 200.0, 200.0
    return min_x, min_y, max_x, max_y


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


def _get_minimap_transform(st: dict | None = None) -> tuple[float, float, float, float, float]:
    """Computes internal transformations representing scale, zoom, and panning inside the minimap."""
    if st is None:
        st = _state()
    rect = st.get("rect", (0, 0, 100, 100))
    bounds = st.get("tree_bounds", (0, 0, 100, 100))
    padding = st.get("padding", 6 * _get_ui_scale())

    if "base_zoom" not in st:
        st["base_zoom"] = st.get("zoom", 1.0)

    base_zoom = st["base_zoom"]
    zoom = base_zoom
    pan = st.get("pan", [0.0, 0.0])

    mx, my, mw, mh = rect
    inner_w = max(mw - 2 * padding, 1.0)
    inner_h = max(mh - 2 * padding, 1.0)

    bbox_w = max(bounds[2] - bounds[0], 1.0)
    bbox_h = max(bounds[3] - bounds[1], 1.0)

    base_scale = min(inner_w / bbox_w, inner_h / bbox_h)

    # Dynamic Auto-Zoom if follow_view is active
    addon = bpy.context.preferences.addons.get(__package__)
    if addon and getattr(addon.preferences.settings, "follow_view", False):
        space = bpy.context.space_data
        region = bpy.context.region
        if space and space.type == "NODE_EDITOR" and region:
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

                st["zoom"] = zoom
                # Execute clamping passively during draw so panning outside the minimap updates bounds
                _clamp_pan_to_viewport(space, region, st)
                pan = st["pan"]

    st["zoom"] = zoom
    scale = base_scale * zoom

    cx = mx + padding + inner_w / 2 + pan[0]
    cy = my + padding + inner_h / 2 + pan[1]

    tree_cx = (bounds[0] + bounds[2]) / 2
    tree_cy = (bounds[1] + bounds[3]) / 2

    return cx, cy, scale, tree_cx, tree_cy


def _clamp_pan_to_viewport(space, region, st) -> None:
    """Clamp *st['pan']* so the editor viewport stays inside the minimap (follow mode).

    No-op when the ``follow_view`` preference is off.
    """
    addon = bpy.context.preferences.addons.get(__package__)
    if not addon or not getattr(addon.preferences.settings, "follow_view", False):
        return

    visible = _get_visible_rect(space, region)
    if not visible:
        return

    rect = st.get("rect", (0, 0, 100, 100))
    bounds = st.get("tree_bounds", (0, 0, 100, 100))
    padding = st.get("padding", 6 * _get_ui_scale())
    zoom = st.get("zoom", 1.0)
    pan = st.get("pan", [0.0, 0.0])

    mx, my, mw, mh = rect
    inner_l = mx + padding
    inner_b = my + padding
    inner_r = mx + mw - padding
    inner_t = my + mh - padding
    inner_w = max(mw - 2 * padding, 1.0)
    inner_h = max(mh - 2 * padding, 1.0)

    bbox_w = bounds[2] - bounds[0]
    bbox_h = bounds[3] - bounds[1]
    base_scale = min(inner_w / max(bbox_w, 1.0), inner_h / max(bbox_h, 1.0))
    scale = base_scale * zoom

    cx = mx + padding + inner_w / 2 + pan[0]
    cy = my + padding + inner_h / 2 + pan[1]
    tree_cx = (bounds[0] + bounds[2]) / 2
    tree_cy = (bounds[1] + bounds[3]) / 2

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
        st["pan"][0] += dx
    if abs(dy) > 0.5:
        st["pan"][1] += dy


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


def frame_all() -> None:
    """Adjust minimap zoom/pan to frame the entire node tree.

    When ``follow_view`` is enabled the editor viewport is included in the
    fit so that clamping cannot clip nodes afterward.
    """
    st = _state()
    space = bpy.context.space_data
    region = bpy.context.region
    if not space or not region:
        return
    node_tree = space.edit_tree
    if not node_tree:
        return

    bounds = _get_node_tree_bounds(node_tree.nodes)
    st["tree_bounds"] = bounds

    addon = bpy.context.preferences.addons.get(__package__)
    follow = addon and getattr(addon.preferences.settings, "follow_view", False)

    if not follow:
        st["base_zoom"] = 1.0
        st["zoom"] = 1.0
        st["pan"] = [0.0, 0.0]
        redraw_ui("NODE_EDITOR")
        return

    visible = _get_visible_rect(space, region)
    if visible:
        c_min_x = min(bounds[0], visible[0])
        c_min_y = min(bounds[1], visible[1])
        c_max_x = max(bounds[2], visible[2])
        c_max_y = max(bounds[3], visible[3])
    else:
        c_min_x, c_min_y, c_max_x, c_max_y = bounds

    rect = st.get("rect", (0, 0, 100, 100))
    padding = st.get("padding", 6 * _get_ui_scale())
    _, _, mw, mh = rect
    inner_w = max(mw - 2 * padding, 1.0)
    inner_h = max(mh - 2 * padding, 1.0)

    bbox_w = max(bounds[2] - bounds[0], 1.0)
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    base_scale = min(inner_w / bbox_w, inner_h / bbox_h)

    combined_w = max(c_max_x - c_min_x, 1.0)
    combined_h = max(c_max_y - c_min_y, 1.0)
    zoom = min(inner_w / (base_scale * combined_w), inner_h / (base_scale * combined_h), 1.0)

    tree_cx = (bounds[0] + bounds[2]) / 2
    tree_cy = (bounds[1] + bounds[3]) / 2
    combined_cx = (c_min_x + c_max_x) / 2
    combined_cy = (c_min_y + c_max_y) / 2

    st["base_zoom"] = zoom
    st["zoom"] = zoom
    st["pan"] = [
        -(combined_cx - tree_cx) * base_scale * zoom,
        -(combined_cy - tree_cy) * base_scale * zoom,
    ]
    redraw_ui("NODE_EDITOR")


def _frame_to_bounds(
    target_bounds: tuple[float, float, float, float],
    fill: bool = False,
) -> None:
    """Adjust minimap zoom/pan to frame the given bounds in tree coordinates.

    When *fill* is True the bounds are zoomed to entirely fill the minimap
    (one axis may clip); when False the bounds fit within the minimap
    (empty space may remain).
    """
    st = _state()
    space = bpy.context.space_data
    region = bpy.context.region
    if not space or not region:
        return

    rect = st.get("rect", (0, 0, 100, 100))
    padding = st.get("padding", 6 * _get_ui_scale())
    _, _, mw, mh = rect
    inner_w = max(mw - 2 * padding, 1.0)
    inner_h = max(mh - 2 * padding, 1.0)

    bounds = st.get("tree_bounds", (0, 0, 100, 100))
    bbox_w = max(bounds[2] - bounds[0], 1.0)
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    base_scale = min(inner_w / bbox_w, inner_h / bbox_h)

    tw = max(target_bounds[2] - target_bounds[0], 1.0)
    th = max(target_bounds[3] - target_bounds[1], 1.0)
    if fill:
        zoom = min(inner_w / (base_scale * tw), inner_h / (base_scale * th))
    else:
        zoom = min(inner_w / (base_scale * tw), inner_h / (base_scale * th), 1.0)

    tree_cx = (bounds[0] + bounds[2]) / 2
    tree_cy = (bounds[1] + bounds[3]) / 2
    target_cx = (target_bounds[0] + target_bounds[2]) / 2
    target_cy = (target_bounds[1] + target_bounds[3]) / 2

    st["base_zoom"] = zoom
    st["zoom"] = zoom
    st["pan"] = [
        -(target_cx - tree_cx) * base_scale * zoom,
        -(target_cy - tree_cy) * base_scale * zoom,
    ]
    redraw_ui("NODE_EDITOR")


def frame_selected() -> None:
    """Adjust minimap zoom/pan to frame the selected node(s)."""
    st = _state()
    space = bpy.context.space_data
    if not space or space.type != "NODE_EDITOR":
        return
    node_tree = space.edit_tree
    if not node_tree:
        return

    selected = [n for n in node_tree.nodes if n.select]
    if not selected:
        frame_all()
        return

    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for node in selected:
        w, h = _get_node_dims(node)
        x, y = node.location_absolute.x, node.location_absolute.y
        min_x = min(min_x, x)
        max_x = max(max_x, x + w)
        min_y = min(min_y, y - h)
        max_y = max(max_y, y)

    st["tree_bounds"] = _get_node_tree_bounds(node_tree.nodes)
    _frame_to_bounds((min_x, min_y, max_x, max_y))


def frame_view() -> None:
    """Adjust minimap zoom/pan to frame the current editor viewport."""
    st = _state()
    space = bpy.context.space_data
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

    st["tree_bounds"] = _get_node_tree_bounds(node_tree.nodes)
    _frame_to_bounds(visible, fill=fill)


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
        "scroll_item": _theme_rgba("user_interface.wcol_scroll.outline", (0.45, 0.45, 0.45, 0.5)),
        "panel_roundness": _theme_float("user_interface.panel_roundness", 0.4) * 15,
        "node_roundness": _theme_float("user_interface.wcol_regular.roundness", 0.2) * 10,
    }


def get_tree_fingerprint(node_tree) -> tuple:
    """Generate a lightweight fingerprint of the node tree structure and selection states."""
    if not node_tree or not hasattr(node_tree, "nodes") or len(node_tree.nodes) == 0:
        return (0, 0.0, 0, 0, 0)
    nodes = node_tree.nodes
    loc_sum = sum(n.location_absolute.x + n.location_absolute.y for n in nodes)
    select_sum = sum(1 for n in nodes if n.select)
    mute_sum = sum(1 for n in nodes if n.mute)
    links_count = len(node_tree.links) if hasattr(node_tree, "links") else 0
    active_name = nodes.active.name if nodes.active else ""
    return (len(nodes), loc_sum, active_name, select_sum, mute_sum, links_count)


def _get_node_initials(name: str) -> str:
    """Extract 1-2 uppercase initials from a node label."""
    name = name.strip()
    if not name:
        return "?"
    words = name.split()
    if len(words) >= 2:
        return "".join(w[0] for w in words).upper()[:2]
    word = words[0]
    for i, ch in enumerate(word):
        if ch.isalpha():
            return word[:i] + ch.upper()
    return word[0].upper()


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
