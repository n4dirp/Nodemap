"""Shared helper utilities for nodes minimap."""

import logging
from typing import Any

import bpy

logger = logging.getLogger(__package__)

LUMINANCE_R: float = 0.299
LUMINANCE_G: float = 0.587
LUMINANCE_B: float = 0.114
OUTLINE_ALPHA: float = 0.8


def redraw_ui(mode: str = "VIEW_3D", area_pointer: int | None = None) -> None:
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
    def _conv(ch: float) -> float:
        return ch / 12.92 if ch <= 0.04045 else ((ch + 0.055) / 1.055) ** 2.4

    return (_conv(c[0]), _conv(c[1]), _conv(c[2]), c[3])


def _rgba(value: tuple[float, ...], alpha: float) -> tuple[float, float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]), float(alpha))


def _get_ui_scale() -> float:
    return float(bpy.context.preferences.system.ui_scale)


def _compute_outline_color(rgb: tuple[float, ...]) -> tuple[float, float, float, float]:
    luminance = rgb[0] * LUMINANCE_R + rgb[1] * LUMINANCE_G + rgb[2] * LUMINANCE_B
    if luminance > 0.5:
        return (0.0, 0.0, 0.0, OUTLINE_ALPHA)
    return (1.0, 1.0, 1.0, OUTLINE_ALPHA)


def color_contrast(color: tuple[float, ...], factor: float = 0.85) -> tuple[float, float, float, float]:
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


def _get_node_color(node, fallback_color: tuple[float, ...]) -> tuple[float, ...]:
    if getattr(node, "use_custom_color", False):
        return _rgba(node.color, fallback_color[3])
    color_tag = getattr(node, "color_tag", "NONE")
    if color_tag != "NONE":
        theme_attr = _COLOR_TAG_TO_THEME_ATTR.get(color_tag)
        if theme_attr:
            return _theme_rgba(f"node_editor.{theme_attr}", fallback_color)
    return fallback_color


NAV_GIZMO_SIZE = 40


def _get_safe_bounds(
    area: bpy.types.Area,
    region: bpy.types.Region,
    space: bpy.types.SpaceNodeEditor | None = None,
    corner: str = "TOP_RIGHT",
) -> tuple[int, int, int, int]:
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

    if space and corner == "TOP_LEFT" and getattr(space, "show_overlays", False):
        top = min(top, region.height - int(30 * scale))

    if getattr(area, "show_region_asset_shelf", False):
        bottom = max(bottom, int(30 * scale))

    nav = int(NAV_GIZMO_SIZE * scale)
    if right - left > nav and top - bottom > nav:
        right -= nav
    return int(left), int(bottom), int(right), int(top)


_minimap_state: dict[int, dict] = {}

_DEFAULT_STATE: dict = {
    "rect": (0, 0, 0, 0),
    "tree_bounds": (0.0, 0.0, 0.0, 0.0),
    "margin": 10,
    "padding": 6,
    "scale": 1.0,
    "hovered_node": None,
    "zoom": 1.0,
    "pan": [0.0, 0.0],
    "modal_active": False,
}


def _state(area_ptr: int | None = None) -> dict:
    if area_ptr is None:
        try:
            area_ptr = bpy.context.area.as_pointer()
        except (AttributeError, ReferenceError):
            return {}
    if area_ptr not in _minimap_state:
        _minimap_state[area_ptr] = dict(_DEFAULT_STATE)
    return _minimap_state[area_ptr]


def _get_node_dims(node) -> tuple[float, float]:
    """Robust extraction of width and height ensuring positive float values."""
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

    return max(w, 5.0), max(h, 5.0)


def _get_node_tree_bounds(nodes: bpy.types.Nodes) -> tuple[float, float, float, float]:
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


def _get_minimap_transform() -> tuple[float, float, float, float, float]:
    """Computes internal transformations representing scale, zoom, and panning inside the minimap."""
    st = _state()
    rect = st.get("rect", (0, 0, 100, 100))
    bounds = st.get("tree_bounds", (0, 0, 100, 100))
    padding = st.get("padding", 6) * _get_ui_scale()
    zoom = st.get("zoom", 1.0)
    pan = st.get("pan", [0.0, 0.0])

    mx, my, mw, mh = rect
    inner_w = mw - 2 * padding
    inner_h = mh - 2 * padding

    bbox_w = bounds[2] - bounds[0]
    bbox_h = bounds[3] - bounds[1]

    base_scale = min(inner_w / max(bbox_w, 1.0), inner_h / max(bbox_h, 1.0))
    scale = base_scale * zoom

    cx = mx + padding + inner_w / 2 + pan[0]
    cy = my + padding + inner_h / 2 + pan[1]

    tree_cx = (bounds[0] + bounds[2]) / 2
    tree_cy = (bounds[1] + bounds[3]) / 2

    return cx, cy, scale, tree_cx, tree_cy


def _find_node_at(nodes, tree_x: float, tree_y: float) -> object | None:
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
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        result = (min(xs), min(ys), max(xs), max(ys))
        logger.log(5, "Visible rect: %s (region %dx%d, %d/4 corners valid)", result, w, h, len(points))
        return result
    except Exception as e:
        logger.log(5, "_get_visible_rect failed: %s", e)
        return None


def _theme_rgba(path: str, default: tuple[float, ...]) -> tuple[float, ...]:
    result = _theme(path, default)
    if len(result) == 3:
        return result + (1.0,)
    return result


def _get_node_editor_theme_colors():
    return {
        "bg": _theme_rgba("node_editor.node_backdrop", (0.22, 0.22, 0.22, 0.85)),
        "bg_border": _theme_rgba("user_interface.wcol_toolbar_item.outline", (1.0, 1.0, 1.0, 0.08)),
        "node": _theme_rgba("user_interface.wcol_regular.inner", (0.25, 0.25, 0.25, 1.0)),
        "node_selected": _theme_rgba("user_interface.wcol_regular.inner_sel", (0.28, 0.45, 0.7, 1.0)),
        "node_border": _theme_rgba("user_interface.wcol_regular.outline", (1.0, 1.0, 1.0, 0.12)),
        "wire": _theme_rgba("user_interface.wcol_regular.text", (0.45, 0.45, 0.45, 0.5)),
        "indicator": _theme_rgba("view_3d.object_active", (1.0, 0.63, 0.16, 1.0)),
        "node_outline": _theme_rgba("node_editor.node_outline", (1.0, 0.37, 0.34, 0.9)),
        "frame_node": _theme_rgba("node_editor.frame_node", (0.22, 0.22, 0.22, 0.85)),
        "text": _theme_rgba("user_interface.wcol_regular.text_sel", (1.0, 1.0, 1.0, 1.0)),
        "panel_roundness": _theme_float("user_interface.panel_roundness", 0.4) * 15,
        "node_roundness": _theme_float("user_interface.wcol_regular.roundness", 0.2) * 10,
    }
