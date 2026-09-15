"""Provide theme and color utilities for the minimap."""

from functools import lru_cache
from typing import Any

import bpy

from .helpers import get_addon_preferences

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


# Bounded: opacity-driven alpha values would grow an unbounded cache without limit.
@lru_cache(maxsize=512)
def _srgb_to_linear(color: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert an sRGB color tuple to linear color space."""

    def _convert_channel(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    return (_convert_channel(color[0]), _convert_channel(color[1]), _convert_channel(color[2]), color[3])


def _rgba(value: tuple[float, ...], alpha: float) -> tuple[float, float, float, float]:
    """Convert a multi-channel tuple to RGBA using the given alpha."""
    return (float(value[0]), float(value[1]), float(value[2]), float(alpha))


def _alpha_mul(color: tuple[float, ...], alpha: float) -> tuple[float, float, float, float]:
    """Return RGBA with the original alpha multiplied by alpha."""
    return (float(color[0]), float(color[1]), float(color[2]), float(color[3] * alpha))


def _color_contrast(color: tuple[float, ...], factor: float = 0.85) -> tuple[float, float, float, float]:
    """Darken a color by the given factor to produce a contrast variant."""
    return (float(color[0] * factor), float(color[1] * factor), float(color[2] * factor), 1.0)


def _theme(path: str, default: tuple[float, ...]) -> tuple[float, ...]:
    """Resolve a dotted theme attribute path to a color tuple, falling back to default."""
    prefs = bpy.context.preferences
    if not prefs.themes:
        return default
    value: Any = prefs.themes[0]
    try:
        for path_part in path.split("."):
            value = getattr(value, path_part)
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
        for path_part in path.split("."):
            value = getattr(value, path_part)
        return float(value)
    except (AttributeError, TypeError, ValueError):
        return default


def _theme_int(path: str, default: int) -> int:
    """Resolve a dotted theme attribute path to an int value, falling back to default."""
    prefs = bpy.context.preferences
    if not prefs.themes:
        return default
    value = prefs.themes[0]
    try:
        for path_part in path.split("."):
            value = getattr(value, path_part)
        return int(value)
    except (AttributeError, TypeError, ValueError):
        return default


def _get_wire_curvature(settings) -> int:
    """Return the effective level of curved-wire rendering (0 = straight).

    Use the add-on custom value when enabled, otherwise use the Blender theme
    ``node_editor.noodle_curving`` so the minimap follows the node-graph look.
    """
    if settings.use_custom_noodle_curving:
        return int(settings.noodle_curving)
    return _theme_int("node_editor.noodle_curving", 0)


def _theme_rgba(path: str, default: tuple[float, ...]) -> tuple[float, ...]:
    """Resolve a dotted theme attribute path to an RGBA tuple, ensuring 4 channels."""
    result = _theme(path, default)
    if len(result) == 3:
        return result + (1.0,)
    return result


def _get_node_color(node: bpy.types.Node, fallback_color: tuple[float, ...]) -> tuple[float, ...]:
    """Return the node's custom color, theme-mapped color_tag color, or fallback."""
    if getattr(node, "use_custom_color", False):
        return _rgba(node.color, fallback_color[3])
    color_tag = getattr(node, "color_tag", "NONE")
    if color_tag != "NONE":
        mapped_theme_attr = _COLOR_TAG_TO_THEME_ATTR.get(color_tag)
        if mapped_theme_attr:
            return _theme_rgba(f"node_editor.{mapped_theme_attr}", fallback_color)
    return fallback_color


def _get_node_editor_theme_colors() -> dict[str, Any]:
    """Fetch theme color palette for the minimap drawing."""
    settings = get_addon_preferences().settings

    theme_background = _theme_rgba("node_editor.space.back", (0.1, 0.1, 0.1, 1.0))
    if settings.use_custom_background:
        background = tuple(settings.high_gradient)
        background_low = tuple(settings.low_gradient)
    else:
        background = theme_background
        background_low = _color_contrast(theme_background, 0.6)

    selected = _theme_rgba("user_interface.wcol_regular.text_sel", (0.28, 0.45, 0.7, 1.0))
    if settings.use_custom_viewport_fill:
        active_view_color = tuple(settings.viewport_fill_color)
    else:
        active_view_color = selected

    result = {
        "background": background,
        "background_low": background_low,
        "background_border": _theme_rgba("user_interface.wcol_toolbar_item.outline", (1.0, 1.0, 1.0, 0.08)),
        "node_backdrop": _theme_rgba("node_editor.node_backdrop", (0.188, 0.188, 0.188, 1.0)),
        "node_selected": _theme_rgba("node_editor.node_selected", (0.929, 0.341, 0.0, 1.0)),
        "node_active": _theme_rgba("node_editor.node_active", (1.0, 1.0, 1.0, 1.0)),
        "node_outline": _theme_rgba("node_editor.node_outline", (1.0, 1.0, 1.0, 0.149)),
        "node_text": _theme_rgba("node_editor.space.text", (0.902, 0.902, 0.902, 1.0)),
        "frame_node": _theme_rgba("node_editor.frame_node", (0.059, 0.059, 0.059, 0.8)),
        # Wire
        "wire_color": _theme_rgba("node_editor.wire_inner", (0.55, 0.55, 0.55, 1.0)),
        "wire_selected": _theme_rgba("node_editor.wire_select", (1.0, 1.0, 1.0, 0.7)),
        "indicator": _theme_rgba("view_3d.object_active", (1.0, 0.63, 0.16, 1.0)),
        "active_view_color": active_view_color,
        # Regular
        "regular_inner": _theme_rgba("user_interface.wcol_regular.inner", (0.329, 0.329, 0.329, 1.0)),
        "regular_selected": _theme_rgba("user_interface.wcol_regular.inner_sel", (0.28, 0.45, 0.7, 1.0)),
        "regular_outline": _theme_rgba("user_interface.wcol_regular.outline", (0.239, 0.239, 0.239, 1.0)),
        "regular_text": _theme_rgba("user_interface.wcol_regular.text", (0.902, 0.902, 0.902, 1.0)),
        "regular_text_selected": _theme_rgba("user_interface.wcol_regular.text_sel", (1.0, 1.0, 1.0, 1.0)),
        # Search
        "search_background": _theme_rgba("user_interface.wcol_text.inner", (0.114, 0.114, 0.114, 1.0)),
        "search_background_selected": _theme_rgba("user_interface.wcol_text.inner_sel", (0.094, 0.094, 0.094, 1.0)),
        "search_text_outline": _theme_rgba("user_interface.wcol_text.outline", (0.239, 0.239, 0.239, 1.0)),
        "search_text": _theme_rgba("user_interface.wcol_text.text", (0.902, 0.902, 0.902, 1.0)),
        "search_text_selected": _theme_rgba("user_interface.wcol_text.text_sel", (1.0, 1.0, 1.0, 1.0)),
        # Tool
        "tool_inner": _theme_rgba("user_interface.wcol_tool.inner", (0.329, 0.329, 0.329, 1.0)),
        "tool_selected": _theme_rgba("user_interface.wcol_tool.inner_sel", (0.28, 0.45, 0.7, 1.0)),
        "tool_outline": _theme_rgba("user_interface.wcol_tool.outline", (0.239, 0.239, 0.239, 1.0)),
        "tool_text": _theme_rgba("user_interface.wcol_tool.text", (0.902, 0.902, 0.902, 1.0)),
        "tool_text_selected": _theme_rgba("user_interface.wcol_tool.text_sel", (1.0, 0.0, 0.0, 1.0)),
        # outliner
        "outliner_back": _theme_rgba("outliner.space.back", (0.157, 0.157, 0.157, 1.0)),
        "outliner_text": _theme_rgba("outliner.space.text", (0.765, 0.765, 0.765, 1.0)),
        "outliner_match": _theme_rgba("outliner.match", (0.2, 0.498, 0.2, 1.0)),
        "outliner_selected_highlight": _theme_rgba("outliner.selected_highlight", (0.114, 0.192, 0.302, 1.0)),
        "outliner_active": _theme_rgba("outliner.active", (0.2, 0.302, 0.502, 1.0)),
        "outliner_selected_object": _theme_rgba("outliner.selected_object", (0.5, 0.4, 0.2, 1.0)),
        "outliner_active_object": _theme_rgba("outliner.active_object", (0.914, 0.416, 0.0, 1.0)),
        "outliner_row_alternate": _theme_rgba("outliner.row_alternate", (1.0, 0.0, 1.0, 0.016)),
        # scroll
        "scroll_item": _theme_rgba("user_interface.wcol_scroll.item", (0.329, 0.329, 0.329, 1.0)),
        "scroll_inner": _theme_rgba("user_interface.wcol_scroll.inner", (0.133, 0.133, 0.133, 0.0)),
        # Roundness
        "panel_roundness": _theme_float("user_interface.panel_roundness", 0.4) * 12,
        "regular_roundness": _theme_float("user_interface.wcol_regular.roundness", 0.4) * 12,
        "node_roundness": _theme_float("user_interface.wcol_regular.roundness", 0.4) * 12,
    }
    return result
