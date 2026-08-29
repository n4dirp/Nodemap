"""Theme and color utilities for the node minimap."""

from typing import Any

import bpy

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


def _theme_int(path: str, default: int) -> int:
    """Resolve a dotted theme attribute path to an int value, falling back to default."""
    prefs = bpy.context.preferences
    if not prefs.themes:
        return default
    value = prefs.themes[0]
    try:
        for part in path.split("."):
            value = getattr(value, part)
        return int(value)
    except (AttributeError, TypeError, ValueError):
        return default


def _get_wire_curvature(settings) -> int:
    """Return the effective level of curved-wire rendering (0 = straight).

    Uses the add-on's custom value when enabled, otherwise the Blender theme's
    ``node_editor.noodle_curving`` so the minimap follows the node-graph look.
    """
    if settings.use_custom_wire_curvature:
        return int(settings.wire_curvature)
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
        theme_attr = _COLOR_TAG_TO_THEME_ATTR.get(color_tag)
        if theme_attr:
            return _theme_rgba(f"node_editor.{theme_attr}", fallback_color)
    return fallback_color


def _get_node_editor_theme_colors() -> dict[str, Any]:
    """Fetch theme color palette for the minimap drawing."""
    addon = bpy.context.preferences.addons.get(__package__)
    theme_bg = _theme_rgba("node_editor.space.back", (0.4, 0.4, 0.4, 0.95))
    if addon and addon.preferences.settings.custom_bg_color:
        bg = tuple(addon.preferences.settings.bg_color)
    else:
        bg = theme_bg

    text = _theme_rgba("node_editor.space.text", (1.0, 1.0, 1.0, 1.0))
    label = _theme_rgba("node_editor.space.text", (1.0, 1.0, 1.0, 1.0))
    if addon and addon.preferences.settings.custom_text_color:
        text = label = tuple(addon.preferences.settings.text_color)

    return {
        "bg": bg,
        "bg_border": _theme_rgba("user_interface.wcol_toolbar_item.outline", (1.0, 1.0, 1.0, 0.08)),
        "node": _theme_rgba("node_editor.node_backdrop", (0.4, 0.4, 0.4, 1.0)),
        "node_selected": _theme_rgba("node_editor.node_selected", (0.28, 0.45, 0.7, 1.0)),
        "node_active": _theme_rgba("node_editor.node_active", (1.0, 1.0, 1.0, 1.0)),
        "node_border": _theme_rgba("node_editor.node_outline", (1.0, 1.0, 1.0, 0.149)),
        "wire": _theme_rgba("node_editor.wire_inner", (0.45, 0.45, 0.45, 0.5)),
        "indicator": _theme_rgba("view_3d.object_active", (1.0, 0.63, 0.16, 1.0)),
        "frame_node": _theme_rgba("node_editor.frame_node", (0.22, 0.22, 0.22, 0.85)),
        "text": text,
        "label": label,
        "scroll_item": _theme_rgba("user_interface.wcol_scroll.item", (0.35, 0.35, 0.35, 0.75)),
        "panel_roundness": _theme_float("user_interface.panel_roundness", 0.4) * 15,
        "node_roundness": _theme_float("user_interface.wcol_regular.roundness", 0.2) * 10,
    }
