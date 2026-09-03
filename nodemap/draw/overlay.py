"""Provide minimap rendering in the Node Editor."""

import logging
import math
import time

import blf
import bpy
import gpu
from mathutils import Matrix

from .. import __package__ as base_package
from ..core.constants import (
    BORDER_POSITIONS,
    BUTTON_HOVER_ALPHA,
    BUTTON_MARGIN,
    BUTTON_SIZE,
    CORNER_POSITIONS,
    FONT_SIZE,
    MIN_MAP_HEIGHT,
    MIN_MAP_WIDTH,
)
from ..core.helpers import (
    _expand_bounds_margin,
    _get_minimap_margins,
    _get_safe_bounds,
    _get_tree_snapshot,
    _get_ui_scale,
    clamp_free_rect,
    get_addon_preferences,
)
from ..core.state import (
    _MINIMAP_BUTTONS,
    MinimapState,
    ResizeHandle,
    _minimap_window_operators,
    _registration_state,
    _state,
)
from ..core.theme import (
    _alpha_mul,
    _get_node_editor_theme_colors,
    _get_wire_curvature,
)
from ..geo.transforms import (
    _clamp_pan_to_viewport,
    _get_map_content_rect,
    _get_minimap_transform,
    _get_visible_rect,
)
from . import content_draw
from .batch_build import _ensure_minimap_batches
from .gpu_draw import (
    _draw_filled_rounded_rect,
    _draw_filled_rounded_rect_clipped,
    _draw_filled_rounded_rect_varying,
    _draw_filled_rounded_rect_with_hole,
    _draw_pill,
    _draw_rounded_rect_border,
    _draw_rounded_rect_border_varying_sides,
    _draw_text_with_shadow,
)
from .tree_compile import (
    _MOVE_REFRESH_MIN_INTERVAL,
    _apply_move_updates,
    _debounced_compile,
    _is_move_only_diff,
    _Timer,
)
from .type_list import _draw_minimap_scrollbars, _draw_type_list, _step_list_width

logger = logging.getLogger(base_package)


def _early_exit(context, space, state: MinimapState) -> bool:
    """Return True if the minimap should not be drawn."""
    if space is None:
        return True
    if space.type != "NODE_EDITOR":
        return True
    if not space.overlay.show_overlays:
        return True
    if not state.enabled:
        return True
    addon = get_addon_preferences(context)
    if not addon:
        return True
    return False


def _dock_to_border(
    corner: str,
    safe_x_min: float,
    safe_y_min: float,
    safe_x_max: float,
    safe_y_max: float,
    map_w: float,
    map_h: float,
    x_margin: float,
    y_margin: float,
    margin: float,
) -> tuple[float, float]:
    """Compute the minimap origin for a border dock, centered on the free axis."""
    center_x = (safe_x_min + safe_x_max) / 2.0
    center_y = (safe_y_min + safe_y_max) / 2.0
    match corner:
        case "TOP_BORDER":
            return center_x - map_w / 2.0, safe_y_max - map_h - y_margin
        case "BOTTOM_BORDER":
            return center_x - map_w / 2.0, safe_y_min + y_margin
        case "LEFT_BORDER":
            return safe_x_min + x_margin, center_y - map_h / 2.0
        case "RIGHT_BORDER":
            return safe_x_max - map_w - x_margin, center_y - map_h / 2.0
        case _:
            return 0.0, 0.0


def _compute_minimap_rect(
    settings, ui_scale, space, region, corner, state: MinimapState
) -> tuple[float, float, float, float, float, float] | None:
    """Compute the minimap rectangle position and dimensions."""
    safe_x_min, safe_y_min, safe_x_max, safe_y_max = _get_safe_bounds(bpy.context.area, region)
    safe_width = safe_x_max - safe_x_min
    safe_height = safe_y_max - safe_y_min

    x_margin, y_margin, margin = _get_minimap_margins(space, corner, ui_scale)

    # Compute desired size, capped to % of safe region (accounting for margins)
    map_w = settings.minimap_width * ui_scale
    map_h = settings.minimap_height * ui_scale
    max_width_pct = settings.max_width_percent / 100.0
    max_height_pct = settings.max_height_percent / 100.0
    map_w = min(map_w, (safe_width - x_margin) * max_width_pct)
    map_h = min(map_h, (safe_height - y_margin - margin) * max_height_pct)

    padding = 6 * ui_scale

    if corner in CORNER_POSITIONS:
        match corner:
            case "TOP_RIGHT":
                map_x = safe_x_max - map_w - x_margin
                map_y = safe_y_max - map_h - y_margin
            case "TOP_LEFT":
                map_x = safe_x_min + x_margin
                map_y = safe_y_max - map_h - y_margin
            case "BOTTOM_RIGHT":
                map_x = safe_x_max - map_w - x_margin
                map_y = safe_y_min + y_margin
            case "BOTTOM_LEFT":
                map_x = safe_x_min + x_margin
                map_y = safe_y_min + y_margin
    elif corner in BORDER_POSITIONS:
        map_x, map_y = _dock_to_border(
            corner, safe_x_min, safe_y_min, safe_x_max, safe_y_max, map_w, map_h, x_margin, y_margin, margin
        )
    else:
        # FREE: apply the persisted offset, then clamp the origin so the map
        # stops cleanly at the safe insets instead of being squeezed past the
        # borders and eventually disappearing.
        map_x = safe_x_min + x_margin + settings.offset_x * ui_scale
        map_y = safe_y_min + y_margin + settings.offset_y * ui_scale
        map_x, map_y = clamp_free_rect(
            map_x, map_y, map_w, map_h, (safe_x_min, safe_y_min, safe_x_max, safe_y_max), x_margin, y_margin, margin
        )
        return map_x, map_y, map_w, map_h, padding, y_margin

    # General edge-aware clamp into safe bounds so a docked minimap never
    # spills outside the drawable region or under the header edge.
    docks_bottom = corner in ("BOTTOM_RIGHT", "BOTTOM_LEFT", "BOTTOM_BORDER")
    top_margin = margin if docks_bottom else y_margin
    bottom_margin = y_margin if docks_bottom else margin
    map_x = max(map_x, float(safe_x_min) + x_margin)
    map_w = min(map_w, float(safe_x_max) - map_x - x_margin)
    map_y = max(map_y, float(safe_y_min) + bottom_margin)
    map_h = min(map_h, float(safe_y_max) - map_y - top_margin)

    # Only bail if the minimap would be too small to be useful
    min_dim_width = MIN_MAP_WIDTH * ui_scale
    min_dim_height = MIN_MAP_HEIGHT * ui_scale
    if map_w < min_dim_width or map_h < min_dim_height:
        state.view.rect = (0.0, 0.0, 0.0, 0.0)
        return None

    return map_x, map_y, map_w, map_h, padding, y_margin


def _draw_background(
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    colors: dict,
    master_alpha: float,
    snapped: bool = False,
    moving: bool = False,
) -> tuple[tuple[float, float, float, float], float]:
    """Draw the minimap backdrop rounded rect and border."""

    bg_color = _alpha_mul(colors["bg"], master_alpha)
    panel_roundness = colors.get("panel_roundness", 4.0)
    shadow_offset = 1
    border_color = _alpha_mul(colors["indicator"] if snapped else colors["bg_border"], master_alpha)
    if moving and not snapped:
        border_color = (1.0, 1.0, 1.0, 0.05)
    border_width = 1.0 if moving else 0.5

    _draw_filled_rounded_rect(map_x, map_y, map_w, map_h, panel_roundness * 1.2, bg_color)
    _draw_rounded_rect_border(
        map_x - shadow_offset,
        map_y - shadow_offset,
        map_w + shadow_offset * 2,
        map_h + shadow_offset * 2,
        panel_roundness,
        (0, 0, 0, 0.15 * master_alpha),
        0.5,
    )
    _draw_rounded_rect_border(map_x, map_y, map_w, map_h, panel_roundness, border_color, border_width)

    return bg_color, panel_roundness


def _setup_scissor(
    map_x: float, map_y: float, map_w: float, map_h: float
) -> tuple[bool, bool, tuple[int, int, int, int]]:
    """Enable scissor test to clip content to minimap interior.

    Return ``(success, was_active, old_rect)`` for restoring later.
    """
    scissor_saved = (False, (0, 0, 0, 0))
    try:
        scissor_was_active = gpu.state.scissor_test_get()
        scissor_saved = (scissor_was_active, gpu.state.scissor_get() if scissor_was_active else (0, 0, 0, 0))
    except Exception:
        pass

    try:
        # Set rect first — scissor_set marks framebuffer dirty on OpenGL,
        # ensuring the subsequent scissor_test_set flush takes effect.
        gpu.state.scissor_set(int(map_x + 1), int(map_y + 1), int(map_w - 2), int(map_h - 2))
        gpu.state.scissor_test_set(True)
        scissor_was_active, scissor_old_rect = scissor_saved
        return True, scissor_was_active, scissor_old_rect
    except Exception:
        return False, False, (0, 0, 0, 0)


def _teardown_scissor(saved_state: tuple[bool, bool, tuple[int, int, int, int]]) -> None:
    """Restore scissor test to its original state before _setup_scissor.

    Workaround for Blender bugs #113310 / #139646: scissor_set marks the
    framebuffer dirty — call it *before* scissor_test_set so the state
    flush actually reaches the GL driver on OpenGL.
    """
    scissor_success, scissor_was_active, scissor_old_rect = saved_state
    if not scissor_success:
        return
    try:
        if scissor_was_active:
            gpu.state.scissor_set(
                int(scissor_old_rect[0]), int(scissor_old_rect[1]), int(scissor_old_rect[2]), int(scissor_old_rect[3])
            )
            gpu.state.scissor_test_set(True)
        else:
            gpu.state.scissor_set(0, 0, 65535, 65535)
            gpu.state.scissor_test_set(False)
    except Exception:
        try:
            gpu.state.scissor_set(0, 0, 65535, 65535)
            gpu.state.scissor_test_set(False)
        except Exception:
            pass


def _draw_resize_handles(
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    state: MinimapState,
) -> None:
    """Draw full-edge resize indicators, colored orange when the percentage cap is active."""
    resize_handle = state.interaction.resize_active
    if not resize_handle:
        return

    width_clamped = state.view.width_clamped
    height_clamped = state.view.height_clamped

    color_base = _alpha_mul(colors["text"], 0.5 * master_alpha)
    color_warn = _alpha_mul(colors["indicator"], master_alpha)
    handle_thickness = 3.0 * ui_scale
    handle_margin = 6 * ui_scale

    if resize_handle == ResizeHandle.LIST:
        zone_rect = state.list.list_zone_rect
        if not zone_rect or not state.view.rect:
            return
        zone_y, zone_height = zone_rect[1], zone_rect[3]
        # Derive the divider x from the live zone width so the pill tracks
        # per-pixel during a drag instead of lagging one frame behind.
        map_left = state.view.rect[0]
        divider_x = (
            map_left
            + state.view.inner_padding
            + state.list.list_width
            - 2.0 * ui_scale
            + 3.0 * ui_scale
            - handle_thickness / 2.0
        )
        # Clamp to zone vertical extent with small margin so pill stays inside.
        zone_margin = 2 * ui_scale
        divider_color = color_warn if state.list.width_clamped else color_base
        _draw_pill(
            divider_x,
            zone_y + zone_margin,
            handle_thickness,
            max(zone_height - 2 * zone_margin, 1.0),
            divider_color,
        )
        return

    width_side = None
    if resize_handle in (ResizeHandle.LEFT, ResizeHandle.TOP_LEFT, ResizeHandle.BOTTOM_LEFT):
        width_side = "left"
    elif resize_handle in (ResizeHandle.RIGHT, ResizeHandle.TOP_RIGHT, ResizeHandle.BOTTOM_RIGHT):
        width_side = "right"
    height_side = None
    if resize_handle in (ResizeHandle.TOP, ResizeHandle.TOP_LEFT, ResizeHandle.TOP_RIGHT):
        height_side = "top"
    elif resize_handle in (ResizeHandle.BOTTOM, ResizeHandle.BOTTOM_LEFT, ResizeHandle.BOTTOM_RIGHT):
        height_side = "bottom"

    if width_side == "left":
        pill_x = map_x + 2 * ui_scale
        _draw_pill(
            pill_x,
            map_y + handle_margin,
            handle_thickness,
            map_h - 2 * handle_margin,
            color_warn if width_clamped else color_base,
        )
    elif width_side == "right":
        pill_x = map_x + map_w - 2 * ui_scale - handle_thickness
        _draw_pill(
            pill_x,
            map_y + handle_margin,
            handle_thickness,
            map_h - 2 * handle_margin,
            color_warn if width_clamped else color_base,
        )

    if height_side == "top":
        pill_y = map_y + map_h - 2 * ui_scale - handle_thickness
        _draw_pill(
            map_x + handle_margin,
            pill_y,
            map_w - 2 * handle_margin,
            handle_thickness,
            color_warn if height_clamped else color_base,
        )
    elif height_side == "bottom":
        pill_y = map_y + 2 * ui_scale
        _draw_pill(
            map_x + handle_margin,
            pill_y,
            map_w - 2 * handle_margin,
            handle_thickness,
            color_warn if height_clamped else color_base,
        )


def _draw_view_fill(
    settings,
    space,
    region,
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    map_anchor_x: float,
    map_anchor_y: float,
    scale: float,
    tree_center_x: float,
    tree_center_y: float,
    colors: dict,
    panel_roundness: float,
    master_alpha: float,
    ui_scale: float,
    visible: tuple[float, float, float, float] | None = None,
) -> None:
    """Draw a filled rect over the active view region, behind nodes and wires."""
    if visible is None:
        visible = _get_visible_rect(space, region)
    if not visible:
        return

    view_x = round(map_anchor_x + (visible[0] - tree_center_x) * scale)
    view_y = round(map_anchor_y + (visible[1] - tree_center_y) * scale)
    view_w = round(max((visible[2] - visible[0]) * scale, 1.0))
    view_h = round(max((visible[3] - visible[1]) * scale, 1.0))

    view_left = max(view_x, map_x)
    view_bottom = max(view_y, map_y)
    view_right = min(view_x + view_w, map_x + map_w)
    view_top = min(view_y + view_h, map_y + map_h)
    hole_width = view_right - view_left
    hole_height = view_top - view_bottom
    if hole_width <= 0 or hole_height <= 0:
        return

    fill_color = colors["node_active"]
    if settings.viewport_fill_rect:
        fill_color = settings.viewport_fill_color
    fill_color = _alpha_mul(fill_color, 0.2 * master_alpha)
    node_roundness = colors.get("node_roundness", 2.0) * ui_scale
    _draw_filled_rounded_rect_clipped(
        view_left,
        view_bottom,
        hole_width,
        hole_height,
        node_roundness,
        fill_color,
        map_x,
        map_y,
        map_w,
        map_h,
        panel_roundness * 1.2,
    )


def _draw_viewport_overlay(
    settings,
    space,
    region,
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    map_anchor_x: float,
    map_anchor_y: float,
    scale: float,
    tree_center_x: float,
    tree_center_y: float,
    colors: dict,
    master_alpha: float,
    panel_roundness: float,
    ui_scale: float,
    scissor_active: bool,
    state: MinimapState | None = None,
    visible: tuple[float, float, float, float] | None = None,
) -> None:
    """Draw the viewport rect outline and optional darkened overlay."""
    if visible is None:
        visible = _get_visible_rect(space, region)
    if not visible:
        return

    view_x = round(map_anchor_x + (visible[0] - tree_center_x) * scale)
    view_y = round(map_anchor_y + (visible[1] - tree_center_y) * scale)
    view_w = round(max((visible[2] - visible[0]) * scale, 1.0))
    view_h = round(max((visible[3] - visible[1]) * scale, 1.0))

    view_left = max(view_x, map_x)
    view_bottom = max(view_y, map_y)
    view_right = min(view_x + view_w, map_x + map_w)
    view_top = min(view_y + view_h, map_y + map_h)

    node_roundness = colors.get("node_roundness", 2.0) * ui_scale
    hole_width = view_right - view_left
    hole_height = view_top - view_bottom

    # Darkened overlay
    if settings.show_viewport_overlay:
        overlay_color = settings.viewport_overlay_color
        overlay = _alpha_mul(overlay_color, master_alpha)

        scissor_temporarily_disabled = scissor_active
        if scissor_temporarily_disabled:
            gpu.state.scissor_test_set(False)

        try:
            if hole_width > 0 and hole_height > 0:
                _draw_filled_rounded_rect_with_hole(
                    map_x,
                    map_y,
                    map_w,
                    map_h,
                    panel_roundness,
                    view_left,
                    view_bottom,
                    hole_width,
                    hole_height,
                    0,
                    overlay,
                )
            else:
                _draw_filled_rounded_rect(map_x, map_y, map_w, map_h, panel_roundness, overlay)
        finally:
            if scissor_temporarily_disabled:
                gpu.state.scissor_test_set(True)

    # Outline the viewport extent when it overlaps the minimap
    if hole_width > 0 and hole_height > 0:
        outline_color = colors["node_active"]
        if settings.viewport_fill_rect:
            outline_color = settings.viewport_fill_color
        border_width = 0.5 * ui_scale
        _draw_rounded_rect_border(
            view_x, view_y, view_w, view_h, node_roundness, _alpha_mul(outline_color, master_alpha), border_width
        )


def _draw_node_count(
    settings,
    node_count: int,
    state: MinimapState,
    map_x: float,
    map_y: float,
    map_w: float,
    padding: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> None:
    """Draw the node count text beside the list toggle at the top-left."""
    if not settings.show_node_count:
        return

    info_text = str(node_count)
    font_id = 0
    font_size = int(FONT_SIZE * ui_scale)
    blf.size(font_id, font_size)
    text_w, _ = blf.dimensions(font_id, info_text)
    text_x = round(map_x + map_w - text_w - padding)
    text_y = round(map_y + padding)

    text_color = _alpha_mul(colors["text"], 0.85 * master_alpha)

    _draw_text_with_shadow(font_id, info_text, text_x, text_y, text_color, font_size, settings.show_text_shadow)


def _paint_frame_all_icon(x: float, y: float, size: float, color, ui_scale: float) -> None:
    """Draw the frame-all corner brackets icon."""
    inset = 5 * ui_scale
    stroke_thickness = max(1, int(1.5 * ui_scale))
    arm_length = size * 0.15

    # Top-left bracket
    _draw_filled_rounded_rect(x + inset, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color)
    _draw_filled_rounded_rect(x + inset, y + inset, stroke_thickness, arm_length, stroke_thickness * 0.5, color)
    # Top-right bracket
    _draw_filled_rounded_rect(
        x + size - inset - arm_length, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color
    )
    _draw_filled_rounded_rect(
        x + size - inset - stroke_thickness, y + inset, stroke_thickness, arm_length, stroke_thickness * 0.5, color
    )
    # Bottom-left bracket
    _draw_filled_rounded_rect(
        x + inset, y + size - inset - stroke_thickness, arm_length, stroke_thickness, stroke_thickness * 0.5, color
    )
    _draw_filled_rounded_rect(
        x + inset, y + size - inset - arm_length, stroke_thickness, arm_length, stroke_thickness * 0.5, color
    )
    # Bottom-right bracket
    _draw_filled_rounded_rect(
        x + size - inset - arm_length,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
    )
    _draw_filled_rounded_rect(
        x + size - inset - stroke_thickness,
        y + size - inset - arm_length,
        stroke_thickness,
        arm_length,
        stroke_thickness * 0.5,
        color,
    )


def _paint_frame_view_icon(x: float, y: float, size: float, color, ui_scale: float) -> None:
    """Draw the frame-view viewport rectangle icon."""
    inset = 5 * ui_scale
    border_thickness = max(4, int(4.0 * ui_scale))
    _draw_rounded_rect_border(
        round(x + inset),
        round(y + inset),
        round(size - 2 * inset),
        round(size - 2 * inset),
        border_thickness,
        color,
        0.5 * ui_scale,
    )


def _paint_frame_selected_icon(x: float, y: float, size: float, color, ui_scale: float) -> None:
    """Draw the frame-selected rails and center box icon."""
    inset = 5 * ui_scale
    stroke_thickness = max(1, int(1.5 * ui_scale))
    arm_length = size * 0.15

    # Left/right rails connecting top and bottom corners
    _draw_filled_rounded_rect(x + inset, y + inset, stroke_thickness, size - 2 * inset, stroke_thickness * 0.5, color)
    _draw_filled_rounded_rect(
        x + size - inset - stroke_thickness,
        y + inset,
        stroke_thickness,
        size - 2 * inset,
        stroke_thickness * 0.5,
        color,
    )
    # Corner arms
    _draw_filled_rounded_rect(x + inset, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color)
    _draw_filled_rounded_rect(
        x + size - inset - arm_length, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color
    )
    _draw_filled_rounded_rect(
        x + inset, y + size - inset - stroke_thickness, arm_length, stroke_thickness, stroke_thickness * 0.5, color
    )
    _draw_filled_rounded_rect(
        x + size - inset - arm_length,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
    )

    # Center box
    center_box_w = center_box_h = 2 * ui_scale
    center_box_x = x + (size - center_box_w) / 2
    center_box_y = y + (size - center_box_h) / 2
    _draw_filled_rounded_rect(center_box_x, center_box_y, center_box_w, center_box_h, 1.5 * ui_scale, color)


def _paint_list_toggle_icon(x: float, y: float, size: float, color, ui_scale: float, active: bool = False) -> None:
    """Draw the list-toggle icon: three horizontal bars, or an X when active."""
    stroke_thickness = max(1, int(1.5 * ui_scale))
    if not active:
        bar_width = size * 0.5
        bar_gap = 2.0 * ui_scale
        bar_x = x + (size - bar_width) / 2
        bar_y = y + (size - (3 * stroke_thickness + 2 * bar_gap)) / 2 - 0.5

        for bar_index in range(3):
            _draw_filled_rounded_rect(
                bar_x,
                bar_y + bar_index * (stroke_thickness + bar_gap),
                bar_width,
                stroke_thickness,
                stroke_thickness * 0.5,
                color,
            )
        return

    # Active state: an X crossing two diagonal rounded bars about the center.
    arm_length = size * 0.25
    center_x = x + size / 2
    center_y = y + size / 2

    gpu.matrix.push()
    try:
        gpu.matrix.translate((center_x, center_y))
        for rotation_sign in (-1, 1):
            gpu.matrix.push()
            gpu.matrix.multiply_matrix(Matrix.Rotation(math.radians(rotation_sign * 45.0), 4, "Z"))
            _draw_filled_rounded_rect(
                -arm_length, -stroke_thickness / 2.0, 2 * arm_length, stroke_thickness, stroke_thickness / 2.0, color
            )
            gpu.matrix.pop()
    finally:
        gpu.matrix.pop()


_BUTTON_ICONS = {
    "ALL": _paint_frame_all_icon,
    "VIEW": _paint_frame_view_icon,
    "SELECTED": _paint_frame_selected_icon,
    "LIST": _paint_list_toggle_icon,
}


def _paint_grip_icon(x: float, y: float, size: float, color, ui_scale: float) -> None:
    """Draw a move grip icon: two rows of four dots (like a drag handle)."""
    dot_size = 1.0 * ui_scale
    gap = 2.0 * ui_scale
    column_gap = 2.0 * ui_scale
    group_w = 4 * dot_size + 3 * column_gap
    group_h = 2 * dot_size + gap
    group_x = round(x + (size - group_w) / 2)
    group_y = round(y + (size - group_h) / 2)
    for column in range(4):
        for row in range(2):
            _draw_filled_rounded_rect(
                round(group_x + column * (dot_size + column_gap)),
                round(group_y + row * (dot_size + gap)),
                dot_size,
                dot_size,
                dot_size / 2,
                color,
            )


def _get_visible_minimap_buttons(settings) -> list[str]:
    """Return ids of enabled minimap buttons in draw order."""
    if not settings or not settings.interactive:
        return []
    visible = [button_id for button_id, pref_attr in _MINIMAP_BUTTONS if getattr(settings, pref_attr, True)]
    # Frame Selected is meaningless with Follow View (the viewport drives framing).
    if settings.follow_view:
        visible = [button_id for button_id in visible if button_id != "SELECTED"]
    return visible


def _layout_minimap_buttons(
    state: MinimapState,
    visible_button_ids: list[str],
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    padding: float,
    ui_scale: float,
    settings=None,
    content_count: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """Return hit-rect origins {id: (x, y, size)} for every visible button.

    The top edge hosts the whole chrome: the list toggle at the left with the
    node count text beside it, and the frame buttons (ALL/VIEW/SELECTED)
    leading into the move-grip drag handle at the right. When the move button
    is disabled the frame row extends to the right padding edge instead.
    Frame buttons are culled progressively (SELECTED → VIEW → ALL) when the
    row would spill past the left padding or collide with the list toggle /
    node count text.
    """
    button_size = BUTTON_SIZE * ui_scale
    button_margin = BUTTON_MARGIN * ui_scale
    gap = padding
    top_y = round(map_y + map_h - padding - button_margin - button_size)

    # Move-grip drag handle: rightmost item of the top row when enabled.
    drag_size = BUTTON_SIZE * ui_scale
    drag_x = round(map_x + map_w - padding - button_margin - drag_size)
    show_move_btn = bool(getattr(settings, "show_move_button", True))

    # Frame buttons (ALL/VIEW/SELECTED) sit left of the drag handle, or flush
    # to the right padding edge when the move button is hidden.
    frame_button_ids = [bid for bid in visible_button_ids if bid != "LIST"]
    row_right_x = round(drag_x - gap - button_size) if show_move_btn else drag_x

    # List toggle at the top-left, sliding right of an open type-list zone.
    list_button_x: float | None = None
    if "LIST" in visible_button_ids:
        list_button_x = round(map_x + padding + button_margin)
        if state.list.list_width > 0:
            list_button_x = max(list_button_x, round(_get_map_content_rect(state)[0] + button_margin))

    # Node count text follows the list toggle and is used as the left chrome
    # boundary when culling the frame row.
    count_width = 0.0
    node_count_anchor: tuple[float, float] | None = None
    if getattr(settings, "show_node_count", False):
        font_id = 0
        font_size = int(FONT_SIZE * ui_scale)
        blf.size(font_id, font_size)
        count_width, count_height = blf.dimensions(font_id, str(content_count))
        count_x = (
            (list_button_x if list_button_x is not None else round(map_x + padding + button_margin)) + button_size + gap
        )
        node_count_anchor = (round(count_x), round(top_y + (button_size - count_height) / 2))
    state.buttons.node_count_anchor = node_count_anchor

    # Avoid overlap between the frame row and the left chrome, and overflow of
    # the row outside the minimap. When space is tight (small map_w), hide
    # frame buttons progressively: SELECTED → VIEW → ALL.
    hide_priority = ["SELECTED", "VIEW", "ALL"]
    # Work on a copy so we can cull without affecting visible_button_ids order.
    culled_frame_button_ids = list(frame_button_ids)
    while culled_frame_button_ids:
        frame_button_count = len(culled_frame_button_ids)
        row_left_x = row_right_x - (frame_button_count - 1) * button_size if frame_button_count else row_right_x
        # 1) row would overflow left padding
        row_overflows_left = row_left_x < (map_x + padding)
        # 2) row would overlap the node count text / list toggle (with clearance)
        if node_count_anchor is not None:
            left_chrome_right = node_count_anchor[0] + count_width
        elif list_button_x is not None:
            left_chrome_right = list_button_x + button_size
        else:
            left_chrome_right = None
        row_overlaps_left_chrome = left_chrome_right is not None and row_left_x < left_chrome_right + gap
        if not row_overflows_left and not row_overlaps_left_chrome:
            break
        # Hide next priority button that is still visible.
        to_hide: str | None = None
        for cand in hide_priority:
            if cand in culled_frame_button_ids:
                to_hide = cand
                break
        if to_hide is None:
            # Fallback: hide leftmost (first in current order).
            to_hide = culled_frame_button_ids[0]
        culled_frame_button_ids = [bid for bid in culled_frame_button_ids if bid != to_hide]

    rects: dict[str, tuple[float, float, float]] = {}
    frame_button_count = len(culled_frame_button_ids)
    for button_index, button_id in enumerate(culled_frame_button_ids):
        x = round(row_right_x - (frame_button_count - 1 - button_index) * button_size)
        rects[button_id] = (x, top_y, button_size)

    if list_button_x is not None:
        rects["LIST"] = (list_button_x, top_y, button_size)

    if show_move_btn:
        rects["DRAG"] = (drag_x, top_y, drag_size)

    return rects


def _draw_minimap_buttons(map_x, map_y, map_w, map_h, padding, colors, ui_scale, master_alpha, content_count=0):
    """Draw the interactive minimap buttons and record their hit rects."""
    addon = get_addon_preferences()
    settings = addon.settings if addon else None
    state = _state()
    state.buttons.rects.clear()

    visible_button_ids = _get_visible_minimap_buttons(settings)
    rects = _layout_minimap_buttons(
        state, visible_button_ids, map_x, map_y, map_w, map_h, padding, ui_scale, settings, content_count
    )
    radius = colors["node_roundness"] * ui_scale
    bg_color = _alpha_mul(colors["bg"], master_alpha)
    border_color = _alpha_mul(colors["bg_border"], master_alpha)

    # Move-grip drag handle is available whenever interactive mode is on, which
    # is the mode that enables repositioning the map.
    if "DRAG" in rects and settings and settings.interactive:
        drag_x, drag_y, drag_size = rects["DRAG"]
        drag_h = BUTTON_SIZE * ui_scale
        _draw_filled_rounded_rect(drag_x, drag_y, drag_size, drag_h, radius, bg_color)
        _draw_rounded_rect_border(drag_x, drag_y, drag_size, drag_h, radius, border_color, 0.5)
        is_moving = state.view.moving
        is_hovered = state.buttons.hovered_button_id == "DRAG"
        if is_moving or is_hovered:
            hover_color = _alpha_mul(colors["text"], BUTTON_HOVER_ALPHA * master_alpha)
            _draw_filled_rounded_rect(
                drag_x + 1, drag_y + 1, drag_size - 2, drag_h - 2, max(2.0, radius - 1), hover_color
            )
        icon_color = (
            _alpha_mul(colors["text"], master_alpha) if is_moving else _alpha_mul(colors["text"], master_alpha * 0.7)
        )
        _paint_grip_icon(drag_x, drag_y, drag_h, icon_color, ui_scale)
        state.buttons.rects["DRAG"] = (drag_x, drag_y, drag_size, drag_h)

    if not visible_button_ids:
        return

    # Frame buttons are drawn as a horizontal row when two or more are shown,
    # each as its own box sharing square inner corners (Blender align style);
    # the list toggle stays standalone.
    # Use culled rects for the frame row (some may have been hidden to
    # avoid overlap with LIST on narrow minimaps).
    frame_button_ids = [bid for bid in visible_button_ids if bid != "LIST" and bid in rects]
    # Fallback: derive from rects if culling removed entries not in visible_button_ids
    if not frame_button_ids:
        frame_button_ids = [bid for bid in rects if bid not in ("LIST", "DRAG")]
    is_combined = len(frame_button_ids) >= 2
    # Sort by x to find the leftmost (first) and rightmost (last) buttons,
    # which carry the only rounded corners of the row.
    frame_buttons_ordered = sorted(frame_button_ids, key=lambda bid: rects[bid][0])
    order_index = {bid: idx for idx, bid in enumerate(frame_buttons_ordered)}
    if is_combined:
        # Draw each frame button as its own box, edge-to-edge with no gap.
        # Only the external corners round, inner corners meet square; each
        # button's border is drawn on its own rect, so neighboring borders
        # coincide at the seam and every interior is equally inset.
        # Buttons that have a left neighbor skip their left border stroke:
        # two coincident strokes would stack into a heavy 2px seam, so the
        # seam line is emitted once by the neighbor's right border only.
        _, _, button_size = rects[frame_buttons_ordered[0]]
        for button_index, button_id in enumerate(frame_buttons_ordered):
            button_x, button_y, _ = rects[button_id]
            if button_index == 0:
                radii = (radius, 0.0, 0.0, radius)
            elif button_index == len(frame_buttons_ordered) - 1:
                radii = (0.0, radius, radius, 0.0)
            else:
                radii = (0.0, 0.0, 0.0, 0.0)
            _draw_filled_rounded_rect_varying(button_x, button_y, button_size, button_size, radii, bg_color)
            _draw_rounded_rect_border_varying_sides(
                button_x,
                button_y,
                button_size,
                button_size,
                radii,
                border_color,
                0.5,
                skip_left=button_index > 0,
            )

    for button_id, _pref_attr in _MINIMAP_BUTTONS:
        if button_id not in rects:
            continue
        button_x, button_y, button_size = rects[button_id]
        if button_id == "LIST" or not is_combined:
            _draw_filled_rounded_rect(button_x, button_y, button_size, button_size, radius, bg_color)
            _draw_rounded_rect_border(button_x, button_y, button_size, button_size, radius, border_color, 0.5)
        is_hovered = state.buttons.hovered_button_id == button_id
        icon_color = _alpha_mul(colors["text"], master_alpha * 0.7)
        if is_hovered:
            hover_color = _alpha_mul(colors["text"], BUTTON_HOVER_ALPHA * master_alpha)
            if is_combined and button_id != "LIST":
                # Per-corner radii (top-left, top-right, bottom-right, bottom-left):
                # only the row's external corners round, inner corners stay square.
                hr = max(2.0, radius - 1)
                button_index = order_index.get(button_id, -1)
                is_first = button_index == 0
                is_last = button_index == len(frame_buttons_ordered) - 1
                if is_first:
                    hover_radii = (hr, 0.0, 0.0, hr)
                    hover_x = button_x + 1
                    hover_width = button_size - 1
                elif is_last:
                    hover_radii = (0.0, hr, hr, 0.0)
                    hover_x = button_x
                    hover_width = button_size - 1
                else:
                    hover_radii = (0.0, 0.0, 0.0, 0.0)
                    hover_x = button_x
                    hover_width = button_size
                _draw_filled_rounded_rect_varying(
                    hover_x, button_y + 1, hover_width, button_size - 2, hover_radii, hover_color
                )
            else:
                _draw_filled_rounded_rect(
                    button_x + 1, button_y + 1, button_size - 2, button_size - 2, max(2.0, radius - 1), hover_color
                )
            icon_color = _alpha_mul(colors["text"], master_alpha)
        if button_id == "LIST":
            _paint_list_toggle_icon(
                button_x, button_y, button_size, icon_color, ui_scale, active=bool(settings and settings.show_type_list)
            )
        else:
            _BUTTON_ICONS[button_id](button_x, button_y, button_size, icon_color, ui_scale)
        state.buttons.rects[button_id] = (button_x, button_y, button_size, button_size)


def draw_minimap() -> None:
    """Orchestrate minimap drawing in the Node Editor."""
    context = bpy.context
    space = context.space_data
    region = context.region

    state = _state()
    if _early_exit(context, space, state):
        show_overlays = space.overlay.show_overlays if space else "?"
        enabled = state.enabled
        logger.debug("draw_minimap: early exit (type=%s overlays=%s enabled=%s)", space.type, show_overlays, enabled)
        return

    prefs = get_addon_preferences(context)
    if prefs is None:
        return
    settings = prefs.settings

    # Defer auto-launch until registration is fully complete
    # to avoid invoking the modal with a stale context.
    if not _registration_state["done"]:
        logger.debug("draw_minimap: registration not done, skipping auto-launch")
    else:
        # Auto-start modal operator for pan/zoom interaction (one per window)
        win = context.window
        window_ptr = win.as_pointer() if win else 0
        has_modal = window_ptr in _minimap_window_operators if win else False

        if settings.interactive:
            if win and not has_modal:
                logger.debug("draw_minimap: invoking nodemap.navigate for window %d", window_ptr)
                try:
                    bpy.ops.nodemap.navigate("INVOKE_DEFAULT")
                    logger.debug("draw_minimap: nodemap.navigate invoked successfully")
                except RuntimeError as e:
                    logger.debug("draw_minimap: nodemap.navigate failed: %s", e)
            elif not win:
                logger.debug("draw_minimap: cannot invoke — context.window is None")

    node_tree = space.edit_tree
    if not node_tree or not node_tree.nodes or len(node_tree.nodes) == 0:
        return

    # Cache the editor viewport rect once for this frame; reused by the
    # transform/clamp logic and the viewport overlay draws below.
    visible = _get_visible_rect(space, region)

    show_borders = settings.show_node_outline
    include_selection = show_borders or settings.highlight_selected_wires
    current_fingerprint, raw_bounds, content_count = _get_tree_snapshot(node_tree, include_selection)
    if raw_bounds[2] - raw_bounds[0] <= 0 or raw_bounds[3] - raw_bounds[1] <= 0:
        return

    logger.trace(
        "SETTINGS %d nodes | show_wires=%d show_node_labels=%d compact_labels=%d"
        " show_node_colors=%d socket_indicators=%d wire_color=%d frame_labels=%d"
        " show_reroutes=%d",
        current_fingerprint[0],
        settings.show_wires,
        settings.show_node_labels,
        settings.compact_node_labels,
        settings.show_node_colors,
        settings.show_socket_indicators,
        settings.show_wire_color,
        settings.show_frame_labels,
        getattr(settings, "show_reroutes", True),
    )

    ui_scale = _get_ui_scale()
    colors = _get_node_editor_theme_colors()
    master_alpha = settings.opacity
    corner = settings.position

    rect = _compute_minimap_rect(settings, ui_scale, space, region, corner, state)
    if rect is None:
        return
    map_x, map_y, map_w, map_h, padding, y_margin = rect

    bounds = _expand_bounds_margin(raw_bounds, ui_scale, map_h, padding)

    state.view.rect = (map_x, map_y, map_w, map_h)
    state.view.tree_bounds = bounds
    state.view.outer_margin = y_margin
    state.view.inner_padding = padding

    # Per-tree view persistence: reset pan/zoom when switching node trees,
    # but restore the saved view when revisiting the same tree.
    try:
        tree_ptr = node_tree.as_pointer() if node_tree else None
    except ReferenceError:
        tree_ptr = None
    if tree_ptr is not None:
        if state.last_tree_ptr is None:
            saved = state.tree_views.get(tree_ptr)
            if saved is not None:
                sz, spx, spy = saved
                state.view.user_zoom = sz
                state.view.anchor_zoom = sz
                state.view.pan = (spx, spy)
            state.last_tree_ptr = tree_ptr
        elif state.last_tree_ptr != tree_ptr:
            # Save view for tree being left.
            state.tree_views[state.last_tree_ptr] = (
                state.view.user_zoom,
                state.view.pan[0],
                state.view.pan[1],
            )
            saved = state.tree_views.get(tree_ptr)
            if saved is not None:
                sz, spx, spy = saved
                state.view.user_zoom = sz
                state.view.anchor_zoom = sz
                state.view.pan = (spx, spy)
            else:
                # No saved view — reset to frame-all for the new tree.
                from ..geo.framing import _compute_frame_all_targets

                area_ptr = None
                try:
                    area_ptr = bpy.context.area.as_pointer()
                except (AttributeError, ReferenceError):
                    pass
                targets = _compute_frame_all_targets(space, region, area_ptr)
                if targets is not None:
                    sz, spx, spy = targets
                    state.view.anchor_zoom = sz
                    state.view.user_zoom = sz
                    state.view.pan = (spx, spy)
            state.last_tree_ptr = tree_ptr

    # Reserve the type-list zone before computing the map transform so
    # node framing and panning never place tree content behind the list.
    with _Timer("type_list_width"):
        _step_list_width(state, settings, map_w, ui_scale)

    _clamp_pan_to_viewport(space, region, state, visible)

    # Refresh tree data: pure position changes (node drags) patch the cached
    # tables immediately; anything else schedules a debounced full compile.
    old_fingerprint = state.cache.fingerprint
    if old_fingerprint != current_fingerprint:
        move_only = _is_move_only_diff(old_fingerprint, current_fingerprint)
        applied = False
        if move_only and (time.perf_counter() - state.cache.last_move_refresh) >= _MOVE_REFRESH_MIN_INTERVAL:
            applied = _apply_move_updates(state, node_tree)
            if applied:
                state.cache.last_move_refresh = time.perf_counter()
                state.cache.pending_settle_flush = True
        if applied:
            state.cache.fingerprint = current_fingerprint
        # Always keep a settle timer armed: it flushes frozen wire/marker
        # batches (forced via pending_settle_flush) or runs the pending full
        # compile. Re-arm (push back) only when the fingerprint changed again
        # since arming; identical-fingerprint redraw streams (list animation,
        # hover) must leave the live timer alone so the settle event cannot be
        # starved by continuous redraws.
        delay = settings.debounce_delay
        now = time.perf_counter()
        if state.cache.pending_timer is not None and state.cache.pending_fingerprint != current_fingerprint:
            if now < state.cache.pending_timer_deadline:
                try:
                    bpy.app.timers.unregister(state.cache.pending_timer)
                except ValueError:
                    pass
                state.cache.pending_timer = None
        if state.cache.pending_timer is None:

            def _settle_fire():
                return _debounced_compile(state, node_tree, colors, settings, master_alpha, ui_scale)

            # An expanding type list needs compiled type stats to measure its
            # target width; compile immediately instead of after the debounce.
            # List click actions also request an immediate compile so the visual
            # feedback is not delayed by the debounce interval.
            immediate = (state.list.anim_active and state.list.anim_target < 0) or state.cache.force_immediate
            interval = 0.0 if immediate else delay
            bpy.app.timers.register(_settle_fire, first_interval=interval)
            state.cache.pending_timer = _settle_fire
            state.cache.pending_timer_deadline = now + delay
            state.cache.pending_fingerprint = current_fingerprint
            state.cache.force_immediate = False

    # Build screen-space batches (cached; applies current zoom/pan via matrix)
    # When a structural preference changed, _batches_dirty forces a batch
    # rebuild using the existing tree_data (which still reflects the old
    # settings). The debounce timer will recompile tree_data on the next
    # event-loop iteration, producing a second rebuild with fresh data.
    if state.cache._batches_dirty:
        state.cache._batches_dirty = False
        state.cache.position_version += 1
    map_anchor_x, map_anchor_y, scale, tree_center_x, tree_center_y = _get_minimap_transform(
        state, space, region, visible
    )
    state.view.map_scale = scale
    highlight_border = (
        _alpha_mul(colors["node_active"], 0.3)
        if (state.list.hovered_type_label or state.interaction.hovered_node_id)
        else None
    )
    wire_curvature = _get_wire_curvature(settings)
    wire_thickness = settings.wire_thickness
    _ensure_minimap_batches(
        state,
        map_x,
        map_y,
        map_w,
        map_h,
        map_anchor_x,
        map_anchor_y,
        scale,
        tree_center_x,
        tree_center_y,
        ui_scale,
        master_alpha,
        show_borders,
        highlight_border,
        wire_curvature,
        wire_thickness,
    )

    try:
        original_blend = gpu.state.blend_get()
    except Exception:
        original_blend = None
    gpu.state.blend_set("ALPHA")

    bg_color, panel_roundness = _draw_background(
        map_x, map_y, map_w, map_h, colors, master_alpha, state.view.snapped, state.view.moving
    )

    scissor_state = _setup_scissor(map_x, map_y, map_w, map_h)
    scissor_was_active = scissor_state[0]

    _draw_view_fill(
        settings,
        space,
        region,
        map_x,
        map_y,
        map_w,
        map_h,
        map_anchor_x,
        map_anchor_y,
        scale,
        tree_center_x,
        tree_center_y,
        colors,
        panel_roundness,
        master_alpha,
        ui_scale,
        visible,
    )

    content_draw.draw_content_batches(
        state,
        settings,
        scale,
        tree_center_x,
        tree_center_y,
        map_anchor_x,
        map_anchor_y,
        master_alpha,
        wire_curvature,
    )

    _draw_viewport_overlay(
        settings,
        space,
        region,
        map_x,
        map_y,
        map_w,
        map_h,
        map_anchor_x,
        map_anchor_y,
        scale,
        tree_center_x,
        tree_center_y,
        colors,
        master_alpha,
        panel_roundness,
        ui_scale,
        scissor_was_active,
        state,
        visible=visible,
    )

    _draw_minimap_scrollbars(
        map_x,
        map_y,
        map_w,
        map_h,
        padding,
        map_anchor_x,
        map_anchor_y,
        scale,
        tree_center_x,
        tree_center_y,
        raw_bounds,
        colors,
        ui_scale,
        master_alpha,
    )

    _draw_minimap_buttons(map_x, map_y, map_w, map_h, padding, colors, ui_scale, master_alpha, content_count)

    _draw_resize_handles(map_x, map_y, map_w, map_h, colors, master_alpha, ui_scale, state)

    _draw_node_count(settings, content_count, state, map_x, map_y, map_w, padding, colors, master_alpha, ui_scale)

    # Persist current view for this tree so it can be restored when revisiting.
    try:
        current_ptr = node_tree.as_pointer() if node_tree else None
    except ReferenceError:
        current_ptr = None
    if current_ptr is not None:
        state.tree_views[current_ptr] = (state.view.user_zoom, state.view.pan[0], state.view.pan[1])

    _teardown_scissor(scissor_state)
    try:
        gpu.state.blend_set(original_blend if original_blend else "NONE")
    except Exception:
        gpu.state.blend_set("NONE")

    # Interactive node-type list zone (drawn unclipped, on top of map content)
    try:
        gpu.state.blend_set("ALPHA")
        with _Timer("draw_type_list"):
            _draw_type_list(settings, state, map_x, map_y, map_h, padding, colors, master_alpha, ui_scale)
    finally:
        try:
            gpu.state.blend_set(original_blend if original_blend else "NONE")
        except Exception:
            gpu.state.blend_set("NONE")
