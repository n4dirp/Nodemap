"""Provide minimap rendering in the Node Editor."""

import logging

# import math
import time
from typing import Any

import blf
import bpy
import gpu

# from mathutils import Matrix
from .. import __package__ as base_package
from ..core.constants import (
    BORDER_POSITIONS,
    BUTTON_HOVER_ALPHA,
    BUTTON_MARGIN,
    BUTTON_SIZE,
    CONTENT_PADDING,
    CORNER_POSITIONS,
    FONT_SIZE,
    MIN_MAP_HEIGHT,
    MIN_MAP_WIDTH,
    TYPE_LIST_LEFT_BUTTON_GAP,
    TYPE_LIST_TOP_BUTTON_GAP,
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
from ..core.list_filter import filter_matching_nodes
from ..core.state import (
    _MINIMAP_BUTTONS,
    MinimapState,
    ResizeHandle,
    _minimap_window_operators,
    _registration_state,
    _shared_tree_cache,
    _state,
    _unregister_pending_timer,
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
    _draw_dashes,
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
    _debounced_compile,
    _is_bounds_stable_diff,
    _is_move_only_diff,
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

    padding = CONTENT_PADDING * ui_scale

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
        # Enforce edge margins on the origin (clamp_free_rect collapses when
        # the map is wider than the safe area, pushing the origin outside the
        # left/bottom margin), then clamp dimensions to the remaining space.
        map_x = round(max(map_x, float(safe_x_min) + x_margin))
        map_w = round(min(map_w, float(safe_x_max) - map_x - x_margin))
        map_y = round(max(map_y, float(safe_y_min) + margin))
        map_h = round(min(map_h, float(safe_y_max) - map_y - y_margin))

        min_dim_width = MIN_MAP_WIDTH * ui_scale
        min_dim_height = MIN_MAP_HEIGHT * ui_scale
        if map_w < min_dim_width or map_h < min_dim_height:
            state.view.rect = (0.0, 0.0, 0.0, 0.0)
            return None

        return map_x, map_y, map_w, map_h, padding, y_margin

    # General edge-aware clamp into safe bounds so a docked minimap never
    # spills outside the drawable region or under the header edge.
    docks_bottom = corner in ("BOTTOM_RIGHT", "BOTTOM_LEFT", "BOTTOM_BORDER")
    top_margin = margin if docks_bottom else y_margin
    bottom_margin = y_margin if docks_bottom else margin
    map_x = round(max(map_x, float(safe_x_min) + x_margin))
    map_w = round(min(map_w, float(safe_x_max) - map_x - x_margin))
    map_y = round(max(map_y, float(safe_y_min) + bottom_margin))
    map_h = round(min(map_h, float(safe_y_max) - map_y - top_margin))

    # Only bail if the minimap would be too small to be useful
    min_dim_width = MIN_MAP_WIDTH * ui_scale
    min_dim_height = MIN_MAP_HEIGHT * ui_scale
    if map_w < min_dim_width or map_h < min_dim_height:
        state.view.rect = (0.0, 0.0, 0.0, 0.0)
        return None

    return map_x, map_y, map_w, map_h, padding, y_margin


def _snap_sides_for(position: str) -> frozenset[str]:
    """Return the map border sides facing the given dock *position*.

    Edges map to a single side, corners to the two adjacent sides, and free
    positioning to no sides.
    """
    corners = {
        "TOP_LEFT": frozenset({"top", "left"}),
        "TOP_RIGHT": frozenset({"top", "right"}),
        "BOTTOM_LEFT": frozenset({"bottom", "left"}),
        "BOTTOM_RIGHT": frozenset({"bottom", "right"}),
    }
    edges = {
        "TOP_BORDER": frozenset({"top"}),
        "BOTTOM_BORDER": frozenset({"bottom"}),
        "LEFT_BORDER": frozenset({"left"}),
        "RIGHT_BORDER": frozenset({"right"}),
    }
    return corners.get(position, edges.get(position, frozenset()))


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


def _draw_background(
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    mvp: Any = None,
) -> tuple[tuple[float, float, float, float], float]:
    """Draw the minimap backdrop rounded rect and border."""

    bg_color = _alpha_mul(colors["background"], master_alpha)
    panel_roundness = colors["panel_roundness"]
    shadow_offset = 1
    border_color = _alpha_mul(colors["background_border"], master_alpha)
    border_width = 0.5 * ui_scale

    _draw_filled_rounded_rect(map_x, map_y, map_w, map_h, panel_roundness * 1.2, bg_color, mvp=mvp)
    _draw_rounded_rect_border(
        map_x - shadow_offset,
        map_y - shadow_offset,
        map_w + shadow_offset * 2,
        map_h + shadow_offset * 2,
        panel_roundness,
        (0, 0, 0, 0.15 * master_alpha),
        border_width,
        mvp=mvp,
    )
    _draw_rounded_rect_border(map_x, map_y, map_w, map_h, panel_roundness, border_color, border_width, mvp=mvp)

    return panel_roundness


def _draw_moving_border(
    map_x,
    map_y,
    map_w,
    map_h,
    panel_roundness,
    colors,
    master_alpha,
    ui_scale,
    moving,
    snapped,
    hovered_minimap,
    position,
    mvp: Any = None,
):
    border_width = 1.5 * ui_scale
    border_color = _alpha_mul(colors["background_border"], master_alpha)

    if moving:
        _draw_filled_rounded_rect(
            map_x, map_y, map_w, map_h, panel_roundness * 1.2, (0, 0, 0, 0.4 * master_alpha), mvp=mvp
        )
        _draw_rounded_rect_border(map_x, map_y, map_w, map_h, panel_roundness, border_color, border_width, mvp=mvp)
    else:
        if not hovered_minimap:
            border_color = _alpha_mul(border_color, 0.2 * master_alpha)
        _draw_rounded_rect_border(map_x, map_y, map_w, map_h, panel_roundness, border_color, 0.5 * ui_scale, mvp=mvp)

    if snapped and (snap_sides := _snap_sides_for(position)):
        snap_color = _alpha_mul(colors["active_view_color"], master_alpha)
        side_colors = {side: snap_color for side in snap_sides}

        if not side_colors:
            return

        thickness = border_width
        margin = CONTENT_PADDING * ui_scale
        inset = 0.5 * ui_scale
        radius = 0

        # Pre-compute all repeated edge coordinates once
        x_inner = map_x
        y_inner = map_y
        x_margin = map_x + margin
        y_margin = map_y + margin
        w_pill = map_w - 2.0 * margin
        h_pill = map_h - 2.0 * margin
        x_right = map_x + map_w - inset - thickness
        y_top = map_y + map_h - inset - thickness

        # (x, y, w, h) for each side; only drawn when color is present
        edges: dict[str, tuple[float, float, float, float]] = {
            "top": (x_margin, y_top, w_pill, thickness),
            "bottom": (x_margin, y_inner, w_pill, thickness),
            "left": (x_inner, y_margin, thickness, h_pill),
            "right": (x_right, y_margin, thickness, h_pill),
        }

        for side, rect in edges.items():
            fill_color = side_colors.get(side)
            if fill_color is not None:
                # _draw_divider(*rect, radius, fill_color, border_color, border_width, mvp=mvp)
                _draw_filled_rounded_rect(*rect, radius, fill_color, mvp)


def _draw_resize_handles(
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    panel_roundness: float,
    state: MinimapState,
    mvp: Any = None,
) -> None:
    """Draw full-edge resize indicators, colored orange when the percentage cap is active."""
    resize_handle = state.interaction.resize_active
    if not resize_handle:
        return

    width_clamped = state.view.width_clamped
    height_clamped = state.view.height_clamped

    color_base = _alpha_mul(colors["regular_text_selected"], 0.5 * master_alpha)
    color_warn = _alpha_mul(colors["regular_selected"], master_alpha)
    thickness = 3.0 * ui_scale

    _draw_filled_rounded_rect(map_x, map_y, map_w, map_h, panel_roundness * 1.2, (0, 0, 0, 0.2 * master_alpha), mvp=mvp)

    if resize_handle == ResizeHandle.LIST:
        list_state = state.list
        zone_rect = list_state.list_zone_rect

        if not zone_rect or not state.view.rect:
            return

        zone_x, zone_y, zone_w, zone_h = zone_rect
        fill_color = color_warn if list_state.width_clamped else color_base

        half_scale = ui_scale / 2.0
        half_handle = thickness / 2.0

        if list_state.list_placement == "TOP":
            x, y = zone_x, round(zone_y - TYPE_LIST_TOP_BUTTON_GAP * half_scale - half_handle)
            w, h = zone_w, thickness
        else:  # LEFT
            view = state.view
            zone_right = view.rect[0] + view.inner_padding + list_state.list_width - 2.0 * ui_scale
            x, y = round(zone_right + TYPE_LIST_LEFT_BUTTON_GAP * half_scale - half_handle), zone_y
            w, h = thickness, zone_h

        _draw_pill(x, y, w, h, fill_color, mvp=mvp)
        return

    # Determine which edges to highlight
    width_side = height_side = None
    if resize_handle in (ResizeHandle.LEFT, ResizeHandle.TOP_LEFT, ResizeHandle.BOTTOM_LEFT):
        width_side = "left"
    elif resize_handle in (ResizeHandle.RIGHT, ResizeHandle.TOP_RIGHT, ResizeHandle.BOTTOM_RIGHT):
        width_side = "right"
    if resize_handle in (ResizeHandle.TOP, ResizeHandle.TOP_LEFT, ResizeHandle.TOP_RIGHT):
        height_side = "top"
    elif resize_handle in (ResizeHandle.BOTTOM, ResizeHandle.BOTTOM_LEFT, ResizeHandle.BOTTOM_RIGHT):
        height_side = "bottom"

    active_sides = {}
    if width_side:
        active_sides[width_side] = color_warn if width_clamped else color_base
    if height_side:
        active_sides[height_side] = color_warn if height_clamped else color_base

    margin = CONTENT_PADDING * ui_scale
    inset = 2.0 * ui_scale

    x_inner = map_x + inset
    y_inner = map_y + inset
    x_margin = map_x + margin
    y_margin = map_y + margin
    w_pill = map_w - 2.0 * margin
    h_pill = map_h - 2.0 * margin
    x_right = map_x + map_w - inset - thickness
    y_top = map_y + map_h - inset - thickness

    edges = {
        "top": (x_margin, y_top, w_pill, thickness),
        "bottom": (x_margin, y_inner, w_pill, thickness),
        "left": (x_inner, y_margin, thickness, h_pill),
        "right": (x_right, y_margin, thickness, h_pill),
    }

    for side, rect in edges.items():
        if (fill_color := active_sides.get(side)) is not None:
            _draw_pill(*rect, fill_color, mvp=mvp)


def _draw_view_fill(
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
    mvp: Any = None,
) -> None:
    """Draw a filled rect over the active view region, behind nodes and wires."""
    if visible is None:
        visible = _get_visible_rect(space, region)
    if not visible:
        return

    view_x = map_anchor_x + (visible[0] - tree_center_x) * scale
    view_y = map_anchor_y + (visible[1] - tree_center_y) * scale
    view_w = max((visible[2] - visible[0]) * scale, 1.0)
    view_h = max((visible[3] - visible[1]) * scale, 1.0)

    view_left = max(view_x, map_x)
    view_bottom = max(view_y, map_y)
    view_right = min(view_x + view_w, map_x + map_w)
    view_top = min(view_y + view_h, map_y + map_h)
    hole_width = view_right - view_left
    hole_height = view_top - view_bottom
    if hole_width <= 0 or hole_height <= 0:
        return

    # Pad the fill 1px inside the viewport outline so the border stroke
    # covers the fill edge instead of the fill spilling past it.
    fill_padding = 1.0
    view_left += fill_padding
    view_bottom += fill_padding
    hole_width -= 2.0 * fill_padding
    hole_height -= 2.0 * fill_padding
    if hole_width <= 0 or hole_height <= 0:
        return

    fill_color = colors["active_view_color"]
    fill_color = _alpha_mul(fill_color, 0.1 * master_alpha)
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
        mvp=mvp,
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
    mvp: Any = None,
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

    node_roundness = colors["node_roundness"] * ui_scale
    hole_width = view_right - view_left
    hole_height = view_top - view_bottom

    # Darkened overlay
    if settings.use_passepartout:
        background = colors["background"]
        overlay_color = (background[0], background[1], background[2], settings.passepartout_alpha)
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
                    mvp=mvp,
                )
            else:
                _draw_filled_rounded_rect(map_x, map_y, map_w, map_h, panel_roundness, overlay, mvp=mvp)
        finally:
            if scissor_temporarily_disabled:
                gpu.state.scissor_test_set(True)

    # Outline the viewport extent when it overlaps the minimap
    if hole_width > 0 and hole_height > 0:
        outline_color = _alpha_mul(colors["active_view_color"], master_alpha)
        border_width = 1.5 * ui_scale

        _draw_rounded_rect_border(view_x, view_y, view_w, view_h, node_roundness, outline_color, border_width, mvp=mvp)


def _draw_node_count(
    settings,
    node_count: int,
    map_x: float,
    map_y: float,
    map_w: float,
    padding: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> None:
    """Draw the node count text in the bottom-right corner of the minimap."""
    if not settings.show_node_count:
        return

    info_text = str(node_count)
    font_id = 0
    font_size = int(FONT_SIZE * ui_scale)
    blf.size(font_id, font_size)
    text_w, _ = blf.dimensions(font_id, info_text)
    text_x = round(map_x + map_w - text_w - padding)
    text_y = round(map_y + padding)

    text_color = _alpha_mul(colors["node_text"], master_alpha)

    _draw_text_with_shadow(font_id, info_text, text_x, text_y, text_color, font_size, settings.show_text_shadow)


def _paint_frame_all_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
    """Draw the frame-all corner brackets icon."""
    inset = 5 * ui_scale
    stroke_thickness = max(1, int(1.5 * ui_scale))
    arm_length = size * 0.15

    # Top-left bracket
    _draw_filled_rounded_rect(
        x + inset, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + inset, y + inset, stroke_thickness, arm_length, stroke_thickness * 0.5, color, mvp=mvp
    )
    # Top-right bracket
    _draw_filled_rounded_rect(
        x + size - inset - arm_length, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + size - inset - stroke_thickness,
        y + inset,
        stroke_thickness,
        arm_length,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    # Bottom-left bracket
    _draw_filled_rounded_rect(
        x + inset,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    _draw_filled_rounded_rect(
        x + inset, y + size - inset - arm_length, stroke_thickness, arm_length, stroke_thickness * 0.5, color, mvp=mvp
    )
    # Bottom-right bracket
    _draw_filled_rounded_rect(
        x + size - inset - arm_length,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    _draw_filled_rounded_rect(
        x + size - inset - stroke_thickness,
        y + size - inset - arm_length,
        stroke_thickness,
        arm_length,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )


def _paint_frame_view_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
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
        mvp=mvp,
    )


def _paint_frame_selected_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
    """Draw the frame-selected rails and center box icon."""
    inset = 5 * ui_scale
    stroke_thickness = max(1, int(1.5 * ui_scale))
    arm_length = size * 0.15

    # Left/right rails connecting top and bottom corners
    _draw_filled_rounded_rect(
        x + inset, y + inset, stroke_thickness, size - 2 * inset, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + size - inset - stroke_thickness,
        y + inset,
        stroke_thickness,
        size - 2 * inset,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    # Corner arms
    _draw_filled_rounded_rect(
        x + inset, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + size - inset - arm_length, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + inset,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    _draw_filled_rounded_rect(
        x + size - inset - arm_length,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )

    # Center box
    center_box_w = center_box_h = 2 * ui_scale
    center_box_x = x + (size - center_box_w) / 2
    center_box_y = y + (size - center_box_h) / 2
    _draw_filled_rounded_rect(center_box_x, center_box_y, center_box_w, center_box_h, 1.5 * ui_scale, color, mvp=mvp)


def _paint_list_toggle_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
    """Draw the list-toggle icon: three horizontal bars, or an X when active."""
    stroke_thickness = max(1, int(1.5 * ui_scale))
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
            mvp=mvp,
        )
    return


_BUTTON_ICONS = {
    "ALL": _paint_frame_all_icon,
    "VIEW": _paint_frame_view_icon,
    "SELECTED": _paint_frame_selected_icon,
    "LIST": _paint_list_toggle_icon,
}


def _paint_grip_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
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
                mvp=mvp,
            )


def _get_visible_minimap_buttons(settings) -> list[str]:
    """Return ids of enabled minimap buttons in draw order."""
    if not settings.use_interactive:
        return []
    visible = [button_id for button_id, pref_attr in _MINIMAP_BUTTONS if getattr(settings, pref_attr, True)]
    # Frame Selected is meaningless with Follow View (the viewport drives framing).
    if settings.use_follow_view:
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
) -> dict[str, tuple[float, float, float]]:
    """Return hit-rect origins {id: (x, y, size)} for every visible button.

    The row hosts the whole chrome: the list toggle at the left and the
    frame buttons (ALL/VIEW/SELECTED) leading into the move-grip drag handle
    at the right. The row sits on the top edge, except in the top-list
    placement where it sits just below the list strip. When the move button is
    disabled the frame row extends to the right padding edge instead.
    Frame buttons are culled progressively (SELECTED → VIEW → ALL) when the
    row would spill past the left padding or collide with the list toggle.
    """
    button_size = BUTTON_SIZE * ui_scale
    button_margin = BUTTON_MARGIN * ui_scale
    gap = padding
    if state.list.list_placement == "TOP" and state.list.list_width > 0:
        # Vertical layout: the row sits just below the top list strip,
        # overlapping the map content filling the bottom, like the left
        # placement.
        strip_h = min(state.list.list_width, map_h - 2 * CONTENT_PADDING * ui_scale)
        zone_bottom = map_y + map_h - CONTENT_PADDING * ui_scale - strip_h
        top_y = round(zone_bottom - TYPE_LIST_TOP_BUTTON_GAP * ui_scale - button_size)
    else:
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
        # 2) row would overlap the list toggle (with clearance)
        left_chrome_right = list_button_x + button_size if list_button_x is not None else None
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


def _draw_minimap_buttons(map_x, map_y, map_w, map_h, padding, colors, ui_scale, master_alpha, mvp: Any = None):
    """Draw the interactive minimap buttons and record their hit rects."""
    settings = get_addon_preferences().settings
    state = _state()
    state.buttons.rects.clear()

    visible_button_ids = _get_visible_minimap_buttons(settings)
    rects = _layout_minimap_buttons(state, visible_button_ids, map_x, map_y, map_w, map_h, padding, ui_scale, settings)
    radius = colors["node_roundness"] * ui_scale
    fill_radius = radius * 1.5
    tool_text = _alpha_mul(colors["tool_text"], master_alpha)
    tool_text_selected = _alpha_mul(colors["tool_text_selected"], master_alpha)

    tool_bg = _alpha_mul(colors["tool_inner"], master_alpha)
    tool_border = _alpha_mul(colors["tool_outline"], master_alpha)
    tool_pressed = _alpha_mul(colors["tool_selected"], master_alpha)
    regular_bg = _alpha_mul(colors["regular_inner"], master_alpha)
    regular_text = _alpha_mul(colors["regular_text"], master_alpha)
    regular_text_selected = _alpha_mul(colors["regular_text_selected"], master_alpha)
    regular_selected = _alpha_mul(colors["regular_selected"], master_alpha)
    regular_border = _alpha_mul(colors["regular_outline"], master_alpha)
    border_width = 0.5 * ui_scale

    # Move-grip drag handle is available whenever interactive mode is on, which
    # is the mode that enables repositioning the map.
    if "DRAG" in rects and settings.use_interactive:
        drag_x, drag_y, drag_size = rects["DRAG"]
        drag_h = BUTTON_SIZE * ui_scale
        drag_pressed = state.buttons.pressed_button_id == "DRAG"
        drag_bg = regular_selected if drag_pressed else regular_bg
        _draw_filled_rounded_rect(drag_x, drag_y, drag_size, drag_h, fill_radius, drag_bg, mvp=mvp)
        _draw_rounded_rect_border(drag_x, drag_y, drag_size, drag_h, radius, regular_border, border_width, mvp=mvp)
        drag_icon = regular_text_selected if drag_pressed else regular_text
        _paint_grip_icon(drag_x, drag_y, drag_h, drag_icon, ui_scale, mvp=mvp)
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
            _draw_filled_rounded_rect_varying(button_x, button_y, button_size, button_size, radii, tool_bg, mvp=mvp)
            _draw_rounded_rect_border_varying_sides(
                button_x,
                button_y,
                button_size,
                button_size,
                radii,
                tool_border,
                border_width,
                skip_left=button_index > 0,
                mvp=mvp,
            )

    for button_id, _pref_attr in _MINIMAP_BUTTONS:
        if button_id not in rects:
            continue
        button_x, button_y, button_size = rects[button_id]
        is_pressed = state.buttons.pressed_button_id == button_id
        is_hovered = (not is_pressed) and state.buttons.hovered_button_id == button_id
        if button_id == "LIST":
            # The type-list toggle acts as an on/off indicator: it shows the
            # selected color whenever the list is open.
            button_fill = regular_selected if settings.show_type_list else regular_bg
            button_border = regular_border
            pressed_fill = regular_selected
            icon_color = regular_text_selected if (settings.show_type_list or is_pressed) else regular_text
        else:
            button_fill = tool_bg
            button_border = tool_border
            pressed_fill = tool_pressed
            icon_color = tool_text_selected if is_pressed else tool_text
        if button_id == "LIST" or not is_combined:
            _draw_filled_rounded_rect(button_x, button_y, button_size, button_size, fill_radius, button_fill, mvp=mvp)
            _draw_rounded_rect_border(
                button_x, button_y, button_size, button_size, radius, button_border, border_width, mvp=mvp
            )
        if is_pressed or is_hovered:
            fill_color = pressed_fill if is_pressed else (1, 1, 1, BUTTON_HOVER_ALPHA * master_alpha)
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
                    hover_x, button_y + 1, hover_width, button_size - 2, hover_radii, fill_color, mvp=mvp
                )
            else:
                _draw_filled_rounded_rect(
                    button_x + 1,
                    button_y + 1,
                    button_size - 2,
                    button_size - 2,
                    max(2.0, radius - 1),
                    fill_color,
                    mvp=mvp,
                )
        if button_id == "LIST":
            _paint_list_toggle_icon(
                button_x,
                button_y,
                button_size,
                icon_color,
                ui_scale,
                mvp=mvp,
            )
        else:
            _BUTTON_ICONS[button_id](button_x, button_y, button_size, icon_color, ui_scale, mvp=mvp)
        state.buttons.rects[button_id] = (button_x, button_y, button_size, button_size)


def _redraw_pressed_move_grip(
    state: MinimapState, colors: dict, master_alpha: float, ui_scale: float, mvp: Any = None
) -> None:
    """Redraw the move-grip button on top of the moving dark overlay.

    While the map is being moved, `_draw_moving_border` lays a translucent black
    rect over the whole map *after* the buttons are drawn, which would smother
    the grip's pressed fill. Repainting the grip last keeps its active_view_color
    press state and icon visible while dragging.
    """
    if state.buttons.pressed_button_id != "DRAG":
        return
    rect = state.buttons.rects.get("DRAG")
    if not rect:
        return
    drag_x, drag_y, drag_size, drag_h = rect
    radius = colors["node_roundness"] * ui_scale
    border_width = 0.5 * ui_scale
    bg_color = _alpha_mul(colors["regular_inner"], master_alpha)
    border_color = _alpha_mul(colors["regular_outline"], master_alpha)
    _draw_filled_rounded_rect(drag_x, drag_y, drag_size, drag_h, radius, bg_color, mvp=mvp)
    _draw_rounded_rect_border(drag_x, drag_y, drag_size, drag_h, radius, border_color, border_width, mvp=mvp)
    press_color = _alpha_mul(colors["regular_selected"], master_alpha)
    _draw_filled_rounded_rect(
        drag_x + 1, drag_y + 1, drag_size - 2, drag_h - 2, max(2.0, radius - 1), press_color, mvp=mvp
    )
    icon_color = _alpha_mul(colors["regular_text_selected"], master_alpha)
    _paint_grip_icon(drag_x, drag_y, drag_h, icon_color, ui_scale, mvp=mvp)


def _draw_marquee(
    state: MinimapState,
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    panel_roundness: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    mvp: Any = None,
) -> None:
    """Draw the frame-region marquee rectangle while it is being dragged."""
    start = state.interaction.marquee_start
    end = state.interaction.marquee_end
    if not state.interaction.marquee_active or start is None or end is None:
        return
    rect_x = min(start[0], end[0])
    rect_y = min(start[1], end[1])
    rect_w = abs(end[0] - start[0])
    rect_h = abs(end[1] - start[1])
    if rect_w < 1 or rect_h < 1:
        return
    base_color = colors["active_view_color"]
    fill_color = _alpha_mul(base_color, 0.02 * master_alpha)
    border_color = _alpha_mul(base_color, master_alpha)
    border_width = 0.5 * ui_scale

    _draw_filled_rounded_rect_clipped(
        rect_x, rect_y, rect_w, rect_h, 0.0, fill_color, map_x, map_y, map_w, map_h, panel_roundness, mvp=mvp
    )
    _draw_rounded_rect_border(rect_x, rect_y, rect_w, rect_h, 0.0, _alpha_mul(border_color, 0.2), border_width, mvp=mvp)

    points = [
        (rect_x, rect_y, 0.0),
        (rect_x + rect_w - 1, rect_y, 0.0),
        (rect_x + rect_w - 1, rect_y + rect_h - 1, 0.0),
        (rect_x, rect_y + rect_h - 1, 0.0),
        (rect_x, rect_y, 0.0),
    ]

    _draw_dashes(
        points,
        4.0 * ui_scale,
        4.0 * ui_scale,
        border_color,
        line_width=border_width,
        mvp=mvp,
    )


def draw_minimap() -> None:
    """Orchestrate minimap drawing in the Node Editor."""
    context = bpy.context
    space = context.space_data
    region = context.region

    state = _state()
    if _early_exit(context, space, state):
        show_overlays = space.overlay.show_overlays if space else "?"
        enabled = state.enabled
        logger.trace("draw_minimap: early exit (type=%s overlays=%s enabled=%s)", space.type, show_overlays, enabled)
        return

    settings = get_addon_preferences(context).settings

    # Defer auto-launch until registration is fully complete
    # to avoid invoking the modal with a stale context.
    if not _registration_state["done"]:
        logger.debug("draw_minimap: registration not done, skipping auto-launch")
    else:
        # Auto-start modal operator for pan/zoom interaction (one per window)
        win = context.window
        window_ptr = win.as_pointer() if win else 0
        has_modal = window_ptr in _minimap_window_operators if win else False

        if settings.use_interactive:
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

    # Compile state (tree data, fingerprint, settle timer) is shared by all
    # areas showing the same node tree, so each tree compiles once instead
    # of once per minimap. GPU batches below stay per-area.
    try:
        tree_ptr = node_tree.as_pointer()
    except ReferenceError:
        return
    shared = _shared_tree_cache(tree_ptr)
    state.shared = shared

    # Cache the editor viewport rect once for this frame; reused by the
    # transform/clamp logic and the viewport overlay draws below.
    visible = _get_visible_rect(space, region)

    show_borders = settings.show_node_outline
    include_selection = show_borders or settings.show_wires_selected
    current_fingerprint, raw_bounds, content_count, selected_bounds = _get_tree_snapshot(node_tree, include_selection)
    if raw_bounds[2] - raw_bounds[0] <= 0 or raw_bounds[3] - raw_bounds[1] <= 0:
        return

    # Cache snapshot results on state so framing functions can reuse them
    # instead of re-scanning the node tree.
    state.view.raw_tree_bounds = raw_bounds
    state.view.snapshot_selected_bounds = selected_bounds

    # logger.trace(
    #     "SETTINGS %d nodes | show_wires=%d show_node_labels=%d compact_labels=%d"
    #     " show_node_colors=%d socket_indicators=%d wire_color=%d frame_labels=%d"
    #     " show_reroutes=%d",
    #     current_fingerprint[0],
    #     settings.show_wires,
    #     settings.show_node_labels,
    #     settings.compact_node_labels,
    #     settings.show_node_colors,
    #     settings.show_socket_indicators,
    #     settings.show_wire_color,
    #     settings.show_frame_labels,
    #     getattr(settings, "show_reroutes", True),
    # )

    ui_scale = _get_ui_scale()
    colors = _get_node_editor_theme_colors()
    master_alpha = settings.opacity
    corner = settings.current_position

    rect = _compute_minimap_rect(settings, ui_scale, space, region, corner, state)
    if rect is None:
        return
    map_x, map_y, map_w, map_h, padding, y_margin = rect

    bounds_frozen = (
        shared.fingerprint is not None
        and shared.fingerprint != current_fingerprint
        and _is_bounds_stable_diff(shared.fingerprint, current_fingerprint)
    )
    # Freeze framing while a bounds-stable diff is pending settle: hold the
    # tree bounds so the map scale/pivot does not creep live during a drag
    # (a grab also flips the active/selection slots), keeping auto-bounds,
    # nodes, and wires uniform on the settle frame instead.
    # When auto-zoom is disabled, keep the frozen framing so moving, adding,
    # or removing nodes never changes the map scale; explicit frame actions
    # still reframe. Initialize once so the first draw has valid bounds.
    use_auto_zoom = getattr(settings, "use_auto_zoom", True)
    current_bounds = state.view.tree_bounds
    bounds_uninitialized = (current_bounds[2] - current_bounds[0] <= 0) or (current_bounds[3] - current_bounds[1] <= 0)
    if not bounds_frozen and (use_auto_zoom or bounds_uninitialized):
        bounds = _expand_bounds_margin(raw_bounds, ui_scale, map_h, padding)
        state.view.tree_bounds = bounds

    state.view.rect = (map_x, map_y, map_w, map_h)
    state.view.outer_margin = y_margin
    state.view.inner_padding = padding

    # Per-tree view persistence: reset pan/zoom when switching node trees,
    # but restore the saved view when revisiting the same tree.
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
            # Type-list expansion/search are not meaningful across trees: clear them
            # so a group expanded in the previous material cannot leak its guide line
            # into this tree.
            state.list.expanded.clear()
            state.list.scroll = 0.0
            state.list.h_scroll = 0.0
            state.list.search_query = ""
            state.list.search_cursor = 0
            # Drop the cached list layout so the new tree rebuilds it from its
            # own compiled data instead of reusing the previous tree's rows.
            state.cache.list_key = None
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
    _step_list_width(state, settings, map_w, map_h, ui_scale)

    _clamp_pan_to_viewport(space, region, state, visible)

    # Refresh tree data: pure position changes (node drags) are deferred to
    # the debounced settle flush so bounds, nodes, and wires update together;
    # anything else schedules a debounced full compile.
    old_fingerprint = shared.fingerprint
    if old_fingerprint != current_fingerprint:
        move_only = _is_move_only_diff(old_fingerprint, current_fingerprint)
        if move_only:
            # Freeze everything until settle: defer node position patches
            # and the wire unfreeze to the debounced settle flush so that
            # auto-bounds, nodes, and wires all update together instead of
            # staggering (bounds -> nodes -> wires) during a drag.
            shared.pending_settle_flush = True
        # An expanding type list needs compiled type stats to measure its
        # target width; list click actions also request an immediate compile
        # so the visual feedback is not delayed by the debounce interval.
        want_immediate = (state.list.anim_active and state.list.anim_target < 0) or shared.force_immediate
        # Always keep a settle timer armed: it flushes frozen wire/marker
        # batches (forced via pending_settle_flush) or runs the pending full
        # compile. Re-arm (push back) only when the fingerprint changed again
        # since arming; identical-fingerprint redraw streams (list animation,
        # hover) must leave the live timer alone so the settle event cannot be
        # starved by continuous redraws.
        delay = settings.debounce_delay
        now = time.perf_counter()
        if shared.pending_timer is not None:
            if shared.pending_fingerprint != current_fingerprint and now < shared.pending_timer_deadline:
                _unregister_pending_timer(shared)
            elif want_immediate and not shared.pending_immediate:
                # A deferred timer cannot serve an immediate request; re-arm
                # so the compile fires on the next event loop iteration.
                _unregister_pending_timer(shared)
        if shared.pending_timer is None:

            def _settle_fire():
                return _debounced_compile(shared, node_tree, colors, settings, master_alpha, ui_scale)

            interval = 0.0 if want_immediate else delay
            bpy.app.timers.register(_settle_fire, first_interval=interval)
            shared.pending_timer = _settle_fire
            shared.pending_immediate = want_immediate
            shared.pending_timer_deadline = now + delay
            shared.pending_fingerprint = current_fingerprint
            shared.force_immediate = False

    # Build screen-space batches (cached; applies current zoom/pan via matrix)
    # When a structural preference changed, _batches_dirty forces a batch
    # rebuild using the existing tree_data (which still reflects the old
    # settings). The debounce timer will recompile tree_data on the next
    # event-loop iteration, producing a second rebuild with fresh data.
    if state.cache._batches_dirty:
        state.cache._batches_dirty = False
        shared.position_version += 1
    map_anchor_x, map_anchor_y, scale, tree_center_x, tree_center_y = _get_minimap_transform(
        state, space, region, visible
    )
    state.view.map_scale = scale
    highlight_border = (
        _alpha_mul(colors["node_active"], 0.4)
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
        colors["node_backdrop"],
        colors["frame_node"],
        show_borders,
        bool(settings.show_type_list),
        highlight_border,
        wire_curvature,
        wire_thickness,
        node_outline=colors["node_outline"],
    )

    try:
        original_blend = gpu.state.blend_get()
    except Exception:
        original_blend = None
    gpu.state.blend_set("ALPHA")

    # Base MVP for the frame's chrome. The top-level matrix is invariant
    # across these draws (every push restores), so thread it down instead
    # of recomposing it per rect.
    base_mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()

    panel_roundness = _draw_background(map_x, map_y, map_w, map_h, colors, master_alpha, ui_scale, mvp=base_mvp)

    scissor_state = _setup_scissor(map_x, map_y, map_w, map_h)
    scissor_was_active = scissor_state[0]

    # _draw_view_fill(
    #     space,
    #     region,
    #     map_x,
    #     map_y,
    #     map_w,
    #     map_h,
    #     map_anchor_x,
    #     map_anchor_y,
    #     scale,
    #     tree_center_x,
    #     tree_center_y,
    #     colors,
    #     panel_roundness,
    #     master_alpha,
    #     ui_scale,
    #     visible,
    #     mvp=base_mvp,
    # )

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
        mvp=base_mvp,
    )

    _draw_marquee(
        state,
        map_x,
        map_y,
        map_w,
        map_h,
        panel_roundness,
        colors,
        master_alpha,
        ui_scale,
        mvp=base_mvp,
    )

    if not state.view.moving:
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
            mvp=base_mvp,
        )

    _draw_minimap_buttons(map_x, map_y, map_w, map_h, padding, colors, ui_scale, master_alpha, mvp=base_mvp)

    # Match the node search highlight gate in _ensure_minimap_batches: the
    # filter is active while the list is shown and the query is non-empty.
    shown_count = content_count
    if settings.show_type_list and state.list.search_query.strip():
        tree_data = state.tree_data() or {}
        shown_count = len(
            filter_matching_nodes(
                tree_data.get("type_stats") or {},
                tree_data.get("type_nodes") or {},
                state.list.search_query,
                tree_data.get("type_search") or None,
            )
        )
    _draw_node_count(settings, shown_count, map_x, map_y, map_w, padding, colors, master_alpha, ui_scale)

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
        _draw_type_list(
            settings, state, map_x, map_y, map_w, map_h, padding, colors, master_alpha, ui_scale, mvp=base_mvp
        )

        _draw_resize_handles(
            map_x, map_y, map_w, map_h, colors, master_alpha, ui_scale, panel_roundness, state, mvp=base_mvp
        )

        dragging = (
            state.view.moving
            or state.interaction.pressed
            or state.interaction.resize_active is not None
            or state.list.dragging_width is not None
            or state.list.scrollbar_dragging
        )
        hovered_minimap = state.interaction.hovered_minimap or dragging
        _draw_moving_border(
            map_x,
            map_y,
            map_w,
            map_h,
            panel_roundness,
            colors,
            master_alpha,
            ui_scale,
            state.view.moving,
            state.view.snapped,
            hovered_minimap,
            settings.current_position,
            mvp=base_mvp,
        )
        _redraw_pressed_move_grip(state, colors, master_alpha, ui_scale, mvp=base_mvp)

    finally:
        try:
            gpu.state.blend_set(original_blend if original_blend else "NONE")
        except Exception:
            gpu.state.blend_set("NONE")
