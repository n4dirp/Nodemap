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
from ..core.buttons import _visible_button_ids
from ..core.constants import (
    BORDER_POSITIONS,
    CONTENT_PADDING,
    CORNER_POSITIONS,
    ELEMENT_GAP,
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
from ..core.list_filter import filter_matching_nodes
from ..core.state import (
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
    _draw_filled_rounded_rect_linear,
    _draw_filled_rounded_rect_vignette,
    _draw_filled_rounded_rect_with_hole,
    _draw_pill,
    _draw_rounded_rect_border,
    _draw_text_with_shadow,
)
from .minimap_buttons import (
    _layout_buttons,
    _paint_buttons,
    _resolve_button_theme,
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
    settings: Any,
    mvp: Any = None,
) -> float:
    """Draw the minimap backdrop rounded rect and border."""

    bg_color = _alpha_mul(colors["background"], master_alpha)
    bg_low_color = _alpha_mul(colors["background_low"], master_alpha)
    panel_roundness = colors["panel_roundness"]
    shadow_offset = 1
    # border_color = _alpha_mul(colors["background_border"], master_alpha)
    border_width = 0.5 * ui_scale

    if settings.background_type == "LINEAR":
        _draw_filled_rounded_rect_linear(
            map_x, map_y, map_w, map_h, panel_roundness * 1.2, bg_color, bg_low_color, mvp=mvp
        )
    elif settings.background_type == "VIGNETTE":
        _draw_filled_rounded_rect_vignette(
            map_x, map_y, map_w, map_h, panel_roundness * 1.2, bg_color, bg_low_color, mvp=mvp
        )
    else:
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
    # _draw_rounded_rect_border(map_x, map_y, map_w, map_h, panel_roundness, border_color, border_width, mvp=mvp)

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
        snap_color = _alpha_mul(colors["regular_selected"], master_alpha)
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
            x, y = zone_x, round(zone_y - ELEMENT_GAP * half_scale - half_handle)
            w, h = zone_w, thickness
        else:  # LEFT
            view = state.view
            zone_right = view.rect[0] + view.inner_padding + list_state.list_width - 2.0 * ui_scale
            x, y = round(zone_right + ELEMENT_GAP * half_scale - half_handle), zone_y
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

    panel_roundness = _draw_background(
        map_x, map_y, map_w, map_h, colors, master_alpha, ui_scale, settings, mvp=base_mvp
    )

    scissor_state = _setup_scissor(map_x, map_y, map_w, map_h)
    scissor_was_active = scissor_state[0]

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
            content_rect=_get_map_content_rect(state),
            mvp=base_mvp,
        )

    button_ids = _visible_button_ids(settings)
    button_layout = _layout_buttons(state, button_ids, map_x, map_y, map_w, map_h, padding, ui_scale, settings)
    state.buttons.rects.clear()
    state.buttons.rects.update(button_layout.rects)
    button_theme = _resolve_button_theme(colors, master_alpha, ui_scale)
    _paint_buttons(button_layout, button_theme, state, settings, ui_scale, master_alpha, mvp=base_mvp)

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
        if state.buttons.pressed_button_id == "DRAG" and "DRAG" in button_layout.rects:
            # The moving overlay above smothers the grip's pressed fill, so
            # repaint the grip last to keep it visible while dragging.
            _paint_buttons(
                button_layout, button_theme, state, settings, ui_scale, master_alpha, mvp=base_mvp, only_id="DRAG"
            )

    finally:
        try:
            gpu.state.blend_set(original_blend if original_blend else "NONE")
        except Exception:
            gpu.state.blend_set("NONE")
