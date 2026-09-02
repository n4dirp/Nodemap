"""Resize helpers for the minimap overlay."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .constants import (
    HANDLE_THICKNESS,
    MIN_MAP_HEIGHT,
    MIN_MAP_WIDTH,
    TYPE_LIST_MAX_WIDTH_PCT,
    TYPE_LIST_MIN_WIDTH,
)
from .helpers import _get_minimap_margins, _get_safe_bounds, _get_ui_scale
from .state import ResizeHandle

if TYPE_CHECKING:
    from bpy.types import Context, Event

    from .minimap_ops import NODEMAP_OT_navigate
    from .state import MinimapState

logger = logging.getLogger(__package__)


def get_list_divider_handle(state: MinimapState, region_x: int, region_y: int, ui_scale: float) -> ResizeHandle | None:
    """Return ``LIST`` when the cursor is over the divider between list and map.

    The divider spans the gap (``6*scale``) between the list zone's right edge
    and the map content's left edge and uses the same hit thickness as the
    outer resize borders.
    """
    if state.list.list_width <= 0 or not state.list.list_zone_rect or not state.view.rect:
        return None
    zone_x, zone_y, zone_w, zone_h = state.list.list_zone_rect
    # Hit zone starts at the zone's right edge (never reaching into the
    # scrollbar) and extends right into the map gutter for reachability.
    zone_right_edge = zone_x + zone_w
    hit_half_width = (HANDLE_THICKNESS - 1) * ui_scale
    if zone_right_edge <= region_x <= zone_right_edge + hit_half_width and zone_y <= region_y <= zone_y + zone_h:
        return ResizeHandle.LIST
    return None


def get_resize_handle(
    state: MinimapState, corner: str, region_x: int, region_y: int, ui_scale: float
) -> ResizeHandle | None:
    map_x, map_y, map_w, map_h = state.view.rect
    if map_w <= 0 or map_h <= 0:
        return None
    half_w = HANDLE_THICKNESS * ui_scale

    def is_near_edge(value, target):
        return target - half_w <= value <= target + half_w

    match corner:
        case "TOP_RIGHT":
            on_left = map_x <= region_x <= map_x + half_w
            on_bottom = map_y <= region_y <= map_y + half_w
            if on_left and on_bottom:
                return ResizeHandle.C
            if on_left:
                return ResizeHandle.W
            if on_bottom:
                return ResizeHandle.H
        case "TOP_LEFT":
            on_right = map_x + map_w - half_w <= region_x <= map_x + map_w
            on_bottom = map_y <= region_y <= map_y + half_w
            if on_right and on_bottom:
                return ResizeHandle.C
            if on_right:
                return ResizeHandle.W
            if on_bottom:
                return ResizeHandle.H
        case "BOTTOM_RIGHT":
            on_left = map_x <= region_x <= map_x + half_w
            on_top = map_y + map_h - half_w <= region_y <= map_y + map_h
            if on_left and on_top:
                return ResizeHandle.C
            if on_left:
                return ResizeHandle.W
            if on_top:
                return ResizeHandle.H
        case "BOTTOM_LEFT":
            on_right = map_x + map_w - half_w <= region_x <= map_x + map_w
            on_top = map_y + map_h - half_w <= region_y <= map_y + map_h
            if on_right and on_top:
                return ResizeHandle.C
            if on_right:
                return ResizeHandle.W
            if on_top:
                return ResizeHandle.H
        case _:
            return None


def resize_apply_delta(op: NODEMAP_OT_navigate, context: Context, event: Event) -> None:
    """Apply a resize drag delta to the minimap settings."""
    addon = context.preferences.addons.get(__package__)
    if not addon:
        return
    settings = addon.preferences.settings
    if not op._resize_start_values:
        return
    w0, h0 = op._resize_start_values
    dx = op._mouse_x - op._resize_start_mouse[0]
    dy = op._mouse_y - op._resize_start_mouse[1]
    corner = settings.position

    ui_scale = _get_ui_scale()
    sx, sy, ex, ey = _get_safe_bounds(op._area, op._region)
    x_margin, y_margin, margin = _get_minimap_margins(op._space, corner, ui_scale)

    safe_w = ex - sx
    safe_h = ey - sy
    max_width_pct = settings.max_width_pct / 100.0
    max_height_pct = settings.max_height_pct / 100.0
    max_w = max(MIN_MAP_WIDTH, int((safe_w - 2 * x_margin) * max_width_pct))
    max_h = max(MIN_MAP_HEIGHT, int((safe_h - y_margin - margin) * max_height_pct))

    # Suppress property update callbacks during drag to avoid clearing
    # tree_data, which causes a one-frame flash while recompiling.
    from .state import suppress_update_callbacks

    with suppress_update_callbacks():
        if op._resize_handle in (ResizeHandle.W, ResizeHandle.C):
            if corner in ("TOP_RIGHT", "BOTTOM_RIGHT"):
                new_w = max(MIN_MAP_WIDTH, min(max_w, int(w0 - dx / ui_scale)))
            else:
                new_w = max(MIN_MAP_WIDTH, min(max_w, int(w0 + dx / ui_scale)))
            settings.minimap_width = new_w

        if op._resize_handle in (ResizeHandle.H, ResizeHandle.C):
            if corner in ("TOP_RIGHT", "TOP_LEFT"):
                new_h = max(MIN_MAP_HEIGHT, min(max_h, int(h0 - dy / ui_scale)))
            else:
                new_h = max(MIN_MAP_HEIGHT, min(max_h, int(h0 + dy / ui_scale)))
            settings.minimap_height = new_h

    state = op._state
    if not state:
        return
    state.interaction.hovered_handle = op._resize_handle
    state.view.width_clamped = settings.minimap_width >= max_w or settings.minimap_width <= MIN_MAP_WIDTH
    state.view.height_clamped = settings.minimap_height >= max_h or settings.minimap_height <= MIN_MAP_HEIGHT


def apply_list_width_drag(op: NODEMAP_OT_navigate, context: Context) -> None:
    """Update the type-list percent width from the current mouse delta."""
    state = op._state
    addon = context.preferences.addons.get(__package__)
    if not state or not addon:
        return
    settings = addon.preferences.settings
    if op._list_width_start_map_w <= 0:
        return
    dx = op._mouse_x - op._list_width_start_x
    map_w = op._list_width_start_map_w
    ui_scale = _get_ui_scale()
    min_w = TYPE_LIST_MIN_WIDTH * ui_scale
    max_w = map_w * TYPE_LIST_MAX_WIDTH_PCT
    start_w = map_w * (op._list_width_start_pct / 100.0)
    start_w = min(max(start_w, min_w), max_w)
    new_w = min(max(start_w + dx, min_w), max_w)
    new_pct = int(round(new_w / max(map_w, 1.0) * 100.0))
    new_pct = min(max(new_pct, 15), 50)
    from .state import suppress_update_callbacks

    with suppress_update_callbacks():
        settings.type_list_width_pct = new_pct
    # Preserve framing so the same world rect stays centered in the
    # reduced/expanded available width (100→75 keeps same relative pos).
    old_w = state.list.list_width
    if abs(new_w - old_w) >= 0.5:
        from .transforms import _preserve_view_for_list_width

        _preserve_view_for_list_width(state, old_w, new_w, ui_scale)
    # Drive the zone width live (per-pixel) so the pill tracks the cursor
    # without the integer-percent quantization or a one-frame zone lag.
    state.list.dragging_width = new_w
    state.list.list_width = new_w
    state.list.width_clamped = new_w <= min_w + 0.5 or new_w >= max_w - 0.5
    # Keep state hover in sync so the pill draws during drag.
    state.interaction.hovered_handle = ResizeHandle.LIST
    state.interaction.resize_active = ResizeHandle.LIST
