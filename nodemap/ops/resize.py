"""Resize helpers for the minimap overlay."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .. import __package__ as base_package
from ..core.constants import (
    HANDLE_THICKNESS,
    MAP_CORNER_SNAP_RADIUS,
    MAP_SNAP_CENTER_ZONE_PCT,
    MAP_SNAP_TOLERANCE,
    MIN_MAP_HEIGHT,
    MIN_MAP_WIDTH,
    TYPE_LIST_MAX_WIDTH_PCT,
    TYPE_LIST_MIN_WIDTH,
)
from ..core.helpers import (
    _get_minimap_margins,
    _get_safe_bounds,
    _get_ui_scale,
    clamp_free_rect,
    get_addon_preferences,
)
from ..core.state import ResizeHandle

if TYPE_CHECKING:
    from bpy.types import Context, Event

    from ..core.state import MinimapState
    from .navigate import NODEMAP_OT_navigate

logger = logging.getLogger(base_package)


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


def get_drag_handle(state: MinimapState, region_x: int, region_y: int) -> bool:
    """Return True when the cursor is over the move-grip drag button."""
    drag_rect = state.buttons.rects.get("DRAG")
    if not drag_rect:
        return False
    drag_x, drag_y, drag_width, drag_height = drag_rect
    return drag_x <= region_x <= drag_x + drag_width and drag_y <= region_y <= drag_y + drag_height


def _pinned_sides(corner: str) -> frozenset[str]:
    """Return the border-fixed sides of a docked minimap position.

    Corner docks pin two sides, border docks pin the one side flush against
    the region border, and ``FREE`` never pins a side.
    """
    pins = {
        "TOP_LEFT": frozenset({"left", "top"}),
        "TOP_RIGHT": frozenset({"right", "top"}),
        "BOTTOM_LEFT": frozenset({"left", "bottom"}),
        "BOTTOM_RIGHT": frozenset({"right", "bottom"}),
        "TOP_BORDER": frozenset({"top"}),
        "BOTTOM_BORDER": frozenset({"bottom"}),
        "LEFT_BORDER": frozenset({"left"}),
        "RIGHT_BORDER": frozenset({"right"}),
    }
    return pins.get(corner, frozenset())


def _resize_open_sides(
    corner: str,
    rect: tuple[float, float, float, float],
    safe: tuple[float, float, float, float] | None,
    margins: tuple[float, float, float] | None,
    ui_scale: float,
) -> tuple[bool, bool, bool, bool]:
    """Return ``(left, right, top, bottom)`` openness for a resize grab.

    Docked positions withhold the sides pinned to a border. FREE keeps every
    side, except a side pressed against the safe bounds collides and is
    withheld; if that would strip an axis of all handles, both sides of the
    axis stay live so the map can still be shrunk.
    """
    left, right, top, bottom = (True, True, True, True)
    pinned = _pinned_sides(corner)
    if "left" in pinned:
        left = False
    if "right" in pinned:
        right = False
    if "top" in pinned:
        top = False
    if "bottom" in pinned:
        bottom = False

    if corner == "FREE" and safe and margins:
        x, y, w, h = rect
        sx, sy, ex, ey = safe
        x_margin, y_margin, margin = margins
        eps = 1.0 * ui_scale
        if left and x <= sx + x_margin + eps:
            left = False
        if right and x + w >= ex - x_margin - eps:
            right = False
        if top and y + h >= ey - y_margin - eps:
            top = False
        if bottom and y <= sy + margin + eps:
            bottom = False
        if not left and not right:
            left = right = True
        if not top and not bottom:
            top = bottom = True

    return left, right, top, bottom


def get_resize_handle(
    state: MinimapState,
    corner: str,
    region_x: int,
    region_y: int,
    ui_scale: float,
    space=None,
    area=None,
    region=None,
) -> ResizeHandle | None:
    """Return the resize handle under the cursor, or None.

    Every edge and corner can start a resize except sides that are pinned to a
    docked border or, in FREE mode, pressed against the region border. A corner
    is only offered when at least one of its two sides is live.
    """
    map_x, map_y, map_w, map_h = state.view.rect
    if map_w <= 0 or map_h <= 0:
        return None
    half_w = HANDLE_THICKNESS * ui_scale

    safe = None
    margins = None
    if corner == "FREE" and space is not None and area is not None and region is not None:
        safe = _get_safe_bounds(area, region)
        margins = _get_minimap_margins(space, corner, ui_scale)

    open_left, open_right, open_top, open_bottom = _resize_open_sides(
        corner, (map_x, map_y, map_w, map_h), safe, margins, ui_scale
    )

    near_left = map_x <= region_x <= map_x + half_w
    near_right = map_x + map_w - half_w <= region_x <= map_x + map_w
    near_top = map_y + map_h - half_w <= region_y <= map_y + map_h
    near_bottom = map_y <= region_y <= map_y + half_w

    if near_left and near_top:
        if open_left or open_top:
            return ResizeHandle.TOP_LEFT
    if near_right and near_top:
        if open_right or open_top:
            return ResizeHandle.TOP_RIGHT
    if near_left and near_bottom:
        if open_left or open_bottom:
            return ResizeHandle.BOTTOM_LEFT
    if near_right and near_bottom:
        if open_right or open_bottom:
            return ResizeHandle.BOTTOM_RIGHT
    if near_left and open_left:
        return ResizeHandle.LEFT
    if near_right and open_right:
        return ResizeHandle.RIGHT
    if near_top and open_top:
        return ResizeHandle.TOP
    if near_bottom and open_bottom:
        return ResizeHandle.BOTTOM
    return None


def _nearest_dock(
    x: float,
    y: float,
    w: float,
    h: float,
    safe: tuple[int, int, int, int],
    x_margin: float,
    y_margin: float,
    margin: float,
    ui_scale: float,
) -> tuple[float, float, str] | None:
    """Return the snapped ``(x, y, position)`` when the rect is near a border corner.

    Uses the same anchor margins as the free rect clamp (``x_margin`` on the
    sides, ``y_margin`` on top, ``margin`` on bottom), so a snapped dock does
    not visibly shift when the position value switches between FREE and the
    dock.
    """
    sx, sy, ex, ey = safe
    tol = MAP_SNAP_TOLERANCE * ui_scale
    corner_tol = MAP_CORNER_SNAP_RADIUS * ui_scale
    left_target = sx + x_margin
    right_target = ex - x_margin
    bottom_target = sy + margin
    top_target = ey - y_margin

    near_left = abs(x - left_target) <= tol
    near_right = abs((x + w) - right_target) <= tol
    near_bottom = abs(y - bottom_target) <= tol
    near_top = abs((y + h) - top_target) <= tol

    # Corner proximity uses a wider radius and takes priority. The centered edge
    # docks are deliberate: the map may only snap to a border's center when its
    # own center is already near the border's midpoint, so freely dragging along
    # an edge never latches the map to the middle by accident.
    corner_left = abs(x - left_target) <= corner_tol
    corner_right = abs((x + w) - right_target) <= corner_tol
    corner_bottom = abs(y - bottom_target) <= corner_tol
    corner_top = abs((y + h) - top_target) <= corner_tol

    if corner_left and corner_top:
        return left_target, top_target - h, "TOP_LEFT"
    if corner_right and corner_top:
        return right_target - w, top_target - h, "TOP_RIGHT"
    if corner_left and corner_bottom:
        return left_target, bottom_target, "BOTTOM_LEFT"
    if corner_right and corner_bottom:
        return right_target - w, bottom_target, "BOTTOM_RIGHT"

    center_tol_x = (ex - sx) * MAP_SNAP_CENTER_ZONE_PCT / 100.0
    center_tol_y = (ey - sy) * MAP_SNAP_CENTER_ZONE_PCT / 100.0
    edge_mid_x = (sx + ex) / 2.0
    edge_mid_y = (sy + ey) / 2.0
    centered_x = abs((x + w / 2.0) - edge_mid_x) <= center_tol_x
    centered_y = abs((y + h / 2.0) - edge_mid_y) <= center_tol_y

    if near_top and centered_x:
        return (sx + ex) / 2.0 - w / 2.0, top_target - h, "TOP_BORDER"
    if near_bottom and centered_x:
        return (sx + ex) / 2.0 - w / 2.0, bottom_target, "BOTTOM_BORDER"
    if near_left and centered_y:
        return left_target, (sy + ey) / 2.0 - h / 2.0, "LEFT_BORDER"
    if near_right and centered_y:
        return right_target - w, (sy + ey) / 2.0 - h / 2.0, "RIGHT_BORDER"
    return None


def _width_growing_side(handle: ResizeHandle) -> str | None:
    """Return ``"left"`` or ``"right"`` when the handle drives the width, else None."""
    if handle in (ResizeHandle.LEFT, ResizeHandle.TOP_LEFT, ResizeHandle.BOTTOM_LEFT):
        return "left"
    if handle in (ResizeHandle.RIGHT, ResizeHandle.TOP_RIGHT, ResizeHandle.BOTTOM_RIGHT):
        return "right"
    return None


def _height_growing_side(handle: ResizeHandle) -> str | None:
    """Return ``"top"`` or ``"bottom"`` when the handle drives the height, else None."""
    if handle in (ResizeHandle.TOP, ResizeHandle.TOP_LEFT, ResizeHandle.TOP_RIGHT):
        return "top"
    if handle in (ResizeHandle.BOTTOM, ResizeHandle.BOTTOM_LEFT, ResizeHandle.BOTTOM_RIGHT):
        return "bottom"
    return None


def _grab_deltas(handle: ResizeHandle, dx: float, dy: float, ui_scale: float) -> tuple[float, float, float, float]:
    """Return ``(d_w, d_h, d_anchor_x, d_anchor_y)`` for a resize grab.

    Width and height always change relative to the grabbed side: dragging the
    left or top edge outward grows the map while the opposite edge stays fixed.
    The anchor deltas are nonzero only when the grab sits on the free map's
    offset-anchored sides (left and bottom), which must shift so the grabbed
    edge follows the cursor instead of pushing the opposite edge away.
    """
    d_w = 0.0
    d_h = 0.0
    d_anchor_x = 0.0
    d_anchor_y = 0.0
    width_side = _width_growing_side(handle)
    if width_side:
        if width_side == "left":
            d_w = -dx
            d_anchor_x = dx
        else:
            d_w = dx
    height_side = _height_growing_side(handle)
    if height_side:
        if height_side == "bottom":
            d_h = -dy
            d_anchor_y = dy
        else:
            d_h = dy
    return d_w, d_h, d_anchor_x, d_anchor_y


def _free_anchor_shift(d_anchor_x: float, d_anchor_y: float, width_change: int, height_change: int) -> tuple[int, int]:
    """Return FREE-mode offset shifts that keep the non-grabbed edge fixed.

    LEFT/BOTTOM grabs anchor the offset to the grabbed edge. The shift couples
    to the effective (clamped) size change, so once the size clamps the map
    freezes instead of continuing to slide along the border.
    """
    shift_x = -width_change if d_anchor_x else 0
    shift_y = -height_change if d_anchor_y else 0
    return shift_x, shift_y


def resize_apply_delta(op: NODEMAP_OT_navigate, context: Context, event: Event) -> None:
    """Apply a resize drag delta to the minimap settings."""
    addon = get_addon_preferences(context)
    if not addon:
        return
    settings = addon.settings
    if not op._resize_start_values:
        return
    w0, h0 = op._resize_start_values
    dx = op._mouse_x - op._resize_start_mouse[0]
    dy = op._mouse_y - op._resize_start_mouse[1]
    corner = settings.current_position

    ui_scale = _get_ui_scale()
    sx, sy, ex, ey = _get_safe_bounds(op._area, op._region)
    x_margin, y_margin, margin = _get_minimap_margins(op._space, corner, ui_scale)

    safe_w = ex - sx
    safe_h = ey - sy
    max_width_pct = settings.max_width_percent / 100.0
    max_height_pct = settings.max_height_percent / 100.0
    max_w = max(MIN_MAP_WIDTH, int((safe_w - 2 * x_margin) * max_width_pct))
    max_h = max(MIN_MAP_HEIGHT, int((safe_h - y_margin - margin) * max_height_pct))

    d_w, d_h, d_anchor_x, d_anchor_y = _grab_deltas(op._resize_handle, dx, dy, ui_scale)

    new_w = max(MIN_MAP_WIDTH, min(max_w, int(w0 + d_w / ui_scale))) if d_w else None
    new_h = max(MIN_MAP_HEIGHT, min(max_h, int(h0 + d_h / ui_scale))) if d_h else None

    # FREE mode anchors the map by its bottom-left offset, so a grabbed edge
    # that presses against a region border must not push the opposite edge.
    # Each grab caps the size by the room beyond its fixed edge (right grabs
    # fix the left edge, left grabs the right edge, top grabs the bottom edge,
    # bottom grabs the top edge), so once the grabbed edge reaches a border
    # the size locks instead of the opposite side growing along.
    free_start_origin: tuple[float, float] | None = None
    start_offset = op._resize_start_offset
    room_capped_w = False
    room_capped_h = False
    if corner == "FREE" and start_offset is not None:
        start_x, start_y = start_offset
        origin_x = sx + x_margin + start_x * ui_scale
        origin_y = sy + y_margin + start_y * ui_scale
        origin_x, origin_y = clamp_free_rect(
            origin_x, origin_y, w0 * ui_scale, h0 * ui_scale, (sx, sy, ex, ey), x_margin, y_margin, margin
        )
        free_start_origin = (origin_x, origin_y)
        width_side = _width_growing_side(op._resize_handle)
        if new_w is not None:
            if width_side == "right":
                room_limit = int((ex - x_margin - origin_x) / ui_scale)
            elif width_side == "left":
                room_limit = int((origin_x + w0 * ui_scale - sx - x_margin) / ui_scale)
            else:
                room_limit = None
            if room_limit is not None:
                room_capped_w = new_w > room_limit
                new_w = max(MIN_MAP_WIDTH, min(new_w, room_limit))
        height_side = _height_growing_side(op._resize_handle)
        if new_h is not None:
            if height_side == "top":
                room_limit = int((ey - y_margin - origin_y) / ui_scale)
            elif height_side == "bottom":
                room_limit = int((origin_y + h0 * ui_scale - sy - margin) / ui_scale)
            else:
                room_limit = None
            if room_limit is not None:
                room_capped_h = new_h > room_limit
                new_h = max(MIN_MAP_HEIGHT, min(new_h, room_limit))

    from ..core.state import suppress_update_callbacks

    old_w = settings.minimap_width
    old_h = settings.minimap_height
    with suppress_update_callbacks():
        if new_w is not None:
            settings.minimap_width = new_w
        if new_h is not None:
            settings.minimap_height = new_h
        # FREE anchors the map's bottom-left edge at offset_x/offset_y. For
        # left/bottom grabs the anchor shifts by the effective size change, so
        # the grabbed edge follows the cursor until the size clamps and then
        # the map stops instead of sliding along the border. Right/top grabs
        # pin the offset to the displayed opposite edge so a border press
        # cannot shove that edge away.
        if free_start_origin is not None and start_offset is not None:
            origin_x, origin_y = free_start_origin
            start_x, start_y = start_offset
            shift_x, shift_y = _free_anchor_shift(
                d_anchor_x,
                d_anchor_y,
                new_w - w0 if new_w is not None else 0,
                new_h - h0 if new_h is not None else 0,
            )
            if _width_growing_side(op._resize_handle) == "right":
                settings.offset_x = int(round((origin_x - sx - x_margin) / ui_scale))
            else:
                settings.offset_x = start_x + shift_x
            if _height_growing_side(op._resize_handle) == "top":
                settings.offset_y = int(round((origin_y - sy - y_margin) / ui_scale))
            else:
                settings.offset_y = start_y + shift_y

    state = op._state
    if not state:
        return
    # Preserve framing so the same world point stays centered in the
    # resized map (100→75 keeps the same relative view).
    from ..geo.transforms import _preserve_view_for_map_resize

    _preserve_view_for_map_resize(state, old_w, old_h, settings.minimap_width, settings.minimap_height, ui_scale)
    state.interaction.hovered_handle = op._resize_handle
    state.view.width_clamped = (
        settings.minimap_width <= MIN_MAP_WIDTH or settings.minimap_width >= max_w or room_capped_w
    )
    state.view.height_clamped = (
        settings.minimap_height <= MIN_MAP_HEIGHT or settings.minimap_height >= max_h or room_capped_h
    )


def apply_list_width_drag(op: NODEMAP_OT_navigate, context: Context) -> None:
    """Update the type-list percent width from the current mouse delta."""
    state = op._state
    addon = get_addon_preferences(context)
    if not state or not addon:
        return
    settings = addon.settings
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
    new_pct = min(max(new_pct, 0), 50)
    from ..core.state import suppress_update_callbacks

    with suppress_update_callbacks():
        settings.type_list_width_percent = new_pct
    # Preserve framing so the same world rect stays centered in the
    # reduced/expanded available width (100→75 keeps same relative pos).
    old_w = state.list.list_width
    if abs(new_w - old_w) >= 0.5:
        from ..geo.transforms import _preserve_view_for_list_width

        _preserve_view_for_list_width(state, old_w, new_w, ui_scale)
    # Drive the zone width live (per-pixel) so the pill tracks the cursor
    # without the integer-percent quantization or a one-frame zone lag.
    state.list.dragging_width = new_w
    state.list.list_width = new_w
    state.list.width_clamped = new_w <= min_w + 0.5 or new_w >= max_w - 0.5
    # Keep state hover in sync so the pill draws during drag.
    state.interaction.hovered_handle = ResizeHandle.LIST
    state.interaction.resize_active = ResizeHandle.LIST
