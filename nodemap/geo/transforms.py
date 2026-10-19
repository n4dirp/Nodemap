"""Provide coordinate transforms and geometry for the minimap."""

import logging
import math
import time
from typing import Any

import bpy

from .. import __package__ as base_package
from ..core.constants import (
    ELEMENT_GAP,
    MAX_MAP_SCALE,
    MIN_MAP_SCALE,
    NODE_ZONE_PADDING,
    TYPE_LIST_ASPECT_THRESHOLD,
    ZONE_FILLET_RADIUS,
    ZONE_FILLET_SEGMENTS,
)
from ..core.helpers import _get_ui_scale, get_addon_preferences
from ..core.state import MinimapState, _state

logger = logging.getLogger(base_package)


def _list_placement(map_w: float, map_h: float, position: str = "AUTO") -> str:
    """Return the type-list placement for a position preference and minimap aspect.

    "AUTO" resolves to "TOP" when the map is taller than wide by
    ``TYPE_LIST_ASPECT_THRESHOLD``, else "LEFT". Explicit "TOP"/"LEFT"
    always win; anything else resolves like "AUTO".
    """
    if position == "TOP":
        return "TOP"
    if position == "LEFT":
        return "LEFT"
    if map_w > 0 and map_h > map_w * TYPE_LIST_ASPECT_THRESHOLD:
        return "TOP"
    return "LEFT"


def _list_top_inset(list_width: float, ui_scale: float) -> float:
    """Return the top inset reserved by the list strip."""
    return ELEMENT_GAP * ui_scale + list_width


def _map_size_pair(settings, ui_scale: float) -> tuple[float, float, float, float]:
    """Return ``(base_w, base_h, list_w, list_h)`` panel sizes in scaled pixels.

    ``minimap_width``/``minimap_height`` apply while the type list is closed,
    ``minimap_width_with_list``/``minimap_height_with_list`` while it is open.
    """
    return (
        settings.minimap_width * ui_scale,
        settings.minimap_height * ui_scale,
        settings.minimap_width_with_list * ui_scale,
        settings.minimap_height_with_list * ui_scale,
    )


def _map_settled_size(minimap_state: MinimapState, settings, ui_scale: float) -> tuple[float, float]:
    """Return the panel size the type-list toggle is heading to.

    The type-list placement and the zone extent resolve against this size
    instead of the previous frame's panel rect, so the reserved content area
    and the framed view are right on the frame the list toggles (and on the
    final frame of its animation) rather than one redraw later.
    """
    base_w, base_h, list_w, list_h = _map_size_pair(settings, ui_scale)
    list_state = minimap_state.list
    if list_state.anim_active and list_state.anim_target >= 0:
        # A resolved animation decides where the panel lands.
        return (list_w, list_h) if list_state.anim_target > 0 else (base_w, base_h)
    show_list = bool(getattr(settings, "show_type_list", False)) and bool(getattr(settings, "use_interactive", False))
    return (list_w, list_h) if show_list else (base_w, base_h)


def _map_duo_size(minimap_state: MinimapState, settings, ui_scale: float) -> tuple[float, float]:
    """Return the frame's minimap size in scaled pixels.

    ``minimap_width``/``minimap_height`` apply while the type list is closed
    and ``minimap_width_with_list``/``minimap_height_with_list`` while it is
    open. While the list open/close animation runs, the two pairs ease into
    each other with the same progress that drives the list zone extent, so
    the panel grows together with the list instead of snapping.
    """
    base_w, base_h, list_w, list_h = _map_size_pair(settings, ui_scale)

    list_state = minimap_state.list
    if list_state.anim_active and list_state.anim_target >= 0:
        progress = min((time.perf_counter() - list_state.anim_start) / max(list_state.anim_duration, 1e-4), 1.0)
        opening = list_state.anim_target >= list_state.anim_from
        eased = 1.0 - (1.0 - progress) ** 3 if opening else progress**3
        factor = eased if opening else 1.0 - eased
    else:
        # Animation settled (or never started; reduce-motion snaps): key the
        # pair on the visible zone extent.
        factor = 1.0 if list_state.list_width > 0 else 0.0

    if factor <= 0.0:
        return base_w, base_h
    if factor >= 1.0:
        return list_w, list_h
    return base_w + (list_w - base_w) * factor, base_h + (list_h - base_h) * factor


def _get_map_content_rect_for_width(
    minimap_state: MinimapState,
    list_width: float,
    ui_scale: float | None = None,
) -> tuple[float, float, float, float]:
    """Return the content rect for an explicit list width without mutating state."""
    map_x, map_y, map_w, map_h = minimap_state.view.rect
    padding = minimap_state.view.inner_padding
    if ui_scale is None:
        ui_scale = _get_ui_scale()
    placement = minimap_state.list.list_placement
    if list_width > 0 and placement == "TOP":
        # Only the top strip is reserved; the button row beneath it overlaps
        # the content, like the left placement. The left/right span the full
        # inner width.
        top_inset = padding + _list_top_inset(list_width, ui_scale)
        return map_x + padding, map_y + padding, max(map_w - 2 * padding, 1.0), max(map_h - padding - top_inset, 1.0)
    left_inset = padding + list_width
    if list_width > 0:
        left_inset += 4.0 * ui_scale
    return map_x + left_inset, map_y + padding, max(map_w - padding - left_inset, 1.0), max(map_h - 2 * padding, 1.0)


def _get_map_content_rect(minimap_state: MinimapState) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, width, height)`` of the map content area.

    Subtract the type-list zone plus a margin from the left edge so node
    framing and panning never place tree content behind the list.
    """
    return _get_map_content_rect_for_width(minimap_state, minimap_state.list.list_width)


def _clamp_minimap_zoom(zoom: float, base_scale: float) -> float:
    """Clamp a minimap ``user_zoom`` to the node-anchored absolute scale range.

    Convert ``MIN_MAP_SCALE``/``MAX_MAP_SCALE`` (pixels per tree unit) into
    zoom multipliers by dividing by ``base_scale`` so the usable range is
    independent of tree size. Zooms of 1.0 or higher are always permitted so
    framing the whole tree is never blocked.
    """
    if base_scale <= 0:
        return zoom
    zoom_min = min(MIN_MAP_SCALE / base_scale, 1.0)
    zoom_max = max(MAX_MAP_SCALE / base_scale, 1.0)
    return max(zoom_min, min(zoom, zoom_max))


# ---------------------------------------------------------------------------
# Framed view
# ---------------------------------------------------------------------------


def _follow_view_enabled() -> bool:
    """Return True when the view tracks the editor viewport (framing is dynamic)."""
    preferences = get_addon_preferences()
    return bool(preferences and preferences.settings.use_follow_view)


def _content_size_key(minimap_state: MinimapState) -> tuple[float, float]:
    """Return the content-rect size the view is fitted against."""
    _left, _bottom, width, height = _get_map_content_rect(minimap_state)
    return (round(width, 3), round(height, 3))


def _view_for_world_rect(
    minimap_state: MinimapState, rect: tuple[float, float, float, float]
) -> tuple[float, tuple[float, float]] | None:
    """Return ``(zoom, pan)`` showing *rect* fitted and centered in the content rect.

    The rect is scaled to fit inside the content area (the tighter axis
    decides), the rule ``frame_view`` and ``frame_all`` use. Return None when
    the geometry is degenerate.
    """
    view = minimap_state.view
    if view.rect[2] <= 1 or view.rect[3] <= 1:
        return None
    bounds = view.tree_bounds
    if bounds[2] - bounds[0] <= 0 or bounds[3] - bounds[1] <= 0:
        return None
    _inner_l, _inner_b, inner_w, inner_h, _bw, _bh, base_scale, tree_cx, tree_cy = _compute_base_map_geom(minimap_state)
    if inner_w <= 1 or inner_h <= 1 or base_scale <= 0:
        return None
    world_w = max(rect[2] - rect[0], 1e-6)
    world_h = max(rect[3] - rect[1], 1e-6)
    zoom = min(inner_w / (base_scale * world_w), inner_h / (base_scale * world_h))
    zoom = _clamp_minimap_zoom(zoom, base_scale)
    scale = base_scale * zoom
    world_cx = (rect[0] + rect[2]) / 2.0
    world_cy = (rect[1] + rect[3]) / 2.0
    return zoom, ((tree_cx - world_cx) * scale, (tree_cy - world_cy) * scale)


def _framed_rect_from_view(minimap_state: MinimapState) -> tuple[float, float, float, float] | None:
    """Return the tree-space rect the running view shows, or None when unavailable."""
    view = minimap_state.view
    if view.rect[2] <= 1 or view.rect[3] <= 1:
        return None
    bounds = view.tree_bounds
    if bounds[2] - bounds[0] <= 0 or bounds[3] - bounds[1] <= 0:
        return None
    _left, _bottom, inner_w, inner_h = _get_map_content_rect(minimap_state)
    if inner_w <= 1 or inner_h <= 1:
        return None
    return _minimap_world_rect(minimap_state, view.user_zoom, view.pan)


def _fit_framed_view(minimap_state: MinimapState) -> bool:
    """Fit the framed world rect into the current content rect.

    Also record the content size the fit was made against and the view pair it
    wrote, so an interaction-driven change can be told apart from a fit. Return
    True when the view was fitted.
    """
    view = minimap_state.view
    if view.framed_rect is None:
        return False
    fitted = _view_for_world_rect(minimap_state, view.framed_rect)
    if fitted is None:
        return False
    zoom, pan = fitted
    view.user_zoom = zoom
    view.anchor_zoom = zoom
    view.pan = pan
    view.framed_key = _content_size_key(minimap_state)
    view.authored_view = (zoom, pan)
    return True


def _mark_view_authored(minimap_state: MinimapState) -> None:
    """Record the running zoom/pan pair as written by the framing logic."""
    view = minimap_state.view
    view.authored_view = (view.user_zoom, view.pan)


def _set_framed_rect(minimap_state: MinimapState, rect: tuple[float, float, float, float]) -> bool:
    """Remember *rect* as the framed world rect and fit it to the content rect.

    Framing actions call this so the same world rect is re-fitted when the type
    list or the panel changes the content rect. Return True when fitted; follow
    view recomputes zoom and pan every draw and is left alone.
    """
    view = minimap_state.view
    view.framed_rect = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
    view.framed_key = None
    if _follow_view_enabled():
        return False
    return _fit_framed_view(minimap_state)


def _adopt_view_framing(minimap_state: MinimapState) -> None:
    """Adopt the running view as the framed rect after an interaction moved it.

    Panning and zooming leave the framed rect behind; the visible rect becomes
    the framing target, so a later type-list toggle keeps showing it. No-op
    while the view still holds the pair the last fit wrote, and in follow view,
    which recomputes zoom and pan every draw.
    """
    if _follow_view_enabled():
        return
    view = minimap_state.view
    if view.framed_rect is not None and view.authored_view == (view.user_zoom, view.pan):
        return
    rect = _framed_rect_from_view(minimap_state)
    if rect is None:
        return
    view.framed_rect = rect
    view.framed_key = _content_size_key(minimap_state)
    _mark_view_authored(minimap_state)


def _refit_framed_view(minimap_state: MinimapState) -> None:
    """Re-fit the framed rect when the content rect changed (list or panel size)."""
    if _follow_view_enabled():
        return
    view = minimap_state.view
    if view.framed_rect is None:
        _adopt_view_framing(minimap_state)
        return
    if view.framed_key != _content_size_key(minimap_state):
        _fit_framed_view(minimap_state)


def _compute_base_map_geom(
    minimap_state: MinimapState,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Return the scale-independent geometry shared by all map transforms.

    Compute ``(inner_l, inner_b, inner_w, inner_h, bbox_w, bbox_h, base_scale,
    tree_center_x, tree_center_y)`` from the current state. Pure: no prefs
    lookups, no mutation of *minimap_state*.
    """
    bounds = minimap_state.view.tree_bounds
    inner_l, inner_b, inner_w, inner_h = _get_map_content_rect(minimap_state)
    bbox_w = max(bounds[2] - bounds[0], 1.0)
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    base_scale = min(inner_w / bbox_w, inner_h / bbox_h)
    tree_center_x = (bounds[0] + bounds[2]) / 2
    tree_center_y = (bounds[1] + bounds[3]) / 2
    return inner_l, inner_b, inner_w, inner_h, bbox_w, bbox_h, base_scale, tree_center_x, tree_center_y


def _compute_map_transform(
    minimap_state: MinimapState | None = None,
) -> tuple[float, float, float, float, float]:
    """Compute the screen mapping ``(map_anchor_x, map_anchor_y, scale, tree_center_x, tree_center_y)`` for the minimap.

    Pure: no side effects, no preference lookups. Callers that need the
    scale-independent geometry (inner rect, base_scale) can use
    :func:`_compute_base_map_geom` directly.
    """
    if minimap_state is None:
        minimap_state = _state()
    inner_l, inner_b, inner_w, inner_h, _bw, _bh, base_scale, tree_center_x, tree_center_y = _compute_base_map_geom(
        minimap_state
    )
    scale = base_scale * minimap_state.view.user_zoom
    map_anchor_x = inner_l + inner_w / 2 + minimap_state.view.pan[0]
    map_anchor_y = inner_b + inner_h / 2 + minimap_state.view.pan[1]
    return map_anchor_x, map_anchor_y, scale, tree_center_x, tree_center_y


def _smooth_view_fac(rect_a: tuple[float, float, float, float], rect_b: tuple[float, float, float, float]) -> float:
    """Return the view-animation magnitude factor between two rects in [0, 1].

    Port ``smooth_view_rect_to_fac``
    (``source/blender/editors/interface/view2d/view2d_ops.cc``): the factor
    grows with the per-axis translation normalized to the view scale and
    with the per-axis scale difference, so small view changes animate in a
    fraction of the duration while large ones take the full duration.
    """
    fac_max = 0.0
    for axis in range(2):
        size_a = max(rect_a[axis + 2] - rect_a[axis], 1e-6)
        size_b = max(rect_b[axis + 2] - rect_b[axis], 1e-6)
        center_a = (rect_a[axis] + rect_a[axis + 2]) / 2.0
        center_b = (rect_b[axis] + rect_b[axis + 2]) / 2.0
        fac_max = max(fac_max, abs(center_a - center_b) / min(size_a, size_b))
        if fac_max >= 1.0:
            break
        fac_max = max(fac_max, (1.0 - min(size_a, size_b) / max(size_a, size_b)) * 2.0)
        if fac_max >= 1.0:
            break
    return min(fac_max, 1.0)


def _interp_rect(
    rect_a: tuple[float, float, float, float] | list[float],
    rect_b: tuple[float, float, float, float] | list[float],
    fac: float,
) -> list[float]:
    """Return the bounds of *rect_a* interpolated toward *rect_b* by *fac*.

    Minimap analog of ``BLI_rctf_interp``: linear interpolation of all four
    bounds; the caller passes the eased progress as *fac*.
    """
    ifac = 1.0 - fac
    return [a * ifac + b * fac for a, b in zip(rect_a, rect_b)]


def _minimap_world_rect(
    minimap_state: MinimapState, zoom: float, pan: tuple[float, float] | list[float]
) -> tuple[float, float, float, float]:
    """Return the tree-space rect visible in the minimap for *zoom* and *pan*.

    Minimap analog of the View2D ``cur`` rect animated by Blender's
    ``view2d_smooth_view``: the world rectangle filling the map content area
    for the given view, derived from the shared base geometry.
    """
    inner_l, inner_b, inner_w, inner_h, _bw, _bh, base_scale, tree_cx, tree_cy = _compute_base_map_geom(minimap_state)
    scale = base_scale * max(zoom, 1e-6)
    world_cx = tree_cx - pan[0] / scale
    world_cy = tree_cy - pan[1] / scale
    half_w = (inner_w / 2.0) / scale
    half_h = (inner_h / 2.0) / scale
    return (world_cx - half_w, world_cy - half_h, world_cx + half_w, world_cy + half_h)


def _get_minimap_transform(
    minimap_state: MinimapState | None = None,
    space: Any = None,
    region: Any = None,
    visible: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float, float]:
    """Compute internal transformations representing scale, zoom, and panning inside the minimap."""
    if minimap_state is None:
        minimap_state = _state()
    anchor_zoom = minimap_state.view.anchor_zoom
    zoom = anchor_zoom

    base_geom = _compute_base_map_geom(minimap_state)
    inner_l, inner_b, inner_w, inner_h, _bw, _bh, base_scale, _tree_center_x, _tree_center_y = base_geom

    # Dynamic Auto-Zoom if use_follow_view is active
    if get_addon_preferences().settings.use_follow_view:
        if space is None:
            space = bpy.context.space_data
        if region is None:
            region = bpy.context.region

        if space and space.type == "NODE_EDITOR" and region:
            if visible is None:
                visible = _get_visible_rect(space, region)
            if visible:
                viewport_w = max(visible[2] - visible[0], 1.0)
                viewport_h = max(visible[3] - visible[1], 1.0)

                req_zoom_w = (inner_w / viewport_w) / base_scale
                req_zoom_h = (inner_h / viewport_h) / base_scale
                min_req_zoom = min(req_zoom_w, req_zoom_h)

                # If viewport indicator exceeds bounds, dynamically zoom out to fit it perfectly
                if min_req_zoom < zoom:
                    zoom = min_req_zoom

                minimap_state.view.user_zoom = zoom
                # Execute clamping passively during draw so panning outside the minimap updates bounds
                _clamp_pan_to_viewport(space, region, minimap_state, visible)

    minimap_state.view.user_zoom = zoom
    scale = base_scale * minimap_state.view.user_zoom
    map_anchor_x = inner_l + inner_w / 2 + minimap_state.view.pan[0]
    map_anchor_y = inner_b + inner_h / 2 + minimap_state.view.pan[1]
    tree_center_x = (minimap_state.view.tree_bounds[0] + minimap_state.view.tree_bounds[2]) / 2
    tree_center_y = (minimap_state.view.tree_bounds[1] + minimap_state.view.tree_bounds[3]) / 2
    return map_anchor_x, map_anchor_y, scale, tree_center_x, tree_center_y


def _clamp_pan_to_viewport(
    space, region, minimap_state: MinimapState, visible: tuple[float, float, float, float] | None = None
) -> None:
    """Clamp *minimap_state.view.pan* so the editor viewport stays inside the minimap (follow mode).

    No-op when the ``use_follow_view`` preference is off.
    """
    if not get_addon_preferences().settings.use_follow_view:
        return

    if visible is None:
        visible = _get_visible_rect(space, region)
    if not visible:
        return

    map_anchor_x, map_anchor_y, scale, tree_center_x, tree_center_y = _compute_map_transform(minimap_state)
    inner_l, inner_b, inner_w, inner_h = _get_map_content_rect(minimap_state)
    inner_r = inner_l + inner_w
    inner_t = inner_b + inner_h

    # Transform viewport corners to minimap pixel space
    visible_l, visible_b, visible_r, visible_t = visible
    view_left_x = map_anchor_x + (visible_l - tree_center_x) * scale
    view_bottom_y = map_anchor_y + (visible_b - tree_center_y) * scale
    viewport_w = (visible_r - visible_l) * scale
    viewport_h = (visible_t - visible_b) * scale

    dx = 0.0
    dy = 0.0

    if viewport_w <= inner_w:
        if view_left_x < inner_l:
            dx = inner_l - view_left_x
        elif view_left_x + viewport_w > inner_r:
            dx = inner_r - (view_left_x + viewport_w)
    else:
        if view_left_x < inner_r - viewport_w:
            dx = inner_r - viewport_w - view_left_x
        elif view_left_x > inner_l:
            dx = inner_l - view_left_x

    if viewport_h <= inner_h:
        if view_bottom_y < inner_b:
            dy = inner_b - view_bottom_y
        elif view_bottom_y + viewport_h > inner_t:
            dy = inner_t - (view_bottom_y + viewport_h)
    else:
        if view_bottom_y < inner_t - viewport_h:
            dy = inner_t - viewport_h - view_bottom_y
        elif view_bottom_y > inner_b:
            dy = inner_b - view_bottom_y

    pan_x, pan_y = minimap_state.view.pan
    if abs(dx) > 0.5:
        pan_x += dx
    if abs(dy) > 0.5:
        pan_y += dy
    minimap_state.view.pan = (pan_x, pan_y)


def _get_visible_rect(
    space: bpy.types.SpaceNodeEditor, region: bpy.types.Region
) -> tuple[float, float, float, float] | None:
    """Return the visible viewport rectangle in tree coordinates, or None if unavailable.

    Account for Blender UI scaling to return unscaled tree coordinates.
    """
    try:
        region_w, region_h = region.width, region.height
        view2d = region.view2d
        if not view2d:
            logger.log(5, "_get_visible_rect: region.view2d unavailable")
            return None

        corner_points = [
            view2d.region_to_view(0, 0),
            view2d.region_to_view(region_w, 0),
            view2d.region_to_view(0, region_h),
            view2d.region_to_view(region_w, region_h),
        ]
        corner_points = [p for p in corner_points if p is not None]
        if not corner_points:
            logger.log(5, "_get_visible_rect: all corners returned None (region %dx%d)", region_w, region_h)
            return None

        ui_scale = _get_ui_scale()
        xs = [p[0] / ui_scale for p in corner_points]
        ys = [p[1] / ui_scale for p in corner_points]
        visible_rect = (min(xs), min(ys), max(xs), max(ys))
        return visible_rect
    except Exception as e:
        logger.log(5, "_get_visible_rect failed: %s", e)
        return None


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Return the CCW convex hull of *points* via the monotone chain algorithm."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return list(pts)

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _fillet_polygon(poly, radius, segments=5):
    """Round a CCW polygon's corners with circular arcs (Blender fillet_curves_poly)."""
    n = len(poly)
    if n < 3 or radius <= 0:
        return list(poly)

    def _norm(a):
        length = math.hypot(a[0], a[1])
        return (0.0, 0.0) if length == 0 else (a[0] / length, a[1] / length)

    in_dir, out_dir, d = [], [], []
    for i in range(n):
        p, prev, nxt = poly[i], poly[i - 1], poly[(i + 1) % n]
        d_in = _norm((p[0] - prev[0], p[1] - prev[1]))
        d_out = _norm((nxt[0] - p[0], nxt[1] - p[1]))
        in_dir.append(d_in)
        out_dir.append(d_out)
        dot = max(-1.0, min(1.0, d_in[0] * d_out[0] + d_in[1] * d_out[1]))
        ang = math.pi - math.acos(-dot)
        d.append(radius * math.tan(ang / 2.0))

    # Clamp adjacent fillets so they don't overlap on short edges.
    for i in range(n):
        j = (i + 1) % n
        ex = poly[j][0] - poly[i][0]
        ey = poly[j][1] - poly[i][1]
        elen = math.hypot(ex, ey)
        s = d[i] + d[j]
        if s > elen and s > 0:
            f = elen / s
            d[i] *= f
            d[j] *= f

    out = []
    for i in range(n):
        p = poly[i]
        di, do, dd = in_dir[i], out_dir[i], d[i]
        start = (p[0] - di[0] * dd, p[1] - di[1] * dd)
        cd = _norm((do[0] - di[0], do[1] - di[1]))
        dc = math.sqrt(radius * radius + dd * dd)
        cx = p[0] + cd[0] * dc
        cy = p[1] + cd[1] * dc
        ang = math.pi - math.acos(max(-1.0, min(1.0, di[0] * do[0] + di[1] * do[1])))
        r = math.hypot(start[0] - cx, start[1] - cy)
        a0 = math.atan2(start[1] - cy, start[0] - cx)
        for k in range(segments + 1):
            a = a0 + (ang * k / segments)
            out.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    return out


def zone_outline(node_rects, widget_unit):
    """Compute the zone fill polygon for a set of node rects in tree space.

    Each rect is ``(xmin, ymin, xmax, ymax)``.  Returns a CCW filleted convex
    hull polygon suitable for triangulation and GPU rendering.
    """
    pad = widget_unit * NODE_ZONE_PADDING
    corners = []
    for x0, y0, x1, y1 in node_rects:
        x0 -= pad
        y0 -= pad
        x1 += pad
        y1 += pad
        corners.extend([(x0, y0), (x0, y1), (x1, y0), (x1, y1)])
    hull = _convex_hull(corners)
    if len(hull) < 3:
        return hull
    return _fillet_polygon(hull, radius=widget_unit * ZONE_FILLET_RADIUS, segments=ZONE_FILLET_SEGMENTS)
