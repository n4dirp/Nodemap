"""Coordinate transforms and geometry for the minimap."""

import logging
from typing import Any

import bpy

from .helpers import _get_ui_scale
from .state import MinimapState, _state

logger = logging.getLogger(__package__)


def _get_map_content_rect(st: MinimapState) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, width, height)`` of the map content area.

    Subtracts the type-list zone plus a margin from the left edge so node
    framing and panning never place tree content behind the list.
    """
    mx, my, mw, mh = st.view.rect
    pad = st.view.padding
    left_inset = pad + st.list.width
    if st.list.width > 0:
        left_inset += 4.0 * _get_ui_scale()
    return mx + left_inset, my + pad, max(mw - pad - left_inset, 1.0), max(mh - 2 * pad, 1.0)


def _compute_base_map_geom(
    st: MinimapState,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Return the scale-independent geometry shared by all map transforms.

    Computes ``(inner_l, inner_b, inner_w, inner_h, bbox_w, bbox_h, base_scale,
    tree_cx, tree_cy)`` from the current state. Pure: no prefs lookups, no
    mutation of *st*.
    """
    bounds = st.view.tree_bounds
    inner_l, inner_b, inner_w, inner_h = _get_map_content_rect(st)
    bbox_w = max(bounds[2] - bounds[0], 1.0)
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    base_scale = min(inner_w / bbox_w, inner_h / bbox_h)
    tree_cx = (bounds[0] + bounds[2]) / 2
    tree_cy = (bounds[1] + bounds[3]) / 2
    return inner_l, inner_b, inner_w, inner_h, bbox_w, bbox_h, base_scale, tree_cx, tree_cy


def _compute_map_transform(
    st: MinimapState | None = None,
) -> tuple[float, float, float, float, float]:
    """Compute the screen mapping ``(cx, cy, scale, tree_cx, tree_cy)`` for the minimap.

    Pure: no side effects, no preference lookups. Callers that need the
    scale-independent geometry (inner rect, base_scale) can use
    :func:`_compute_base_map_geom` directly.
    """
    if st is None:
        st = _state()
    inner_l, inner_b, inner_w, inner_h, _bw, _bh, base_scale, tree_cx, tree_cy = _compute_base_map_geom(st)
    scale = base_scale * st.view.zoom
    cx = inner_l + inner_w / 2 + st.view.pan[0]
    cy = inner_b + inner_h / 2 + st.view.pan[1]
    return cx, cy, scale, tree_cx, tree_cy


def _get_minimap_transform(
    st: MinimapState | None = None,
    space: Any = None,
    region: Any = None,
    visible: tuple[float, float, float, float] | None = None,
) -> tuple[float, float, float, float, float]:
    """Computes internal transformations representing scale, zoom, and panning inside the minimap."""
    if st is None:
        st = _state()
    base_zoom = st.view.base_zoom
    zoom = base_zoom

    geom = _compute_base_map_geom(st)
    inner_l, inner_b, inner_w, inner_h, _bw, _bh, base_scale, _tcx, _tcy = geom

    # Dynamic Auto-Zoom if follow_view is active
    addon = bpy.context.preferences.addons.get(__package__)
    if addon and addon.preferences.settings.follow_view:
        if space is None:
            space = bpy.context.space_data
        if region is None:
            region = bpy.context.region

        if space and space.type == "NODE_EDITOR" and region:
            if visible is None:
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

                st.view.zoom = zoom
                # Execute clamping passively during draw so panning outside the minimap updates bounds
                _clamp_pan_to_viewport(space, region, st, visible)

    st.view.zoom = zoom
    scale = base_scale * st.view.zoom
    cx = inner_l + inner_w / 2 + st.view.pan[0]
    cy = inner_b + inner_h / 2 + st.view.pan[1]
    tree_cx = (st.view.tree_bounds[0] + st.view.tree_bounds[2]) / 2
    tree_cy = (st.view.tree_bounds[1] + st.view.tree_bounds[3]) / 2
    return cx, cy, scale, tree_cx, tree_cy


def _clamp_pan_to_viewport(
    space, region, st: MinimapState, visible: tuple[float, float, float, float] | None = None
) -> None:
    """Clamp *st.view.pan* so the editor viewport stays inside the minimap (follow mode).

    No-op when the ``follow_view`` preference is off.
    """
    addon = bpy.context.preferences.addons.get(__package__)
    if not addon or not addon.preferences.settings.follow_view:
        return

    if visible is None:
        visible = _get_visible_rect(space, region)
    if not visible:
        return

    cx, cy, scale, tree_cx, tree_cy = _compute_map_transform(st)
    inner_l, inner_b, inner_w, inner_h = _get_map_content_rect(st)
    inner_r = inner_l + inner_w
    inner_t = inner_b + inner_h

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

    pan_x, pan_y = st.view.pan
    if abs(dx) > 0.5:
        pan_x += dx
    if abs(dy) > 0.5:
        pan_y += dy
    st.view.pan = (pan_x, pan_y)


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
