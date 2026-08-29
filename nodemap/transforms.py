"""Coordinate transforms and geometry for the minimap."""

import logging
from typing import Any

import bpy

from .helpers import _get_ui_scale
from .state import MinimapState, _state

logger = logging.getLogger(__package__)


def _get_map_content_rect_for_width(
    st: MinimapState, list_width: float, ui_scale: float | None = None
) -> tuple[float, float, float, float]:
    """Return the content rect for an explicit list width without mutating state."""
    mx, my, mw, mh = st.view.rect
    pad = st.view.padding
    if ui_scale is None:
        ui_scale = _get_ui_scale()
    left_inset = pad + list_width
    if list_width > 0:
        left_inset += 4.0 * ui_scale
    return mx + left_inset, my + pad, max(mw - pad - left_inset, 1.0), max(mh - 2 * pad, 1.0)


def _get_map_content_rect(st: MinimapState) -> tuple[float, float, float, float]:
    """Return ``(left, bottom, width, height)`` of the map content area.

    Subtracts the type-list zone plus a margin from the left edge so node
    framing and panning never place tree content behind the list.
    """
    return _get_map_content_rect_for_width(st, st.list.width)


def _preserve_view_for_list_width(
    st: MinimapState, old_width: float, new_width: float, ui_scale: float | None = None
) -> None:
    """Adjust ``st.view.pan/zoom`` so the same world rect stays framed after width change.

    Keeps the world rectangle that previously filled the map content rect
    filling the new content rect with the same relative position (centered).
    This is the automatic equivalent of re-running ``frame_view``/``frame_all``
    after toggling the type list, as requested: if width goes 100→75 the
    nodes translate and scale to occupy the same position inside the reduced
    available space.

    No-op when rect/bounds are degenerate or widths are equal within 0.5px.
    """
    if abs(new_width - old_width) < 0.5:
        return
    if not st.view.rect or st.view.rect[2] <= 1 or st.view.rect[3] <= 1:
        return
    bounds = st.view.tree_bounds
    if not bounds or (bounds[2] - bounds[0] <= 0) or (bounds[3] - bounds[1] <= 0):
        return
    if ui_scale is None:
        ui_scale = _get_ui_scale()

    # Follow-view mode already recomputes zoom/clamp every draw; the generic
    # world-rect preservation would fight that dynamic. Skip automatic
    # compensation there and let _get_minimap_transform / _clamp_pan_to_viewport
    # handle it.
    try:
        addon = bpy.context.preferences.addons.get(__package__)
        if addon and addon.preferences.settings.follow_view:
            return
    except Exception:
        pass

    bbox_w = max(bounds[2] - bounds[0], 1.0)
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    tree_cx = (bounds[0] + bounds[2]) / 2
    tree_cy = (bounds[1] + bounds[3]) / 2

    old_inner_l, old_inner_b, old_inner_w, old_inner_h = _get_map_content_rect_for_width(st, old_width, ui_scale)
    new_inner_l, new_inner_b, new_inner_w, new_inner_h = _get_map_content_rect_for_width(st, new_width, ui_scale)

    # Old scale / center derived from current zoom/pan and old geometry.
    old_base = min(old_inner_w / bbox_w, old_inner_h / bbox_h)
    if old_base <= 0:
        return
    old_scale = old_base * max(st.view.zoom, 1e-6)
    if old_scale <= 0:
        return
    old_cx = old_inner_l + old_inner_w / 2 + st.view.pan[0]
    old_cy = old_inner_b + old_inner_h / 2 + st.view.pan[1]

    # World rect that filled the old inner.
    wl = tree_cx + (old_inner_l - old_cx) / old_scale
    wr = tree_cx + (old_inner_l + old_inner_w - old_cx) / old_scale
    wb = tree_cy + (old_inner_b - old_cy) / old_scale
    wt = tree_cy + (old_inner_b + old_inner_h - old_cy) / old_scale

    # If the old view was framing the full tree (world covers bounds),
    # preserve the tree bounds instead of the letterboxed world. This
    # avoids drift when toggling 100→75→100: the world for a framed
    # view includes empty letterbox margins, so preserving it makes
    # zoom shrink each toggle (0.116→0.072→0.072). Using bounds keeps
    # zoom stable (1.0) and matches re-running frame_all.
    eps = 1.0
    covers_w = wl <= bounds[0] + eps and wr >= bounds[2] - eps
    covers_h = wb <= bounds[1] + eps and wt >= bounds[3] - eps
    new_base = min(new_inner_w / bbox_w, new_inner_h / bbox_h)
    if new_base <= 0:
        return

    if covers_w and covers_h:
        # Re-frame the full tree: same as frame_all with new inner.
        # Keep zoom at 1 (or min to fit) and center on tree.
        tw = bbox_w
        th = bbox_h
        target_cx = tree_cx
        target_cy = tree_cy
        req_zoom_w = (new_inner_w / tw) / new_base
        req_zoom_h = (new_inner_h / th) / new_base
        new_zoom = min(req_zoom_w, req_zoom_h)
        # Clamp like elsewhere (0.1..20) and keep at least 1 for
        # frame_all style (don't magnify small trees).
        # Use the same cap as _compute_frame_to_bounds_targets(fill=False).
        new_zoom = max(0.1, min(new_zoom, 20.0))
        if new_zoom > 1.0:
            new_zoom = 1.0
        new_scale = new_base * new_zoom
        pan_x = -(target_cx - tree_cx) * new_scale
        pan_y = -(target_cy - tree_cy) * new_scale
    else:
        # Custom panned view: preserve the same world center by
        # scaling pan proportionally to the width change only (list
        # affects width, not height). This keeps the same area at
        # the same relative X position and is symmetric for
        # toggle off (75→100 restores zoom).
        scale_ratio = new_inner_w / max(old_inner_w, 1.0)
        new_scale = old_scale * scale_ratio
        new_zoom = new_scale / new_base
        new_zoom = max(0.1, min(new_zoom, 20.0))
        # Re-derive scale after clamp.
        new_scale = new_base * new_zoom
        # Preserve world center proportionally: pan scales with scale.
        # Equivalent to pan_new = pan_old * (new_scale/old_scale).
        ratio = new_scale / max(old_scale, 1e-6)
        pan_x = st.view.pan[0] * ratio
        pan_y = st.view.pan[1] * ratio
        # Also account for the inner center shift caused by the
        # list. The pan definition is offset from inner center, so
        # the world center stays at inner center. The proportional
        # scaling above already keeps world center stable; no extra
        # shift is needed because inner center movement is absorbed
        # by the pan scaling. For pure offset preservation, the
        # formula pan_new = pan_old * ratio is sufficient.

    st.view.zoom = new_zoom
    st.view.base_zoom = new_zoom
    st.view.pan = (pan_x, pan_y)


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
