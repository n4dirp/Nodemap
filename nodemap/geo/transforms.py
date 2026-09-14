"""Provide coordinate transforms and geometry for the minimap."""

import logging
from typing import Any

import bpy

from .. import __package__ as base_package
from ..core.constants import (
    ELEMENT_GAP,
    MAX_MAP_SCALE,
    MIN_MAP_SCALE,
    TYPE_LIST_ASPECT_THRESHOLD,
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


def _get_map_content_rect_for_width(
    minimap_state: MinimapState,
    list_width: float,
    ui_scale: float | None = None,
    placement: str | None = None,
) -> tuple[float, float, float, float]:
    """Return the content rect for an explicit list width without mutating state.

    ``placement`` overrides the stored zone placement, for preserving views
    across a placement flip.
    """
    map_x, map_y, map_w, map_h = minimap_state.view.rect
    padding = minimap_state.view.inner_padding
    if ui_scale is None:
        ui_scale = _get_ui_scale()
    if placement is None:
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


def _preserve_view_for_list_width(
    minimap_state: MinimapState,
    old_width: float,
    new_width: float,
    ui_scale: float | None = None,
    min_delta: float = 0.5,
    old_placement: str | None = None,
    new_placement: str | None = None,
) -> None:
    """Adjust ``minimap_state.view.pan/zoom`` so the same world rect stays framed after width change.

    Keep the world rectangle that previously filled the map content rect
    filling the new content rect with the same relative position (centered).
    This is the automatic equivalent of re-running ``frame_view``/``frame_all``
    after toggling the type list, as requested: if width goes 100→75 the
    nodes translate and scale to occupy the same position inside the reduced
    available space.

    No-op when rect/bounds are degenerate or widths are equal within ``min_delta`` px.
    The animation path passes ``min_delta=0.0`` so eased steps map the list
    width change exactly; interactive deltas keep the 0.5px tolerance. A
    placement flip is expressed as equal widths with ``min_delta=0.0`` and
    explicit old/new placements.
    """
    if abs(new_width - old_width) < min_delta:
        return
    if not minimap_state.view.rect or minimap_state.view.rect[2] <= 1 or minimap_state.view.rect[3] <= 1:
        return
    bounds = minimap_state.view.tree_bounds
    if not bounds or (bounds[2] - bounds[0] <= 0) or (bounds[3] - bounds[1] <= 0):
        return
    if ui_scale is None:
        ui_scale = _get_ui_scale()

    # Follow-view mode already recomputes zoom/clamp every draw; the generic
    # world-rect preservation would fight that dynamic. Skip automatic
    # compensation there and let _get_minimap_transform / _clamp_pan_to_viewport
    # handle it.
    if get_addon_preferences().settings.use_follow_view:
        return

    bbox_w = max(bounds[2] - bounds[0], 1.0)
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    tree_center_x = (bounds[0] + bounds[2]) / 2
    tree_center_y = (bounds[1] + bounds[3]) / 2

    old_inner_l, old_inner_b, old_inner_w, old_inner_h = _get_map_content_rect_for_width(
        minimap_state, old_width, ui_scale, old_placement
    )
    new_inner_l, new_inner_b, new_inner_w, new_inner_h = _get_map_content_rect_for_width(
        minimap_state, new_width, ui_scale, new_placement
    )

    # Old scale / center derived from current zoom/pan and old geometry.
    old_base = min(old_inner_w / bbox_w, old_inner_h / bbox_h)
    if old_base <= 0:
        return
    old_scale = old_base * max(minimap_state.view.user_zoom, 1e-6)
    if old_scale <= 0:
        return
    old_cx = old_inner_l + old_inner_w / 2 + minimap_state.view.pan[0]
    old_cy = old_inner_b + old_inner_h / 2 + minimap_state.view.pan[1]

    # World rect that filled the old inner.
    world_l = tree_center_x + (old_inner_l - old_cx) / old_scale
    world_r = tree_center_x + (old_inner_l + old_inner_w - old_cx) / old_scale
    world_b = tree_center_y + (old_inner_b - old_cy) / old_scale
    world_t = tree_center_y + (old_inner_b + old_inner_h - old_cy) / old_scale

    # If the old view was framing the full tree (world covers bounds),
    # preserve the tree bounds instead of the letterboxed world. This
    # avoids drift when toggling 100→75→100: the world for a framed
    # view includes empty letterbox margins, so preserving it makes
    # zoom shrink each toggle (0.116→0.072→0.072). Using bounds keeps
    # zoom stable (1.0) and matches re-running frame_all.
    eps = 1.0
    covers_w = world_l <= bounds[0] + eps and world_r >= bounds[2] - eps
    covers_h = world_b <= bounds[1] + eps and world_t >= bounds[3] - eps
    new_base = min(new_inner_w / bbox_w, new_inner_h / bbox_h)
    if new_base <= 0:
        return

    if covers_w and covers_h:
        # Re-frame the full tree: same as frame_all with new inner.
        # Keep zoom at 1 (or min to fit) and center on tree.
        target_w = bbox_w
        target_h = bbox_h
        target_cx = tree_center_x
        target_cy = tree_center_y
        req_zoom_w = (new_inner_w / target_w) / new_base
        req_zoom_h = (new_inner_h / target_h) / new_base
        new_zoom = min(req_zoom_w, req_zoom_h)
        # Clamp to the node-anchored scale range and cap at 1 for
        # frame_all style (don't magnify small trees).
        new_zoom = _clamp_minimap_zoom(new_zoom, new_base)
        if new_zoom > 1.0:
            new_zoom = 1.0
        new_scale = new_base * new_zoom
        pan_x = -(target_cx - tree_center_x) * new_scale
        pan_y = -(target_cy - tree_center_y) * new_scale
    else:
        # Custom panned view: preserve the same world center by
        # scaling pan proportionally to the width change only (list
        # affects width, not height). This keeps the same area at
        # the same relative X position and is symmetric for
        # toggle off (75→100 restores zoom). A placement flip changes
        # both axes, so scale by the tighter axis ratio.
        if old_placement is not None and old_placement != new_placement:
            scale_ratio = min(new_inner_w / max(old_inner_w, 1.0), new_inner_h / max(old_inner_h, 1.0))
        else:
            scale_ratio = new_inner_w / max(old_inner_w, 1.0)
        new_scale = old_scale * scale_ratio
        new_zoom = new_scale / new_base
        new_zoom = _clamp_minimap_zoom(new_zoom, new_base)
        # Re-derive scale after clamp.
        new_scale = new_base * new_zoom
        # Preserve world center proportionally: pan scales with scale.
        # Equivalent to pan_new = pan_old * (new_scale/old_scale).
        ratio = new_scale / max(old_scale, 1e-6)
        pan_x = minimap_state.view.pan[0] * ratio
        pan_y = minimap_state.view.pan[1] * ratio

    minimap_state.view.user_zoom = new_zoom
    minimap_state.view.anchor_zoom = new_zoom
    minimap_state.view.pan = (pan_x, pan_y)


def _preserve_view_for_map_resize(
    minimap_state: MinimapState,
    old_w: float,
    old_h: float,
    new_w: float,
    new_h: float,
    ui_scale: float | None = None,
) -> None:
    """Adjust ``minimap_state.view.pan`` so the same world point stays centered after a map resize.

    Scale pan by the base-scale ratio between the old and new content rects,
    keeping the world point at the content-rect center fixed. Zoom is left
    untouched. No-op when use_follow_view is active, sizes are equal within 0.5px,
    or geometry is degenerate.
    """
    if abs(new_w - old_w) < 0.5 and abs(new_h - old_h) < 0.5:
        return
    if not minimap_state.view.rect or minimap_state.view.rect[2] <= 1 or minimap_state.view.rect[3] <= 1:
        return
    bounds = minimap_state.view.tree_bounds
    if not bounds or (bounds[2] - bounds[0] <= 0) or (bounds[3] - bounds[1] <= 0):
        return
    if ui_scale is None:
        ui_scale = _get_ui_scale()

    # Follow-view mode recomputes zoom/clamp every draw; compensating here
    # would fight that dynamic. Skip, like _preserve_view_for_list_width.
    if get_addon_preferences().settings.use_follow_view:
        return

    bbox_w = max(bounds[2] - bounds[0], 1.0)
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    padding = minimap_state.view.inner_padding
    list_width = minimap_state.list.list_width

    def _base_scale(map_w: float, map_h: float) -> float:
        if list_width > 0 and minimap_state.list.list_placement == "TOP":
            top_inset = padding + _list_top_inset(list_width, ui_scale)
            inner_w = max(map_w * ui_scale - 2.0 * padding, 1.0)
            inner_h = max(map_h * ui_scale - padding - top_inset, 1.0)
            return min(inner_w / bbox_w, inner_h / bbox_h)
        left_inset = padding + list_width
        if list_width > 0:
            left_inset += 4.0 * ui_scale
        inner_w = max(map_w * ui_scale - padding - left_inset, 1.0)
        inner_h = max(map_h * ui_scale - 2.0 * padding, 1.0)
        return min(inner_w / bbox_w, inner_h / bbox_h)

    old_base = _base_scale(old_w, old_h)
    new_base = _base_scale(new_w, new_h)
    if old_base <= 0 or new_base <= 0:
        return
    ratio = new_base / old_base
    if abs(ratio - 1.0) < 1e-6:
        return
    pan_x, pan_y = minimap_state.view.pan
    minimap_state.view.pan = (pan_x * ratio, pan_y * ratio)


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


def _minimap_view_from_world_rect(
    minimap_state: MinimapState, rect: tuple[float, float, float, float] | list[float]
) -> tuple[float, tuple[float, float]]:
    """Return the ``(zoom, pan)`` view showing *rect* in the minimap.

    Inverse of :func:`_minimap_world_rect`; interpolating rects and mapping
    back couples pan and zoom exactly like Blender's rect animation.
    """
    _il, _ib, inner_w, _ih, _bw, _bh, base_scale, tree_cx, tree_cy = _compute_base_map_geom(minimap_state)
    world_w = max(rect[2] - rect[0], 1e-6)
    scale = inner_w / world_w
    zoom = scale / max(base_scale, 1e-6)
    world_cx = (rect[0] + rect[2]) / 2.0
    world_cy = (rect[1] + rect[3]) / 2.0
    return zoom, ((tree_cx - world_cx) * scale, (tree_cy - world_cy) * scale)


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
