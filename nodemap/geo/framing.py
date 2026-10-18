"""Provide viewport and node-tree framing logic for the minimap."""

import bpy

from ..core.constants import CONTENT_PADDING, EDITOR_FIT_MARGIN, MAX_FRAME_ZOOM
from ..core.helpers import (
    _expand_bounds_margin,
    _get_node_dims,
    _get_node_tree_bounds,
    _get_ui_scale,
)
from ..core.state import _state
from .transforms import (
    _compute_map_transform,
    _follow_view_enabled,
    _get_visible_rect,
    _mark_view_authored,
    _minimap_world_rect,
    _set_framed_rect,
)


def _compute_frame_all_rect(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> tuple[float, float, float, float] | None:
    """Return the tree-space rect that frames the entire node tree.

    Update ``minimap_state.view.tree_bounds`` immediately and return the rect,
    or ``None`` when data is unavailable. When ``use_follow_view`` is enabled
    the editor viewport is included in the rect so that clamping cannot clip
    nodes afterward.
    """
    minimap_state = _state(area_ptr)
    if space is None:
        space = bpy.context.space_data
    if region is None:
        region = bpy.context.region
    if not space or not region:
        return None
    node_tree = space.edit_tree
    if not node_tree:
        return None

    bounds = minimap_state.view.raw_tree_bounds
    if bounds is None:
        bounds = _get_node_tree_bounds(node_tree.nodes)

    _, _, _, map_h = minimap_state.view.rect
    bounds = _expand_bounds_margin(bounds, _get_ui_scale(), map_h, minimap_state.view.inner_padding)
    minimap_state.view.tree_bounds = bounds

    if not _follow_view_enabled():
        return bounds

    visible = _get_visible_rect(space, region)
    if not visible:
        return bounds
    return (
        min(bounds[0], visible[0]),
        min(bounds[1], visible[1]),
        max(bounds[2], visible[2]),
        max(bounds[3], visible[3]),
    )


def frame_all(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the entire node tree."""
    target = _compute_frame_all_rect(space, region, area_ptr)
    if target is None:
        return
    minimap_state = _state(area_ptr)
    _set_framed_rect(minimap_state, target)
    _mark_view_authored(minimap_state)
    # Force a batch rebuild on the next draw: the one-shot zoom/pan change must
    # re-bake nodes/text at the new scale immediately, not wait for a settle
    # window that may never flush (operators stop after a single redraw).
    minimap_state.cache._batches_dirty = True
    _redraw()


def _frame_to_bounds(
    target_bounds: tuple[float, float, float, float],
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the given bounds in tree coordinates."""
    minimap_state = _state(area_ptr)
    _set_framed_rect(minimap_state, target_bounds)
    _mark_view_authored(minimap_state)
    # Force a batch rebuild on the next draw (see frame_all).
    minimap_state.cache._batches_dirty = True
    _redraw()


def _compute_center_pan(tree_x: float, tree_y: float, area_ptr: int | None = None) -> tuple[float, float]:
    """Compute minimap pan values that center the given tree point, keeping zoom."""
    minimap_state = _state(area_ptr)
    _, _, scale, tree_center_x, tree_center_y = _compute_map_transform(minimap_state)
    return -(tree_x - tree_center_x) * scale, -(tree_y - tree_center_y) * scale


def _get_selected_bounds(nodes) -> tuple[float, float, float, float] | None:
    """Return the ``(min_x, min_y, max_x, max_y)`` bounds of the selected nodes, or None."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for node in nodes:
        if not node.select:
            continue
        node_w, node_h = _get_node_dims(node)
        node_x, node_y = node.location_absolute.x, node.location_absolute.y
        min_x = min(min_x, node_x)
        max_x = max(max_x, node_x + node_w)
        min_y = min(min_y, node_y - node_h)
        max_y = max(max_y, node_y)
    if min_x == float("inf"):
        return None
    return min_x, min_y, max_x, max_y


def _compute_frame_selected_targets(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> tuple[float, float, float, float] | None:
    """Return the tree-space rect that frames the selected nodes.

    Fit multiple selections or a single frame node with a margin; a single
    regular node yields the visible rect centered on it, which keeps the
    current zoom. Return ``None`` when nothing is selected or data is
    unavailable.
    """
    minimap_state = _state(area_ptr)
    if space is None:
        space = bpy.context.space_data
    if not space or space.type != "NODE_EDITOR":
        return None
    node_tree = space.edit_tree
    if not node_tree:
        return None

    selected = [n for n in node_tree.nodes if n.select]
    if not selected:
        return None

    # Use cached bounds from the snapshot when available, otherwise compute.
    bounds = minimap_state.view.snapshot_selected_bounds
    if bounds is None:
        bounds = _get_selected_bounds(selected)
    if bounds is None:
        return None
    min_x, min_y, max_x, max_y = bounds

    rect = minimap_state.view.rect
    _, _, map_w, map_h = rect
    raw_bounds = minimap_state.view.raw_tree_bounds
    if raw_bounds is None:
        raw_bounds = _get_node_tree_bounds(node_tree.nodes)
    minimap_state.view.tree_bounds = _expand_bounds_margin(
        raw_bounds, _get_ui_scale(), map_h, minimap_state.view.inner_padding
    )

    if len(selected) > 1 or selected[0].type == "FRAME":
        margin_px = (CONTENT_PADDING * 2) * _get_ui_scale()
        bounds_h = max(max_y - min_y, 1.0)
        inner_h = max(rect[3] - 2 * minimap_state.view.inner_padding, 1.0)
        margin = margin_px * bounds_h / inner_h
        return (min_x - margin, min_y - margin, max_x + margin, max_y + margin)

    pan_x, pan_y = _compute_center_pan((min_x + max_x) / 2, (min_y + max_y) / 2, area_ptr)
    return _minimap_world_rect(minimap_state, minimap_state.view.user_zoom, (pan_x, pan_y))


def _compute_editor_frame_selected_targets(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
) -> tuple[float, float, float, float] | None:
    """Compute the editor viewport rect that frames the selected nodes.

    Fit multiple selections or a single frame node with a margin; center a
    single regular node while keeping the current viewport size. Return
    tree-space ``(left, bottom, right, top)`` or ``None`` when unavailable.
    """
    if space is None:
        space = bpy.context.space_data
    if region is None:
        region = bpy.context.region
    if not space or space.type != "NODE_EDITOR" or not region:
        return None
    node_tree = space.edit_tree
    if not node_tree:
        return None

    visible = _get_visible_rect(space, region)
    if not visible:
        return None

    selected = [n for n in node_tree.nodes if n.select]
    if not selected:
        return None
    bounds = _get_selected_bounds(selected)
    if bounds is None:
        return None
    min_x, min_y, max_x, max_y = bounds
    sel_cx = (min_x + max_x) / 2
    sel_cy = (min_y + max_y) / 2
    viewport_w = visible[2] - visible[0]
    viewport_h = visible[3] - visible[1]

    if len(selected) > 1 or selected[0].type == "FRAME":
        bounds_w = max(max_x - min_x, 1.0)
        bounds_h = max(max_y - min_y, 1.0)
        margin_x = bounds_w * EDITOR_FIT_MARGIN
        margin_y = bounds_h * EDITOR_FIT_MARGIN
        left, bottom, right, top = min_x - margin_x, min_y - margin_y, max_x + margin_x, max_y + margin_y

        # Limit zoom-in so tiny selections do not magnify excessively.
        half_w = max((right - left) / 2, viewport_w / MAX_FRAME_ZOOM / 2)
        half_h = max((top - bottom) / 2, viewport_h / MAX_FRAME_ZOOM / 2)
        cx = (left + right) / 2
        cy = (bottom + top) / 2
        return cx - half_w, cy - half_h, cx + half_w, cy + half_h

    return sel_cx - viewport_w / 2, sel_cy - viewport_h / 2, sel_cx + viewport_w / 2, sel_cy + viewport_h / 2


def frame_selected(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom and pan to frame the selected nodes.

    Zoom to fit multiple selections or a single frame; center a single
    regular node without changing the zoom level.
    """
    targets = _compute_frame_selected_targets(space, region, area_ptr)
    if targets is None:
        return
    minimap_state = _state(area_ptr)
    _set_framed_rect(minimap_state, targets)
    _mark_view_authored(minimap_state)
    # Force a batch rebuild on the next draw (see frame_all).
    minimap_state.cache._batches_dirty = True
    _redraw()


def frame_view(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the current editor viewport."""
    minimap_state = _state(area_ptr)
    if space is None:
        space = bpy.context.space_data
    if region is None:
        region = bpy.context.region
    if not space or not region:
        return
    node_tree = space.edit_tree
    if not node_tree:
        return

    visible = _get_visible_rect(space, region)
    if not visible:
        return

    rect = minimap_state.view.rect
    _, _, map_w, map_h = rect
    raw_bounds = minimap_state.view.raw_tree_bounds
    if raw_bounds is None:
        raw_bounds = _get_node_tree_bounds(node_tree.nodes)
    minimap_state.view.tree_bounds = _expand_bounds_margin(
        raw_bounds, _get_ui_scale(), map_h, minimap_state.view.inner_padding
    )
    _frame_to_bounds(visible, area_ptr)


def _redraw() -> None:
    """Redraw all NODE_EDITOR areas."""
    from ..core.helpers import redraw_ui

    redraw_ui("NODE_EDITOR")
