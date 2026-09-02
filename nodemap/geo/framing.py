"""Provide viewport and node-tree framing logic for the minimap."""

import bpy

from ..core.constants import EDITOR_FIT_MARGIN, MAX_FRAME_ZOOM
from ..core.helpers import (
    _expand_bounds_margin,
    _get_node_dims,
    _get_node_tree_bounds,
    _get_ui_scale,
    get_addon_preferences,
)
from ..core.state import _state
from .transforms import (
    _compute_base_map_geom,
    _compute_map_transform,
    _get_visible_rect,
)


def _compute_frame_all_targets(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> tuple[float, float, float] | None:
    """Compute target zoom and pan to frame the entire node tree.

    Update ``minimap_state.view.tree_bounds`` immediately and return
    ``(zoom, pan_x, pan_y)`` or ``None`` when data is unavailable.
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

    bounds = _get_node_tree_bounds(node_tree.nodes)

    _, _, _, map_h = minimap_state.view.rect
    bounds = _expand_bounds_margin(bounds, _get_ui_scale(), map_h, minimap_state.view.inner_padding)
    minimap_state.view.tree_bounds = bounds

    addon_prefs_block = get_addon_preferences()
    follow = addon_prefs_block and addon_prefs_block.settings.follow_view

    if not follow:
        return 1.0, 0.0, 0.0

    visible = _get_visible_rect(space, region)
    if visible:
        combined_min_x = min(bounds[0], visible[0])
        combined_min_y = min(bounds[1], visible[1])
        combined_max_x = max(bounds[2], visible[2])
        combined_max_y = max(bounds[3], visible[3])
    else:
        combined_min_x, combined_min_y, combined_max_x, combined_max_y = bounds

    _, _, inner_w, inner_h, _, _, base_scale, tree_center_x, tree_center_y = _compute_base_map_geom(minimap_state)

    combined_w = max(combined_max_x - combined_min_x, 1.0)
    combined_h = max(combined_max_y - combined_min_y, 1.0)
    zoom = min(inner_w / (base_scale * combined_w), inner_h / (base_scale * combined_h), 1.0)

    combined_cx = (combined_min_x + combined_max_x) / 2
    combined_cy = (combined_min_y + combined_max_y) / 2

    pan_x = -(combined_cx - tree_center_x) * base_scale * zoom
    pan_y = -(combined_cy - tree_center_y) * base_scale * zoom
    return zoom, pan_x, pan_y


def frame_all(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the entire node tree.

    When ``follow_view`` is enabled the editor viewport is included in the
    frame so that clamping cannot clip nodes afterward.
    """
    targets = _compute_frame_all_targets(space, region, area_ptr)
    if targets is None:
        return
    minimap_state = _state(area_ptr)
    zoom, pan_x, pan_y = targets
    minimap_state.view.anchor_zoom = zoom
    minimap_state.view.user_zoom = zoom
    minimap_state.view.pan = (pan_x, pan_y)
    _redraw()


def _compute_frame_to_bounds_targets(
    target_bounds: tuple[float, float, float, float],
    fill: bool = False,
    area_ptr: int | None = None,
) -> tuple[float, float, float]:
    """Compute target zoom and pan to frame the given bounds without applying them.

    Return ``(zoom, pan_x, pan_y)``.
    """
    minimap_state = _state(area_ptr)

    _, _, inner_w, inner_h, _, _, base_scale, tree_center_x, tree_center_y = _compute_base_map_geom(minimap_state)

    target_w = max(target_bounds[2] - target_bounds[0], 1.0)
    target_h = max(target_bounds[3] - target_bounds[1], 1.0)
    if fill:
        zoom = min(inner_w / (base_scale * target_w), inner_h / (base_scale * target_h))
    else:
        zoom = min(inner_w / (base_scale * target_w), inner_h / (base_scale * target_h), 1.0)

    target_cx = (target_bounds[0] + target_bounds[2]) / 2
    target_cy = (target_bounds[1] + target_bounds[3]) / 2

    pan_x = -(target_cx - tree_center_x) * base_scale * zoom
    pan_y = -(target_cy - tree_center_y) * base_scale * zoom
    return zoom, pan_x, pan_y


def _frame_to_bounds(
    target_bounds: tuple[float, float, float, float],
    fill: bool = False,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the given bounds in tree coordinates.

    When *fill* True the bounds are zoomed to entirely fill the minimap
    (one axis may clip); when False the bounds frame within the minimap
    (empty space may remain).
    """
    minimap_state = _state(area_ptr)
    zoom, pan_x, pan_y = _compute_frame_to_bounds_targets(target_bounds, fill, area_ptr)
    minimap_state.view.anchor_zoom = zoom
    minimap_state.view.user_zoom = zoom
    minimap_state.view.pan = (pan_x, pan_y)
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
) -> tuple[float | None, float, float] | None:
    """Compute target zoom and pan for the selected nodes without applying them.

    Return ``(zoom, pan_x, pan_y)`` where *zoom* is ``None`` when the current
    zoom should be kept. Return ``None`` when nothing is selected or data is
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

    bounds = _get_selected_bounds(selected)
    if bounds is None:
        return None
    min_x, min_y, max_x, max_y = bounds

    rect = minimap_state.view.rect
    _, _, map_w, map_h = rect
    minimap_state.view.tree_bounds = _expand_bounds_margin(
        _get_node_tree_bounds(node_tree.nodes), _get_ui_scale(), map_h, minimap_state.view.inner_padding
    )

    if len(selected) > 1 or selected[0].type == "FRAME":
        zoom, pan_x, pan_y = _compute_frame_to_bounds_targets(
            (min_x, min_y, max_x, max_y), fill=True, area_ptr=area_ptr
        )
        return min(zoom, MAX_FRAME_ZOOM), pan_x, pan_y

    pan_x, pan_y = _compute_center_pan((min_x + max_x) / 2, (min_y + max_y) / 2, area_ptr)
    return None, pan_x, pan_y


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
    zoom, pan_x, pan_y = targets
    minimap_state = _state(area_ptr)
    if zoom is not None:
        minimap_state.view.anchor_zoom = zoom
        minimap_state.view.user_zoom = zoom
    minimap_state.view.pan = (pan_x, pan_y)
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

    addon_prefs_block = get_addon_preferences()
    fill = addon_prefs_block and addon_prefs_block.settings.frame_view_fill

    rect = minimap_state.view.rect
    _, _, map_w, map_h = rect
    minimap_state.view.tree_bounds = _expand_bounds_margin(
        _get_node_tree_bounds(node_tree.nodes), _get_ui_scale(), map_h, minimap_state.view.inner_padding
    )
    _frame_to_bounds(visible, fill=fill, area_ptr=area_ptr)


def _redraw() -> None:
    """Redraw all NODE_EDITOR areas."""
    from ..core.helpers import redraw_ui

    redraw_ui("NODE_EDITOR")
