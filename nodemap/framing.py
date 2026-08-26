"""Viewport and node-tree framing logic for the minimap."""

import bpy

from .helpers import (
    _EDITOR_FIT_MARGIN,
    MAX_FRAME_ZOOM,
    _expand_bounds_margin,
    _get_node_dims,
    _get_node_tree_bounds,
    _get_ui_scale,
)
from .state import _state
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

    Updates ``st.tree_bounds`` immediately (required for correct targets and
    drawing during animation). Returns ``(zoom, pan_x, pan_y)`` or ``None``
    when data is unavailable.
    """
    st = _state(area_ptr)
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

    _, _, _, mh = st.rect
    bounds = _expand_bounds_margin(bounds, _get_ui_scale(), mh, st.padding)
    st.tree_bounds = bounds

    addon = bpy.context.preferences.addons.get(__package__)
    follow = addon and getattr(addon.preferences.settings, "follow_view", False)

    if not follow:
        return 1.0, 0.0, 0.0

    visible = _get_visible_rect(space, region)
    if visible:
        c_min_x = min(bounds[0], visible[0])
        c_min_y = min(bounds[1], visible[1])
        c_max_x = max(bounds[2], visible[2])
        c_max_y = max(bounds[3], visible[3])
    else:
        c_min_x, c_min_y, c_max_x, c_max_y = bounds

    _, _, inner_w, inner_h, _, _, base_scale, tree_cx, tree_cy = _compute_base_map_geom(st)

    combined_w = max(c_max_x - c_min_x, 1.0)
    combined_h = max(c_max_y - c_min_y, 1.0)
    zoom = min(inner_w / (base_scale * combined_w), inner_h / (base_scale * combined_h), 1.0)

    combined_cx = (c_min_x + c_max_x) / 2
    combined_cy = (c_min_y + c_max_y) / 2

    pan_x = -(combined_cx - tree_cx) * base_scale * zoom
    pan_y = -(combined_cy - tree_cy) * base_scale * zoom
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
    st = _state(area_ptr)
    zoom, pan_x, pan_y = targets
    st.base_zoom = zoom
    st.zoom = zoom
    st.pan = [pan_x, pan_y]
    _redraw()


def _compute_frame_to_bounds_targets(
    target_bounds: tuple[float, float, float, float],
    fill: bool = False,
    area_ptr: int | None = None,
) -> tuple[float, float, float]:
    """Compute target zoom and pan to frame the given bounds without applying them.

    Returns ``(zoom, pan_x, pan_y)``.
    """
    st = _state(area_ptr)

    _, _, inner_w, inner_h, _, _, base_scale, tree_cx, tree_cy = _compute_base_map_geom(st)

    tw = max(target_bounds[2] - target_bounds[0], 1.0)
    th = max(target_bounds[3] - target_bounds[1], 1.0)
    if fill:
        zoom = min(inner_w / (base_scale * tw), inner_h / (base_scale * th))
    else:
        zoom = min(inner_w / (base_scale * tw), inner_h / (base_scale * th), 1.0)

    target_cx = (target_bounds[0] + target_bounds[2]) / 2
    target_cy = (target_bounds[1] + target_bounds[3]) / 2

    pan_x = -(target_cx - tree_cx) * base_scale * zoom
    pan_y = -(target_cy - tree_cy) * base_scale * zoom
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
    st = _state(area_ptr)
    zoom, pan_x, pan_y = _compute_frame_to_bounds_targets(target_bounds, fill, area_ptr)
    st.base_zoom = zoom
    st.zoom = zoom
    st.pan = [pan_x, pan_y]
    _redraw()


def _compute_center_pan(tree_x: float, tree_y: float, area_ptr: int | None = None) -> tuple[float, float]:
    """Compute minimap pan values that center the given tree point, keeping zoom."""
    st = _state(area_ptr)
    _, _, scale, tree_cx, tree_cy = _compute_map_transform(st)
    return -(tree_x - tree_cx) * scale, -(tree_y - tree_cy) * scale


def _get_selected_bounds(nodes) -> tuple[float, float, float, float] | None:
    """Return the ``(min_x, min_y, max_x, max_y)`` bounds of the selected nodes, or None."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for node in nodes:
        if not node.select:
            continue
        w, h = _get_node_dims(node)
        x, y = node.location_absolute.x, node.location_absolute.y
        min_x = min(min_x, x)
        max_x = max(max_x, x + w)
        min_y = min(min_y, y - h)
        max_y = max(max_y, y)
    if min_x == float("inf"):
        return None
    return min_x, min_y, max_x, max_y


def _compute_frame_selected_targets(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> tuple[float | None, float, float] | None:
    """Compute target zoom and pan for the selected nodes without applying them.

    Returns ``(zoom, pan_x, pan_y)`` where *zoom* is ``None`` when the current
    zoom should be kept (single regular node selected). Returns ``None`` when
    nothing is selected or data is unavailable.
    """
    st = _state(area_ptr)
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

    rect = st.rect
    _, _, mw, mh = rect
    st.tree_bounds = _expand_bounds_margin(_get_node_tree_bounds(node_tree.nodes), _get_ui_scale(), mh, st.padding)

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

    Fits multiple selections or a single frame node with a margin; centers a
    single regular node while keeping the current viewport size. Returns
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
    vw = visible[2] - visible[0]
    vh = visible[3] - visible[1]

    if len(selected) > 1 or selected[0].type == "FRAME":
        bw = max(max_x - min_x, 1.0)
        bh = max(max_y - min_y, 1.0)
        mx = bw * _EDITOR_FIT_MARGIN
        my = bh * _EDITOR_FIT_MARGIN
        left, bottom, right, top = min_x - mx, min_y - my, max_x + mx, max_y + my

        # Limit zoom-in so tiny selections do not magnify excessively.
        hw = max((right - left) / 2, vw / MAX_FRAME_ZOOM / 2)
        hh = max((top - bottom) / 2, vh / MAX_FRAME_ZOOM / 2)
        cx = (left + right) / 2
        cy = (bottom + top) / 2
        return cx - hw, cy - hh, cx + hw, cy + hh

    return sel_cx - vw / 2, sel_cy - vh / 2, sel_cx + vw / 2, sel_cy + vh / 2


def frame_selected(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the selected node(s).

    Zooms to fit multiple selections or a single frame; centers a single
    regular node without changing the zoom level.
    """
    targets = _compute_frame_selected_targets(space, region, area_ptr)
    if targets is None:
        return
    zoom, pan_x, pan_y = targets
    st = _state(area_ptr)
    if zoom is not None:
        st.base_zoom = zoom
        st.zoom = zoom
    st.pan = [pan_x, pan_y]
    _redraw()


def frame_view(
    space: bpy.types.SpaceNodeEditor | None = None,
    region: bpy.types.Region | None = None,
    area_ptr: int | None = None,
) -> None:
    """Adjust minimap zoom/pan to frame the current editor viewport."""
    st = _state(area_ptr)
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

    addon = bpy.context.preferences.addons.get(__package__)
    fill = addon and getattr(addon.preferences.settings, "frame_view_fill", False)

    rect = st.rect
    _, _, mw, mh = rect
    st.tree_bounds = _expand_bounds_margin(_get_node_tree_bounds(node_tree.nodes), _get_ui_scale(), mh, st.padding)
    _frame_to_bounds(visible, fill=fill, area_ptr=area_ptr)


def _redraw() -> None:
    """Trigger a redraw of all NODE_EDITOR areas."""
    from .helpers import redraw_ui

    redraw_ui("NODE_EDITOR")
