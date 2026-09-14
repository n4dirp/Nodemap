"""Selection helpers for node editor interaction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import bpy

from .. import __package__ as base_package
from ..core.constants import BUTTON_SIZE, ELEMENT_GAP
from ..core.helpers import _find_node_at, _get_node_dims, _get_ui_scale, get_addon_preferences
from ..core.list_filter import (
    _ROW_CHILD,
    _ROW_HEADER,
    _iter_type_list_layout,
    filter_type_list,
    flat_type_nodes,
    iter_flat_list_layout,
)

if TYPE_CHECKING:
    from bpy.types import Context, Region

    from ..core.state import MinimapState
    from .navigate import NODEMAP_OT_navigate

logger = logging.getLogger(base_package)


def node_select_location(node) -> tuple[float, float]:
    """Tree-space coordinate to emulate a click on *node*.

    Native Blender only selects frame nodes when clicked on their
    header/border, so frames are redirected to their header. Other nodes
    reuse the minimap hit-test dims so collapsed or never-drawn nodes
    probe inside their actual drawn bounds.
    """
    if node.type == "FRAME":
        return node.location_absolute.x + 15, node.location_absolute.y - 15
    w, h = _get_node_dims(node)
    return node.location_absolute.x + w / 2.0, node.location_absolute.y - h / 2.0


def node_fallback_location(node) -> tuple[float, float]:
    """Tree-space point near the node's top-left interior edge.

    Falls inside a collapsed node's header strip whatever its drawn label
    width, covering center probes that miss due to stale dimensions.
    """
    x, y = node.location_absolute.x, node.location_absolute.y
    if node.type == "FRAME":
        return x + 15, y - 15
    return x + 10.0, y - 10.0


def project_tree_to_region(region: Region | None, tree_x: float, tree_y: float) -> tuple[int, int] | None:
    """Project a tree-space point to editor region pixels via view2d.

    View2d coordinates are tree coordinates scaled by the UI scale factor
    (mirroring ``_get_visible_rect``), so the point is scaled here first.
    """
    view2d = region.view2d if region else None
    if not view2d:
        return None
    ui = _get_ui_scale()
    pt = view2d.view_to_region(tree_x * ui, tree_y * ui, clip=False)
    if not pt:
        return None
    return int(pt[0]), int(pt[1])


def select_node_via_operator(
    op: NODEMAP_OT_navigate,
    context: Context,
    node,
    extend: bool,
    deselect_all: bool,
) -> bool:
    """Select *node* via the native ``node.select`` operator.

    Projects candidate tree positions into the editor's region coordinates
    and passes those to ``bpy.ops.node.select``, emulating a standard UI
    click. This avoids the NodeTree "modified" tag that Python property
    assignment (``node.select = True``) triggers, which forces a full EEVEE
    material rebuild. Probes run from the node center to its header edge
    and each pick is verified against ``node.select``, so a silent miss
    retries instead of reporting success; returns False only when no probe
    selects the node so callers can fall back to the property API.
    """
    if not op._region or not op._region.view2d:
        return False
    probes = [node_select_location(node)]
    fallback = node_fallback_location(node)
    if fallback != probes[0]:
        probes.append(fallback)

    tree_nodes = getattr(getattr(node, "id_data", None), "nodes", None)
    for probe_index, (probe_tx, probe_ty) in enumerate(probes):
        projected = project_tree_to_region(op._region, probe_tx, probe_ty)
        if projected is None:
            continue
        proj_x, proj_y = projected
        keep = None
        if probe_index and extend and tree_nodes:
            keep = {n.name for n in tree_nodes if n.select}
        kwargs: dict = {"extend": extend}
        if bpy.app.version >= (3, 0, 0):
            kwargs["location"] = (proj_x, proj_y)
            kwargs["deselect_all"] = deselect_all
        else:
            kwargs["mouse_x"] = proj_x
            kwargs["mouse_y"] = proj_y
        try:
            with op._override_ctx(context):
                bpy.ops.node.select(**kwargs)
        except Exception as e:
            logger.debug("Failed to select via operator: %s", e)
            continue
        try:
            if not node.select:
                continue
        except ReferenceError:
            return False
        if keep is not None and tree_nodes:
            # Release neighbors an overlapping retry probe picked up unintentionally.
            for other in tree_nodes:
                if other.select and other.name != node.name and other.name not in keep:
                    other.select = False
        return True
    return False


def handle_click_selection(
    op: NODEMAP_OT_navigate,
    context: Context,
    state: MinimapState,
    frame: bool = False,
    extend: bool = False,
    toggle: bool = False,
) -> None:
    """Handle a click on the minimap: find the node under cursor and select it.

    *extend* adds the node to the current selection, *toggle* flips its
    selection state, and *frame* eases the editor onto the node after the
    selection change.
    """
    from .navigate import _region_to_tree

    space = op._space
    if not space or space.type != "NODE_EDITOR":
        return
    node_tree = space.edit_tree
    if not node_tree or not node_tree.nodes:
        return

    tree_coord = _region_to_tree(op._mouse_x, op._mouse_y, state)
    if tree_coord is None:
        return

    node = _find_node_at(node_tree.nodes, tree_coord[0], tree_coord[1])
    if node:
        if toggle:
            select_single_node(op, context, node.name, toggle=True)
        elif not select_node_via_operator(op, context, node, extend=extend, deselect_all=not extend):
            # Fallback for API changes (may trigger EEVEE compile)
            if extend:
                node.select = not node.select
                if node.select:
                    node_tree.nodes.active = node
            else:
                for n in node_tree.nodes:
                    n.select = False
                node.select = True
                node_tree.nodes.active = node

        if frame:
            if not op._anim.view_selected_animated(context):
                try:
                    with op._override_ctx(context):
                        bpy.ops.node.view_selected()
                except RuntimeError:
                    pass

    state.list.hovered_list_row = None
    state.interaction.hovered_node_id = None
    op._redraw_ui()


def apply_list_range(
    op: NODEMAP_OT_navigate,
    context: Context,
    state: MinimapState,
    target_key: tuple,
    last_row_index: int,
) -> None:
    """Select all visible rows between the last-clicked and *target_key*.

    Replaces the current selection with the contiguous range, matching
    standard file-explorer Shift-click behaviour. The anchor
    (``_list_last_row_index``) is **not** moved — it stays at the last
    plain-clicked row so repeated Shift-clicks expand from the same origin.
    """
    keys = state.list.visible_row_keys
    index_map = state.list.visible_row_index_map
    target_idx = index_map.get(target_key)
    if target_idx is None:
        # Fallback for stale map.
        try:
            target_idx = keys.index(target_key)
        except ValueError:
            return
    last = last_row_index
    if last < 0 or last >= len(keys):
        lo, hi = target_idx, target_idx
    else:
        lo, hi = min(last, target_idx), max(last, target_idx)

    # Deselect everything first so the range *replaces* the selection.
    space = op._space
    node_tree = space.edit_tree if space else None
    if not node_tree:
        return
    try:
        with op._override_ctx(context):
            bpy.ops.node.select_all(action="DESELECT")
    except RuntimeError:
        pass

    for key_idx in range(lo, hi + 1):
        key = keys[key_idx]
        if key[0] == "header":
            select_type_nodes(op, context, key[1], extend=True)
        elif key[0] == "child":
            node = node_tree.nodes.get(key[2])
            if node:
                if not select_node_via_operator(op, context, node, extend=True, deselect_all=False):
                    node.select = True
    state.list.arrow_key = target_key
    op._redraw_ui()


def select_type_nodes(
    op: NODEMAP_OT_navigate,
    context: Context,
    label: str,
    extend: bool = False,
    toggle: bool = False,
) -> None:
    """Select all editor nodes whose compiled type label matches *label*.

    When *extend* is True the current selection is preserved and the
    matching nodes are added. When *toggle* is True the behaviour
    depends on the current state: if every matching node is already
    selected they are all deselected, otherwise they are all selected.
    Selecting leaves the group's first node active.
    """
    space = op._space
    state = op._state
    if not space or space.type != "NODE_EDITOR" or not state:
        return
    node_tree = space.edit_tree
    if not node_tree:
        return
    type_nodes = (state.tree_data() or {}).get("type_nodes") or {}
    names = type_nodes.get(label)
    if not names:
        return

    if toggle:
        all_sel = all((node_tree.nodes.get(n) is not None and node_tree.nodes[n].select) for n in names)
        if all_sel:
            # Deselect only this type group, preserving other selections.
            for name in names:
                node = node_tree.nodes.get(name)
                if node and node.select:
                    node.select = False
            op._redraw_ui()
            return
        else:
            deselect = False
            extend = True
    else:
        deselect = not extend

    if deselect:
        try:
            with op._override_ctx(context):
                bpy.ops.node.select_all(action="DESELECT")
        except RuntimeError:
            pass
    # After the upfront deselect the selection is empty, so every
    # addition must use extend to accumulate the whole group. Passing
    # extend=False would replace the previous node on each iteration
    # and leave only the last one selected (and framed).
    node_extend = extend or deselect
    first_node = None
    for name in names:
        node = node_tree.nodes.get(name)
        if node:
            if first_node is None:
                first_node = node
            # Native operator keeps selection/additive state and sets the
            # active node without tagging the NodeTree for an EEVEE rebuild.
            if not select_node_via_operator(op, context, node, extend=node_extend, deselect_all=False):
                node.select = True
    if first_node is not None:
        # Each operator pick above re-targets the active node, leaving the
        # last one active; pin the group's first item instead.
        node_tree.nodes.active = first_node
    op._redraw_ui()


def select_single_node(
    op: NODEMAP_OT_navigate,
    context: Context,
    node_name: str,
    extend: bool = False,
    toggle: bool = False,
) -> None:
    """Select only the editor node whose compiled name matches *node_name*.

    When *extend* is True the current selection is preserved and the
    node is added. When *toggle* is True the node's selection state
    is flipped instead of replaced.
    """
    space = op._space
    state = op._state
    if not space or space.type != "NODE_EDITOR" or not state:
        return
    node_tree = space.edit_tree
    if not node_tree:
        return
    node = node_tree.nodes.get(node_name)
    if not node:
        return
    if toggle:
        node.select = not node.select
        if node.select:
            node_tree.nodes.active = node
    else:
        if not extend:
            try:
                with op._override_ctx(context):
                    bpy.ops.node.select_all(action="DESELECT")
            except RuntimeError:
                pass
        if not select_node_via_operator(op, context, node, extend=extend, deselect_all=False):
            node.select = True
            node_tree.nodes.active = node
    op._redraw_ui()


def _find_list_row_index(
    rows: list[tuple],
    label: str,
    node_name: str,
    target_is_child: bool,
) -> int | None:
    """Return the index of *label*'s target row in *rows*, or None.

    The node's own child row is preferred when it is listed; the header row
    is the fallback (single-node types list no child rows, and a child may be
    hidden by an active search). Returns None when the label has no row at all.
    """
    header_index = None
    child_index = None
    for index, (kind, row_label, row_node_name, _local_y) in enumerate(rows):
        if row_label != label:
            continue
        if kind == _ROW_HEADER:
            if header_index is None:
                header_index = index
        elif kind == _ROW_CHILD and row_node_name == node_name:
            if child_index is None:
                child_index = index
    if target_is_child and child_index is not None:
        return child_index
    if header_index is not None:
        return header_index
    return child_index


def _scroll_list_to_row(
    state: MinimapState,
    row_count: int,
    row_index: int,
    row_h: float,
    settings,
) -> bool:
    """Scroll the type list to reveal *row_index*; return True when it changed.

    The row is aligned to the nearest viewport edge: a row below the current
    view lands flush with the bottom, a row above flush with the top, so a
    row one past the edge scrolls the list by exactly one row. A row already
    visible leaves the scroll untouched. The scroll is clamped to
    *row_count*'s range; the next draw pass re-clamps to the exact (possibly
    grown) scroll max.
    """
    zone_rect = state.list.list_zone_rect
    if zone_rect is None:
        return False
    ui_scale = _get_ui_scale()
    search_h = BUTTON_SIZE * ui_scale if settings.show_search_bar else 0.0
    search_gap = ELEMENT_GAP * ui_scale if settings.show_search_bar else 0.0
    row_pad_v = ui_scale
    view_h = max(zone_rect[3] - search_h - search_gap - 2 * row_pad_v - ui_scale, row_h)
    scroll_max = max(0.0, row_count * row_h - view_h)

    scroll = state.list.scroll
    row_top = row_index * row_h
    row_bottom = (row_index + 1) * row_h
    if row_top < scroll + view_h and row_bottom > scroll:
        return False

    if row_bottom <= scroll:
        new_scroll = row_top
    else:
        new_scroll = row_bottom - view_h
    new_scroll = min(max(new_scroll, 0.0), scroll_max)
    if new_scroll == scroll:
        return False
    state.list.scroll = new_scroll
    return True


def _full_list_rows(state: MinimapState, settings) -> list[tuple]:
    """Return all type-list rows as ``(kind, label, node_name)`` in display order.

    Mirrors the row build the draw pass uses, but without viewport culling so
    keyboard navigation can reach rows above and below the scrolled view.
    """
    tree_data = state.tree_data() or {}
    type_nodes = tree_data.get("type_nodes") or {}
    type_stats = tree_data.get("type_stats") or {}
    search_texts = tree_data.get("type_search") or None

    if not bool(getattr(settings, "use_group_by_type", True)):
        nodes = flat_type_nodes(type_nodes, state.list.search_query, search_texts=search_texts)
        row_h = state.list.row_height
        rows = iter_flat_list_layout(nodes, row_h)
        return [(kind, label, node_name) for kind, label, node_name, _local_y in rows]

    visible, effective_expanded, filtered_children = filter_type_list(
        type_stats,
        type_nodes,
        state.list.expanded,
        state.list.search_query,
        search_texts=search_texts,
    )
    effective_expanded = effective_expanded or set()
    filtered_children = filtered_children or {}

    if not state.list.search_query.strip():
        if settings.type_list_sort == "NAME":
            visible.sort(key=lambda label_count: label_count[0].lower())
        else:
            visible.sort(key=lambda label_count: (-label_count[1], label_count[0]))

    entries: list[tuple[str, str, float, int]] = []
    for entry_label, display_count in visible:
        full_count = type_stats.get(entry_label, display_count)
        entries.append((entry_label, str(display_count), 0.0, full_count))

    row_h = state.list.row_height
    rows = _iter_type_list_layout(entries, filtered_children, effective_expanded, row_h)
    return [(kind, label, node_name) for kind, label, node_name, _local_y in rows]


def handle_list_arrow(
    op: NODEMAP_OT_navigate,
    context: Context,
    state: MinimapState,
    settings,
    direction: int,
) -> bool:
    """Move the type-list selection one row along *direction* and select it.

    *direction* is -1 for up and +1 for down. The move starts from the
    editor active node's row, falling back to the last arrow-selected row,
    then to the hovered row, then to the list edge — mouse hover alone
    never redirects the walk. While the walk cursor sits inside the active
    node's group it stays authoritative, so stepping onto a header (which
    activates the group's first item) still continues past it on the next
    press. When the active node sits in a collapsed group that group is
    expanded first — but only with Follow Active enabled — so the walk steps
    through its items instead of hopping header to header. The target row is scrolled into view (aligned to the
    viewport edge when it lies outside the visible area) and hovered (so
    the minimap highlights it like a mouse hover), and selected with
    everything else deselected. Return True when the key was handled,
    False when the list is empty so the caller lets the key pass through
    to the Node Editor.
    """
    rows = _full_list_rows(state, settings)
    if not rows:
        return False

    index_of: dict[tuple, int] = {}
    for row_index, (kind, label, node_name) in enumerate(rows):
        if kind == _ROW_HEADER:
            index_of.setdefault(("header", label), row_index)
        else:
            index_of.setdefault(("child", label, node_name), row_index)

    start = None
    active_label = None
    active_row = None
    node_tree = op._space.edit_tree if op._space else None
    active_node = node_tree.nodes.active if node_tree else None
    if active_node is not None:
        tree_data = state.tree_data() or {}
        type_nodes = tree_data.get("type_nodes") or {}
        for type_label, names in type_nodes.items():
            if active_node.name in names:
                active_label = type_label
                active_row = index_of.get(("child", type_label, active_node.name))
                if active_row is None:
                    active_row = index_of.get(("header", type_label))
                break
    arrow_key = state.list.arrow_key
    if arrow_key is not None and active_label is not None and tuple(arrow_key)[1] == active_label:
        # The walk cursor is inside the active node's group: it stays
        # authoritative so stepping onto a header (which activates the
        # group's first item) still continues past it on the next press
        # instead of bouncing back to that same header.
        start = index_of.get(tuple(arrow_key))
    if start is None:
        start = active_row
    if (
        active_label is not None
        and start is not None
        and start == index_of.get(("header", active_label))
        and bool(getattr(settings, "use_group_by_type", True))
        and not state.list.search_query.strip()
        and bool(getattr(settings, "use_follow_active", False))
    ):
        type_stats = (state.tree_data() or {}).get("type_stats") or {}
        if type_stats.get(active_label, 0) > 1 and active_label not in state.list.expanded:
            # The editor selection lives in a collapsed group: expand it so
            # the walk starts at the active node's own row and steps through
            # the group's items instead of hopping header to header.
            state.list.expanded.add(active_label)
            state.cache.list_key = None
            rows = _full_list_rows(state, settings)
            index_of = {}
            for row_index, (kind, label, node_name) in enumerate(rows):
                if kind == _ROW_HEADER:
                    index_of.setdefault(("header", label), row_index)
                else:
                    index_of.setdefault(("child", label, node_name), row_index)
            start = index_of.get(("child", active_label, active_node.name), index_of.get(("header", active_label)))
    if start is None and arrow_key is not None:
        start = index_of.get(tuple(arrow_key))
    if start is None:
        child_hover = state.list.hovered_list_row
        if child_hover is not None:
            start = index_of.get(("child", child_hover[0], child_hover[1]))
        if start is None and state.list.hovered_type_label is not None:
            start = index_of.get(("header", state.list.hovered_type_label))
    if start is None:
        start = -1 if direction > 0 else len(rows)

    target = min(max(start + direction, 0), len(rows) - 1)
    kind, label, node_name = rows[target]
    if kind == _ROW_HEADER:
        target_key: tuple = ("header", label)
    else:
        target_key = ("child", label, node_name)

    state.request_immediate_compile()
    if kind == _ROW_HEADER:
        state.list.hovered_type_label = label
        state.list.hovered_list_row = None
        state.interaction.hovered_node_id = None
        select_type_nodes(op, context, label)
    else:
        state.list.hovered_type_label = None
        state.list.hovered_list_row = (label, node_name)
        state.interaction.hovered_node_id = node_name
        select_single_node(op, context, node_name)
    _scroll_list_to_row(state, len(rows), target, state.list.row_height, settings)
    state.list.arrow_key = target_key
    op._list_last_row_index = state.list.visible_row_index_map.get(target_key, op._list_last_row_index)
    op._redraw_ui()
    return True


def handle_list_expand(
    op: NODEMAP_OT_navigate,
    context: Context,
    state: MinimapState,
    settings,
    expand: bool,
) -> bool:
    """Collapse or expand the active type-list group and select it.

    *expand* is True for the right arrow (expand / drill in) and False for
    the left arrow (collapse). The active group resolves from the active
    node's group, then the arrow cursor, then the hovered row — matching
    :func:`handle_list_arrow`. Expanding selects the group's first child,
    collapsing selects the group header so the active node stays visible.
    Return True when the key was handled, False when the list is empty or
    no active group resolves, so the caller lets the key pass through.
    """
    if not bool(getattr(settings, "use_group_by_type", True)):
        return False
    rows = _full_list_rows(state, settings)
    if not rows:
        return False

    index_of: dict[tuple, int] = {}
    for row_index, (kind, label, _node_name) in enumerate(rows):
        if kind == _ROW_HEADER:
            index_of.setdefault(("header", label), row_index)

    label = None
    node_tree = op._space.edit_tree if op._space else None
    active_node = node_tree.nodes.active if node_tree else None
    if active_node is not None:
        tree_data = state.tree_data() or {}
        type_nodes = tree_data.get("type_nodes") or {}
        for type_label, names in type_nodes.items():
            if active_node.name in names:
                if index_of.get(("header", type_label)) is not None:
                    label = type_label
                break
    if label is None:
        arrow_key = state.list.arrow_key
        if arrow_key is not None and index_of.get(("header", arrow_key[1])) is not None:
            label = arrow_key[1]
    if label is None:
        child_hover = state.list.hovered_list_row
        if child_hover is not None and index_of.get(("header", child_hover[0])) is not None:
            label = child_hover[0]
        if label is None and state.list.hovered_type_label is not None:
            if index_of.get(("header", state.list.hovered_type_label)) is not None:
                label = state.list.hovered_type_label
    if label is None:
        return False

    expanded = state.list.expanded

    if expand:
        tree_data = state.tree_data() or {}
        type_stats = tree_data.get("type_stats") or {}
        type_nodes = tree_data.get("type_nodes") or {}
        type_count = type_stats.get(label, len(type_nodes.get(label, ())))
        if type_count <= 1:
            # A single-node group lists no child rows, so expanding it would
            # only pollute ``expanded`` and draw a stray guide line over the
            # following row. Select its header instead.
            select_type_nodes(op, context, label)
            target_key: tuple = ("header", label)
            state.list.hovered_type_label = label
            state.list.hovered_list_row = None
            state.interaction.hovered_node_id = None
            state.list.arrow_key = target_key
            op._list_last_row_index = state.list.visible_row_index_map.get(target_key, op._list_last_row_index)
            op._redraw_ui()
            return True
        changed = label not in expanded
        expanded.add(label)
    else:
        changed = label in expanded
        expanded.discard(label)
    if changed:
        state.cache.list_key = None
    state.request_immediate_compile()
    if expand:
        first_child = None
        for kind, child_label, child_name in _full_list_rows(state, settings):
            if kind != _ROW_HEADER and child_label == label:
                first_child = child_name
                break
        if first_child is None:
            select_type_nodes(op, context, label)
            target_key = ("header", label)
            state.list.hovered_type_label = label
            state.list.hovered_list_row = None
            state.interaction.hovered_node_id = None
        else:
            select_single_node(op, context, first_child)
            target_key = ("child", label, first_child)
            state.list.hovered_type_label = None
            state.list.hovered_list_row = (label, first_child)
            state.interaction.hovered_node_id = first_child
    else:
        select_type_nodes(op, context, label)
        target_key = ("header", label)
        state.list.hovered_type_label = label
        state.list.hovered_list_row = None
        state.interaction.hovered_node_id = None
    state.list.arrow_key = target_key
    op._list_last_row_index = state.list.visible_row_index_map.get(target_key, op._list_last_row_index)
    op._redraw_ui()
    return True


def handle_list_toggle_all(
    op: NODEMAP_OT_navigate,
    context: Context,
    state: MinimapState,
    settings,
) -> bool:
    """Expand or collapse every expandable group in the type list.

    Groups holding more than one node are targeted; all are expanded
    unless every one of them is already expanded, in which case all are
    collapsed. The scroll is adjusted so the top visible row stays in
    place. Return True when the key was handled, False when no group
    can be expanded so the caller lets the key pass through to the Node
    Editor.
    """
    if not bool(getattr(settings, "use_group_by_type", True)):
        return False
    tree_data = state.tree_data() or {}
    type_nodes = tree_data.get("type_nodes") or {}
    type_stats = tree_data.get("type_stats") or {}
    search_texts = tree_data.get("type_search") or None

    visible, _effective_expanded, _filtered_children = filter_type_list(
        type_stats,
        type_nodes,
        state.list.expanded,
        state.list.search_query,
        search_texts=search_texts,
    )

    expandable = {label for label, _display_count in visible if type_stats.get(label, 0) > 1}
    if not expandable:
        return False

    # Anchor the top visible row (with its fractional offset) so the view
    # stays put once the row count changes below.
    row_h = state.list.row_height
    old_rows = _full_list_rows(state, settings)
    anchor = None
    anchor_offset = 0.0
    if old_rows and row_h > 0:
        top_index = min(max(int(state.list.scroll / row_h), 0), len(old_rows) - 1)
        anchor = old_rows[top_index]
        anchor_offset = state.list.scroll - top_index * row_h

    if len(state.list.expanded & expandable) == len(expandable):
        state.list.expanded.difference_update(expandable)
    else:
        state.list.expanded.update(expandable)
    state.cache.list_key = None

    if anchor is not None and row_h > 0:
        new_rows = _full_list_rows(state, settings)
        anchor_index = None
        for row_index, row in enumerate(new_rows):
            if row == anchor:
                anchor_index = row_index
                break
        if anchor_index is None and anchor[0] != _ROW_HEADER:
            # A collapsed child row is gone; hold its group header instead.
            for row_index, row in enumerate(new_rows):
                if row[0] == _ROW_HEADER and row[1] == anchor[1]:
                    anchor_index = row_index
                    break
        if anchor_index is not None:
            new_scroll = anchor_index * row_h + anchor_offset
            zone_rect = state.list.list_zone_rect
            if zone_rect is not None:
                ui_scale = _get_ui_scale()
                search_h = BUTTON_SIZE * ui_scale if settings.show_search_bar else 0.0
                search_gap = ELEMENT_GAP * ui_scale if settings.show_search_bar else 0.0
                view_h = max(zone_rect[3] - search_h - search_gap - 3 * ui_scale, row_h)
                scroll_max = max(0.0, len(new_rows) * row_h - view_h)
                new_scroll = min(max(new_scroll, 0.0), scroll_max)
            state.list.scroll = new_scroll
    state.request_immediate_compile()
    op._redraw_ui()
    return True


def focus_list_on_active_node(op: NODEMAP_OT_navigate, context: Context) -> None:
    """Expand and scroll the type list to reveal the active node's row.

    No-op when the type list is hidden or there is no active node. When the
    active node's type group holds more than one node the group is expanded so
    the node gets its own child row. The list scroll is set to bring the target
    row into view without touching the node editor view.
    """
    prefs = get_addon_preferences(context)
    settings = prefs.settings if prefs else None
    state = op._state
    if settings is None or state is None:
        return
    if state.list.list_width <= 0:
        return
    node_tree = op._space.edit_tree if op._space else None
    if not node_tree:
        return
    active_node = node_tree.nodes.active
    if active_node is None:
        return

    tree_data = state.tree_data() or {}
    type_nodes = tree_data.get("type_nodes") or {}
    type_stats = tree_data.get("type_stats") or {}
    search_texts = tree_data.get("type_search") or None

    if not bool(getattr(settings, "use_group_by_type", True)):
        nodes = flat_type_nodes(type_nodes, state.list.search_query, search_texts=search_texts)
        row_h = state.list.row_height
        rows = list(iter_flat_list_layout(nodes, row_h))
        row_index = next(
            (index for index, (_kind, _label, node_name, _local_y) in enumerate(rows) if node_name == active_node.name),
            None,
        )
        if row_index is None:
            return
        if _scroll_list_to_row(state, len(rows), row_index, row_h, settings):
            op._redraw_ui()
        return

    label = None
    for key, names in type_nodes.items():
        if active_node.name in names:
            label = key
            break
    if label is None:
        label = tree_data.get("type_active_label")
    if not label:
        return

    target_is_child = type_stats.get(label, 0) > 1

    changed = False
    if target_is_child and label not in state.list.expanded:
        state.list.expanded.add(label)
        state.cache.list_key = None
        changed = True

    # Rebuild the rows exactly as the next draw pass will so the target index
    # lines up with the rendered layout.
    visible, effective_expanded, filtered_children = filter_type_list(
        type_stats,
        type_nodes,
        state.list.expanded,
        state.list.search_query,
        search_texts=search_texts,
    )
    effective_expanded = effective_expanded or set()
    filtered_children = filtered_children or {}

    if not state.list.search_query.strip():
        if settings.type_list_sort == "NAME":
            visible.sort(key=lambda label_count: label_count[0].lower())
        else:
            visible.sort(key=lambda label_count: (-label_count[1], label_count[0]))

    entries: list[tuple[str, str, float, int]] = []
    for entry_label, display_count in visible:
        full_count = type_stats.get(entry_label, display_count)
        entries.append((entry_label, str(display_count), 0.0, full_count))

    row_h = state.list.row_height
    rows = list(_iter_type_list_layout(entries, filtered_children, effective_expanded, row_h))
    row_index = _find_list_row_index(rows, label, active_node.name, target_is_child)
    if row_index is None:
        return

    if _scroll_list_to_row(state, len(rows), row_index, row_h, settings):
        changed = True

    if changed:
        op._redraw_ui()
