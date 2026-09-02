"""Selection helpers for node editor interaction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import bpy

from .helpers import _find_node_at, _get_node_dims, _get_ui_scale

if TYPE_CHECKING:
    from bpy.types import Context, Event, Region

    from .minimap_ops import NODEMAP_OT_navigate
    from .state import MinimapState

logger = logging.getLogger(__package__)


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
    event: Event,
    state: MinimapState,
    frame: bool = False,
) -> None:
    """Handle a click on the minimap: find the node under cursor and select it."""
    from .minimap_ops import _region_to_tree

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
        if not select_node_via_operator(op, context, node, extend=event.shift, deselect_all=not event.shift):
            # Fallback for API changes (may trigger EEVEE compile)
            if event.shift:
                node.select = not node.select
                if node.select:
                    node_tree.nodes.active = node
            else:
                for n in node_tree.nodes:
                    n.select = False
                node.select = True
                node_tree.nodes.active = node

        if frame:
            addon = context.preferences.addons.get(__package__)
            settings = addon.preferences.settings if addon else None
            if not (settings and op._anim.view_selected_animated(context, settings)):
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
    """
    space = op._space
    state = op._state
    if not space or space.type != "NODE_EDITOR" or not state:
        return
    node_tree = space.edit_tree
    if not node_tree:
        return
    type_nodes = (state.cache.tree_data or {}).get("type_nodes") or {}
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
    for name in names:
        node = node_tree.nodes.get(name)
        if node:
            # Native operator keeps selection/additive state and sets the
            # active node without tagging the NodeTree for an EEVEE rebuild.
            if not select_node_via_operator(op, context, node, extend=node_extend, deselect_all=False):
                node.select = True
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
