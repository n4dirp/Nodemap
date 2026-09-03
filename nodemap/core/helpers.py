"""Provide shared helper utilities for the node minimap."""

from __future__ import annotations

import logging
import time

import blf
import bpy

from .. import __package__ as base_package
from .constants import (
    EMPTY_FINGERPRINT,
    LABEL_MARGIN_PX,
    LIST_ANIM_FRAMES,
    TYPE_LIST_FONT_SIZE,
    TYPE_LIST_MAX_WIDTH_PCT,
    TYPE_LIST_MIN_WIDTH,
)

logger = logging.getLogger(base_package)


def get_addon_preferences(context=None):
    """Get addon preferences from Blender context."""
    try:
        ctx = context or bpy.context
        if not ctx or not getattr(ctx, "preferences", None):
            return None
        addon = ctx.preferences.addons.get(base_package)
        if addon is None:
            return None
        return addon.preferences
    except (AttributeError, ReferenceError):
        return None


def redraw_ui(mode: str = "VIEW_3D", area_pointer: int | None = None) -> None:
    """Redraw all areas matching the given mode, or a specific area by pointer."""
    ctx = bpy.context
    if not ctx or not ctx.window_manager:
        return
    for window in ctx.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area_pointer is not None:
                try:
                    if area.as_pointer() != area_pointer:
                        continue
                except ReferenceError:
                    continue
            if mode == "ALL" or area.type == mode:
                area.tag_redraw()


def _get_ui_scale() -> float:
    """Return the Blender UI scale factor from preferences."""
    return float(bpy.context.preferences.system.ui_scale)


def _get_node_dims(node: bpy.types.Node, ui_scale: float | None = None) -> tuple[float, float]:
    """Return width and height ensuring positive float values."""
    if ui_scale is None:
        ui_scale = _get_ui_scale()
    if getattr(node, "hide", False):
        return 100.0, 30.0
    dims = getattr(node, "dimensions", None)
    if dims is not None and len(dims) >= 2:
        try:
            node_w = abs(float(dims[0]))
            if node_w == 0:
                node_w = abs(float(node.width))
        except (TypeError, ValueError, IndexError):
            node_w = abs(float(node.width))
        try:
            node_h = abs(float(dims[1]))
            if node_h == 0:
                node_h = abs(float(getattr(node, "height", 30.0)))
        except (TypeError, ValueError, IndexError):
            node_h = abs(float(getattr(node, "height", 30.0)))
    else:
        node_w = abs(float(node.width))
        node_h = abs(float(getattr(node, "height", 30.0)))

    return max(node_w / ui_scale, 5.0), max(node_h / ui_scale, 5.0)


def _get_node_tree_bounds(nodes: bpy.types.Nodes) -> tuple[float, float, float, float]:
    """Compute the bounding box of all nodes in a node tree as (min_x, min_y, max_x, max_y)."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    ui_scale = _get_ui_scale()
    for node in nodes:
        node_w, node_h = _get_node_dims(node, ui_scale)
        node_x, node_y = node.location_absolute.x, node.location_absolute.y
        min_x = min(min_x, node_x)
        max_x = max(max_x, node_x + node_w)
        min_y = min(min_y, node_y - node_h)
        max_y = max(max_y, node_y)

    if min_x == float("inf"):
        return 0.0, 0.0, 200.0, 200.0
    return min_x, min_y, max_x, max_y


def _expand_bounds_margin(
    bounds: tuple[float, float, float, float], ui_scale: float, map_h: float, padding: float
) -> tuple[float, float, float, float]:
    """Expand tree bounds by a small margin so frame labels stay inside the minimap."""
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    inner_h = max(map_h - 2 * padding, 1.0)
    margin = LABEL_MARGIN_PX * ui_scale * bbox_h / inner_h
    return (bounds[0] - margin - 50, bounds[1] - margin, bounds[2] + margin + 100, bounds[3] + margin)


def _find_node_at(nodes: bpy.types.Nodes, tree_x: float, tree_y: float) -> bpy.types.Node | None:
    """Find the hovered node via true box intersection, favoring top-level over frames."""
    best_node = None
    for node in nodes:
        node_w, node_h = _get_node_dims(node)
        node_x, node_y = node.location_absolute.x, node.location_absolute.y

        # Checking exact bounds since layout is strictly Y-down
        if node_x <= tree_x <= node_x + node_w and (node_y - node_h) <= tree_y <= node_y:
            if node.type != "FRAME":
                return node
            else:
                best_node = node
    return best_node


def _get_area_and_region_under_mouse(context, event) -> tuple:
    """Return the area and WINDOW region under the mouse cursor."""
    window = getattr(context, "window", None)
    if not window:
        return None, None
    mouse_x, mouse_y = event.mouse_x, event.mouse_y
    for area in window.screen.areas:
        if area.x <= mouse_x <= area.x + area.width and area.y <= mouse_y <= area.y + area.height:
            for region in area.regions:
                if (
                    region.type == "WINDOW"
                    and region.x <= mouse_x <= region.x + region.width
                    and region.y <= mouse_y <= region.y + region.height
                ):
                    return area, region
    return None, None


def _get_safe_bounds(
    area: bpy.types.Area,
    region: bpy.types.Region,
) -> tuple[int, int, int, int]:
    """Compute drawable region bounds excluding toolbars, shelves, headers, and UI panels."""
    left = 0
    bottom = 0
    right = region.width
    top = region.height

    for sub_region in area.regions:
        if sub_region.type == "TOOLS":
            left = max(left, sub_region.width)
        elif "ASSET_SHELF" in sub_region.type:
            bottom = max(bottom, sub_region.height)
        elif sub_region.type == "UI":
            right = min(right, region.width - sub_region.width)

    return int(left), int(bottom), int(right), int(top)


def clamp_free_rect(
    x: float,
    y: float,
    w: float,
    h: float,
    safe: tuple[float, float, float, float],
    x_margin: float,
    y_margin: float,
    margin: float,
) -> tuple[float, float]:
    """Return a ``(x, y)`` origin clamped so the ``(w, h)`` rect stays fully inside safe bounds.

    Only the origin moves; the rect is never shrunk, so free dragging stops at
    the borders instead of squeezing the map away. ``y_margin`` guards the top
    edge (context path) and ``margin`` the bottom edge (asset shelf), matching
    the insets the docked positions use.
    """
    sx, sy, ex, ey = safe
    min_x = sx + x_margin
    max_x = ex - x_margin - w
    min_y = sy + margin
    max_y = ey - y_margin - h
    return min(max(x, min_x), max_x), min(max(y, min_y), max_y)


def _get_minimap_margins(space, corner: str, ui_scale: float) -> tuple[float, float, float]:
    """Return ``(x_margin, y_margin, margin_bottom)`` based on corner and visible UI elements.

    Adjust margins when the breadcrumb context path or compositing asset shelf
    occupies space near the minimap corner. ``margin_bottom`` is the
    additional margin on the edge opposite the header.
    """
    is_compositor = space.node_tree is not None and space.node_tree.type == "COMPOSITING"
    show_asset_shelf = getattr(space, "show_region_asset_shelf", False)
    show_context_path = getattr(space.overlay, "show_context_path", False)

    map_padding = 12.0
    x_margin = map_padding * ui_scale
    y_margin = x_margin
    margin_bottom = x_margin

    adjusted_margin = (map_padding + 29) * ui_scale

    # Classify the dock by which vertical edge it sits near: top corners and
    # the top border treat the context path above and the asset shelf below;
    # bottom docks mirror that.
    docks_bottom = corner in ("BOTTOM_RIGHT", "BOTTOM_LEFT", "BOTTOM_BORDER")
    if docks_bottom:
        if is_compositor and show_asset_shelf:
            y_margin = adjusted_margin
        if show_context_path:
            margin_bottom = adjusted_margin
    else:
        if show_context_path:
            y_margin = adjusted_margin
        if is_compositor and show_asset_shelf:
            margin_bottom = adjusted_margin

    return x_margin, y_margin, margin_bottom


def _get_node_initials(name: str) -> str:
    """Return uppercase initials from each word of a node label."""
    name = name.strip()
    if not name:
        return "?"
    words = name.split()
    if len(words) >= 2:
        initials = "".join(word[0] for word in words if word[0].isalnum()).upper()
        if initials:
            return initials
    for character in name:
        if character.isalnum():
            return character.upper()
    return name[0].upper()


def _get_node_label_lines(label: str, font_id: int, font_size: int, max_width: float, max_lines: int = 3) -> list[str]:
    """Wrap a label into up to max_lines, each fitting within max_width pixels."""
    blf.size(font_id, font_size)
    words = label.split()
    if not words:
        return []
    if blf.dimensions(font_id, label)[0] <= max_width:
        return [label]
    lines = []
    word_index = 0
    while word_index < len(words) and len(lines) < max_lines:
        line_words = [words[word_index]]
        word_index += 1
        while word_index < len(words):
            candidate = " ".join(line_words + [words[word_index]])
            candidate_width, _ = blf.dimensions(font_id, candidate)
            if candidate_width > max_width:
                break
            line_words.append(words[word_index])
            word_index += 1
        lines.append(" ".join(line_words))
    return lines


def _get_type_list_width(
    settings, minimap_state, map_w: float, ui_scale: float, font_size: int = TYPE_LIST_FONT_SIZE
) -> float:
    """Return the type-list zone width as a percentage of *map_w* (0 when disabled).

    Width is driven by ``type_list_width_percent`` (``TYPE_LIST_MIN_WIDTH`` to
    ``TYPE_LIST_MAX_WIDTH_PCT`` clamp) and does not depend on content
    measurement; content clips or shows extra padding instead.
    Called before the map transform so node framing can reserve the zone.
    """
    if not settings or not settings.show_type_list or not settings.interactive:
        return 0.0
    tree_data = minimap_state.cache.tree_data
    type_stats = tree_data.get("type_stats") if tree_data else None
    if not type_stats:
        return 0.0

    percent = settings.type_list_width_percent / 100.0
    raw_width = map_w * percent
    return min(max(raw_width, TYPE_LIST_MIN_WIDTH * ui_scale), map_w * TYPE_LIST_MAX_WIDTH_PCT)


def start_list_width_animation(minimap_state, settings) -> None:
    """Begin animating the type-list zone width after a toggle-button click.

    An expansion defers the target measurement to the draw step because
    measurable type stats only exist once the pending tree compile lands.
    Skipped when Reduce Motion is enabled so the zone snaps instantly.
    """
    try:
        if bpy.context.preferences.view.use_reduce_motion:
            return
    except AttributeError:
        pass
    if not settings or not settings.use_animations:
        return
    minimap_state.list.anim_active = True
    minimap_state.list.anim_from = minimap_state.list.list_width
    minimap_state.list.anim_target = 0.0 if not settings.show_type_list else -1.0
    frames = LIST_ANIM_FRAMES.get(settings.pan_speed, 24)
    minimap_state.list.anim_duration = frames / 60.0
    minimap_state.list.anim_start = time.perf_counter()


def _list_anim_tick(minimap_state) -> None:
    minimap_state.list.anim_timer = None
    if minimap_state.list.anim_active:
        redraw_ui("NODE_EDITOR")


def _schedule_list_anim_redraw(minimap_state) -> None:
    """Schedule a one-shot timer tick that forces a redraw while the list animates."""
    if minimap_state.list.anim_timer is not None:
        return
    try:
        bpy.app.timers.register(lambda: _list_anim_tick(minimap_state), first_interval=1 / 60)
        minimap_state.list.anim_timer = True
    except (RuntimeError, ValueError):
        pass


def _get_tree_snapshot(
    node_tree, include_selection: bool = True
) -> tuple[tuple, tuple[float, float, float, float], int]:
    """Return ``(fingerprint, raw bounds, drawable node count)`` in a single RNA pass.

    The fingerprint matches :func:`get_tree_fingerprint`; bounds use the same
    clamped dims as :func:`_get_node_tree_bounds`; the count excludes FRAME
    and REROUTE nodes.
    """
    if not node_tree or not hasattr(node_tree, "nodes") or len(node_tree.nodes) == 0:
        return EMPTY_FINGERPRINT, (0.0, 0.0, 200.0, 200.0), 0
    nodes = node_tree.nodes
    ui_scale = _get_ui_scale()

    loc_sum = 0.0
    width_sum = 0.0
    height_sum = 0.0
    mute_sum = 0
    hide_sum = 0
    select_sum = 0
    content_count = 0

    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")

    for node in nodes:
        node_loc_abs = node.location_absolute
        node_x = node_loc_abs.x
        node_y = node_loc_abs.y
        loc_sum += node_x + node_y

        width_rna = abs(node.width)
        width_sum += node.width

        try:
            dims = node.dimensions
            height_raw = abs(dims[1])
            dimensions_width = abs(dims[0])
        except (AttributeError, TypeError, IndexError):
            height_raw = abs(getattr(node, "height", 30.0))
            dimensions_width = width_rna
        height_sum += height_raw

        if node.mute:
            mute_sum += 1
        if node.hide:
            hide_sum += 1
            bounds_w, bounds_h = 100.0, 30.0
        else:
            bounds_w = dimensions_width if dimensions_width > 0 else width_rna
            bounds_h = height_raw if height_raw > 0 else abs(getattr(node, "height", 30.0))
            bounds_w = max(bounds_w / ui_scale, 5.0)
            bounds_h = max(bounds_h / ui_scale, 5.0)

        if include_selection and node.select:
            select_sum += 1
        if node.type not in ("FRAME", "REROUTE"):
            content_count += 1

        right_x = node_x + bounds_w
        bottom_y = node_y - bounds_h
        if node_x < min_x:
            min_x = node_x
        if right_x > max_x:
            max_x = right_x
        if bottom_y < min_y:
            min_y = bottom_y
        if node_y > max_y:
            max_y = node_y

    links_count = len(node_tree.links) if hasattr(node_tree, "links") else 0
    active_node_name = ""
    if include_selection:
        active_node = nodes.active
        if active_node:
            active_node_name = active_node.name

    fingerprint = (
        len(nodes),
        loc_sum,
        active_node_name,
        select_sum,
        mute_sum,
        hide_sum,
        width_sum,
        height_sum,
        links_count,
    )
    bounds = (min_x, min_y, max_x, max_y) if min_x != float("inf") else (0.0, 0.0, 200.0, 200.0)
    return fingerprint, bounds, content_count


def get_tree_fingerprint(node_tree, include_selection: bool = True) -> tuple:
    """Return a lightweight fingerprint of the node tree structure and selection states."""
    return _get_tree_snapshot(node_tree, include_selection)[0]
