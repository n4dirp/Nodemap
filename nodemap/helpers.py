"""Shared helper utilities for node minimap."""

from __future__ import annotations

import logging
import time

import blf
import bpy

logger = logging.getLogger(__package__)

_HANDLE_THICKNESS: int = 6
MAX_FRAME_ZOOM: float = 20.0
_EDITOR_FIT_MARGIN: float = 0.15

MIN_MAP_WIDTH: int = 120
MIN_MAP_HEIGHT: int = 80

STATS_FONT_ID = 0
STATS_FONT_SIZE = 10
_TYPE_LIST_MIN_WIDTH = 70.0
_TYPE_LIST_MAX_WIDTH_PCT = 0.35
_LIST_PAD_X = 6.0
_LIST_SWATCH = 8.0
_LIST_SWATCH_GAP = 5.0
_LIST_COUNT_GAP = 8.0
_SCROLLBAR_HIT_PAD = 6.0
_EMPTY_FINGERPRINT = (0, 0.0, "", 0, 0, 0, 0.0, 0.0, 0)
_LIST_ANIM_FRAMES: dict[str, int] = {"FAST": 10, "MEDIUM": 20, "SLOW": 30}


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
    """Robust extraction of width and height ensuring positive float values."""
    if ui_scale is None:
        ui_scale = _get_ui_scale()
    if getattr(node, "hide", False):
        return 100.0, 30.0
    try:
        dims = node.dimensions
        w = abs(dims[0])
        if w == 0:
            w = abs(node.width)
    except (AttributeError, TypeError, IndexError):
        w = abs(node.width)

    try:
        dims = node.dimensions
        h = abs(dims[1])
        if h == 0:
            h = abs(getattr(node, "height", 30.0))
    except (AttributeError, TypeError, IndexError):
        h = abs(getattr(node, "height", 30.0))

    return max(w / ui_scale, 5.0), max(h / ui_scale, 5.0)


def _get_node_tree_bounds(nodes: bpy.types.Nodes) -> tuple[float, float, float, float]:
    """Compute the bounding box of all nodes in a node tree as (min_x, min_y, max_x, max_y)."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    ui = _get_ui_scale()
    for node in nodes:
        w, h = _get_node_dims(node, ui)
        x, y = node.location_absolute.x, node.location_absolute.y
        min_x = min(min_x, x)
        max_x = max(max_x, x + w)
        min_y = min(min_y, y - h)
        max_y = max(max_y, y)

    if min_x == float("inf"):
        return 0.0, 0.0, 200.0, 200.0
    return min_x, min_y, max_x, max_y


def _expand_bounds_margin(
    bounds: tuple[float, float, float, float], ui_scale: float, mh: float, padding: float
) -> tuple[float, float, float, float]:
    """Expand tree bounds by a small margin so frame labels stay inside the minimap."""
    LABEL_MARGIN_PX = 12 * ui_scale
    bbox_h = max(bounds[3] - bounds[1], 1.0)
    inner_h = max(mh - 2 * padding, 1.0)
    margin = LABEL_MARGIN_PX * bbox_h / inner_h
    return (bounds[0] - margin - 50, bounds[1] - margin, bounds[2] + margin + 100, bounds[3] + margin)


def _find_node_at(nodes: bpy.types.Nodes, tree_x: float, tree_y: float) -> bpy.types.Node | None:
    """Accurately finds hovered node via true box intersection, favoring top-level over frames."""
    best_node = None
    for node in nodes:
        w, h = _get_node_dims(node)
        x, y = node.location_absolute.x, node.location_absolute.y

        # Checking exact bounds since layout is strictly Y-down
        if x <= tree_x <= x + w and (y - h) <= tree_y <= y:
            if node.type != "FRAME":
                return node
            else:
                best_node = node
    return best_node


def _get_area_and_region_under_mouse(context, event) -> tuple:
    """Find the area and WINDOW region under the mouse cursor using screen coordinates."""
    window = getattr(context, "window", None)
    if not window:
        return None, None
    mx, my = event.mouse_x, event.mouse_y
    for area in window.screen.areas:
        if area.x <= mx <= area.x + area.width and area.y <= my <= area.y + area.height:
            for region in area.regions:
                if (
                    region.type == "WINDOW"
                    and region.x <= mx <= region.x + region.width
                    and region.y <= my <= region.y + region.height
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

    for r in area.regions:
        if r.type == "TOOLS":
            left = max(left, r.width)
        elif "ASSET_SHELF" in r.type:
            bottom = max(bottom, r.height)
        elif r.type == "UI":
            right = min(right, region.width - r.width)

    return int(left), int(bottom), int(right), int(top)


def _get_minimap_margins(space, corner: str, ui_scale: float) -> tuple[float, float, float]:
    """Return ``(x_margin, y_margin, margin_bottom)`` based on corner and visible UI elements.

    Adjusts margins when the breadcrumb context path or compositing asset shelf
    occupies space near the minimap's corner.
    ``margin_bottom`` is the additional margin on the edge opposite the header.
    """
    is_compositor = space.node_tree is not None and space.node_tree.type == "COMPOSITING"
    show_asset_shelf = getattr(space, "show_region_asset_shelf", False)
    show_context_path = getattr(space.overlay, "show_context_path", False)

    MAP_PADDING = 12.0
    x_margin = MAP_PADDING * ui_scale
    y_margin = x_margin
    margin_bottom = x_margin

    adjusted = (MAP_PADDING + 25) * ui_scale

    match corner:
        case "TOP_RIGHT" | "TOP_LEFT":
            if show_context_path:
                y_margin = adjusted
            if is_compositor and show_asset_shelf:
                margin_bottom = adjusted

        case "BOTTOM_RIGHT" | "BOTTOM_LEFT":
            if is_compositor and show_asset_shelf:
                y_margin = adjusted
            if show_context_path:
                margin_bottom = adjusted

    return x_margin, y_margin, margin_bottom


def _get_node_initials(name: str) -> str:
    """Extract uppercase initials from each word of a node label."""
    name = name.strip()
    if not name:
        return "?"
    words = name.split()
    if len(words) >= 2:
        initials = "".join(w[0] for w in words if w[0].isalnum()).upper()
        if initials:
            return initials
    for ch in name:
        if ch.isalnum():
            return ch.upper()
    return name[0].upper()


def _get_node_label_lines(label: str, font_id: int, font_size: int, max_width: float, max_lines: int = 3) -> list[str]:
    """Word-wrap a label into up to max_lines, each fitting within max_width pixels."""
    blf.size(font_id, font_size)
    words = label.split()
    if not words:
        return []
    if blf.dimensions(font_id, label)[0] <= max_width:
        return [label]
    lines = []
    i = 0
    while i < len(words) and len(lines) < max_lines:
        line_words = [words[i]]
        i += 1
        while i < len(words):
            candidate = " ".join(line_words + [words[i]])
            w, _ = blf.dimensions(font_id, candidate)
            if w > max_width:
                break
            line_words.append(words[i])
            i += 1
        lines.append(" ".join(line_words))
    return lines


def _get_type_list_width(settings, st, mw: float, ui_scale: float, font_size: int = STATS_FONT_SIZE) -> float:
    """Measure the type-list zone width for the current tree data (0 when disabled).

    Called before the map transform is computed so node framing can reserve
    the zone; ``st.list_width`` must be assigned from its result.
    """
    if not getattr(settings, "show_type_list", False):
        return 0.0
    tree_data = st.cache.tree_data
    type_stats = tree_data.get("type_stats") if tree_data else None
    if not type_stats:
        return 0.0

    font_id = STATS_FONT_ID
    blf.size(font_id, int(font_size * ui_scale))
    pad_x = _LIST_PAD_X * ui_scale
    swatch = _LIST_SWATCH * ui_scale
    swatch_gap = _LIST_SWATCH_GAP * ui_scale
    count_gap = _LIST_COUNT_GAP * ui_scale
    widest_label = max(blf.dimensions(font_id, label)[0] for label in type_stats)
    widest_count = max(blf.dimensions(font_id, str(count))[0] for count in type_stats.values())
    # Leading icon columns: expand-toggle slot, plus color-swatch slot when enabled.
    icon_cols = 2 if getattr(settings, "colored_nodes", True) else 1
    content_w = pad_x * 2 + icon_cols * (swatch + swatch_gap) + widest_label + count_gap + widest_count
    return min(max(content_w, _TYPE_LIST_MIN_WIDTH * ui_scale), mw * _TYPE_LIST_MAX_WIDTH_PCT)


def start_list_width_animation(st, settings) -> None:
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
    if not getattr(settings, "animations", True):
        return
    st.list.anim_active = True
    st.list.anim_from = st.list.width
    st.list.anim_target = 0.0 if not getattr(settings, "show_type_list", False) else -1.0
    frames = _LIST_ANIM_FRAMES.get(getattr(settings, "pan_speed", "MEDIUM"), 24)
    st.list.anim_duration = frames / 60.0
    st.list.anim_start = time.perf_counter()


def _list_anim_tick(st) -> None:
    st.list.anim_timer = None
    if st.list.anim_active:
        redraw_ui("NODE_EDITOR")


def _schedule_list_anim_redraw(st) -> None:
    """Schedule a one-shot timer tick that forces a redraw while the list animates."""
    if st.list.anim_timer is not None:
        return
    try:
        bpy.app.timers.register(lambda: _list_anim_tick(st), first_interval=1 / 60)
        st.list.anim_timer = True
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
        return _EMPTY_FINGERPRINT, (0.0, 0.0, 200.0, 200.0), 0
    nodes = node_tree.nodes
    ui = _get_ui_scale()

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
        loc = node.location_absolute
        x = loc.x
        y = loc.y
        loc_sum += x + y

        w_rna = abs(node.width)
        width_sum += node.width

        try:
            dims = node.dimensions
            h_abs = abs(dims[1])
            w_dim = abs(dims[0])
        except (AttributeError, TypeError, IndexError):
            h_abs = abs(getattr(node, "height", 30.0))
            w_dim = w_rna
        height_sum += h_abs

        if node.mute:
            mute_sum += 1
        if node.hide:
            hide_sum += 1
            bw, bh = 100.0, 30.0
        else:
            bw = w_dim if w_dim > 0 else w_rna
            bh = h_abs if h_abs > 0 else abs(getattr(node, "height", 30.0))
            bw = max(bw / ui, 5.0)
            bh = max(bh / ui, 5.0)

        if include_selection and node.select:
            select_sum += 1
        if node.type not in ("FRAME", "REROUTE"):
            content_count += 1

        rx = x + bw
        by = y - bh
        if x < min_x:
            min_x = x
        if rx > max_x:
            max_x = rx
        if by < min_y:
            min_y = by
        if y > max_y:
            max_y = y

    links_count = len(node_tree.links) if hasattr(node_tree, "links") else 0
    active_name = ""
    if include_selection:
        active = nodes.active
        if active:
            active_name = active.name

    fingerprint = (
        len(nodes),
        loc_sum,
        active_name,
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
    """Generate a lightweight fingerprint of the node tree structure and selection states."""
    return _get_tree_snapshot(node_tree, include_selection)[0]
