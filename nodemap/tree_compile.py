"""Tree compilation and incremental position updates."""

import io
import logging
import time

import bpy

from .helpers import (
    _get_node_dims,
    _get_node_initials,
    get_tree_fingerprint,
)
from .preferences import TRACE_LEVEL
from .state import MinimapState
from .theme import (
    _COLOR_TAG_TO_THEME_ATTR,
    _alpha_mul,
    _compute_outline_color,
    _srgb_to_linear,
    _theme_rgba,
)

logger = logging.getLogger(__package__)

try:
    import cProfile

    _HAS_C_PROFILE = True
except ImportError:
    _HAS_C_PROFILE = False

_NODE_ROUNDNESS_DEFAULT = 2.0

# Minimum interval between live position-only refreshes during drags (seconds).
# Skipped frames fall through to the debounced compile, which flushes the
# final position once movement settles.
_MOVE_REFRESH_MIN_INTERVAL = 0.016

# Profile for N frames, then dump sorted stats via logger.trace
_PROFILE_FRAMES = 300


class _Timer:
    """Context manager that logs elapsed milliseconds at TRACE level.

    Becomes a no-op when TRACE logging is not enabled (zero overhead).
    """

    __slots__ = ("_name", "_start", "_active")

    def __init__(self, name: str):
        self._name = name
        self._active = logger.isEnabledFor(TRACE_LEVEL)

    def __enter__(self):
        if self._active:
            self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self._active:
            elapsed = (time.perf_counter() - self._start) * 1000
            logger.trace("TIMER %s: %.3f ms", self._name, elapsed)


def _maybe_start_profiler(st: MinimapState) -> None:
    """Start cProfile if TRACE is enabled and profiling is not already active.

    Stores the profiler in *st* so each area gets its own session.
    """
    if not _HAS_C_PROFILE:
        return
    if not logger.isEnabledFor(TRACE_LEVEL):
        return
    if st._profiling_active:
        return
    prefs = bpy.context.preferences.addons[__package__].preferences
    if not prefs.logging_enabled or prefs.logging_level != "TRACE":
        return
    try:
        profiler = cProfile.Profile()
        profiler.enable()
    except ValueError:
        st._profiler = None
        st._profiling_active = False
        return
    st._profiler = profiler
    st._profiling_active = True
    st._profiling_frame_count = 0
    logger.trace("PROFILER: started (will dump after %d frames)", _PROFILE_FRAMES)


def _maybe_stop_profiler(st: MinimapState) -> None:
    """Increment frame count; dump profile stats after *_PROFILE_FRAMES* frames."""
    if not _HAS_C_PROFILE:
        return
    if not st._profiling_active:
        return
    if not logger.isEnabledFor(TRACE_LEVEL):
        st._profiling_active = False
        return
    st._profiling_frame_count += 1
    if st._profiling_frame_count < _PROFILE_FRAMES:
        return
    profiler = st._profiler
    if profiler is None:
        st._profiling_active = False
        return
    try:
        profiler.disable()
        profiler.create_stats()

        if not profiler.stats:
            return

        s = io.StringIO()
        sorted_funcs = sorted(profiler.stats.items(), key=lambda x: x[1][3], reverse=True)
        for func, (cc, nc, tt, ct, callers) in sorted_funcs[:40]:
            filename, lineno, funcname = func
            label = f"{funcname}:{lineno}" if funcname else f"{filename}:{lineno}"
            s.write(f"{label:<50s} {tt:8.3f}s {ct:8.3f}s {nc:6d}\n")
        logger.trace("PROFILER: stats after %d frames\n%s", _PROFILE_FRAMES, s.getvalue())
    finally:
        st._profiling_active = False


def _is_move_only_diff(old: tuple | None, current: tuple) -> bool:
    """True when two fingerprints differ only in the position-sum slot."""
    return old is not None and len(old) == len(current) and old[:1] == current[:1] and old[2:] == current[2:]


def _debounced_compile(st: MinimapState, node_tree, colors, settings, master_alpha, ui_scale):
    """Timer callback: compile tree data after fingerprint settles, then force redraw.

    When ``st.cache.pending_settle_flush`` is set (drag position refreshes happened),
    an unchanged fingerprint only needs the tree-data generation bumped so
    frozen wire/marker batches snap to the already-patched positions. A
    position-only diff is patched incrementally; anything else recompiles.
    """
    include_selection = settings.show_node_borders
    current_fingerprint = get_tree_fingerprint(node_tree, include_selection=include_selection)
    old_fingerprint = st.cache.fingerprint
    unchanged = old_fingerprint == current_fingerprint
    trace = logger.isEnabledFor(TRACE_LEVEL)
    if unchanged and not st.cache.pending_settle_flush:
        st.cache.pending_timer = None
        st.cache.pending_timer_deadline = 0.0
        st.cache.pending_fingerprint = None
        if trace:
            logger.trace("SETTLE skip: fingerprint unchanged, nothing pending")
        return None
    applied = False
    path = "compile"
    if unchanged and st.cache.tree_data:
        # Positions were fully patched by _apply_move_updates; rebaking the
        # frozen wire/marker generation skips the full recompile.
        st.cache.tree_version += 1
        applied = True
        path = "settle_bump"
    elif _is_move_only_diff(old_fingerprint, current_fingerprint) and st.cache.tree_data:
        with _Timer("move_update"):
            applied = _apply_move_updates(st, node_tree)
        if applied:
            st.cache.fingerprint = current_fingerprint
            # Movement settled: unfreeze wire/marker batches so they snap to
            # the patched positions without a full recompile.
            st.cache.tree_version += 1
            path = "move_patch"
    if not applied:
        with _Timer("compile_tree"):
            _compile_tree_data(st, node_tree, colors, settings, master_alpha, ui_scale)
            st.cache.fingerprint = current_fingerprint
    st.cache.pending_timer = None
    st.cache.pending_timer_deadline = 0.0
    st.cache.pending_fingerprint = None
    st.cache.pending_settle_flush = False
    if trace:
        logger.trace("SETTLE %s", path)
    from .helpers import redraw_ui

    redraw_ui("NODE_EDITOR")
    return None


def _compile_tree_data(st: MinimapState, node_tree, colors, settings, master_alpha, ui_scale):
    """Compute tree-space data for nodes, wires, sockets, and labels.

    Called only when the node tree fingerprint changes (tree topology,
    selection, mute, active node).  Screen-space transforms (zoom/pan)
    are NOT applied here — content batches are baked in map-local space
    by ``_ensure_minimap_batches()`` and placed with a matrix transform.

    Stores result in ``st.cache.tree_data``.
    """
    nodes = node_tree.nodes
    active_node = nodes.active
    zoom = st.view.zoom

    tree_data: dict = {}

    # Hoisted settings lookups (avoid repeated attribute lookups in loops)
    show_frames = settings.show_frames
    show_names = settings.show_names
    show_socket_indicators = settings.show_socket_indicators
    show_wires = settings.show_wires
    show_wire_color = settings.show_wire_color
    show_frame_labels = settings.show_frame_labels
    colored_nodes = settings.colored_nodes
    node_label_mode = settings.node_label_mode
    show_type_list = settings.show_type_list

    # Single pre-pass: classify nodes + cache dims/location + compute bounds
    frames = []
    unselected_nodes = []
    selected_nodes = []
    active_node_item = None
    node_data: dict[int, dict] = {}
    group_markers: dict[tuple, list[tuple[float, float, float]]] = {}
    type_counts: dict[str, int] = {}
    type_colors: dict[str, tuple[float, float, float, float]] = {}
    type_nodes: dict[str, list[str]] = {}
    type_node_colors: dict[str, dict[str, tuple[float, float, float, float]]] = {}
    type_selected_counts: dict[str, int] = {}
    type_active_label: str | None = None

    bounds_min_x = float("inf")
    bounds_min_y = float("inf")
    bounds_max_x = float("-inf")
    bounds_max_y = float("-inf")

    with _Timer("compile_tree.pre_pass"):
        for node in nodes:
            ptr = node.as_pointer()
            w, h = _get_node_dims(node, ui_scale)
            loc = node.location_absolute
            loc_x, loc_y = loc.x, loc.y

            node_data[ptr] = {"dims": (w, h), "loc": (loc_x, loc_y)}

            # Track bounding box
            if loc_x < bounds_min_x:
                bounds_min_x = loc_x
            if loc_y > bounds_max_y:
                bounds_max_y = loc_y
            rx = loc_x + w
            if rx > bounds_max_x:
                bounds_max_x = rx
            ty = loc_y - h
            if ty < bounds_min_y:
                bounds_min_y = ty

            if node.type == "FRAME":
                if show_frames:
                    frames.append(node)
            elif node.type == "REROUTE":
                pass
            else:
                if node.select:
                    if node == active_node:
                        active_node_item = node
                    else:
                        selected_nodes.append(node)
                else:
                    unselected_nodes.append(node)

        if bounds_min_x == float("inf"):
            tree_data["bounds"] = (0.0, 0.0, 200.0, 200.0)
        else:
            tree_data["bounds"] = (bounds_min_x, bounds_min_y, bounds_max_x, bounds_max_y)
        # Stable local-space origin for batch baking (independent of later
        # bound drift so screen transforms stay exact between rebuilds).
        tree_data["origin"] = (
            (bounds_min_x + bounds_max_x) / 2,
            (bounds_min_y + bounds_max_y) / 2,
        )

        # Build sorted Z-order (frames first, then unselected, selected, active)
        sorted_items = []
        for node in frames:
            sorted_items.append((node, True))
        for node in unselected_nodes:
            sorted_items.append((node, False))
        for node in selected_nodes:
            sorted_items.append((node, False))
        if active_node_item:
            sorted_items.append((active_node_item, False))

    # ------------------------------------------------------------------
    # Combined pass: node data + sockets + wire endpoints (tree-space)
    # ------------------------------------------------------------------

    with _Timer("compile_tree.combined"):
        # Pre-compute theme colors by color_tag (avoids per-node _theme_rgba call)
        color_tag_cache: dict[str, tuple[float, float, float, float]] = {}
        for tag, theme_attr in _COLOR_TAG_TO_THEME_ATTR.items():
            color_tag_cache[tag] = _theme_rgba(f"node_editor.{theme_attr}", colors["node"])

        node_infos: list[dict] = []
        default_socket_color = (*colors["wire"][:3], master_alpha)
        default_wire_color = _alpha_mul(colors["wire"], master_alpha)
        out_pos: dict[str, dict] = {}
        in_pos: dict[str, dict] = {}
        # Socket draw colors keyed by socket pointer, shared across nodes and
        # persisted so position-only refreshes skip draw_color() calls.
        sock_color_cache: dict[int, tuple[float, float, float, float]] = {}
        # Per-node socket dots so drag refreshes rebuild only the moved nodes;
        # grouped by color afterwards via _group_socket_dots().
        socket_items_by_node: dict[int, list[tuple[tuple, float, float]]] = {}

        for node, is_frame in sorted_items:
            ptr = node.as_pointer()
            w, h = node_data[ptr]["dims"]
            loc_x, loc_y_top = node_data[ptr]["loc"]
            ty = loc_y_top - h

            info: dict = {
                "ptr": ptr,
                "tree_x": loc_x,
                "tree_y": ty,
                "tree_w": w,
                "tree_h": h,
                "is_frame": is_frame,
                "border_w": 0.5,
            }

            if is_frame:
                frame_alpha = 0.6 * master_alpha
                if colored_nodes:
                    if getattr(node, "use_custom_color", False):
                        nc = node.color
                        frame_color = (float(nc[0]), float(nc[1]), float(nc[2]), colors["node"][3])
                    else:
                        tag = getattr(node, "color_tag", "NONE")
                        frame_color = color_tag_cache.get(tag, colors.get("frame_node", colors["node"]))
                else:
                    frame_color = colors.get("frame_node", colors["node"])
                info["fill_color"] = _srgb_to_linear((frame_color[0], frame_color[1], frame_color[2], frame_alpha))

                border_col = frame_color
                if node.select:
                    border_col = colors["node_active"] if node == active_node else colors["node_selected"]
                frame_border_alpha = master_alpha if node.select else master_alpha * 0.9
                info["border_color"] = _srgb_to_linear(_alpha_mul(border_col, frame_border_alpha))
                info["frame_color"] = frame_color
                info["name"] = node.name
                info["node_r_base"] = _NODE_ROUNDNESS_DEFAULT
                if show_type_list and show_frames:
                    if "Frame" not in type_counts:
                        type_colors["Frame"] = frame_color
                    info["type_label"] = "Frame"
                    type_counts["Frame"] = type_counts.get("Frame", 0) + 1
                    type_nodes.setdefault("Frame", []).append(node.name)
                    type_node_colors.setdefault("Frame", {})[node.name] = frame_color
                    if node.select:
                        type_selected_counts["Frame"] = type_selected_counts.get("Frame", 0) + 1
                    if node == active_node:
                        type_active_label = "Frame"
            else:
                if colored_nodes:
                    if getattr(node, "use_custom_color", False):
                        nc = node.color
                        node_color = (float(nc[0]), float(nc[1]), float(nc[2]), colors["node"][3])
                    else:
                        tag = getattr(node, "color_tag", "NONE")
                        node_color = color_tag_cache.get(tag, colors["node"])
                else:
                    node_color = colors["node"]

                if show_type_list:
                    label = node.bl_label or node.type.replace("_", " ").title()
                    if label not in type_counts:
                        type_colors[label] = node_color
                    info["type_label"] = label
                    type_counts[label] = type_counts.get(label, 0) + 1
                    type_nodes.setdefault(label, []).append(node.name)
                    type_node_colors.setdefault(label, {})[node.name] = node_color
                    if node.select:
                        type_selected_counts[label] = type_selected_counts.get(label, 0) + 1
                    if node == active_node:
                        type_active_label = label

                if node.mute:
                    bg_color = colors["bg"]
                    info["fill_color"] = _srgb_to_linear(
                        (
                            node_color[0] * 0.15 + bg_color[0] * 0.85,
                            node_color[1] * 0.15 + bg_color[1] * 0.85,
                            node_color[2] * 0.15 + bg_color[2] * 0.85,
                            node_color[3] * master_alpha,
                        )
                    )
                else:
                    info["fill_color"] = _srgb_to_linear(_alpha_mul(node_color, master_alpha))

                border_col = colors["node_border"]
                if node.select:
                    border_col = colors["node_active"] if node == active_node else colors["node_selected"]
                border_alpha = master_alpha
                if not node.select:
                    border_alpha *= 0.6
                if node.mute:
                    border_alpha = 0.35 * master_alpha
                info["border_color"] = _srgb_to_linear(_alpha_mul(border_col, border_alpha))
                info["node_r_base"] = _NODE_ROUNDNESS_DEFAULT * 2
                info["name"] = node.name

                if node.type == "GROUP":
                    marker_col = node_color if colored_nodes and not node.select else border_col
                    marker_color = _alpha_mul(marker_col, border_alpha)
                    group_markers.setdefault(marker_color, []).append((loc_x + w / 2, ty, w))
                    info["group_marker_col"] = marker_color

            # Labels (tree-space positions computed in build)
            text_alpha = 0.35 if node.mute else 1.0
            if is_frame:
                frame_label = node.label
                if frame_label and show_frame_labels and zoom >= 0.8:
                    text_color = _alpha_mul(colors["text"], master_alpha)
                    fc = info["frame_color"]
                    bg_color_lbl = _srgb_to_linear((fc[0], fc[1], fc[2], 0.4 * master_alpha))
                    info["frame_label"] = (frame_label, text_color, bg_color_lbl)
            else:
                if show_names:
                    label = node.label
                    if not label and getattr(node, "node_tree", None):
                        label = node.node_tree.name
                    if not label:
                        label = node.bl_label

                    if node_label_mode == "FULL" and label:
                        info["node_label_type"] = "full"
                        info["node_label_text"] = label
                    else:
                        initials = _get_node_initials(label)
                        if initials:
                            info["node_label_type"] = "initials"
                            info["node_label_text"] = initials

                    fill_for_contrast = info["fill_color"]
                    lbl_contrast = _compute_outline_color(fill_for_contrast)
                    info["node_label_color"] = (*lbl_contrast[:3], fill_for_contrast[3] * text_alpha * master_alpha)

            node_infos.append(info)

            # Sockets + wire endpoints for this node (skip frames)
            if is_frame or node.type == "REROUTE":
                continue

            body_top = loc_y_top
            body_bot = body_top - h
            body_range = body_top - body_bot

            if show_socket_indicators:
                dots: list[tuple[tuple, float, float]] = []
                for is_output, sock_list in [(False, node.inputs), (True, node.outputs)]:
                    try:
                        visible = [s for s in sock_list if not s.hide and s.enabled]
                    except AttributeError:
                        visible = [
                            s for s in sock_list if getattr(s, "hide", False) is False and getattr(s, "enabled", True)
                        ]

                    x_base = loc_x + (w if is_output else 0)
                    num = len(visible)
                    for idx, socket in enumerate(visible):
                        if body_range <= 0 or num <= 1:
                            sy_tree = (body_top + body_bot) * 0.5
                        else:
                            sy_tree = body_top - body_range * (idx + 1) / (num + 1)

                        sptr = socket.as_pointer()
                        if sptr not in sock_color_cache:
                            if show_wire_color:
                                try:
                                    sc = socket.draw_color(bpy.context, node)
                                    sock_color_cache[sptr] = (float(sc[0]), float(sc[1]), float(sc[2]), master_alpha)
                                except Exception:
                                    sock_color_cache[sptr] = default_socket_color
                            else:
                                sock_color_cache[sptr] = default_socket_color
                        dots.append((sock_color_cache[sptr], x_base, sy_tree))
                socket_items_by_node[ptr] = dots

            if show_wires:
                visible_outs = [
                    s for s in node.outputs if not getattr(s, "hide", False) and getattr(s, "enabled", True)
                ]
                if visible_outs:
                    x_base = loc_x + w
                    num = len(visible_outs)
                    out_dict = {}
                    for idx, sock in enumerate(visible_outs):
                        if body_range <= 0 or num <= 1:
                            sy = (body_top + body_bot) * 0.5
                        else:
                            sy = body_top - body_range * (idx + 1) / (num + 1)
                        sptr = sock.as_pointer()
                        if sptr in sock_color_cache:
                            wire_color = sock_color_cache[sptr]
                        else:
                            wire_color = default_wire_color
                            if show_wire_color:
                                try:
                                    sc = sock.draw_color(bpy.context, node)
                                    wire_color = (float(sc[0]), float(sc[1]), float(sc[2]), master_alpha)
                                except Exception:
                                    pass
                        out_dict[sock.identifier] = (x_base, sy, wire_color)
                    out_pos[node.name] = out_dict

                visible_ins = [s for s in node.inputs if not getattr(s, "hide", False) and getattr(s, "enabled", True)]
                if visible_ins:
                    x_base = loc_x
                    num = len(visible_ins)
                    in_dict = {}
                    for idx, sock in enumerate(visible_ins):
                        if body_range <= 0 or num <= 1:
                            sy = (body_top + body_bot) * 0.5
                        else:
                            sy = body_top - body_range * (idx + 1) / (num + 1)
                        in_dict[sock.identifier] = (x_base, sy, default_wire_color)
                    in_pos[node.name] = in_dict

        tree_data["node_infos"] = node_infos
        tree_data["socket_items"] = _group_socket_dots(socket_items_by_node)
        tree_data["socket_ph_base"] = 8.0
        tree_data["group_markers"] = group_markers
        tree_data["type_stats"] = type_counts
        tree_data["type_colors"] = type_colors
        tree_data["type_node_colors"] = type_node_colors
        # Stable child order (by name) so selecting a node — which recompiles
        # and re-iterates node_tree.nodes — never reshuffles the sub-list.
        for _lbl in type_nodes:
            type_nodes[_lbl].sort()
        tree_data["type_nodes"] = type_nodes
        tree_data["type_selected_counts"] = type_selected_counts
        tree_data["type_active_label"] = type_active_label
        # Position-refresh support (see _apply_move_updates)
        tree_data["out_pos"] = out_pos
        tree_data["in_pos"] = in_pos
        tree_data["socket_draw_colors"] = sock_color_cache
        tree_data["default_socket_color"] = default_socket_color
        tree_data["default_wire_color"] = default_wire_color
        tree_data["socket_indicators_on"] = show_socket_indicators
        tree_data["socket_items_by_node"] = socket_items_by_node

    # ------------------------------------------------------------------
    # REROUTE wire endpoints (not in sorted_items, handled separately)
    # ------------------------------------------------------------------
    reroute_meta: dict[str, tuple[float, float, tuple[float, float, float, float]]] = {}
    with _Timer("compile_tree.reroute"):
        if show_wires:
            for node in nodes:
                if node.type != "REROUTE":
                    continue
                ptr = node.as_pointer()
                w, h = node_data[ptr]["dims"]
                loc_x, loc_y_top = node_data[ptr]["loc"]
                cx_n = loc_x + w / 2
                cy_n = loc_y_top - h / 2

                wire_color = default_wire_color
                if show_wire_color:
                    try:
                        sock = node.outputs[0] if node.outputs else node.inputs[0]
                        sc = sock.draw_color(bpy.context, node)
                        wire_color = (float(sc[0]), float(sc[1]), float(sc[2]), master_alpha)
                    except Exception:
                        pass

                reroute_meta[node.name] = (w / 2, h / 2, wire_color)
                out_pos[node.name] = {s.identifier: (cx_n, cy_n, wire_color) for s in node.outputs}
                in_pos[node.name] = {s.identifier: (cx_n, cy_n, wire_color) for s in node.inputs}

    tree_data["reroute_meta"] = reroute_meta

    # ------------------------------------------------------------------
    # Wire connections (using wire endpoints)
    # ------------------------------------------------------------------
    with _Timer("compile_tree.wire_links"):
        raw_links = _extract_raw_links(node_tree) if show_wires else []
        wire_items = _resolve_wire_items(raw_links, out_pos, in_pos)

    # Persisted so position-only refreshes skip the links RNA pass entirely
    tree_data["raw_links"] = raw_links
    tree_data["wire_items"] = wire_items
    st.cache.tree_data = tree_data
    st.cache.tree_version += 1
    st.cache.position_version += 1


def _extract_raw_links(node_tree) -> list[tuple[str, str, str, str]]:
    """Extract ``(from_name, from_id, to_name, to_id)`` tuples for all links.

    Pure RNA pass; only needed when topology changes since results are
    persisted on ``tree_data["raw_links"]``.
    """
    raw_links: list[tuple[str, str, str, str]] = []
    for link in node_tree.links:
        from_node = link.from_node
        if from_node and from_node.type != "FRAME":
            raw_links.append(
                (
                    from_node.name,
                    link.from_socket.identifier,
                    link.to_node.name,
                    link.to_socket.identifier,
                )
            )
    return raw_links


def _resolve_wire_items(
    raw_links: list[tuple[str, str, str, str]],
    out_pos: dict[str, dict],
    in_pos: dict[str, dict],
) -> dict[tuple, list[tuple[float, float, float, float]]]:
    """Resolve persisted links to per-color wire segment lists (pure dict ops)."""
    wire_items: dict[tuple, list[tuple[float, float, float, float]]] = {}
    for from_name, from_id, to_name, to_id in raw_links:
        out_pos_node = out_pos.get(from_name)
        if not out_pos_node:
            continue
        out_tuple = out_pos_node.get(from_id)
        if not out_tuple:
            continue
        in_pos_node = in_pos.get(to_name)
        if not in_pos_node:
            continue
        in_tuple = in_pos_node.get(to_id)
        if not in_tuple:
            continue
        out_x, out_y, wire_color = out_tuple
        in_x, in_y, _ = in_tuple
        wire_items.setdefault(wire_color, []).append((out_x, out_y, in_x, in_y))
    return wire_items


def _group_socket_dots(by_node: dict[int, list[tuple[tuple, float, float]]]) -> dict[tuple, list[tuple[float, float]]]:
    """Group per-node socket dots into color-keyed position lists (pure dict ops)."""
    grouped: dict[tuple, list[tuple[float, float]]] = {}
    for dots in by_node.values():
        for color, x, y in dots:
            grouped.setdefault(color, []).append((x, y))
    return grouped


def _apply_move_updates(st: MinimapState, node_tree) -> bool:
    """Patch cached tree data in place after pure position changes (drag).

    Refreshes node positions, socket/wire endpoints, and group markers
    without recomputing colors, labels, or type stats. Socket indicator
    dots are rebuilt only for the moved nodes and regrouped by color.
    Returns True when applied; False when cached tables are missing and a
    full recompile is required.
    """
    tree_data = st.cache.tree_data
    if not tree_data:
        return False
    infos = tree_data.get("node_infos")
    out_pos = tree_data.get("out_pos")
    in_pos = tree_data.get("in_pos")
    reroute_meta = tree_data.get("reroute_meta")
    default_socket_color = tree_data.get("default_socket_color")
    default_wire_color = tree_data.get("default_wire_color")
    if infos is None or out_pos is None or in_pos is None:
        return False
    if reroute_meta is None or default_socket_color is None or default_wire_color is None:
        return False

    info_by_ptr: dict[int, dict] = {}
    for info in infos:
        ptr = info.get("ptr")
        if ptr:
            info_by_ptr[ptr] = info

    show_indicators = bool(tree_data.get("socket_indicators_on"))
    by_node = tree_data.get("socket_items_by_node")
    sock_colors = tree_data.get("socket_draw_colors") or {}
    if show_indicators and by_node is None:
        return False

    moved_any = False
    # TRACE-only sub-timers split RNA-heavy socket patching from wire re-resolution.
    trace = logger.isEnabledFor(TRACE_LEVEL)
    sockets_t = 0.0

    with _Timer("move_update"):
        for node in node_tree.nodes:
            ptr = node.as_pointer()
            loc = node.location_absolute
            lx = loc.x
            ly = loc.y

            ntype = node.type
            if ntype == "REROUTE":
                meta = reroute_meta.get(node.name)
                if meta:
                    hw_off, hh_off, wire_color = meta
                    cx_n = lx + hw_off
                    cy_n = ly - hh_off
                    o_entry = out_pos.get(node.name)
                    i_entry = in_pos.get(node.name)
                    # Flag movement so the tail re-resolves wire_items and
                    # bumps the position generation; reroutes have no info
                    # entry, so nothing else would mark them as moved.
                    entry = o_entry or i_entry
                    if entry:
                        old_x, old_y, _old_col = next(iter(entry.values()))
                        if old_x != cx_n or old_y != cy_n:
                            moved_any = True
                    if o_entry is not None:
                        for sid in o_entry:
                            o_entry[sid] = (cx_n, cy_n, wire_color)
                    if i_entry is not None:
                        for sid in i_entry:
                            i_entry[sid] = (cx_n, cy_n, wire_color)
                continue

            info = info_by_ptr.get(ptr)
            if info is None:
                continue

            w = info["tree_w"]
            body_top = ly
            body_range = info["tree_h"]
            new_y = body_top - body_range
            # Endpoint and dot geometry only depends on position (dims are
            # unchanged on move-only diffs), so untouched nodes skip all
            # socket RNA; only the moved nodes' dots get rebuilt per node.
            moved = lx != info["tree_x"] or new_y != info["tree_y"]
            info["tree_x"] = lx
            info["tree_y"] = new_y

            if not moved:
                continue
            moved_any = True

            if ntype == "FRAME":
                continue

            if trace:
                t0 = time.perf_counter()

            name = node.name
            o_entry = out_pos.get(name)
            if o_entry:
                visible_outs = [
                    s for s in node.outputs if not getattr(s, "hide", False) and getattr(s, "enabled", True)
                ]
                x_base = lx + w
                num = len(visible_outs)
                for idx, sock in enumerate(visible_outs):
                    if body_range <= 0 or num <= 1:
                        sy = body_top - body_range * 0.5
                    else:
                        sy = body_top - body_range * (idx + 1) / (num + 1)
                    sid = sock.identifier
                    old = o_entry.get(sid)
                    color = old[2] if old else default_wire_color
                    o_entry[sid] = (x_base, sy, color)

            i_entry = in_pos.get(name)
            if i_entry:
                visible_ins = [s for s in node.inputs if not getattr(s, "hide", False) and getattr(s, "enabled", True)]
                num = len(visible_ins)
                for idx, sock in enumerate(visible_ins):
                    if body_range <= 0 or num <= 1:
                        sy = body_top - body_range * 0.5
                    else:
                        sy = body_top - body_range * (idx + 1) / (num + 1)
                    sid = sock.identifier
                    old = i_entry.get(sid)
                    color = old[2] if old else default_wire_color
                    i_entry[sid] = (lx, sy, color)

            if show_indicators:
                dots: list[tuple[tuple, float, float]] = []
                for is_output, sock_list in ((False, node.inputs), (True, node.outputs)):
                    try:
                        visible = [s for s in sock_list if not s.hide and s.enabled]
                    except AttributeError:
                        visible = [
                            s for s in sock_list if getattr(s, "hide", False) is False and getattr(s, "enabled", True)
                        ]
                    x_base = lx + (w if is_output else 0.0)
                    num = len(visible)
                    for idx, socket in enumerate(visible):
                        if body_range <= 0 or num <= 1:
                            sy_tree = (body_top + new_y) * 0.5
                        else:
                            sy_tree = body_top - body_range * (idx + 1) / (num + 1)
                        color = sock_colors.get(socket.as_pointer(), default_socket_color)
                        dots.append((color, x_base, sy_tree))
                by_node[ptr] = dots

            if trace:
                sockets_t += time.perf_counter() - t0

        # Group underline markers follow their nodes
        markers: dict[tuple, list[tuple[float, float, float]]] = {}
        for info in infos:
            marker_col = info.get("group_marker_col")
            if marker_col:
                markers.setdefault(marker_col, []).append(
                    (info["tree_x"] + info["tree_w"] / 2, info["tree_y"], info["tree_w"])
                )
        tree_data["group_markers"] = markers
        wires_t = 0.0
        if moved_any:
            if show_indicators:
                tree_data["socket_items"] = _group_socket_dots(by_node)
            raw_links = tree_data.get("raw_links")
            if raw_links is None:
                raw_links = _extract_raw_links(node_tree)
            if trace:
                t1 = time.perf_counter()
            tree_data["wire_items"] = _resolve_wire_items(raw_links, out_pos, in_pos)
            st.cache.position_version += 1
            if trace:
                wires_t = time.perf_counter() - t1
        if trace:
            logger.trace("TIMER move_update.sockets: %.3f ms", sockets_t * 1000)
            if moved_any:
                logger.trace("TIMER move_update.wires: %.3f ms", wires_t * 1000)
    return True
