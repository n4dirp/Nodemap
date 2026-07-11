"""Minimap rendering in the Node Editor."""

import io
import logging
import math
import time

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

try:
    import cProfile

    _HAS_C_PROFILE = True
except ImportError:
    _HAS_C_PROFILE = False

from .gpu_draw import (
    _batch_draw_pills,
    _draw_filled_rounded_rect,
    _draw_filled_rounded_rect_with_hole,
    _draw_pill,
    _draw_pill_border,
    _draw_rounded_rect_border,
    _draw_text_with_shadow,
    _get_batch_rect_border_shader,
    _get_batch_rect_shader,
)
from .helpers import (
    _COLOR_TAG_TO_THEME_ATTR,
    MinimapState,
    _alpha_mul,
    _clamp_pan_to_viewport,
    _compute_outline_color,
    _get_minimap_margins,
    _get_minimap_transform,
    _get_node_dims,
    _get_node_editor_theme_colors,
    _get_node_initials,
    _get_node_label_lines,
    _get_node_tree_bounds,
    _get_safe_bounds,
    _get_ui_scale,
    _get_visible_rect,
    _minimap_window_operators,
    _registration_state,
    _srgb_to_linear,
    _state,
    _theme_rgba,
    get_tree_fingerprint,
)
from .preferences import TRACE_LEVEL

logger = logging.getLogger(__package__)

FONT_SIZE = 11
FRAME_ALL_BTN_SIZE = 20
FRAME_ALL_BTN_MARGIN = 1
_MIN_SOCKET_SCALE = 0.15


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


# Profile for N frames, then dump sorted stats via logger.trace
_PROFILE_FRAMES = 300


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
    if not getattr(prefs, "logging_enabled", False) or getattr(prefs, "logging_level", "INFO") != "TRACE":
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


def _early_exit(context, space, st: MinimapState) -> bool:
    """Return True if the minimap should not be drawn."""
    if space is None:
        return True
    if space.type != "NODE_EDITOR":
        return True
    if not space.overlay.show_overlays:
        return True
    if not st.enabled:
        return True
    addon = context.preferences.addons.get(__package__)
    if not addon:
        return True
    return False


def _compute_minimap_rect(
    settings, ui_scale, space, region, corner, st: MinimapState
) -> tuple[float, float, float, float, float, float] | None:
    """Compute the minimap rectangle position and dimensions."""
    sx, sy, ex, ey = _get_safe_bounds(bpy.context.area, region)
    safe_w = ex - sx
    safe_h = ey - sy

    x_margin, y_margin, margin = _get_minimap_margins(space, corner, ui_scale)

    # Compute desired size, capped to % of safe region (accounting for margins)
    mw = getattr(settings, "minimap_width", 256) * ui_scale
    mh = getattr(settings, "minimap_height", 128) * ui_scale
    max_mw_pct = getattr(settings, "max_width_pct", 50) / 100.0
    max_mh_pct = getattr(settings, "max_height_pct", 50) / 100.0
    mw = min(mw, (safe_w - x_margin) * max_mw_pct)
    mh = min(mh, (safe_h - y_margin - margin) * max_mh_pct)

    padding = 6 * ui_scale

    # Position minimap in the chosen corner of the safe region
    match corner:
        case "TOP_RIGHT":
            mx = ex - mw - x_margin
            my = ey - mh - y_margin
        case "TOP_LEFT":
            mx = sx + x_margin
            my = ey - mh - y_margin
        case "BOTTOM_RIGHT":
            mx = ex - mw - x_margin
            my = sy + y_margin
        case "BOTTOM_LEFT":
            mx = sx + x_margin
            my = sy + y_margin

    # Clamp to safe bounds instead of bailing
    mx = max(mx, float(sx))
    my = max(my, float(sy))
    mw = min(mw, float(ex) - mx - x_margin)
    mh = min(mh, float(ey) - my - y_margin)

    # Only bail if the minimap would be too small to be useful
    MIN_DIM = 50 * ui_scale
    if mw < MIN_DIM or mh < MIN_DIM:
        st.rect = (0.0, 0.0, 0.0, 0.0)
        return None

    return mx, my, mw, mh, padding, y_margin


def _draw_background(
    mx: float, my: float, mw: float, mh: float, colors: dict, master_alpha: float
) -> tuple[tuple[float, float, float, float], float]:
    """Draw the minimap backdrop rounded rect and border."""

    bg_color = _alpha_mul(colors["bg"], master_alpha)
    panel_r = colors.get("panel_roundness", 4.0)
    shadow_w = 1

    _draw_filled_rounded_rect(mx, my, mw, mh, panel_r * 1.2, bg_color)
    border_color = _alpha_mul(colors["bg_border"], master_alpha)

    _draw_rounded_rect_border(
        mx - shadow_w, my - shadow_w, mw + shadow_w * 2, mh + shadow_w * 2, panel_r, (0, 0, 0, 0.15 * master_alpha), 0.5
    )

    _draw_rounded_rect_border(mx, my, mw, mh, panel_r, border_color, 0.5)
    return bg_color, panel_r


def _setup_scissor(mx: float, my: float, mw: float, mh: float) -> tuple[bool, bool, tuple[int, int, int, int]]:
    """Enable scissor test to clip content to minimap interior.

    Returns ``(success, was_active, old_rect)`` for restoring later.
    """
    saved = (False, (0, 0, 0, 0))
    try:
        was_active = gpu.state.scissor_test_get()
        saved = (was_active, gpu.state.scissor_get() if was_active else (0, 0, 0, 0))
    except Exception:
        pass

    try:
        # Set rect first — scissor_set marks framebuffer dirty on OpenGL,
        # ensuring the subsequent scissor_test_set flush takes effect.
        gpu.state.scissor_set(int(mx + 1), int(my + 1), int(mw - 2), int(mh - 2))
        gpu.state.scissor_test_set(True)
        was_active, old_rect = saved
        return True, was_active, old_rect
    except Exception:
        return False, False, (0, 0, 0, 0)


def _teardown_scissor(saved_state: tuple[bool, bool, tuple[int, int, int, int]]) -> None:
    """Restore scissor test to its original state before _setup_scissor.

    Workaround for Blender bugs #113310 / #139646: scissor_set marks the
    framebuffer dirty — call it *before* scissor_test_set so the state
    flush actually reaches the GL driver on OpenGL.
    """
    success, was_active, old_rect = saved_state
    if not success:
        return
    try:
        if was_active:
            gpu.state.scissor_set(int(old_rect[0]), int(old_rect[1]), int(old_rect[2]), int(old_rect[3]))
            gpu.state.scissor_test_set(True)
        else:
            gpu.state.scissor_set(0, 0, 65535, 65535)
            gpu.state.scissor_test_set(False)
    except Exception:
        try:
            gpu.state.scissor_set(0, 0, 65535, 65535)
            gpu.state.scissor_test_set(False)
        except Exception:
            pass


def _draw_resize_handles(
    mx: float,
    my: float,
    mw: float,
    mh: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    corner: str,
    st: MinimapState,
) -> None:
    """Draw full-edge resize indicators, colored orange when the percentage cap is active."""
    handle = st.resize_active
    if not handle:
        return

    w_clamped = st.width_clamped
    h_clamped = st.height_clamped

    col_base = _alpha_mul(colors["text"], 0.5 * master_alpha)
    col_warn = _alpha_mul(colors["indicator"], master_alpha)
    thick = 3.0 * ui_scale
    margin = 6 * ui_scale

    match handle:
        case "W":
            wx = mx + 2 * ui_scale if corner in ("TOP_RIGHT", "BOTTOM_RIGHT") else mx + mw - 2 * ui_scale - thick
            _draw_pill(wx, my + margin, thick, mh - 2 * margin, col_warn if w_clamped else col_base)
        case "H":
            hy = my + 2 * ui_scale if corner in ("TOP_RIGHT", "TOP_LEFT") else my + mh - 2 * ui_scale - thick
            _draw_pill(mx + margin, hy, mw - 2 * margin, thick, col_warn if h_clamped else col_base)
        case "C":
            wx = mx + 2 * ui_scale if corner in ("TOP_RIGHT", "BOTTOM_RIGHT") else mx + mw - 2 * ui_scale - thick
            _draw_pill(wx, my + margin, thick, mh - 2 * margin, col_warn if w_clamped else col_base)

            hy = my + 2 * ui_scale if corner in ("TOP_RIGHT", "TOP_LEFT") else my + mh - 2 * ui_scale - thick
            _draw_pill(mx + margin, hy, mw - 2 * margin, thick, col_warn if h_clamped else col_base)


def _draw_viewport_overlay(
    settings,
    space,
    region,
    mx: float,
    my: float,
    mw: float,
    mh: float,
    cx: float,
    cy: float,
    scale: float,
    tree_cx: float,
    tree_cy: float,
    colors: dict,
    master_alpha: float,
    panel_r: float,
    ui_scale: float,
    scissor_active: bool,
    st: MinimapState | None = None,
) -> None:
    """Draw the viewport rect outline and optional darkened overlay."""
    visible = _get_visible_rect(space, region)
    if not visible:
        return

    # Transform visible viewport rect from tree coords to minimap pixel coords
    vx = round(cx + (visible[0] - tree_cx) * scale)
    vy = round(cy + (visible[1] - tree_cy) * scale)
    vw = round(max((visible[2] - visible[0]) * scale, 1.0))
    vh = round(max((visible[3] - visible[1]) * scale, 1.0))

    # Clamp viewport rect to minimap interior
    v_left = max(vx, mx)
    v_bottom = max(vy, my)
    v_right = min(vx + vw, mx + mw)
    v_top = min(vy + vh, my + mh)

    node_r = colors.get("node_roundness", 2.0) * ui_scale
    hole_w = v_right - v_left
    hole_h = v_top - v_bottom

    # border_alpha_mul = 0.5 if st and st.pressed else 1.0

    # Darkened overlay (optional)
    if getattr(settings, "show_viewport_overlay", True):
        overlay_color = getattr(settings, "viewport_overlay_color", (0.0, 0.0, 0.0, 0.5))
        overlay = _alpha_mul(overlay_color, master_alpha)

        scissor_overlay = scissor_active
        if scissor_overlay:
            gpu.state.scissor_test_set(False)

        try:
            if hole_w > 0 and hole_h > 0:
                _draw_filled_rounded_rect_with_hole(
                    mx,
                    my,
                    mw,
                    mh,
                    panel_r,
                    v_left,
                    v_bottom,
                    hole_w,
                    hole_h,
                    0,
                    overlay,
                )
            else:
                _draw_filled_rounded_rect(mx, my, mw, mh, panel_r, overlay)
        finally:
            if scissor_overlay:
                gpu.state.scissor_test_set(True)
                gpu.state.scissor_set(int(mx + 1), int(my + 1), int(mw - 2), int(mh - 2))

    # Outline the viewport extent when it overlaps the minimap
    if hole_w > 0 and hole_h > 0:
        if st and st.pressed:
            outline_col = _alpha_mul(colors["node_active"], master_alpha)
        else:
            outline_col = _alpha_mul(colors["node_outline"], master_alpha)
        border = 0.5 * ui_scale
        shadow = (0, 0, 0, 0.15 * master_alpha)
        _draw_rounded_rect_border(vx - 1, vy - 1, vw + 2, vh + 2, node_r, shadow, border)
        _draw_rounded_rect_border(vx, vy, vw, vh, node_r, outline_col, border)


def _draw_node_count(
    settings,
    nodes,
    mx: float,
    my: float,
    mw: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    font_id: int,
) -> None:
    """Draw the node count text centered below the minimap."""
    if not getattr(settings, "show_node_count", True):
        return

    info_text = f"{len(nodes)}"
    font_size = int(FONT_SIZE * ui_scale)
    blf.size(font_id, font_size)
    text_w, _ = blf.dimensions(font_id, info_text)

    pad = 1
    tx = mx + (mw - text_w) - 10 * ui_scale
    ty = my + (FONT_SIZE * ui_scale) - 3

    st = _state()
    btn_bottom = None
    if st.frame_view_btn:
        btn_bottom = st.frame_view_btn[1]
    elif st.frame_all_btn:
        btn_bottom = st.frame_all_btn[1]
    if btn_bottom is not None and btn_bottom <= ty + font_size:
        return

    text_color = _alpha_mul(colors["text"], 0.85 * master_alpha)

    _draw_text_with_shadow(font_id, info_text, tx + pad, ty + pad, text_color, font_size)


def _create_quad_indices(n: int) -> list[tuple[int, int, int]]:
    """Helper to populate triangular indices sequentially for quad batches."""
    indices = []
    for i in range(n):
        base = i * 4
        indices.append((base, base + 1, base + 2))
        indices.append((base + 2, base + 3, base))
    return indices


_NODE_ROUNDNESS_DEFAULT = 2.0


def _debounced_compile(st: MinimapState, node_tree, colors, settings, master_alpha, ui_scale):
    """Timer callback: compile tree data after fingerprint settles, then force redraw."""
    current_fingerprint = get_tree_fingerprint(node_tree)
    if st.cached_fingerprint == current_fingerprint:
        st.pending_timer = None
        return None
    with _Timer("compile_tree"):
        _compile_tree_data(st, node_tree, colors, settings, master_alpha, ui_scale)
        st.cached_fingerprint = current_fingerprint
    st.pending_timer = None
    screen = bpy.context.screen
    if screen:
        for area in screen.areas:
            if area.type == "NODE_EDITOR":
                area.tag_redraw()
    return None


def _compile_tree_data(st: MinimapState, node_tree, colors, settings, master_alpha, ui_scale):
    """Compute tree-space data for nodes, wires, sockets, and labels.

    Called only when the node tree fingerprint changes (tree topology,
    selection, mute, active node).  Screen-space transforms (zoom/pan)
    are NOT applied here — they are handled each frame by
    ``_build_minimap_batches()``.

    Stores result in ``st.tree_data``.
    """
    nodes = node_tree.nodes
    active_node = nodes.active
    zoom = st.zoom

    tree_data: dict = {}

    # Hoisted settings lookups (avoid repeated getattr in loops)
    show_frames = getattr(settings, "show_frames", True)
    show_names = getattr(settings, "show_names", True)
    show_socket_indicators = getattr(settings, "show_socket_indicators", False)
    show_wires = getattr(settings, "show_wires", True)
    show_wire_color = getattr(settings, "show_wire_color", True)
    show_frame_labels = getattr(settings, "show_frame_labels", True)
    colored_nodes = getattr(settings, "colored_nodes", True)
    node_label_mode = getattr(settings, "node_label_mode", "COMPACT")

    # Single pre-pass: classify nodes + cache dims/location + compute bounds
    frames = []
    unselected_nodes = []
    selected_nodes = []
    active_node_item = None
    node_data: dict[int, dict] = {}

    bounds_min_x = float("inf")
    bounds_min_y = float("inf")
    bounds_max_x = float("-inf")
    bounds_max_y = float("-inf")

    with _Timer("compile_tree.pre_pass"):
        for node in nodes:
            ptr = node.as_pointer()
            w, h = _get_node_dims(node)
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
        socket_items: dict[tuple, list[tuple[float, float]]] = {}
        default_socket_color = (*colors["wire"][:3], master_alpha)
        default_wire_color = _alpha_mul(colors["wire"], master_alpha)
        out_pos: dict[str, dict] = {}
        in_pos: dict[str, dict] = {}

        for node, is_frame in sorted_items:
            ptr = node.as_pointer()
            w, h = node_data[ptr]["dims"]
            loc_x, loc_y_top = node_data[ptr]["loc"]
            ty = loc_y_top - h

            info: dict = {
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
                info["border_color"] = _srgb_to_linear(_alpha_mul(border_col, master_alpha))
                info["frame_color"] = frame_color
                info["node_r_base"] = _NODE_ROUNDNESS_DEFAULT
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
                if node.mute:
                    info["border_color"] = _srgb_to_linear(_alpha_mul(border_col, 0.35 * master_alpha))
                else:
                    info["border_color"] = _srgb_to_linear(_alpha_mul(border_col, master_alpha))
                info["node_r_base"] = _NODE_ROUNDNESS_DEFAULT * 2

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

            sock_color_cache: dict[int, tuple[float, float, float, float]] = {}

            if show_socket_indicators:
                for is_output, sock_list in [(False, node.inputs), (True, node.outputs)]:
                    try:
                        visible = [s for s in sock_list if not s.hide and s.enabled]
                    except AttributeError:
                        visible = [
                            s for s in sock_list if getattr(s, "hide", False) is False and getattr(s, "enabled", True)
                        ]
                    if not visible:
                        continue

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
                        color = sock_color_cache[sptr]
                        socket_items.setdefault(color, []).append((x_base, sy_tree))

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
        tree_data["socket_items"] = socket_items
        tree_data["socket_ph_base"] = 8.0

    # ------------------------------------------------------------------
    # REROUTE wire endpoints (not in sorted_items, handled separately)
    # ------------------------------------------------------------------
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

                out_pos[node.name] = {s.identifier: (cx_n, cy_n, wire_color) for s in node.outputs}
                in_pos[node.name] = {s.identifier: (cx_n, cy_n, wire_color) for s in node.inputs}

    # ------------------------------------------------------------------
    # Wire connections (using wire endpoints)
    # ------------------------------------------------------------------
    wire_items: dict[tuple, list[tuple[float, float, float, float]]] = {}
    with _Timer("compile_tree.wire_links"):
        if show_wires:
            # Phase 1: extract all link data once (Blender API calls)
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

            # Phase 2: resolve to wire endpoints (pure Python dict ops)
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

    tree_data["wire_items"] = wire_items
    st.tree_data = tree_data


def _build_minimap_batches(st: MinimapState, rect, cx, cy, scale, tree_cx, tree_cy, ui_scale, master_alpha):
    """Transform tree-space data to screen-space and compile GPU draw batches.

    Must be called every frame after ``_compile_tree_data()`` has stored
    ``st.tree_data``.
    """
    tree_data = st.tree_data
    if tree_data is None:
        return

    mx, my, mw, mh, padding, y_margin = rect
    st.rect = (mx, my, mw, mh)
    st.margin = y_margin
    st.padding = padding
    st.scale = scale

    font_id = 0
    min_dim = 3.0 * ui_scale
    node_infos = tree_data["node_infos"]

    all_pos_fill = []
    all_uv_fill = []
    all_half_size_fill = []
    all_radius_fill = []
    all_color_fill = []

    all_pos_border = []
    all_uv_border = []
    all_half_size_border = []
    all_radius_border = []
    all_color_border = []
    all_line_width_border = []

    frame_pos_fill = []
    frame_uv_fill = []
    frame_half_size_fill = []
    frame_radius_fill = []
    frame_color_fill = []

    frame_pos_border = []
    frame_uv_border = []
    frame_half_size_border = []
    frame_radius_border = []
    frame_color_border = []
    frame_line_width_border = []

    cached_text = []

    for info in node_infos:
        nx = round(cx + (info["tree_x"] - tree_cx) * scale)
        ny = round(cy + (info["tree_y"] - tree_cy) * scale)
        nw_s = max(info["tree_w"] * scale, 1.0)
        nh_s = max(info["tree_h"] * scale, 1.0)
        is_frame = info["is_frame"]

        if is_frame:
            node_r = info["node_r_base"] * ui_scale * 1.6
        else:
            node_r = info["node_r_base"] * ui_scale * (scale * 4)

        is_tiny = (nw_s < min_dim or nh_s < min_dim) and not is_frame

        if is_tiny:
            nw_s_final = max(nw_s, min_dim)
            nh_s_final = max(nh_s, min_dim)
            hw = nw_s_final / 2
            hh = nh_s_final / 2
            all_pos_fill.extend(
                [
                    (nx, ny, 0.0),
                    (nx + nw_s_final, ny, 0.0),
                    (nx + nw_s_final, ny + nh_s_final, 0.0),
                    (nx, ny + nh_s_final, 0.0),
                ]
            )
            all_uv_fill.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
            all_half_size_fill.extend([(hw, hh)] * 4)
            all_radius_fill.extend([node_r] * 4)
            all_color_fill.extend([info["fill_color"]] * 4)
        else:
            hw = nw_s / 2
            hh = nh_s / 2

            pos_fill = frame_pos_fill if is_frame else all_pos_fill
            uv_fill = frame_uv_fill if is_frame else all_uv_fill
            hs_fill = frame_half_size_fill if is_frame else all_half_size_fill
            rad_fill = frame_radius_fill if is_frame else all_radius_fill
            col_fill = frame_color_fill if is_frame else all_color_fill
            pos_fill.extend(
                [
                    (nx, ny, 0.0),
                    (nx + nw_s, ny, 0.0),
                    (nx + nw_s, ny + nh_s, 0.0),
                    (nx, ny + nh_s, 0.0),
                ]
            )
            uv_fill.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
            hs_fill.extend([(hw, hh)] * 4)
            rad_fill.extend([node_r] * 4)
            col_fill.extend([info["fill_color"]] * 4)

            bw = info["border_w"]
            pb = frame_pos_border if is_frame else all_pos_border
            ub = frame_uv_border if is_frame else all_uv_border
            hsb = frame_half_size_border if is_frame else all_half_size_border
            rb = frame_radius_border if is_frame else all_radius_border
            cb = frame_color_border if is_frame else all_color_border
            lwb = frame_line_width_border if is_frame else all_line_width_border
            pb.extend(
                [
                    (nx, ny, 0.0),
                    (nx + nw_s, ny, 0.0),
                    (nx + nw_s, ny + nh_s, 0.0),
                    (nx, ny + nh_s, 0.0),
                ]
            )
            ub.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
            hsb.extend([(hw, hh)] * 4)
            rb.extend([node_r] * 4)
            cb.extend([info["border_color"]] * 4)
            lwb.extend([bw] * 4)

            # Labels
            if is_frame:
                frame_lbl = info.get("frame_label")
                if frame_lbl:
                    text, text_color, bg_color_lbl = frame_lbl
                    label_font_size = max(6, min(11, int(11 * ui_scale * scale * 8)))
                    blf.size(font_id, label_font_size)
                    tw, th = blf.dimensions(font_id, text)
                    lx = nx + (nw_s - tw) / 2
                    ly = ny + nh_s + 3 * ui_scale
                    label_pad = 2 * ui_scale

                    frame_pos_fill.extend(
                        [
                            (lx - label_pad, ly - label_pad, 0.0),
                            (lx + tw + label_pad, ly - label_pad, 0.0),
                            (lx + tw + label_pad, ly + th + label_pad, 0.0),
                            (lx - label_pad, ly + th + label_pad, 0.0),
                        ]
                    )
                    hw_lp = (tw + 2 * label_pad) / 2
                    hh_lp = (th + 2 * label_pad) / 2
                    frame_uv_fill.extend([(-hw_lp, -hh_lp), (hw_lp, -hh_lp), (hw_lp, hh_lp), (-hw_lp, hh_lp)])
                    frame_half_size_fill.extend([(hw_lp, hh_lp)] * 4)
                    frame_radius_fill.extend([node_r] * 4)
                    frame_color_fill.extend([bg_color_lbl] * 4)
                    cached_text.append((font_id, text, lx, ly, text_color, label_font_size))
            else:
                lbl_type = info.get("node_label_type")
                lbl_text = info.get("node_label_text")
                if lbl_type and lbl_text and nw_s > 6 * ui_scale and nh_s > 6 * ui_scale:
                    text_color = info["node_label_color"]
                    if lbl_type == "full":
                        font_size = max(6, min(int(11 * ui_scale), int(min(nw_s, nh_s) * 0.35)))
                        lines = _get_node_label_lines(lbl_text, font_id, font_size, nw_s - 4 * ui_scale, 3)
                        if lines:
                            blf.size(font_id, font_size)
                            line_h = blf.dimensions(font_id, "Ay")[1] + 1
                            asc_h = blf.dimensions(font_id, "A")[1]
                            vis_h = (len(lines) - 1) * line_h + asc_h
                            start_y = ny + (nh_s - vis_h) / 2
                            for i, line in enumerate(lines):
                                lw, _ = blf.dimensions(font_id, line)
                                lx = nx + (nw_s - lw) / 2
                                ly = start_y + (len(lines) - 1 - i) * line_h
                                cached_text.append((font_id, line, lx, ly, text_color, font_size))
                    else:
                        font_size = max(6, min(int(11 * ui_scale), int(min(nw_s, nh_s) * 0.45)))
                        blf.size(font_id, font_size)
                        tw, th = blf.dimensions(font_id, lbl_text)
                        tx = nx + (nw_s - tw) / 2
                        ty = ny + (nh_s - th) / 2
                        cached_text.append((font_id, lbl_text, tx, ty, text_color, font_size))

    # Compile GPU batches
    num_fills = len(all_pos_fill) // 4
    if num_fills > 0:
        shader = _get_batch_rect_shader()
        st.cached_backdrops_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": all_pos_fill,
                "uv": all_uv_fill,
                "halfSize": all_half_size_fill,
                "radius": all_radius_fill,
                "color": all_color_fill,
            },
            indices=_create_quad_indices(num_fills),
        )
    else:
        st.cached_backdrops_batch = None

    num_borders = len(all_pos_border) // 4
    if num_borders > 0:
        shader = _get_batch_rect_border_shader()
        st.cached_borders_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": all_pos_border,
                "uv": all_uv_border,
                "halfSize": all_half_size_border,
                "radius": all_radius_border,
                "color": all_color_border,
                "lineWidth": all_line_width_border,
            },
            indices=_create_quad_indices(num_borders),
        )
    else:
        st.cached_borders_batch = None

    num_frame_fills = len(frame_pos_fill) // 4
    if num_frame_fills > 0:
        shader = _get_batch_rect_shader()
        st.cached_frames_fill_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": frame_pos_fill,
                "uv": frame_uv_fill,
                "halfSize": frame_half_size_fill,
                "radius": frame_radius_fill,
                "color": frame_color_fill,
            },
            indices=_create_quad_indices(num_frame_fills),
        )
    else:
        st.cached_frames_fill_batch = None

    num_frame_borders = len(frame_pos_border) // 4
    if num_frame_borders > 0:
        shader = _get_batch_rect_border_shader()
        st.cached_frames_border_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": frame_pos_border,
                "uv": frame_uv_border,
                "halfSize": frame_half_size_border,
                "radius": frame_radius_border,
                "color": frame_color_border,
                "lineWidth": frame_line_width_border,
            },
            indices=_create_quad_indices(num_frame_borders),
        )
    else:
        st.cached_frames_border_batch = None

    st.cached_text = cached_text

    # Sockets — unified batch with per-vertex color + auto-hide by zoom
    ph = max(1, tree_data["socket_ph_base"] * scale * ui_scale)
    pw = ph
    st.cached_socket_ph = ph
    if tree_data["socket_items"] and scale >= _MIN_SOCKET_SCALE:
        half_w = pw / 2
        half_h = ph / 2
        r = ph / 2
        socket_all_pos = []
        socket_all_uv = []
        socket_all_hs = []
        socket_all_r = []
        socket_all_c = []
        for color, positions in tree_data["socket_items"].items():
            for sx_tree, sy_tree in positions:
                sx = cx + (sx_tree - tree_cx) * scale
                sy = cy + (sy_tree - tree_cy) * scale
                _pad = 1.5
                socket_all_pos.extend(
                    [
                        (sx - half_w - _pad, sy - half_h - _pad, 0.0),
                        (sx + half_w + _pad, sy - half_h - _pad, 0.0),
                        (sx + half_w + _pad, sy + half_h + _pad, 0.0),
                        (sx - half_w - _pad, sy + half_h + _pad, 0.0),
                    ]
                )
                socket_all_uv.extend(
                    [
                        (-half_w - _pad, -half_h - _pad),
                        (half_w + _pad, -half_h - _pad),
                        (half_w + _pad, half_h + _pad),
                        (-half_w - _pad, half_h + _pad),
                    ]
                )
                socket_all_hs.extend([(half_w, half_h)] * 4)
                socket_all_r.extend([r] * 4)
                socket_all_c.extend([_srgb_to_linear(color)] * 4)
        num_s = len(socket_all_pos) // 4
        if num_s > 0:
            shader = _get_batch_rect_shader()
            st.cached_socket_batch = batch_for_shader(
                shader,
                "TRIS",
                {
                    "pos": socket_all_pos,
                    "uv": socket_all_uv,
                    "halfSize": socket_all_hs,
                    "radius": socket_all_r,
                    "color": socket_all_c,
                },
                indices=_create_quad_indices(num_s),
            )
        else:
            st.cached_socket_batch = None
    else:
        st.cached_socket_batch = None

    # Wires
    wires_by_color = {}
    thickness = max(1.0, 2.0 * scale)
    for color, items in tree_data["wire_items"].items():
        group = []
        for out_x, out_y, in_x, in_y in items:
            x1 = cx + (out_x - tree_cx) * scale
            y1 = cy + (out_y - tree_cy) * scale
            x2 = cx + (in_x - tree_cx) * scale
            y2 = cy + (in_y - tree_cy) * scale
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 0.5:
                continue
            angle = math.atan2(dy, dx)
            mx_w = (x1 + x2) / 2
            my_w = (y1 + y2) / 2
            group.append((mx_w, my_w, length, angle))
        if group:
            wires_by_color[color] = group
    st.cached_wires = wires_by_color
    st.cached_wire_thickness = thickness


def draw_minimap() -> None:
    """Main entry point — orchestrate minimap drawing in the Node Editor."""
    context = bpy.context
    space = context.space_data
    region = context.region

    # Early exit checks
    st = _state()
    if _early_exit(context, space, st):
        show_overlays = space.overlay.show_overlays if space else "?"
        enabled = st.enabled
        logger.debug("draw_minimap: early exit (type=%s overlays=%s enabled=%s)", space.type, show_overlays, enabled)
        return

    addon = context.preferences.addons.get(__package__)
    settings = addon.preferences.settings

    # Defer auto-launch until registration is fully complete
    # to avoid invoking the modal with a stale context.
    if not _registration_state["done"]:
        logger.debug("draw_minimap: registration not done, skipping auto-launch")
    else:
        # Auto-start modal operator for pan/zoom interaction (one per window)
        win = context.window
        win_ptr = win.as_pointer() if win else 0
        has_modal = win_ptr in _minimap_window_operators if win else False
        logger.debug(
            "draw_minimap: area=%d win=%d modal_ops=%s has_modal=%s interactive=%s",
            context.area.as_pointer() if context.area else 0,
            win_ptr,
            list(_minimap_window_operators.keys()),
            has_modal,
            getattr(settings, "interactive", True),
        )
        if getattr(settings, "interactive", True):
            if win and not has_modal:
                logger.debug("draw_minimap: invoking nodemap.navigate for window %d", win_ptr)
                try:
                    bpy.ops.nodemap.navigate("INVOKE_DEFAULT")
                    logger.debug("draw_minimap: nodemap.navigate invoked successfully")
                except RuntimeError as e:
                    logger.debug("draw_minimap: nodemap.navigate failed: %s", e)
            elif not win:
                logger.debug("draw_minimap: cannot invoke — context.window is None")

    # Guard: must have a valid node tree with nodes
    node_tree = space.edit_tree
    if not node_tree or not node_tree.nodes or len(node_tree.nodes) == 0:
        return
    nodes = node_tree.nodes
    bounds = _get_node_tree_bounds(nodes)
    if bounds[2] - bounds[0] <= 0 or bounds[3] - bounds[1] <= 0:
        return

    # Start cProfile for this area (only when TRACE logging is on)
    _maybe_start_profiler(st)

    # Log active settings every frame at TRACE level
    logger.trace(
        "SETTINGS %d nodes | show_wires=%d show_names=%d label_mode=%s"
        " colored_nodes=%d socket_indicators=%d wire_color=%d frame_labels=%d",
        len(nodes),
        getattr(settings, "show_wires", True),
        getattr(settings, "show_names", True),
        getattr(settings, "node_label_mode", "COMPACT"),
        getattr(settings, "colored_nodes", True),
        getattr(settings, "show_socket_indicators", False),
        getattr(settings, "show_wire_color", True),
        getattr(settings, "show_frame_labels", True),
    )

    # Compute dimensions and layout
    with _Timer("setup"):
        ui_scale = _get_ui_scale()
        colors = _get_node_editor_theme_colors()
        master_alpha = getattr(settings, "opacity", 0.85)
        corner = getattr(settings, "position", "TOP_RIGHT")

        rect = _compute_minimap_rect(settings, ui_scale, space, region, corner, st)
        if rect is None:
            return
        mx, my, mw, mh, padding, y_margin = rect

        st.rect = (mx, my, mw, mh)
        st.tree_bounds = bounds
        st.margin = y_margin
        st.padding = padding

        _clamp_pan_to_viewport(space, region, st)

    # Debounce: schedule compile after fingerprint settles (via bpy.app.timers)
    current_fingerprint = get_tree_fingerprint(node_tree)
    if st.cached_fingerprint != current_fingerprint:
        if st.pending_timer is not None:
            try:
                bpy.app.timers.unregister(st.pending_timer)
            except ValueError:
                pass
        delay = getattr(settings, "debounce_interval", 0.15)
        timer = bpy.app.timers.register(
            lambda: _debounced_compile(st, node_tree, colors, settings, master_alpha, ui_scale),
            first_interval=delay,
        )
        st.pending_timer = timer

    # Build screen-space batches every frame (applies current zoom/pan)
    cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform(st, space, region)
    with _Timer("build_batches"):
        _build_minimap_batches(st, rect, cx, cy, scale, tree_cx, tree_cy, ui_scale, master_alpha)

    # Draw minimap panel
    try:
        original_blend = gpu.state.blend_get()
    except Exception:
        original_blend = None
    gpu.state.blend_set("ALPHA")

    with _Timer("draw_background"):
        bg_color, panel_r = _draw_background(mx, my, mw, mh, colors, master_alpha)

    # Clip node/wire content to minimap interior
    with _Timer("setup_scissor"):
        scissor_state = _setup_scissor(mx, my, mw, mh)
        scissor_active = scissor_state[0]

    # Layer 1: Frame nodes (behind wires)
    frames_fill_batch = st.cached_frames_fill_batch
    frames_border_batch = st.cached_frames_border_batch
    if frames_fill_batch or frames_border_batch:
        with _Timer("draw_frames"):
            fill_shader = _get_batch_rect_shader()
            border_shader = _get_batch_rect_border_shader()
            mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
            if frames_fill_batch:
                fill_shader.bind()
                fill_shader.uniform_float("ModelViewProjectionMatrix", mvp)
                frames_fill_batch.draw(fill_shader)
            if frames_border_batch:
                border_shader.bind()
                border_shader.uniform_float("ModelViewProjectionMatrix", mvp)
                frames_border_batch.draw(border_shader)

    # Layer 2: Connection wires
    wires_by_color = st.cached_wires or {}
    thickness = st.cached_wire_thickness
    if getattr(settings, "show_wires", True) and wires_by_color:
        with _Timer("draw_wires"):
            shadow_alpha = 0.35 * master_alpha
            if shadow_alpha > 0:
                shadow_group = [
                    (wx, wy, length, angle) for group in wires_by_color.values() for wx, wy, length, angle in group
                ]
                _batch_draw_pills(shadow_group, thickness * 2.5, (0.0, 0.0, 0.0, shadow_alpha))
            for wire_color, group in wires_by_color.items():
                _batch_draw_pills(group, thickness, wire_color)

    # Layer 3: Node backgrounds
    backdrops_batch = st.cached_backdrops_batch
    if backdrops_batch:
        with _Timer("draw_backdrops"):
            fill_shader = _get_batch_rect_shader()
            fill_shader.bind()
            fill_shader.uniform_float(
                "ModelViewProjectionMatrix", gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
            )
            backdrops_batch.draw(fill_shader)

    # Layer 4: Node borders
    borders_batch = st.cached_borders_batch
    if borders_batch:
        with _Timer("draw_borders"):
            border_shader = _get_batch_rect_border_shader()
            border_shader.bind()
            border_shader.uniform_float(
                "ModelViewProjectionMatrix", gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
            )
            borders_batch.draw(border_shader)

    # Layer 5: Socket indicator pills (single batch with per-vertex color)
    socket_batch = st.cached_socket_batch
    if getattr(settings, "show_socket_indicators", False) and socket_batch:
        with _Timer("draw_sockets"):
            shader = _get_batch_rect_shader()
            shader.bind()
            shader.uniform_float(
                "ModelViewProjectionMatrix",
                gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
            )
            socket_batch.draw(shader)

    # Layer 6: Text labels
    cached_text = st.cached_text or []
    if cached_text:
        with _Timer("draw_text"):
            gpu.state.blend_set("ALPHA")
            for font_id, text, lx, ly, text_color, font_size in cached_text:
                _draw_text_with_shadow(font_id, text, lx, ly, text_color, font_size)
            gpu.state.blend_set("ALPHA")

    # Layer 7: Viewport overlay with cutout hole
    with _Timer("draw_viewport"):
        _draw_viewport_overlay(
            settings,
            space,
            region,
            mx,
            my,
            mw,
            mh,
            cx,
            cy,
            scale,
            tree_cx,
            tree_cy,
            colors,
            master_alpha,
            panel_r,
            ui_scale,
            scissor_active,
            st,
        )

    # Layer 8: Scrollbars
    with _Timer("draw_scrollbars"):
        _draw_minimap_scrollbars(
            mx,
            my,
            mw,
            mh,
            padding,
            cx,
            cy,
            scale,
            tree_cx,
            tree_cy,
            bounds,
            colors,
            ui_scale,
            master_alpha,
        )

    # Top corner frame-all brackets
    with _Timer("_draw_frame_all_button"):
        _draw_frame_all_button(mx, my, mw, mh, padding, bounds, colors, ui_scale, master_alpha)

    # Frame-view button (below frame-all)
    with _Timer("_draw_frame_view_button"):
        _draw_frame_view_button(mx, my - 1, mw, mh, padding, colors, ui_scale, master_alpha)

    # Edge resize handle pills
    with _Timer("draw_resize_handles"):
        _draw_resize_handles(mx, my, mw, mh, colors, master_alpha, ui_scale, corner, st)

    # Node count overlay text
    with _Timer("draw_node_count"):
        content_nodes = [n for n in nodes if n.type not in ("FRAME", "REROUTE")]
        font_id = 0
        _draw_node_count(settings, content_nodes, mx, my, mw, colors, master_alpha, ui_scale, font_id)

    # Restore GPU state
    _teardown_scissor(scissor_state)
    try:
        gpu.state.blend_set(original_blend if original_blend else "NONE")
    except Exception:
        gpu.state.blend_set("NONE")

    # Stop & dump profile stats after N frames
    _maybe_stop_profiler(st)


def _get_socket_pos(node, socket, is_output):
    """Return (x, y) in tree coordinates for a socket position."""
    w, h = _get_node_dims(node)

    if node.type == "REROUTE":
        return node.location_absolute.x + w / 2, node.location_absolute.y - h / 2

    x = node.location_absolute.x + (w if is_output else 0)

    visible = [
        s
        for s in (node.outputs if is_output else node.inputs)
        if not getattr(s, "hide", False) and getattr(s, "enabled", True)
    ]

    if not visible:
        return x, node.location_absolute.y - h * 0.5

    try:
        idx = visible.index(socket)
    except (ValueError, AttributeError):
        idx = 0

    top_y = node.location_absolute.y
    body_top = top_y
    body_bot = top_y - h

    num = len(visible)
    body_range = body_top - body_bot
    if body_range <= 0 or num <= 1:
        y = (body_top + body_bot) * 0.5
    else:
        y = body_top - body_range * (idx + 1) / (num + 1)

    return x, y


def _draw_wires(nodes, tree_cx, tree_cy, scale, cx, cy, colors, master_alpha=1.0, use_socket_color=False):
    """Fallback wire drawing handler (compiled wires take precedence)."""
    pass


def _draw_minimap_scrollbars(
    mx, my, mw, mh, padding, cx, cy, scale, tree_cx, tree_cy, bounds, colors, ui_scale, master_alpha
):
    """Draw horizontal/vertical scrollbar thumbs when zoomed in."""
    inner_l = mx + padding
    inner_r = mx + mw - padding
    inner_b = my + padding
    inner_t = my + mh - padding
    inner_w = mw - 2 * padding
    inner_h = mh - 2 * padding

    bbox_l, bbox_b, bbox_r, bbox_t = bounds
    bbox_w = bbox_r - bbox_l
    bbox_h = bbox_t - bbox_b
    if bbox_w <= 0 or bbox_h <= 0:
        return

    # Convert minimap inner rect corners back to tree coords to find visible extent
    tree_l = tree_cx + (inner_l - cx) / scale
    tree_r = tree_cx + (inner_r - cx) / scale
    tree_b = tree_cy + (inner_b - cy) / scale
    tree_t = tree_cy + (inner_t - cy) / scale

    # Clamp visible area to bbox (viewport cannot extend past tree bounds)
    v_left = max(bbox_l, min(bbox_r, tree_l))
    v_right = max(bbox_l, min(bbox_r, tree_r))
    v_bottom = max(bbox_b, min(bbox_t, tree_b))
    v_top = max(bbox_b, min(bbox_t, tree_t))

    visible_w = v_right - v_left
    visible_h = v_top - v_bottom
    if visible_w >= bbox_w and visible_h >= bbox_h:
        return

    bar_thick = max(2, int(3 * ui_scale))
    bar_off = int(2 * ui_scale)
    min_thumb = int(6 * ui_scale)
    scroll_color = _alpha_mul(colors["scroll_item"], master_alpha * 0.65)

    # Horizontal scrollbar (bottom edge)
    if visible_w < bbox_w:
        track_w = inner_w
        thumb_w = max(min_thumb, int(track_w * visible_w / bbox_w))
        thumb_x = inner_l + int(track_w * (v_left - bbox_l) / bbox_w)
        thumb_y = my + bar_off
        _draw_pill(thumb_x, thumb_y, thumb_w, bar_thick, scroll_color)

    # Vertical scrollbar (right edge)
    if visible_h < bbox_h:
        track_h = inner_h
        thumb_h = max(min_thumb, int(track_h * visible_h / bbox_h))
        thumb_x2 = mx + mw - bar_off - bar_thick
        thumb_y2 = inner_b + int(track_h * (v_bottom - bbox_b) / bbox_h)
        _draw_pill(thumb_x2, thumb_y2, bar_thick, thumb_h, scroll_color)


BTN_BG_COLOR = (0, 0, 0, 0.3)


def _draw_frame_all_button(mx, my, mw, mh, padding, bounds, colors, ui_scale, master_alpha):
    """Draw a frame-all button at the top-right of the minimap inner area."""
    addon = bpy.context.preferences.addons.get(__package__)
    settings = getattr(addon.preferences, "settings", None) if addon else None
    if not settings or not getattr(settings, "show_frame_all_btn", True) or not getattr(settings, "interactive", True):
        _state().frame_all_btn = None
        return
    inner_l = mx
    inner_t = my + mh - padding

    bbox_l, bbox_b, bbox_r, bbox_t = bounds
    bbox_w = bbox_r - bbox_l
    bbox_h = bbox_t - bbox_b
    if bbox_w <= 0 or bbox_h <= 0:
        return

    st = _state()

    btn_size = FRAME_ALL_BTN_SIZE * ui_scale
    margin = FRAME_ALL_BTN_MARGIN * ui_scale
    x = inner_l + mw - btn_size - margin - padding
    y = inner_t - btn_size - margin
    ico_color = _alpha_mul(colors["text"], master_alpha * 0.7)
    border_color = _alpha_mul(BTN_BG_COLOR, master_alpha * 0.2)

    show_frame_view = getattr(settings, "show_frame_view_btn", True) and getattr(settings, "interactive", True)

    if show_frame_view:
        gap = 3 * ui_scale
        fy = y - gap - btn_size
        _draw_pill(x, fy, btn_size, btn_size * 2 + gap, _alpha_mul(BTN_BG_COLOR, master_alpha))
        _draw_pill_border(x, fy, btn_size, btn_size * 2 + gap, border_color, 0.5)

    else:
        _draw_pill(x, y, btn_size, btn_size, _alpha_mul(BTN_BG_COLOR, master_alpha))
        _draw_pill_border(x, y, btn_size, btn_size, border_color, 0.5)

    # Corner brackets icon (four brackets pointing outward)
    i = 5 * ui_scale
    t = max(1, int(1.5 * ui_scale))
    arm = btn_size * 0.15

    # Top-left bracket
    _draw_filled_rounded_rect(x + i, y + i, arm, t, t * 0.5, ico_color)
    _draw_filled_rounded_rect(x + i, y + i, t, arm, t * 0.5, ico_color)
    # Top-right bracket
    _draw_filled_rounded_rect(x + btn_size - i - arm, y + i, arm, t, t * 0.5, ico_color)
    _draw_filled_rounded_rect(x + btn_size - i - t, y + i, t, arm, t * 0.5, ico_color)
    # Bottom-left bracket
    _draw_filled_rounded_rect(x + i, y + btn_size - i - t, arm, t, t * 0.5, ico_color)
    _draw_filled_rounded_rect(x + i, y + btn_size - i - arm, t, arm, t * 0.5, ico_color)
    # Bottom-right bracket
    _draw_filled_rounded_rect(x + btn_size - i - arm, y + btn_size - i - t, arm, t, t * 0.5, ico_color)
    _draw_filled_rounded_rect(x + btn_size - i - t, y + btn_size - i - arm, t, arm, t * 0.5, ico_color)

    st.frame_all_btn = (x, y, btn_size, btn_size)


def _draw_frame_view_button(mx, my, mw, mh, padding, colors, ui_scale, master_alpha):
    """Draw a frame-view button below the frame-all button."""
    addon = bpy.context.preferences.addons.get(__package__)
    settings = getattr(addon.preferences, "settings", None) if addon else None
    if not settings or not getattr(settings, "show_frame_view_btn", True) or not getattr(settings, "interactive", True):
        _state().frame_view_btn = None
        return

    inner_l = mx
    inner_t = my + mh - padding

    st = _state()

    btn_size = FRAME_ALL_BTN_SIZE * ui_scale
    margin = FRAME_ALL_BTN_MARGIN * ui_scale
    gap = 2 * ui_scale
    x = inner_l + mw - btn_size - margin - padding
    y = inner_t - btn_size - margin  # frame-all y
    has_frame_all = getattr(settings, "show_frame_all_btn", True) and getattr(settings, "interactive", True)
    fy = y - gap - btn_size if has_frame_all else y
    ico_color = _alpha_mul(colors["text"], master_alpha * 0.8)

    # Viewport rectangle icon
    inset = 5 * ui_scale
    rx = round(x + inset)
    ry = round(fy + inset)
    rw = btn_size - 2 * inset
    rh = btn_size - 2 * inset
    t = max(1, int(1.5 * ui_scale))

    if not _state().frame_all_btn:
        _draw_pill(x, fy, btn_size, btn_size, BTN_BG_COLOR)

    _draw_rounded_rect_border(rx, ry, rw, rh, t, ico_color, 0.1)

    st.frame_view_btn = (x, fy, btn_size, btn_size)
