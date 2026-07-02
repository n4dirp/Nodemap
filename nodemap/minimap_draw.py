"""Minimap rendering in the Node Editor."""

import io
import logging
import math
import time

import blf
import bpy
import gpu

try:
    import cProfile

    _HAS_C_PROFILE = True
except ImportError:
    _HAS_C_PROFILE = False

from .gpu_draw import (
    _batch_draw_pills,
    _draw_filled_rounded_rect,
    _draw_filled_rounded_rect_with_hole,
    _draw_rounded_rect_border,
    _draw_text_with_shadow,
)
from .helpers import (
    _clamp_pan_to_viewport,
    _compute_outline_color,
    _get_minimap_transform,
    _get_node_color,
    _get_node_dims,
    _get_node_editor_theme_colors,
    _get_node_tree_bounds,
    _get_safe_bounds,
    _get_ui_scale,
    _get_visible_rect,
    _state,
)
from .preferences import TRACE_LEVEL

logger = logging.getLogger(__package__)

FONT_SIZE = 11
MAP_PADDING = 10
FRAME_ALL_BTN_SIZE = 16
FRAME_ALL_BTN_MARGIN = 1


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


def _maybe_start_profiler(st: dict) -> None:
    """Start cProfile if TRACE is enabled and profiling is not already active.

    Stores the profiler in *st* so each area gets its own session.
    """
    if not _HAS_C_PROFILE:
        return
    if not logger.isEnabledFor(TRACE_LEVEL):
        return
    if st.get("_profiling_active"):
        return
    prefs = bpy.context.preferences.addons[__package__].preferences
    if not getattr(prefs, "logging_enabled", False) or getattr(prefs, "logging_level", "INFO") != "TRACE":
        return
    try:
        profiler = cProfile.Profile()
        profiler.enable()
    except ValueError:
        st["_profiler"] = None
        st["_profiling_active"] = False
        return
    st["_profiler"] = profiler
    st["_profiling_active"] = True
    st["_profiling_frame_count"] = 0
    logger.trace("PROFILER: started (will dump after %d frames)", _PROFILE_FRAMES)


def _maybe_stop_profiler(st: dict) -> None:
    """Increment frame count; dump profile stats after *_PROFILE_FRAMES* frames."""
    if not _HAS_C_PROFILE:
        return
    if not st.get("_profiling_active"):
        return
    st["_profiling_frame_count"] = st.get("_profiling_frame_count", 0) + 1
    if st["_profiling_frame_count"] < _PROFILE_FRAMES:
        return
    profiler = st.get("_profiler")
    if profiler is None:
        st["_profiling_active"] = False
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
        st["_profiling_active"] = False


def _early_exit(context, space, st) -> bool:
    """Return True if the minimap should not be drawn."""
    if space.type != "NODE_EDITOR":
        return True
    if not space.overlay.show_overlays:
        return True
    if not st.get("enabled", True):
        return True
    addon = context.preferences.addons.get(__package__)
    if not addon:
        return True
    return False


def _compute_minimap_rect(
    settings, ui_scale, space, region, corner, st
) -> tuple[float, float, float, float, float, float] | None:
    """Compute the minimap rectangle position and dimensions."""
    sx, sy, ex, ey = _get_safe_bounds(bpy.context.area, region, space, corner)
    safe_w = ex - sx
    safe_h = ey - sy

    # Compute desired size, capped to % of safe region
    mw = getattr(settings, "minimap_width", 200) * ui_scale
    mh = getattr(settings, "minimap_height", 200) * ui_scale
    max_mw_pct = getattr(settings, "max_width_pct", 30) / 100.0
    max_mh_pct = getattr(settings, "max_height_pct", 40) / 100.0
    mw = min(mw, safe_w * max_mw_pct)
    mh = min(mh, safe_h * max_mh_pct)

    x_margin = MAP_PADDING * ui_scale
    y_margin = x_margin

    # Adjust top margin when breadcrumb path or compositing asset shelf is visible
    match corner:
        case "TOP_RIGHT" | "TOP_LEFT":
            if getattr(space.overlay, "show_context_path", False):
                y_margin = (MAP_PADDING + 20) * ui_scale
        case "BOTTOM_RIGHT" | "BOTTOM_LEFT":
            if space.node_tree and space.node_tree.type == "COMPOSITING":
                if getattr(space, "show_region_asset_shelf", False):
                    y_margin = (MAP_PADDING + 25) * ui_scale

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
        st["rect"] = (0, 0, 0, 0)
        return None

    return mx, my, mw, mh, padding, y_margin


def _draw_background(
    mx: float, my: float, mw: float, mh: float, colors: dict, master_alpha: float
) -> tuple[tuple[float, float, float, float], float]:
    """Draw the minimap backdrop rounded rect and border."""
    bg_color = colors["bg"][:3] + (master_alpha,)
    panel_r = colors.get("panel_roundness", 4.0)
    _draw_filled_rounded_rect(mx, my, mw, mh, panel_r, bg_color)
    border_color = (*colors["bg_border"][:3], colors["bg_border"][3] * master_alpha)
    _draw_rounded_rect_border(mx, my, mw, mh, panel_r, border_color, 0.5)
    return bg_color, panel_r


def _setup_scissor(mx: float, my: float, mw: float, mh: float) -> bool:
    """Enable scissor test to clip content to minimap interior."""
    try:
        gpu.state.scissor_test_set(True)
        gpu.state.scissor_set(int(mx + 1), int(my + 1), int(mw - 2), int(mh - 2))
        return True
    except Exception:
        return False


def _teardown_scissor(active: bool) -> None:
    """Disable scissor test if it was active."""
    if active:
        try:
            gpu.state.scissor_test_set(False)
        except Exception:
            pass


def _draw_frame_nodes(
    frames: list,
    cx: float,
    cy: float,
    scale: float,
    tree_cx: float,
    tree_cy: float,
    colors: dict,
    settings,
    master_alpha: float,
    hovered_node_name: str | None,
    ui_scale: float,
    font_id: int,
) -> None:
    """Draw frame nodes as transparent rounded rects with optional labels."""

    node_r = colors.get("node_roundness", 2.0) * ui_scale

    for node in frames:
        w, h = _get_node_dims(node)

        # Transform node tree coords to minimap pixel coords
        nx = cx + (node.location_absolute.x - tree_cx) * scale
        ny = cy + (node.location_absolute.y - h - tree_cy) * scale
        nw_s = max(w * scale, 1.0)
        nh_s = max(h * scale, 1.0)

        is_hovered = node.name == hovered_node_name
        frame_alpha = (0.6 if is_hovered else 0.5) * master_alpha

        # Fill: use theme or per-node custom color
        if getattr(settings, "colored_nodes", True):
            frame_color = _get_node_color(node, colors.get("frame_node", colors["node"]))
        else:
            frame_color = colors.get("frame_node", colors["node"])
        bg_frame = (frame_color[0], frame_color[1], frame_color[2], frame_alpha)
        _draw_filled_rounded_rect(nx, ny, nw_s, nh_s, node_r, bg_frame)

        # Border: highlight hovered frames with the indicator color
        border_col = colors["node_selected"] if node.select else frame_color
        border_col = (*border_col[:3], border_col[3] * master_alpha)
        _draw_rounded_rect_border(nx, ny, nw_s, nh_s, node_r, border_col, 0.5)

        # Label: centered above the frame, only if large enough
        frame_label = node.label
        if (
            frame_label
            and getattr(settings, "show_frame_labels", True)
            and nw_s > 20 * ui_scale
            and nh_s > 14 * ui_scale
        ):
            label_font_size = max(6, min(int(11 * ui_scale), int(nh_s * 0.2)))
            text_color = (*colors["text"][:3], colors["text"][3] * master_alpha)
            blf.size(font_id, label_font_size)
            tw, _ = blf.dimensions(font_id, frame_label)
            if tw < nw_s - 4 * ui_scale:
                lx = nx + (nw_s - tw) / 2
                ly = ny + nh_s + 2 * ui_scale
                _draw_text_with_shadow(font_id, frame_label, lx, ly, text_color, label_font_size)
                gpu.state.blend_set("ALPHA")


def _draw_regular_nodes(
    regular_nodes: list,
    cx: float,
    cy: float,
    scale: float,
    tree_cx: float,
    tree_cy: float,
    colors: dict,
    settings,
    master_alpha: float,
    hovered_node_name: str | None,
    ui_scale: float,
    bg_color: tuple[float, float, float, float],
    font_id: int,
    show_socket_indicators: bool = False,
    use_socket_color: bool = True,
) -> None:
    """Draw regular nodes with fill, border, label, and mute overlay."""
    node_r = colors.get("node_roundness", 2.0) * ui_scale * (scale * 2)
    min_dim = 3.0 * ui_scale

    pills_by_color: dict[tuple[float, ...], list[tuple[float, float, float, float]]] = {}
    if show_socket_indicators:
        pw = max(2.0, 8.0 * scale * ui_scale)
        ph = max(2.0, 8.0 * scale * ui_scale)

    for node in regular_nodes:
        if node.type == "REROUTE":
            continue

        w, h = _get_node_dims(node)

        # Transform node tree coords to minimap pixel coords
        nx = round(cx + (node.location_absolute.x - tree_cx) * scale)
        ny = round(cy + (node.location_absolute.y - h - tree_cy) * scale)
        nw_s = max(w * scale, 1.0)
        nh_s = max(h * scale, 1.0)

        is_hovered = node.name == hovered_node_name

        # Resolve fill color: custom, theme-mapped by color_tag, or default
        if getattr(settings, "colored_nodes", True):
            fill_color = _get_node_color(node, colors["node"])
        else:
            fill_color = colors["node"]
        fill_color = (*fill_color[:3], fill_color[3] * master_alpha)
        if is_hovered:
            # Brighten hovered nodes for visual feedback
            fill_color = (
                min(fill_color[0] * 1.35, 1.0),
                min(fill_color[1] * 1.35, 1.0),
                min(fill_color[2] * 1.35, 1.0),
                fill_color[3],
            )

        if nw_s < min_dim or nh_s < min_dim:
            # Tiny nodes: draw as minimal dots, skip border/label
            _draw_filled_rounded_rect(nx, ny, max(nw_s, min_dim), max(nh_s, min_dim), node_r, fill_color)
        else:
            _draw_filled_rounded_rect(nx, ny, nw_s, nh_s, node_r, fill_color)

            # Border: thicker + selected color for active or hovered nodes
            border_w = 0.5  # * ui_scale if (node.select or is_hovered) else 0.5 * ui_scale
            border_c = colors["node_selected"] if node.select else colors["node_border"]
            border_c = (*border_c[:3], border_c[3])
            if node.mute:
                border_c = (border_c[0], border_c[1], border_c[2], border_c[3] * 0.35)
            border_c = (*border_c[:3], border_c[3] * master_alpha)
            _draw_rounded_rect_border(nx, ny, nw_s, nh_s, node_r, border_c, border_w)

            # Mute overlay: dim the node with a semi-transparent backdrop
            text_alpha = 1.0
            if node.mute:
                muted_overlay = (bg_color[0], bg_color[1], bg_color[2], 0.85 * master_alpha)
                _draw_filled_rounded_rect(nx, ny, nw_s, nh_s, node_r, muted_overlay)
                text_alpha = 0.35

            # Node label rendering
            node_label_mode = getattr(settings, "node_label_mode", "COMPACT")
            if getattr(settings, "show_names", True) and nw_s > 6 * ui_scale and nh_s > 6 * ui_scale:
                label = node.label
                if not label and getattr(node, "node_tree", None):
                    label = node.node_tree.name
                if not label:
                    label = node.bl_label

                if node_label_mode == "FULL" and label:
                    # Full label: word-wrap into up to 3 centered lines
                    font_size = max(6, min(int(11 * ui_scale), int(min(nw_s, nh_s) * 0.35)))
                    text_color = _compute_outline_color(fill_color)
                    text_color = (*text_color[:3], text_color[3] * text_alpha * master_alpha)
                    lines = _get_node_label_lines(label, font_id, font_size, nw_s - 4 * ui_scale, 3)
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
                            _draw_text_with_shadow(font_id, line, lx, ly, text_color, font_size)
                            gpu.state.blend_set("ALPHA")
                else:
                    # Compact: render 1-2 uppercase initials
                    initials = _get_node_initials(label)
                    if initials:
                        font_size = max(6, min(int(11 * ui_scale), int(min(nw_s, nh_s) * 0.45)))
                        text_color = _compute_outline_color(fill_color)
                        text_color = (*text_color[:3], text_color[3] * text_alpha * master_alpha)
                        blf.size(font_id, font_size)
                        tw, th = blf.dimensions(font_id, initials)
                        tx = nx + (nw_s - tw) / 2
                        ty = ny + (nh_s - th) / 2
                        _draw_text_with_shadow(font_id, initials, tx, ty, text_color, font_size)
                        gpu.state.blend_set("ALPHA")

            # Socket indicator pills (collected by color for batched draw)
            if show_socket_indicators:
                default_color = (*colors["wire"][:3], master_alpha)
                body_top = node.location_absolute.y
                body_bot = body_top - h
                body_range = body_top - body_bot
                for is_output, sock_list in [(False, node.inputs), (True, node.outputs)]:
                    visible = [s for s in sock_list if not getattr(s, "hide", False) and getattr(s, "enabled", True)]
                    if not visible:
                        continue
                    x_base = node.location_absolute.x + (w if is_output else 0)
                    num = len(visible)
                    for idx, socket in enumerate(visible):
                        if body_range <= 0 or num <= 1:
                            sy_tree = (body_top + body_bot) * 0.5
                        else:
                            sy_tree = body_top - body_range * (idx + 1) / (num + 1)
                        sx = cx + (x_base - tree_cx) * scale
                        sy = cy + (sy_tree - tree_cy) * scale
                        if use_socket_color:
                            try:
                                socket_color = socket.draw_color(bpy.context, node)
                                color = (
                                    float(socket_color[0]),
                                    float(socket_color[1]),
                                    float(socket_color[2]),
                                    master_alpha,
                                )
                            except Exception:
                                color = default_color
                        else:
                            color = default_color
                        pills_by_color.setdefault(color, []).append((sx, sy, pw, 0.0))

    # Batched socket indicator pills (one draw call per unique color)
    if show_socket_indicators and pills_by_color:
        for color, group in pills_by_color.items():
            _batch_draw_pills(group, ph, color)


def _draw_resize_handles(
    mx: float,
    my: float,
    mw: float,
    mh: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    corner: str,
) -> None:
    """Draw full-edge resize indicators, colored orange when the percentage cap is active."""
    st = _state()
    handle = st.get("resize_active")
    if not handle:
        return

    w_clamped = st.get("width_clamped", False)
    h_clamped = st.get("height_clamped", False)

    base = (*colors["node_selected"][:3], colors["node_selected"][3] * 0.5 * master_alpha)
    warn = (*colors["indicator"][:3], colors["indicator"][3] * master_alpha)
    thick = 3.0 * ui_scale
    r = thick * 0.5

    margin = 6 * ui_scale

    match handle:
        case "W":
            wx = mx + 2 * ui_scale if corner in ("TOP_RIGHT", "BOTTOM_RIGHT") else mx + mw - 2 * ui_scale - thick
            _draw_filled_rounded_rect(wx, my + margin, thick, mh - 2 * margin, r, warn if w_clamped else base)
        case "H":
            hy = my + 2 * ui_scale if corner in ("TOP_RIGHT", "TOP_LEFT") else my + mh - 2 * ui_scale - thick
            _draw_filled_rounded_rect(mx + margin, hy, mw - 2 * margin, thick, r, warn if h_clamped else base)
        case "C":
            wx = mx + 2 * ui_scale if corner in ("TOP_RIGHT", "BOTTOM_RIGHT") else mx + mw - 2 * ui_scale - thick
            _draw_filled_rounded_rect(wx, my + margin, thick, mh - 2 * margin, r, warn if w_clamped else base)
            hy = my + 2 * ui_scale if corner in ("TOP_RIGHT", "TOP_LEFT") else my + mh - 2 * ui_scale - thick
            _draw_filled_rounded_rect(mx + margin, hy, mw - 2 * margin, thick, r, warn if h_clamped else base)


def _draw_viewport_overlay(
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
) -> None:
    """Draw the darkened overlay with a viewport hole and outline border."""
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

    hole_w = v_right - v_left
    hole_h = v_top - v_bottom

    overlay = (0.0, 0.0, 0.0, 0.5 * master_alpha)

    # Temporarily disable scissor so the overlay covers the full rounded panel edge
    scissor_overlay = scissor_active
    if scissor_overlay:
        gpu.state.scissor_test_set(False)

    if hole_w > 0 and hole_h > 0:
        # Single draw: rounded panel rect with a rectangular cutout for the viewport
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
        # Viewport entirely outside minimap; darken the full panel
        _draw_filled_rounded_rect(mx, my, mw, mh, panel_r, overlay)

    # Re-enable scissor for subsequent drawing layers
    if scissor_overlay:
        gpu.state.scissor_test_set(True)
        gpu.state.scissor_set(int(mx + 1), int(my + 1), int(mw - 2), int(mh - 2))

    # Outline the viewport extent when it overlaps the minimap
    if hole_w > 0 and hole_h > 0:
        node_r = colors.get("node_roundness", 2.0) * ui_scale
        outline_col = (*colors["node_outline"][:3], colors["node_outline"][3] * master_alpha * 0.5)
        _draw_rounded_rect_border(vx, vy, vw, vh, node_r, outline_col, 0.5 * ui_scale)


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
    text_color = (*colors["text"][:3], colors["text"][3] * 0.85 * master_alpha)

    # btn_size = FRAME_ALL_BTN_SIZE * ui_scale
    # panel_r = colors.get("panel_roundness", 4.0)
    # _draw_filled_rounded_rect(tx, ty, text_w + pad * 2, btn_size, panel_r, colors["node"])

    _draw_text_with_shadow(font_id, info_text, tx + pad, ty + pad, text_color, font_size)


def draw_minimap() -> None:
    """Main entry point — orchestrate minimap drawing in the Node Editor."""
    context = bpy.context
    space = context.space_data
    region = context.region

    #    Early exit checks
    st = _state()
    if _early_exit(context, space, st):
        return

    addon = context.preferences.addons.get(__package__)
    settings = addon.preferences.settings

    # Auto-start modal operator for pan/zoom interaction
    if getattr(settings, "interactive", True):
        if not st.get("modal_active", False):
            try:
                bpy.ops.nodemap.navigate("INVOKE_DEFAULT")
            except RuntimeError:
                pass

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

    #    Compute dimensions and layout
    with _Timer("setup"):
        ui_scale = _get_ui_scale()
        colors = _get_node_editor_theme_colors()
        master_alpha = getattr(settings, "opacity", 0.85)
        corner = getattr(settings, "position", "TOP_RIGHT")

        rect = _compute_minimap_rect(settings, ui_scale, space, region, corner, st)
        if rect is None:
            return
        mx, my, mw, mh, padding, y_margin = rect

        st["rect"] = (mx, my, mw, mh)
        st["tree_bounds"] = bounds
        st["margin"] = y_margin
        st["padding"] = padding

        _clamp_pan_to_viewport(space, region, st)

    #    Draw minimap panel
    gpu.state.blend_set("ALPHA")

    with _Timer("draw_background"):
        bg_color, panel_r = _draw_background(mx, my, mw, mh, colors, master_alpha)

    cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform()
    st["scale"] = scale

    # Clip node/wire content to minimap interior
    with _Timer("setup_scissor"):
        scissor_active = _setup_scissor(mx, my, mw, mh)

    hovered_node_name = st.get("hovered_node")
    font_id = 0

    #    Draw layers back to front
    with _Timer("classify"):
        frames = [n for n in nodes if n.type == "FRAME"]
        regular_nodes = [n for n in nodes if n.type != "FRAME"]
        if not getattr(settings, "show_wires", True):
            regular_nodes = [n for n in regular_nodes if n.type != "REROUTE"]

    # Layer 1: frame node backgrounds
    with _Timer("draw_frames"):
        _draw_frame_nodes(
            frames,
            cx,
            cy,
            scale,
            tree_cx,
            tree_cy,
            colors,
            settings,
            master_alpha,
            hovered_node_name,
            ui_scale,
            font_id,
        )

    # Layer 2: connection wires
    if getattr(settings, "show_wires", True):
        with _Timer("draw_wires"):
            _draw_wires(
                nodes,
                tree_cx,
                tree_cy,
                scale,
                cx,
                cy,
                colors,
                master_alpha,
                getattr(settings, "show_wire_color", True),
            )

    # Layer 3: regular (non-frame) nodes
    with _Timer("draw_nodes"):
        _draw_regular_nodes(
            regular_nodes,
            cx,
            cy,
            scale,
            tree_cx,
            tree_cy,
            colors,
            settings,
            master_alpha,
            hovered_node_name,
            ui_scale,
            bg_color,
            font_id,
            show_socket_indicators=getattr(settings, "show_socket_indicators", False),
            use_socket_color=getattr(settings, "show_wire_color", True),
        )

    # Layer 4: viewport extent overlay with cutout hole
    with _Timer("draw_viewport"):
        _draw_viewport_overlay(
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
        )

    # Layer 5: scrollbar thumbs when zoomed past tree bounds
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

    # Frame-all button in the top-left corner (only when scrollbars visible)
    _draw_frame_all_button(
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

    # Resize handle indicators
    with _Timer("draw_resize_handles"):
        _draw_resize_handles(mx, my, mw, mh, colors, master_alpha, ui_scale, corner)

    _teardown_scissor(scissor_active)
    gpu.state.blend_set("NONE")

    # Overlay: node count text
    with _Timer("draw_node_count"):
        content_nodes = [n for n in nodes if n.type not in ("FRAME", "REROUTE")]
        _draw_node_count(settings, content_nodes, mx, my, mw, colors, master_alpha, ui_scale, font_id)

    # Stop & dump profile stats after N frames
    _maybe_stop_profiler(st)


def _get_socket_pos(node, socket, is_output):
    """Return (x, y) in tree coordinates for a socket position.

    Computes position from the node's layout geometry rather than relying
    on the (unexposed) socket.location API.
    """
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

    # header_h = 28
    top_y = node.location_absolute.y
    body_top = top_y  # - header_h
    body_bot = top_y - h  # + 8

    num = len(visible)
    body_range = body_top - body_bot
    if body_range <= 0 or num <= 1:
        y = (body_top + body_bot) * 0.5
    else:
        y = body_top - body_range * (idx + 1) / (num + 1)

    return x, y


def _draw_wires(nodes, tree_cx, tree_cy, scale, cx, cy, colors, master_alpha=1.0, use_socket_color=False):
    """Draw connection lines between nodes as pill-shaped wires.

    Uses actual socket positions for endpoints.  Wires are grouped by
    color and drawn in a single GPU batch per color group.
    """

    master_alpha = master_alpha * 1.0
    default_wire_color = (*colors["wire"][:3], colors["wire"][3] * master_alpha)
    thickness = max(1.0, 2.0 * scale)

    # Precompute socket positions per node (O(N) per side, no redundant index())
    out_pos: dict[str, dict[str, tuple[float, float]]] = {}
    in_pos: dict[str, dict[str, tuple[float, float]]] = {}
    for node in nodes:
        if node.type == "FRAME":
            continue
        w, h = _get_node_dims(node)
        if node.type == "REROUTE":
            cx_n = node.location_absolute.x + w / 2
            cy_n = node.location_absolute.y - h / 2
            out_pos[node.name] = {s.identifier: (cx_n, cy_n) for s in node.outputs}
            in_pos[node.name] = {s.identifier: (cx_n, cy_n) for s in node.inputs}
        else:
            body_top = node.location_absolute.y
            body_bot = body_top - h
            body_range = body_top - body_bot
            visible_outs = [s for s in node.outputs if not getattr(s, "hide", False) and getattr(s, "enabled", True)]
            if visible_outs:
                x_base = node.location_absolute.x + w
                num = len(visible_outs)
                out_dict: dict[str, tuple[float, float]] = {}
                for idx, sock in enumerate(visible_outs):
                    if body_range <= 0 or num <= 1:
                        sy = (body_top + body_bot) * 0.5
                    else:
                        sy = body_top - body_range * (idx + 1) / (num + 1)
                    out_dict[sock.identifier] = (x_base, sy)
                out_pos[node.name] = out_dict
            visible_ins = [s for s in node.inputs if not getattr(s, "hide", False) and getattr(s, "enabled", True)]
            if visible_ins:
                x_base = node.location_absolute.x
                num = len(visible_ins)
                in_dict: dict[str, tuple[float, float]] = {}
                for idx, sock in enumerate(visible_ins):
                    if body_range <= 0 or num <= 1:
                        sy = (body_top + body_bot) * 0.5
                    else:
                        sy = body_top - body_range * (idx + 1) / (num + 1)
                    in_dict[sock.identifier] = (x_base, sy)
                in_pos[node.name] = in_dict

    # Collect wires grouped by color for batched drawing
    wires_by_color: dict[tuple[float, ...], list[tuple[float, float, float, float]]] = {}

    for node in nodes:
        if node.type == "FRAME" or not getattr(node, "outputs", None):
            continue
        node_out_pos = out_pos.get(node.name)
        if not node_out_pos:
            continue

        for output in node.outputs:
            if not getattr(output, "is_linked", False) or not getattr(output, "links", None):
                continue
            out_pos_tuple = node_out_pos.get(output.identifier)
            if not out_pos_tuple:
                continue
            out_x, out_y = out_pos_tuple

            for link in output.links:
                to_node = link.to_node
                if not to_node or to_node.name not in nodes.keys() or to_node.type == "FRAME":
                    continue
                node_in_pos = in_pos.get(to_node.name)
                if not node_in_pos:
                    continue
                in_pos_tuple = node_in_pos.get(link.to_socket.identifier)
                if not in_pos_tuple:
                    continue
                in_x, in_y = in_pos_tuple

                x1 = cx + (out_x - tree_cx) * scale
                y1 = cy + (out_y - tree_cy) * scale
                x2 = cx + (in_x - tree_cx) * scale
                y2 = cy + (in_y - tree_cy) * scale

                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                if length < 0.5:
                    continue

                if use_socket_color:
                    try:
                        socket_color = output.draw_color(bpy.context, node)
                        wire_color = (
                            float(socket_color[0]),
                            float(socket_color[1]),
                            float(socket_color[2]),
                            master_alpha,
                        )
                    except Exception:
                        wire_color = default_wire_color
                else:
                    wire_color = default_wire_color

                angle = math.atan2(dy, dx)
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2

                wires_by_color.setdefault(wire_color, []).append((mx, my, length, angle))

    # Draw one batch per unique color
    for wire_color, group in wires_by_color.items():
        _batch_draw_pills(group, thickness, wire_color)


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
    scroll_color = (*colors["scroll_item"][:3], colors["scroll_item"][3] * master_alpha * 0.5)

    # Horizontal scrollbar (bottom edge)
    if visible_w < bbox_w:
        track_w = inner_w
        thumb_w = max(min_thumb, int(track_w * visible_w / bbox_w))
        thumb_x = inner_l + int(track_w * (v_left - bbox_l) / bbox_w)
        thumb_y = my + bar_off
        _draw_filled_rounded_rect(thumb_x, thumb_y, thumb_w, bar_thick, bar_thick * 0.5, scroll_color)

    # Vertical scrollbar (right edge)
    if visible_h < bbox_h:
        track_h = inner_h
        thumb_h = max(min_thumb, int(track_h * visible_h / bbox_h))
        thumb_x2 = mx + mw - bar_off - bar_thick
        thumb_y2 = inner_b + int(track_h * (v_bottom - bbox_b) / bbox_h)
        _draw_filled_rounded_rect(thumb_x2, thumb_y2, bar_thick, thumb_h, bar_thick * 0.5, scroll_color)


def _draw_frame_all_button(
    mx, my, mw, mh, padding, cx, cy, scale, tree_cx, tree_cy, bounds, colors, ui_scale, master_alpha
):
    """Draw a frame-all button at the top-left of the minimap inner area when scrollbars are visible."""
    addon = bpy.context.preferences.addons.get(__package__)
    settings = getattr(addon.preferences, "settings", None) if addon else None
    if not settings or not getattr(settings, "show_frame_all_btn", True):
        _state()["frame_all_btn"] = None
        return
    inner_l = mx + padding
    # inner_r = mx + mw - padding
    # inner_b = my + padding
    inner_t = my + mh - padding

    bbox_l, bbox_b, bbox_r, bbox_t = bounds
    bbox_w = bbox_r - bbox_l
    bbox_h = bbox_t - bbox_b
    if bbox_w <= 0 or bbox_h <= 0:
        return

    # tree_l = tree_cx + (inner_l - cx) / scale
    # tree_r = tree_cx + (inner_r - cx) / scale
    # tree_b = tree_cy + (inner_b - cy) / scale
    # tree_t = tree_cy + (inner_t - cy) / scale

    # v_left = max(bbox_l, min(bbox_r, tree_l))
    # v_right = max(bbox_l, min(bbox_r, tree_r))
    # v_bottom = max(bbox_b, min(bbox_t, tree_b))
    # v_top = max(bbox_b, min(bbox_t, tree_t))

    # visible_w = v_right - v_left
    # visible_h = v_top - v_bottom

    st = _state()
    # if visible_w >= bbox_w and visible_h >= bbox_h:
    #     st["frame_all_btn"] = None
    #     return

    btn_size = FRAME_ALL_BTN_SIZE * ui_scale
    margin = FRAME_ALL_BTN_MARGIN * ui_scale
    x = inner_l + margin
    y = inner_t - btn_size - margin
    # r = colors.get("panel_roundness", 4.0)
    # btn_color = (*colors["bg"][:3], colors["bg"][3] * master_alpha * 0.3)
    ico_color = (*colors["text"][:3], colors["text"][3] * master_alpha * 0.7)
    # _draw_filled_rounded_rect(x, y, btn_size, btn_size, 4, btn_color)

    # Corner brackets icon (four brackets pointing outward)
    i = 2 * ui_scale
    t = max(1, int(1.5 * ui_scale))
    arm = btn_size * 0.3

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

    st["frame_all_btn"] = (x, y, btn_size, btn_size)


def _get_node_initials(name: str) -> str:
    """Extract 1-2 uppercase initials from a node label."""
    name = name.strip()
    if not name:
        return "?"
    words = name.split()
    if len(words) >= 2:
        return "".join(w[0] for w in words).upper()[:2]
    word = words[0]
    for i, ch in enumerate(word):
        if ch.isalpha():
            return word[:i] + ch.upper()
    return word[0].upper()


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
