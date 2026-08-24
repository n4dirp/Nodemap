"""Minimap rendering in the Node Editor."""

import io
import logging
import math
import time

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

try:
    import cProfile

    _HAS_C_PROFILE = True
except ImportError:
    _HAS_C_PROFILE = False

from .gpu_draw import (
    _build_pill_batch,
    _draw_filled_rounded_rect,
    _draw_filled_rounded_rect_clipped,
    _draw_filled_rounded_rect_with_hole,
    _draw_pill,
    _draw_pill_border,
    _draw_rounded_rect_border,
    _draw_text_with_shadow,
    _get_batch_pill_shader,
    _get_batch_rect_border_shader,
    _get_batch_rect_shader,
)
from .helpers import (
    _COLOR_TAG_TO_THEME_ATTR,
    _HANDLE_THICKNESS,
    _LIST_COUNT_GAP,
    _LIST_PAD_X,
    _LIST_SWATCH,
    _LIST_SWATCH_GAP,
    _MINIMAP_BUTTONS,
    MIN_MAP_HEIGHT,
    MIN_MAP_WIDTH,
    STATS_FONT_ID,
    STATS_FONT_SIZE,
    MinimapState,
    _alpha_mul,
    _clamp_pan_to_viewport,
    _compute_outline_color,
    _expand_bounds_margin,
    _get_map_content_rect,
    _get_minimap_margins,
    _get_minimap_transform,
    _get_node_dims,
    _get_node_editor_theme_colors,
    _get_node_initials,
    _get_node_label_lines,
    _get_safe_bounds,
    _get_tree_snapshot,
    _get_type_list_width,
    _get_ui_scale,
    _get_visible_rect,
    _minimap_window_operators,
    _registration_state,
    _schedule_list_anim_redraw,
    _srgb_to_linear,
    _state,
    _theme_rgba,
    get_tree_fingerprint,
)
from .preferences import TRACE_LEVEL

logger = logging.getLogger(__package__)

# Variables
FONT_SIZE = 11
FRAME_ALL_BTN_SIZE = 20
FRAME_ALL_BTN_MARGIN = 1
FRAME_BTN_GAP = 2
_MIN_SOCKET_SCALE = 0.15

BTN_HOVER_ALPHA = 0.015

_SCROLLBAR_THICKNESS = 3.0
_SCROLLBAR_THICKNESS_HOVER = 6.0
_SCROLLBAR_INSET = 2.0
_SCROLLBAR_MIN_THUMB = 6.0
_SCROLLBAR_ALPHA = 0.65

_TYPE_LIST_ANIM_AWAIT_TIMEOUT = 1.0
# Minimum label width (device-independent px); below it the count column is
# dropped so type names keep the full row instead of clipping.
_TYPE_LIST_MIN_LABEL_W = 32.0

_NODE_ROUNDNESS_DEFAULT = 2.0

# Rebuild cached batches when the map scale drifts this far from the baked
# scale (relative); only radius/thickness/font buckets depend on it.
_SCALE_REBUILD_REL = 0.002
# Force a batch rebuild when the per-frame anchor drifts this far from the
# bake-time anchor; bounds how stale rect culling may become (px).
_BATCH_DRIFT_PX = 256.0
_CULL_MARGIN_PX = _BATCH_DRIFT_PX + 32.0
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
    mx = max(mx, float(sx) + x_margin)
    mw = min(mw, float(ex) - mx - x_margin)
    if corner in ("TOP_RIGHT", "TOP_LEFT"):
        # Top corners: bottom is limited by margin (margin_bottom), top is limited by y_margin
        my = max(my, float(sy) + margin)
        mh = min(mh, float(ey) - my - y_margin)
    else:
        # Bottom corners: bottom is limited by y_margin, top is limited by margin
        my = max(my, float(sy) + y_margin)
        mh = min(mh, float(ey) - my - margin)

    # Only bail if the minimap would be too small to be useful
    min_dim_w = MIN_MAP_WIDTH * ui_scale
    min_dim_h = MIN_MAP_HEIGHT * ui_scale
    if mw < min_dim_w or mh < min_dim_h:
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


def _draw_view_fill(
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
    panel_r: float,
    master_alpha: float,
    ui_scale: float,
    visible: tuple[float, float, float, float] | None = None,
) -> None:
    """Draw a filled rect over the active view region, behind nodes and wires."""
    if not getattr(settings, "viewport_fill_rect", False):
        return
    if visible is None:
        visible = _get_visible_rect(space, region)
    if not visible:
        return

    vx = round(cx + (visible[0] - tree_cx) * scale)
    vy = round(cy + (visible[1] - tree_cy) * scale)
    vw = round(max((visible[2] - visible[0]) * scale, 1.0))
    vh = round(max((visible[3] - visible[1]) * scale, 1.0))

    v_left = max(vx, mx)
    # st_fill = _state()
    # if st_fill.list_width > 0:
    #     # Keep the fill out of the type-list zone.
    #     v_left = max(v_left, _get_map_content_rect(st_fill)[0])
    v_bottom = max(vy, my)
    v_right = min(vx + vw, mx + mw)
    v_top = min(vy + vh, my + mh)
    hole_w = v_right - v_left
    hole_h = v_top - v_bottom
    if hole_w <= 0 or hole_h <= 0:
        return

    fill_color = getattr(settings, "viewport_fill_color", (0.28, 0.45, 0.7, 1.0))
    fill = _alpha_mul(fill_color, master_alpha)
    node_r = colors.get("node_roundness", 2.0) * ui_scale
    _draw_filled_rounded_rect_clipped(v_left, v_bottom, hole_w, hole_h, node_r, fill, mx, my, mw, mh, panel_r * 1.2)


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
    visible: tuple[float, float, float, float] | None = None,
) -> None:
    """Draw the viewport rect outline and optional darkened overlay."""
    if visible is None:
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
    node_count: int,
    mx: float,
    my: float,
    mw: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> None:
    """Draw the node count text centered below the minimap."""
    if not getattr(settings, "show_node_count", True):
        return

    info_text = str(node_count)
    font_id = 0
    font_size = int(FONT_SIZE * ui_scale)
    blf.size(font_id, font_size)
    text_w, _ = blf.dimensions(font_id, info_text)

    pad = 1
    tx = mx + (mw - text_w) - 10 * ui_scale
    ty = my + (FONT_SIZE * ui_scale) - 3

    st = _state()
    btn_bottoms = [btn[1] for btn in (st.frame_all_btn, st.frame_view_btn, st.frame_selected_btn) if btn]
    if btn_bottoms and min(btn_bottoms) <= ty + font_size:
        return

    text_color = _alpha_mul(colors["text"], 0.85 * master_alpha)

    _draw_text_with_shadow(font_id, info_text, tx + pad, ty + pad, text_color, font_size)


def _step_list_width(st: MinimapState, settings, mw: float, ui_scale: float) -> None:
    """Advance the animated type-list zone width for this frame.

    Snaps directly when no animation is running; otherwise eases from the
    recorded start width toward the locked target. An expansion waits (up to
    a timeout) for the pending compile to expose measurable type stats
    before starting its clock.
    """
    list_font_size = getattr(settings, "type_list_font_size", STATS_FONT_SIZE)
    target_now = _get_type_list_width(settings, st, mw, ui_scale, list_font_size)
    if not st.list_anim_active:
        st.list_width = target_now
        return

    if st.list_anim_target < 0:
        if target_now > 0:
            st.list_anim_target = target_now
            st.list_anim_start = time.perf_counter()
        elif time.perf_counter() - st.list_anim_start > _TYPE_LIST_ANIM_AWAIT_TIMEOUT:
            st.list_anim_active = False
            st.list_width = target_now
            return
        else:
            st.list_width = st.list_anim_from
            _schedule_list_anim_redraw(st)
            return

    progress = min((time.perf_counter() - st.list_anim_start) / max(st.list_anim_duration, 1e-4), 1.0)
    eased = 1.0 - (1.0 - progress) ** 3
    st.list_width = st.list_anim_from + (st.list_anim_target - st.list_anim_from) * eased
    if progress >= 1.0:
        st.list_width = st.list_anim_target
        st.list_anim_active = False
    else:
        _schedule_list_anim_redraw(st)


def _type_list_cache_key(st: MinimapState, settings, colors: dict, master_alpha: float, ui_scale: float) -> tuple:
    """Return the invalidation key for the cached type-list layout and swatch batch."""
    # The color-tag palette feeds type swatch colors at compile time, so track
    # it here to catch theme edits that do not change the tree fingerprint.
    palette = tuple(
        _theme_rgba(f"node_editor.{attr}", colors["node"])[:3] for attr in _COLOR_TAG_TO_THEME_ATTR.values()
    )
    return (
        st.tree_data_version,
        getattr(settings, "type_list_sort", "COUNT"),
        getattr(settings, "colored_nodes", True),
        frozenset(st.list_expanded),
        ui_scale,
        master_alpha,
        tuple(colors["node"]),
        palette,
    )


def _build_type_list_cache(
    st: MinimapState, settings, key: tuple, colors: dict, master_alpha: float, ui_scale: float
) -> None:
    """Sort type entries once and bake layout metrics plus all swatch pills.

    Swatch vertices are stored in list-local space — x relative to the content
    origin, y downward from the top row at scroll=0 — so scrolling and width
    animation only move the matrix translate, never rebuild the batch.
    """
    tree_data = st.tree_data or {}
    type_stats = tree_data.get("type_stats") or {}

    font_id = STATS_FONT_ID
    font_size = int(getattr(settings, "type_list_font_size", STATS_FONT_SIZE) * ui_scale)
    blf.size(font_id, font_size)

    if getattr(settings, "type_list_sort", "COUNT") == "NAME":
        items = sorted(type_stats.items(), key=lambda kv: kv[0].lower())
    else:
        items = sorted(type_stats.items(), key=lambda kv: (-kv[1], kv[0]))

    entries: list[tuple[str, str, float, int]] = []
    widest_count = 0.0
    for label, count in items:
        count_text = str(count)
        count_w = blf.dimensions(font_id, count_text)[0]
        widest_count = max(widest_count, count_w)
        entries.append((label, count_text, count_w, count))

    _, line_h = blf.dimensions(font_id, "Ay")
    row_h = line_h + 4 * ui_scale

    # Icons (swatch / +/− / child circle) are drawn live in _draw_type_list so
    # selection and active state can recolor them per frame; no baked batch.
    st.list_cache_key = key
    st.cached_list_entries = entries
    st.cached_list_children = tree_data.get("type_nodes") or {}
    st.cached_list_layout = {"font_size": font_size, "line_h": line_h, "row_h": row_h, "widest_count": widest_count}
    st.cached_list_swatches_batch = None


def _iter_type_list_layout(
    entries: list[tuple[str, str, float, int]],
    children: dict[str, list[str]],
    expanded: set,
    row_h: float,
):
    """Yield ``(kind, label, node_name, local_y_top)`` for each visible list row.

    ``kind`` is ``"header"`` for a type row or ``"child"`` for an individual
    node row. ``local_y_top`` is the list-local top coordinate (0 at the top,
    negative downward) so the same model drives the baked swatch batch and the
    per-frame text/hit-test layout. Only type groups with count > 1 that are
    present in *expanded* emit child rows.
    """
    y = 0.0
    for label, _count_text, _count_w, count in entries:
        yield ("header", label, None, y)
        y -= row_h
        if count > 1 and label in expanded:
            for node_name in children.get(label, ()):
                yield ("child", label, node_name, y)
                y -= row_h


def _draw_type_list(
    settings,
    st: MinimapState,
    mx: float,
    my: float,
    mh: float,
    padding: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> None:
    """Draw the interactive node-type list zone along the minimap's left edge."""
    st.list_row_rects = []
    st.list_node_rects = []
    st.list_toggle_rects = {}
    st.list_scroll_max = 0.0
    # Drawn whenever the zone has width, including while it animates shut,
    # so the content slides out with the panel instead of vanishing.
    if st.list_width <= 0:
        st.hovered_type_label = None
        st.hovered_list_node = None
        st.hovered_list_scrollbar = False
        st.list_scrollbar_thumb = None
        st.list_scrollbar_track = None
        st.list_zone_rect = None
        return
    tree_data = st.tree_data
    type_stats = tree_data.get("type_stats") if tree_data else None
    if not type_stats:
        st.hovered_type_label = None
        st.hovered_list_node = None
        st.hovered_list_scrollbar = False
        st.list_scrollbar_thumb = None
        st.list_scrollbar_track = None
        st.list_zone_rect = None
        return

    key = _type_list_cache_key(st, settings, colors, master_alpha, ui_scale)
    if key != st.list_cache_key or not st.cached_list_layout:
        with _Timer("type_list_cache_build"):
            _build_type_list_cache(st, settings, key, colors, master_alpha, ui_scale)
    entries = st.cached_list_entries or []
    layout = st.cached_list_layout or {}
    font_size = layout.get("font_size", int(getattr(settings, "type_list_font_size", STATS_FONT_SIZE) * ui_scale))
    row_h = layout.get("row_h", 16.0)
    line_h = layout.get("line_h", 12.0)
    widest_count = layout.get("widest_count", 0.0)
    st.list_row_h = row_h

    pad_x = _LIST_PAD_X * ui_scale
    swatch = _LIST_SWATCH * ui_scale
    swatch_gap = _LIST_SWATCH_GAP * ui_scale
    count_gap = _LIST_COUNT_GAP * ui_scale

    # Compute total rows height so the background can shrink-wrap when the
    # list is shorter than the minimap.
    row_pad_v = 3 * ui_scale
    _children = st.cached_list_children or {}
    total_h = 0.0
    for _ in _iter_type_list_layout(entries, _children, st.list_expanded, row_h):
        total_h += row_h

    # Zone geometry: inset by the resize-handle thickness so edge resize
    # borders stay reachable around the list
    handle_pad = _HANDLE_THICKNESS * ui_scale
    zone_x = mx + handle_pad
    zone_w = mx + padding + st.list_width - 2 * ui_scale - zone_x
    zone_h = min(mh - 2 * handle_pad, total_h + 2 * row_pad_v)
    zone_y = my + mh - zone_h - handle_pad
    st.list_zone_rect = (zone_x, zone_y, zone_w, zone_h)

    zone_r = colors.get("panel_roundness", 4.0) * 0.6
    _draw_filled_rounded_rect(zone_x, zone_y, zone_w, zone_h, zone_r, _alpha_mul(colors["bg"], master_alpha))
    _draw_rounded_rect_border(
        zone_x, zone_y, zone_w, zone_h, zone_r, _alpha_mul(colors["bg_border"], master_alpha), 0.5
    )

    # Scrollable rows viewport inside the zone
    view_t = zone_y + zone_h - row_pad_v
    view_b = zone_y + row_pad_v
    view_h = max(view_t - view_b, row_h)
    scroll_max = max(0.0, total_h - view_h)
    st.list_scroll = min(max(st.list_scroll, 0.0), scroll_max)
    st.list_scroll_max = scroll_max

    show_swatch = getattr(settings, "colored_nodes", True)

    content_x = zone_x + pad_x
    # Static extra margin so counts stay clear of the expanded scrollbar.
    count_right = zone_x + zone_w - pad_x - 4 * ui_scale
    # Always reserve the swatch/toggle column so multi-node + icons never
    # overlap the type label, even when colored nodes are disabled.
    label_x = content_x + (swatch + swatch_gap)
    # Hide counts when reserving them would squeeze the label below the
    # minimum; names then reclaim the full row width.
    show_counts = count_right - widest_count - count_gap - label_x >= _TYPE_LIST_MIN_LABEL_W * ui_scale
    label_max_w = max(0.0, (count_right - widest_count - count_gap if show_counts else count_right) - label_x)
    text_y_off = (row_h - line_h) / 2

    text_col = _alpha_mul(colors["text"], 0.65 * master_alpha)
    count_col = _alpha_mul(colors["text"], 0.3 * master_alpha)
    sel_col = _alpha_mul(colors["node_selected"], 0.95 * master_alpha)
    active_col = _alpha_mul(colors["node_active"], master_alpha)

    # Per-type selection state (compiled) drives font (not icon) recoloring.
    type_colors = tree_data.get("type_colors") or {}
    type_selected = tree_data.get("type_selected_counts") or {}
    type_active = tree_data.get("type_active_label")
    hover_col = _alpha_mul(colors["text"], 0.02 * master_alpha)

    pill_x = zone_x + 2 * ui_scale
    pill_w = zone_w - 4 * ui_scale

    # Clip rows to the zone interior so partial rows never bleed onto the map
    saved_scissor = None
    try:
        was_active = gpu.state.scissor_test_get()
        saved_scissor = (was_active, gpu.state.scissor_get() if was_active else None)
    except Exception:
        saved_scissor = None
    try:
        gpu.state.scissor_set(int(zone_x + 1), int(zone_y + 1), max(0, int(zone_w - 2)), max(0, int(zone_h - 2)))
        gpu.state.scissor_test_set(True)
        gpu.state.blend_set("ALPHA")

        entry_map = {lbl: (ct, cw, cnt) for (lbl, ct, cw, cnt) in entries}
        expanded = st.list_expanded
        hovered = st.hovered_type_label
        hovered_child = st.hovered_list_node

        # Walk the shared layout model; cull rows outside the viewport.
        visible_rows = []
        for row_idx, (kind, label, node_name, local_y) in enumerate(
            _iter_type_list_layout(entries, _children, expanded, row_h)
        ):
            s_top = view_t + st.list_scroll + local_y
            s_bottom = s_top - row_h
            if s_top <= view_b or s_bottom >= view_t:
                continue
            visible_rows.append((kind, label, node_name, s_top, s_bottom, row_idx))

        # Zebra bands keyed on the absolute layout index so they stay attached
        # to rows while scrolling; drawn beneath pills, text, and icons.
        band_col = (0.0, 0.0, 0.0, 0.15 * master_alpha)
        for _kind, _label, _node_name, s_top, s_bottom, row_idx in visible_rows:
            if row_idx % 2 == 1:
                _draw_filled_rounded_rect(pill_x, s_bottom, pill_w, row_h, 0.0, band_col)

        # Header hover pills + hit rects (rows + expand toggle slots)
        header_rects = []
        toggle_rects = {}
        for kind, label, _node_name, _s_top, s_bottom, _row_idx in visible_rows:
            if kind != "header":
                continue
            if hovered == label:
                _draw_filled_rounded_rect(pill_x, s_bottom, pill_w, row_h, 4.0 * ui_scale, hover_col)
            if label == type_active:
                # Active outline drawn here (pre-text) so BLF cannot clobber the
                # alpha blend state the SDF fill relies on.
                _draw_rounded_rect_border(pill_x, s_bottom, pill_w, row_h, 4.0 * ui_scale, sel_col, 0.5 * ui_scale)
            header_rects.append((pill_x, s_bottom, pill_w, row_h, label))
            if entry_map.get(label, ("", 0.0, 1))[2] > 1:
                toggle_rects[label] = (content_x, s_bottom, swatch + swatch_gap, row_h)
        st.list_row_rects = header_rects
        st.list_toggle_rects = toggle_rects

        # Active child hit-test lookups run up here so the outline can be drawn
        # in this pre-text pass (BLF disables alpha blending after glyph draws).
        node_tree = bpy.context.space_data.edit_tree if bpy.context.space_data else None
        active_node = node_tree.nodes.active if node_tree else None

        # Child hover pills + hit rects
        child_rects = []
        for kind, label, node_name, _s_top, s_bottom, _row_idx in visible_rows:
            if kind != "child":
                continue
            if hovered_child == (label, node_name):
                _draw_filled_rounded_rect(pill_x, s_bottom, pill_w, row_h, 4.0 * ui_scale, hover_col)
            # else:
            child_active = False
            try:
                node = node_tree.nodes.get(node_name) if node_tree else None
                child_active = bool(active_node and node == active_node)
            except Exception:
                node = None
            if child_active:
                _draw_rounded_rect_border(pill_x, s_bottom, pill_w, row_h, 4.0 * ui_scale, sel_col, 0.5 * ui_scale)
            child_rects.append((pill_x, s_bottom, pill_w, row_h, label, node_name))
        st.list_node_rects = child_rects

        # Hoisted BLF state (size, shadow, clip box); calling the shared text
        # helper per row would redo this setup twice per visible row.
        font_id = STATS_FONT_ID
        blf.size(font_id, font_size)
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 3, 0, 0, 0, 255)
        blf.shadow_offset(font_id, 0, -1)

        child_indent = swatch + swatch_gap
        child_label_x = label_x + child_indent
        # Child rows never draw a count, so their names always run to the
        # right edge instead of reserving the header count column.
        child_label_max_w = max(0.0, count_right - child_label_x)
        child_clip_l = int(child_label_x)
        child_clip_r = int(child_label_x + child_label_max_w)
        clip_t = int(zone_y - row_h)
        clip_b = int(zone_y + zone_h + row_h)

        child_swatch_x = content_x + child_indent

        for kind, label, node_name, _s_top, s_bottom, _row_idx in visible_rows:
            text_y = s_bottom + text_y_off
            if kind == "header":
                # Icons keep the type color; selection/active state shows in the
                # row text color instead (active brightest, then selected).
                is_active = label == type_active
                is_sel = type_selected.get(label, 0) > 0
                if is_active:
                    label_col = active_col
                elif is_sel:
                    label_col = sel_col
                else:
                    label_col = text_col

                blf.clipping(font_id, int(label_x), clip_t, int(label_x + label_max_w), clip_b)
                blf.enable(font_id, blf.CLIPPING)
                blf.position(font_id, label_x, text_y, 0)
                blf.color(font_id, *label_col)
                blf.draw(font_id, label)
                # Counts sit right of the label clip box; BLF discards glyphs
                # past the box instead of clipping them.
                if show_counts:
                    blf.disable(font_id, blf.CLIPPING)
                    count_text, count_w, _cnt = entry_map.get(label, ("", 0.0, 1))
                    blf.position(font_id, count_right - count_w, text_y, 0)
                    blf.color(font_id, *count_col)
                    blf.draw(font_id, count_text)
                    blf.enable(font_id, blf.CLIPPING)

                # Icon sits over the text; restore alpha blending after BLF.
                gpu.state.blend_set("ALPHA")

                # Icon in the swatch slot stays the type color for every state:
                # +/− toggle for multi-node types, colored swatch for single-node.
                icon_color = type_colors.get(label, colors["node"])
                icon_rgba = _alpha_mul(icon_color, master_alpha)
                if entry_map.get(label, ("", 0.0, 1))[2] > 1:
                    _paint_expand_icon(
                        content_x + swatch / 2, s_bottom + row_h / 2, swatch, icon_rgba, ui_scale, label in expanded
                    )
                elif show_swatch:
                    _draw_filled_rounded_rect(
                        content_x, s_bottom + (row_h - swatch) / 2, swatch, swatch, swatch / 2, icon_rgba
                    )
            else:
                # Child row: icon is the type color; text shows selection state.
                is_active = False
                is_sel = False
                try:
                    node = node_tree.nodes.get(node_name) if node_tree else None
                    is_active = bool(active_node and node == active_node)
                    is_sel = bool(node and node.select)
                except Exception:
                    node = None
                if is_active:
                    label_col = active_col
                elif is_sel:
                    label_col = sel_col
                else:
                    label_col = text_col

                blf.clipping(font_id, child_clip_l, clip_t, child_clip_r, clip_b)
                blf.enable(font_id, blf.CLIPPING)
                # Child icon matches the normal item's swatch (same size/shape).
                # Restore alpha blending after the row's text draw.
                gpu.state.blend_set("ALPHA")
                if show_swatch:
                    _draw_filled_rounded_rect(
                        child_swatch_x,
                        s_bottom + (row_h - swatch) / 2,
                        swatch,
                        swatch,
                        swatch / 2,
                        _alpha_mul(type_colors.get(label, colors["node"]), master_alpha),
                    )
                blf.position(font_id, child_label_x, text_y, 0)
                blf.color(font_id, *label_col)
                blf.draw(font_id, node_name)
                blf.enable(font_id, blf.CLIPPING)
        blf.disable(font_id, blf.CLIPPING)
        blf.disable(font_id, blf.SHADOW)
    finally:
        try:
            was_active, old_rect = saved_scissor or (False, None)
            if was_active and old_rect:
                gpu.state.scissor_set(*old_rect)
                gpu.state.scissor_test_set(True)
            else:
                gpu.state.scissor_set(0, 0, 65535, 65535)
                gpu.state.scissor_test_set(False)
        except Exception:
            pass

    # Scrollbar thumb when the list overflows (same style as the map scrollbars)
    st.list_scrollbar_thumb = None
    st.list_scrollbar_track = None
    if scroll_max > 0:
        gpu.state.blend_set("ALPHA")
        _bar_thick, bar_off = _get_scrollbar_style(ui_scale)
        frac = st.list_scroll / scroll_max
        # Hover and drag share one expanded look (Blender overlay style).
        active = st.hovered_list_scrollbar or st.list_scroll_dragging
        thick = _scrollbar_thickness(ui_scale, active)
        thumb_rect, track_rect = _draw_scrollbar_thumb(
            zone_x + zone_w - thick - bar_off,
            zone_y + bar_off,
            zone_h - 2 * bar_off,
            view_h / total_h,
            1.0 - frac,
            colors,
            master_alpha,
            ui_scale,
            active=active,
        )
        st.list_scrollbar_thumb = thumb_rect
        st.list_scrollbar_track = track_rect


def _create_quad_indices(n: int) -> list[tuple[int, int, int]]:
    """Helper to populate triangular indices sequentially for quad batches."""
    indices = []
    for i in range(n):
        base = i * 4
        indices.append((base, base + 1, base + 2))
        indices.append((base + 2, base + 3, base))
    return indices


def _is_move_only_diff(old: tuple | None, current: tuple) -> bool:
    """True when two fingerprints differ only in the position-sum slot."""
    return old is not None and len(old) == len(current) and old[:1] == current[:1] and old[2:] == current[2:]


def _debounced_compile(st: MinimapState, node_tree, colors, settings, master_alpha, ui_scale):
    """Timer callback: compile tree data after fingerprint settles, then force redraw.

    When ``st.pending_settle_flush`` is set (drag position refreshes happened),
    an unchanged fingerprint only needs the tree-data generation bumped so
    frozen wire/marker batches snap to the already-patched positions. A
    position-only diff is patched incrementally; anything else recompiles.
    """
    include_selection = getattr(settings, "show_node_borders", True)
    current_fingerprint = get_tree_fingerprint(node_tree, include_selection=include_selection)
    old_fingerprint = st.cached_fingerprint
    unchanged = old_fingerprint == current_fingerprint
    trace = logger.isEnabledFor(TRACE_LEVEL)
    if unchanged and not st.pending_settle_flush:
        st.pending_timer = None
        st.pending_timer_deadline = 0.0
        st.pending_fingerprint = None
        if trace:
            logger.trace("SETTLE skip: fingerprint unchanged, nothing pending")
        return None
    applied = False
    path = "compile"
    if unchanged and st.tree_data:
        # Positions were fully patched by _apply_move_updates; rebaking the
        # frozen wire/marker generation skips the full recompile.
        st.tree_data_version += 1
        applied = True
        path = "settle_bump"
    elif _is_move_only_diff(old_fingerprint, current_fingerprint) and st.tree_data:
        with _Timer("move_update"):
            applied = _apply_move_updates(st, node_tree)
        if applied:
            st.cached_fingerprint = current_fingerprint
            # Movement settled: unfreeze wire/marker batches so they snap to
            # the patched positions without a full recompile.
            st.tree_data_version += 1
            path = "move_patch"
    if not applied:
        with _Timer("compile_tree"):
            _compile_tree_data(st, node_tree, colors, settings, master_alpha, ui_scale)
            st.cached_fingerprint = current_fingerprint
    st.pending_timer = None
    st.pending_timer_deadline = 0.0
    st.pending_fingerprint = None
    st.pending_settle_flush = False
    if trace:
        logger.trace("SETTLE %s", path)
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
    are NOT applied here — content batches are baked in map-local space
    by ``_ensure_minimap_batches()`` and placed with a matrix transform.

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
    show_type_list = getattr(settings, "show_type_list", False)

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

                if show_type_list:
                    label = node.bl_label or node.type.replace("_", " ").title()
                    if label not in type_counts:
                        type_colors[label] = node_color
                    info["type_label"] = label
                    type_counts[label] = type_counts.get(label, 0) + 1
                    type_nodes.setdefault(label, []).append(node.name)
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
    st.tree_data = tree_data
    st.tree_data_version += 1
    st.pos_data_version += 1


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
    tree_data = st.tree_data
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
            st.pos_data_version += 1
            if trace:
                wires_t = time.perf_counter() - t1
        if trace:
            logger.trace("TIMER move_update.sockets: %.3f ms", sockets_t * 1000)
            if moved_any:
                logger.trace("TIMER move_update.wires: %.3f ms", wires_t * 1000)
    return True


def _ensure_minimap_batches(
    st: MinimapState,
    mx,
    my,
    mw,
    mh,
    cx,
    cy,
    scale,
    tree_cx,
    tree_cy,
    ui_scale,
    master_alpha,
    show_borders,
    highlight_border=None,
):
    """Bake content batches in map-local space, rebuilding only when stale.

    Vertex data is stored relative to ``tree_data["origin"]`` at the bake-time
    scale, so pan/drag frames only need the matrix transform applied by the
    caller (see draw_minimap). Rebuilds happen when tree positions change,
    the scale drifts past the bucket width (radius/thickness/font buckets),
    styling keys change, or the anchor drifts too far for culling to stay
    conservative. When *highlight_border* is an RGBA color, nodes whose type
    matches ``st.hovered_type_label`` get a highlighted border.
    """
    tree_data = st.tree_data
    if tree_data is None:
        return
    origin = tree_data.get("origin")
    if not origin:
        return

    key = (
        st.pos_data_version,
        round(ui_scale, 3),
        show_borders,
        st.hovered_type_label,
        st.hovered_node,
        bool(highlight_border),
    )
    sb = st.batch_scale
    anchor_x, anchor_y = st.batch_anchor
    # A settle bump changes only tree_data_version, so wire/marker freshness
    # must gate the early return too — otherwise wires stay frozen at their
    # pre-drag positions until an unrelated rebuild trigger fires.
    wire_key = (st.tree_data_version, round(ui_scale, 3))
    wires_fresh = wire_key == st.wire_cache_key and st.wire_scale == sb
    if (
        key == st.batch_cache_key
        and wires_fresh
        and sb > 0.0
        and abs(scale - sb) <= _SCALE_REBUILD_REL * max(sb, 1e-6)
        and abs(cx - anchor_x) <= _BATCH_DRIFT_PX
        and abs(cy - anchor_y) <= _BATCH_DRIFT_PX
    ):
        return

    ocx, ocy = origin
    # Sticky bake scale: adopt the live scale only when the drift budget is
    # exceeded, so fill and wire generations always share one bake scale
    # (and thus one content-matrix factor) between bucket crossings.
    prev_sb = st.batch_scale
    if prev_sb > 0.0 and abs(scale - prev_sb) <= _SCALE_REBUILD_REL * max(prev_sb, 1e-6):
        sb = prev_sb
    else:
        sb = scale

    font_id = 0
    min_dim = 3.0 * ui_scale
    node_infos = tree_data["node_infos"]
    hovered_type = st.hovered_type_label
    hovered_node_name = st.hovered_node
    hl_color = None
    if (hovered_type or hovered_node_name) and highlight_border is not None:
        hl_color = _srgb_to_linear(_alpha_mul(highlight_border, master_alpha))

    # Cull window in baked space: the map interior plus slack for anchor
    # drift between rebuilds (nodes outside never reach the GPU batches).
    piv_bx = (tree_cx - ocx) * sb
    piv_by = (tree_cy - ocy) * sb
    cul_l = mx - _CULL_MARGIN_PX - cx + piv_bx
    cul_r = mx + mw + _CULL_MARGIN_PX - cx + piv_bx
    cul_b = my - _CULL_MARGIN_PX - cy + piv_by
    cul_t = my + mh + _CULL_MARGIN_PX - cy + piv_by

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
        bw_raw = info["tree_w"] * sb
        bh_raw = info["tree_h"] * sb
        bx = (info["tree_x"] - ocx) * sb
        by = (info["tree_y"] - ocy) * sb
        bw = max(bw_raw, 1.0)
        bh = max(bh_raw, 1.0)
        is_frame = info["is_frame"]

        # Cull nodes whose quads cannot intersect the minimap interior
        if bx >= cul_r or bx + bw <= cul_l or by >= cul_t or by + bh <= cul_b:
            continue

        if is_frame:
            node_r = info["node_r_base"] * ui_scale * 1.6
        else:
            node_r = info["node_r_base"] * ui_scale * (sb * 2)

        is_tiny = (bw < min_dim or bh < min_dim) and not is_frame

        border_color = info["border_color"]
        border_w = info["border_w"]
        if hl_color and not is_frame:
            # Guard on hovered_type so a hidden type list (infos without
            # "type_label") can't match None == None for every node.
            if hovered_type is not None and info.get("type_label") == hovered_type:
                border_color = hl_color
                border_w = 1.25
            elif hovered_node_name is not None and info.get("name") == hovered_node_name:
                border_color = hl_color
                border_w = 1.25

        # Borders always emit vertices regardless of on-screen size so they
        # stay visible at any zoom (hover and normal alike); the SDF shader
        # clamps the line width for tiny nodes.
        draw_border = show_borders

        if is_tiny:
            bw_final = max(bw, min_dim)
            bh_final = max(bh, min_dim)
            hw = bw_final / 2
            hh = bh_final / 2
            all_pos_fill.extend(
                [
                    (bx, by, 0.0),
                    (bx + bw_final, by, 0.0),
                    (bx + bw_final, by + bh_final, 0.0),
                    (bx, by + bh_final, 0.0),
                ]
            )
            all_uv_fill.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
            all_half_size_fill.extend([(hw, hh)] * 4)
            all_radius_fill.extend([node_r] * 4)
            all_color_fill.extend([info["fill_color"]] * 4)

            if draw_border:
                all_pos_border.extend(
                    [
                        (bx, by, 0.0),
                        (bx + bw_final, by, 0.0),
                        (bx + bw_final, by + bh_final, 0.0),
                        (bx, by + bh_final, 0.0),
                    ]
                )
                all_uv_border.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
                all_half_size_border.extend([(hw, hh)] * 4)
                all_radius_border.extend([node_r] * 4)
                all_color_border.extend([border_color] * 4)
                all_line_width_border.extend([border_w] * 4)
        else:
            hw = bw / 2
            hh = bh / 2

            pos_fill = frame_pos_fill if is_frame else all_pos_fill
            uv_fill = frame_uv_fill if is_frame else all_uv_fill
            hs_fill = frame_half_size_fill if is_frame else all_half_size_fill
            rad_fill = frame_radius_fill if is_frame else all_radius_fill
            col_fill = frame_color_fill if is_frame else all_color_fill
            pos_fill.extend(
                [
                    (bx, by, 0.0),
                    (bx + bw, by, 0.0),
                    (bx + bw, by + bh, 0.0),
                    (bx, by + bh, 0.0),
                ]
            )
            uv_fill.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
            hs_fill.extend([(hw, hh)] * 4)
            rad_fill.extend([node_r] * 4)
            col_fill.extend([info["fill_color"]] * 4)

            if draw_border:
                pb = frame_pos_border if is_frame else all_pos_border
                ub = frame_uv_border if is_frame else all_uv_border
                hsb = frame_half_size_border if is_frame else all_half_size_border
                rb = frame_radius_border if is_frame else all_radius_border
                cb = frame_color_border if is_frame else all_color_border
                lwb = frame_line_width_border if is_frame else all_line_width_border
                pb.extend(
                    [
                        (bx, by, 0.0),
                        (bx + bw, by, 0.0),
                        (bx + bw, by + bh, 0.0),
                        (bx, by + bh, 0.0),
                    ]
                )
                ub.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
                hsb.extend([(hw, hh)] * 4)
                rb.extend([node_r] * 4)
                cb.extend([border_color] * 4)
                lwb.extend([border_w] * 4)

            # Labels
            if is_frame:
                frame_lbl = info.get("frame_label")
                if frame_lbl:
                    text, text_color, bg_color_lbl = frame_lbl
                    label_font_size = max(6, min(11, int(11 * ui_scale * sb * 8)))
                    blf.size(font_id, label_font_size)
                    tw, th = blf.dimensions(font_id, text)
                    lx = bx + (bw - tw) / 2
                    ly = by + bh + 3 * ui_scale
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
                if lbl_type and lbl_text and bw > 6 * ui_scale and bh > 6 * ui_scale:
                    text_color = info["node_label_color"]
                    if lbl_type == "full":
                        font_size = max(6, min(int(11 * ui_scale), int(min(bw, bh) * 0.35)))
                        lines = _get_node_label_lines(lbl_text, font_id, font_size, bw - 4 * ui_scale, 3)
                        if lines:
                            blf.size(font_id, font_size)
                            line_h = blf.dimensions(font_id, "Ay")[1] + 1
                            asc_h = blf.dimensions(font_id, "A")[1]
                            vis_h = (len(lines) - 1) * line_h + asc_h
                            start_y = by + (bh - vis_h) / 2
                            for i, line in enumerate(lines):
                                lw, _ = blf.dimensions(font_id, line)
                                lx = bx + (bw - lw) / 2
                                ly = start_y + (len(lines) - 1 - i) * line_h
                                cached_text.append((font_id, line, lx, ly, text_color, font_size))
                    else:
                        font_size = max(6, min(int(11 * ui_scale), int(min(bw, bh) * 0.45)))
                        blf.size(font_id, font_size)
                        tw, th = blf.dimensions(font_id, lbl_text)
                        tx = bx + (bw - tw) / 2
                        ty = by + (bh - th) / 2
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
    ph = max(1, tree_data["socket_ph_base"] * sb * ui_scale)
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
            linear_color = _srgb_to_linear(color)
            for sx_tree, sy_tree in positions:
                sxb = (sx_tree - ocx) * sb
                syb = (sy_tree - ocy) * sb
                _pad = 1.5
                socket_all_pos.extend(
                    [
                        (sxb - half_w - _pad, syb - half_h - _pad, 0.0),
                        (sxb + half_w + _pad, syb - half_h - _pad, 0.0),
                        (sxb + half_w + _pad, syb + half_h + _pad, 0.0),
                        (sxb - half_w - _pad, syb + half_h + _pad, 0.0),
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
                socket_all_c.extend([linear_color] * 4)
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

    # Wires and markers get their own cache generation so position-only
    # refreshes (drags) skip the O(links) pill rebake entirely. Rebuilds
    # track the sticky bake scale exactly, keeping the shared matrix factor
    # consistent.
    if wire_key != st.wire_cache_key or st.wire_scale != sb:
        _rebuild_wire_marker_batches(st, tree_data, ocx, ocy, sb, ui_scale, min_dim)
        st.wire_cache_key = wire_key
        st.wire_scale = sb

    st.batch_cache_key = key
    st.batch_scale = sb
    st.batch_anchor = (cx, cy)


def _rebuild_wire_marker_batches(
    st: MinimapState,
    tree_data: dict,
    ocx: float,
    ocy: float,
    sb: float,
    ui_scale: float,
    min_dim: float,
) -> None:
    """Bake wire pill batches (per color + shadow underlay) and group markers.

    Called only when tree structure, UI scale, or the scale bucket changed —
    never on position-only drag refreshes.
    """
    # Wires — baked pill batches per color plus a merged thicker shadow underlay
    thickness = max(1.0, 2.0 * sb)
    wire_batches = []
    shadow_points = []
    for color, items in tree_data["wire_items"].items():
        group = []
        for out_x, out_y, in_x, in_y in items:
            x1 = (out_x - ocx) * sb
            y1 = (out_y - ocy) * sb
            x2 = (in_x - ocx) * sb
            y2 = (in_y - ocy) * sb
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 0.5:
                continue
            angle = math.atan2(dy, dx)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            group.append((mid_x, mid_y, length, angle))
        if group:
            _shader, batch = _build_pill_batch(group, thickness)
            if batch is not None:
                wire_batches.append((color, batch))
                shadow_points.extend(group)
    shadow_batch = None
    if shadow_points:
        _shadow_shader, shadow_batch = _build_pill_batch(shadow_points, thickness * 2.5)
    st.cached_wire_batches = wire_batches
    st.cached_wire_shadow_batch = shadow_batch

    # Group node underline markers — baked like wires
    marker_batches = []
    group_markers = tree_data.get("group_markers")
    if group_markers:
        marker_offset = 3 * ui_scale
        marker_thick = max(1.0, 1.5 * ui_scale)
        for marker_color, items in group_markers.items():
            group = []
            for x_mid, y_bot, length in items:
                ln = length * sb
                if ln < min_dim:
                    continue
                mxb = (x_mid - ocx) * sb
                myb = (y_bot - ocy) * sb - marker_offset
                group.append((mxb, myb, ln, 0.0))
            if group:
                _mshader, mbatch = _build_pill_batch(group, marker_thick)
                if mbatch is not None:
                    marker_batches.append((marker_color, mbatch))
    st.cached_marker_batches = marker_batches


def _get_scrollbar_style(ui_scale: float) -> tuple[int, int]:
    """Return the shared scrollbar (thickness, inset) scaled for the UI."""
    return max(2, int(_SCROLLBAR_THICKNESS * ui_scale)), int(_SCROLLBAR_INSET * ui_scale)


def _scrollbar_thickness(ui_scale: float, active: bool = False) -> int:
    """Return the scrollbar thumb thickness; expanded while hovered or dragged."""
    thick, _ = _get_scrollbar_style(ui_scale)
    if not active:
        return thick
    return max(thick + 1, int(_SCROLLBAR_THICKNESS_HOVER * ui_scale))


def _draw_scrollbar_thumb(
    x: float,
    y: float,
    track_len: float,
    visible_frac: float,
    pos_frac: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    horizontal: bool = False,
    active: bool = False,
    pressed: bool = False,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Draw a scrollbar thumb pill; the origin is the track's start end.

    *visible_frac* is the visible/content ratio sizing the thumb;
    *pos_frac* (0..1) slides it along the track from its start end
    (left when horizontal, bottom otherwise). Hovering (*active*) fades
    the thumb to full opacity; dragging (*pressed*) additionally lifts
    each channel by 5/255 like SCROLL_PRESSED in Blender's widget code.
    Returns the drawn ``(thumb_rect, track_rect)`` as ``(x, y, w, h)``
    for hit-testing.
    """
    thick = _scrollbar_thickness(ui_scale, active)
    if not active:
        color = _alpha_mul(colors["scroll_item"], master_alpha * _SCROLLBAR_ALPHA)
    else:
        rgba = colors["scroll_item"]
        if pressed:
            lift = 5.0 / 255.0
            rgba = (
                min(rgba[0] + lift, 1.0),
                min(rgba[1] + lift, 1.0),
                min(rgba[2] + lift, 1.0),
                rgba[3],
            )
        color = _alpha_mul(rgba, master_alpha)
    min_thumb = int(_SCROLLBAR_MIN_THUMB * ui_scale)
    thumb_len = max(min_thumb, int(track_len * visible_frac))
    offset = int((track_len - thumb_len) * min(max(pos_frac, 0.0), 1.0))
    if horizontal:
        _draw_pill(x + offset, y, thumb_len, thick, color)
        return (x + offset, y, thumb_len, thick), (x, y, track_len, thick)
    _draw_pill(x, y + offset, thick, thumb_len, color)
    return (x, y + offset, thick, thumb_len), (x, y, thick, track_len)


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

    bar_thick, bar_off = _get_scrollbar_style(ui_scale)

    # Horizontal scrollbar (bottom edge)
    if visible_w < bbox_w:
        pos = (v_left - bbox_l) / (bbox_w - visible_w)
        _draw_scrollbar_thumb(
            inner_l,
            my + bar_off,
            inner_w,
            visible_w / bbox_w,
            pos,
            colors,
            master_alpha,
            ui_scale,
            horizontal=True,
        )

    # Vertical scrollbar (right edge)
    if visible_h < bbox_h:
        pos = (v_bottom - bbox_b) / (bbox_h - visible_h)
        _draw_scrollbar_thumb(
            mx + mw - bar_off - bar_thick,
            inner_b,
            inner_h,
            visible_h / bbox_h,
            pos,
            colors,
            master_alpha,
            ui_scale,
        )


def _paint_frame_all_icon(x: float, y: float, size: float, color, ui_scale: float) -> None:
    """Draw the frame-all corner brackets icon."""
    i = 5 * ui_scale
    t = max(1, int(1.5 * ui_scale))
    arm = size * 0.15

    # Top-left bracket
    _draw_filled_rounded_rect(x + i, y + i, arm, t, t * 0.5, color)
    _draw_filled_rounded_rect(x + i, y + i, t, arm, t * 0.5, color)
    # Top-right bracket
    _draw_filled_rounded_rect(x + size - i - arm, y + i, arm, t, t * 0.5, color)
    _draw_filled_rounded_rect(x + size - i - t, y + i, t, arm, t * 0.5, color)
    # Bottom-left bracket
    _draw_filled_rounded_rect(x + i, y + size - i - t, arm, t, t * 0.5, color)
    _draw_filled_rounded_rect(x + i, y + size - i - arm, t, arm, t * 0.5, color)
    # Bottom-right bracket
    _draw_filled_rounded_rect(x + size - i - arm, y + size - i - t, arm, t, t * 0.5, color)
    _draw_filled_rounded_rect(x + size - i - t, y + size - i - arm, t, arm, t * 0.5, color)


def _paint_frame_view_icon(x: float, y: float, size: float, color, ui_scale: float) -> None:
    """Draw the frame-view viewport rectangle icon."""
    inset = 5 * ui_scale
    t = max(4, int(4.0 * ui_scale))
    _draw_rounded_rect_border(
        round(x + inset),
        round(y + inset),
        round(size - 2 * inset),
        round(size - 2 * inset),
        t,
        color,
        0.5 * ui_scale,
    )


def _paint_frame_selected_icon(x: float, y: float, size: float, color, ui_scale: float) -> None:
    """Draw the frame-selected rails and center box icon."""
    i = 5 * ui_scale
    t = max(1, int(1.5 * ui_scale))
    arm = size * 0.15

    # Left/right rails connecting top and bottom corners
    _draw_filled_rounded_rect(x + i, y + i, t, size - 2 * i, t * 0.5, color)
    _draw_filled_rounded_rect(x + size - i - t, y + i, t, size - 2 * i, t * 0.5, color)
    # Corner arms
    _draw_filled_rounded_rect(x + i, y + i, arm, t, t * 0.5, color)
    _draw_filled_rounded_rect(x + size - i - arm, y + i, arm, t, t * 0.5, color)
    _draw_filled_rounded_rect(x + i, y + size - i - t, arm, t, t * 0.5, color)
    _draw_filled_rounded_rect(x + size - i - arm, y + size - i - t, arm, t, t * 0.5, color)

    # Center box
    box_w = box_h = 2 * ui_scale
    box_x = x + (size - box_w) / 2
    box_y = y + (size - box_h) / 2
    _draw_filled_rounded_rect(box_x, box_y, box_w, box_h, 1.5 * ui_scale, color)


def _paint_list_toggle_icon(x: float, y: float, size: float, color, ui_scale: float) -> None:
    """Draw the list-toggle three horizontal bars icon."""
    t = max(1, int(1.5 * ui_scale))
    bar_w = size * 0.5
    bar_gap = 2.0 * ui_scale
    bar_x = x + (size - bar_w) / 2
    bar_y = y + (size - (3 * t + 2 * bar_gap)) / 2 - 0.5

    for i in range(3):
        _draw_filled_rounded_rect(bar_x, bar_y + i * (t + bar_gap), bar_w, t, t * 0.5, color)


def _paint_expand_icon(x: float, y: float, size: float, color, ui_scale: float, expanded: bool) -> None:
    """Draw a plus (collapsed) or minus (expanded) glyph centered at ``(x, y)``."""
    t = max(1, int(1.2 * ui_scale))
    arm = size * 0.5
    _draw_filled_rounded_rect(x - arm, y - t / 2, arm * 2, t, t * 0.5, color)
    if not expanded:
        _draw_filled_rounded_rect(x - t / 2, y - arm, t, arm * 2, t * 0.5, color)


_BUTTON_ICONS = {
    "ALL": _paint_frame_all_icon,
    "VIEW": _paint_frame_view_icon,
    "SELECTED": _paint_frame_selected_icon,
    "LIST": _paint_list_toggle_icon,
}


def _get_visible_minimap_buttons(settings) -> list[str]:
    """Return ids of enabled minimap buttons in draw order."""
    if not settings or not getattr(settings, "interactive", True):
        return []
    return [btn_id for btn_id, pref_attr, _state_attr in _MINIMAP_BUTTONS if getattr(settings, pref_attr, True)]


def _layout_minimap_buttons(
    st: MinimapState,
    visible_ids: list[str],
    mx: float,
    my: float,
    mw: float,
    mh: float,
    padding: float,
    ui_scale: float,
) -> dict[str, tuple[float, float, float]]:
    """Return hit-rect origins {id: (x, y, size)} for every visible button.

    Frame buttons stack top-down along the right edge; the list toggle
    sits at the top-left and slides right of an open type-list zone.
    """
    size = FRAME_ALL_BTN_SIZE * ui_scale
    margin = FRAME_ALL_BTN_MARGIN * ui_scale
    gap = FRAME_BTN_GAP * ui_scale
    top_y = round(my + mh - padding - margin - size)
    stack_x = round(mx + mw - padding - margin - size)

    rects: dict[str, tuple[float, float, float]] = {}
    stack_index = 0
    for btn_id in visible_ids:
        if btn_id == "LIST":
            lx = round(mx + padding + margin)
            if st.list_width > 0:
                # Slide right of the list zone: list, padding, button.
                lx = max(lx, round(_get_map_content_rect(st)[0] + margin))
            rects[btn_id] = (lx, top_y, size)
        else:
            rects[btn_id] = (stack_x, round(top_y - stack_index * (gap + size)), size)
            stack_index += 1
    return rects


def _draw_minimap_buttons(mx, my, mw, mh, padding, colors, ui_scale, master_alpha):
    """Draw the interactive minimap buttons and record their hit rects."""
    addon = bpy.context.preferences.addons.get(__package__)
    settings = getattr(addon.preferences, "settings", None) if addon else None
    st = _state()
    st.frame_all_btn = None
    st.frame_view_btn = None
    st.frame_selected_btn = None
    st.list_toggle_btn = None

    visible_ids = _get_visible_minimap_buttons(settings)
    if not visible_ids:
        return
    rects = _layout_minimap_buttons(st, visible_ids, mx, my, mw, mh, padding, ui_scale)

    bg_color = _alpha_mul(colors["bg"], master_alpha)
    border_color = _alpha_mul(colors["bg"], master_alpha * 0.25)

    # Shared capsule behind the right-edge stack, anchored at its bottom button
    stack = [btn_id for btn_id in visible_ids if btn_id != "LIST"]
    if stack:
        sx, sy, size = rects[stack[-1]]
        span_h = len(stack) * size + (len(stack) - 1) * FRAME_BTN_GAP * ui_scale
        _draw_pill(sx, sy, size, span_h, bg_color)
        _draw_pill_border(sx, sy, size, span_h, border_color, 0.5)

    for btn_id, _pref_attr, state_attr in _MINIMAP_BUTTONS:
        if btn_id not in rects:
            continue
        bx, by, size = rects[btn_id]
        if btn_id == "LIST":
            # Standalone capsule outside the shared stack
            _draw_pill(bx, by, size, size, bg_color)
            _draw_pill_border(bx, by, size, size, border_color, 0.5)
        hovered = st.hovered_frame_btn == btn_id
        ico_color = _alpha_mul(colors["text"], master_alpha * 0.7)
        if hovered:
            _draw_pill(bx + 1, by + 1, size - 2, size - 2, _alpha_mul(colors["text"], BTN_HOVER_ALPHA * master_alpha))
            ico_color = _alpha_mul(colors["text"], master_alpha)
        _BUTTON_ICONS[btn_id](bx, by, size, ico_color, ui_scale)
        setattr(st, state_attr, (bx, by, size, size))


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
        # logger.debug(
        #     "draw_minimap: area=%d win=%d modal_ops=%s has_modal=%s interactive=%s",
        #     context.area.as_pointer() if context.area else 0,
        #     win_ptr,
        #     list(_minimap_window_operators.keys()),
        #     has_modal,
        #     getattr(settings, "interactive", True),
        # )
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

    # Cache the editor viewport rect once for this frame; reused by the
    # transform/clamp logic and the viewport overlay draws below.
    visible = _get_visible_rect(space, region)

    # Single RNA pass: fingerprint, raw tree bounds, and drawable node count.
    show_borders = getattr(settings, "show_node_borders", True)
    with _Timer("tree_snapshot"):
        current_fingerprint, raw_bounds, content_count = _get_tree_snapshot(node_tree, show_borders)
    if raw_bounds[2] - raw_bounds[0] <= 0 or raw_bounds[3] - raw_bounds[1] <= 0:
        return

    # Start cProfile for this area (only when TRACE logging is on)
    _maybe_start_profiler(st)

    # Log active settings every frame at TRACE level
    logger.trace(
        "SETTINGS %d nodes | show_wires=%d show_names=%d label_mode=%s"
        " colored_nodes=%d socket_indicators=%d wire_color=%d frame_labels=%d",
        current_fingerprint[0],
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

        bounds = _expand_bounds_margin(raw_bounds, ui_scale, mh, padding)

        st.rect = (mx, my, mw, mh)
        st.tree_bounds = bounds
        st.margin = y_margin
        st.padding = padding

        # Reserve the type-list zone before computing the map transform so
        # node framing and panning never place tree content behind the list.
        with _Timer("type_list_width"):
            _step_list_width(st, settings, mw, ui_scale)

        _clamp_pan_to_viewport(space, region, st, visible)

    # Refresh tree data: pure position changes (node drags) patch the cached
    # tables immediately; anything else schedules a debounced full compile.
    old_fingerprint = st.cached_fingerprint
    if old_fingerprint != current_fingerprint:
        move_only = _is_move_only_diff(old_fingerprint, current_fingerprint)
        applied = False
        if move_only and (time.perf_counter() - st.last_move_refresh) >= _MOVE_REFRESH_MIN_INTERVAL:
            with _Timer("apply_move_updates"):
                applied = _apply_move_updates(st, node_tree)
            if applied:
                st.last_move_refresh = time.perf_counter()
                st.pending_settle_flush = True
        if applied:
            st.cached_fingerprint = current_fingerprint
        # Always keep a settle timer armed: it flushes frozen wire/marker
        # batches (forced via pending_settle_flush) or runs the pending full
        # compile. Re-arm (push back) only when the fingerprint changed again
        # since arming; identical-fingerprint redraw streams (list animation,
        # hover) must leave the live timer alone so the settle event cannot be
        # starved by continuous redraws.
        delay = getattr(settings, "debounce_interval", 0.15)
        now = time.perf_counter()
        if st.pending_timer is not None and st.pending_fingerprint != current_fingerprint:
            if now < st.pending_timer_deadline:
                try:
                    bpy.app.timers.unregister(st.pending_timer)
                except ValueError:
                    pass
                st.pending_timer = None
        if st.pending_timer is None:

            def _settle_fire():
                return _debounced_compile(st, node_tree, colors, settings, master_alpha, ui_scale)

            # An expanding type list needs compiled type stats to measure its
            # target width; compile immediately instead of after the debounce.
            interval = 0.0 if st.list_anim_active and st.list_anim_target < 0 else delay
            bpy.app.timers.register(_settle_fire, first_interval=interval)
            st.pending_timer = _settle_fire
            st.pending_timer_deadline = now + delay
            st.pending_fingerprint = current_fingerprint

    # Build screen-space batches (cached; applies current zoom/pan via matrix)
    cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform(st, space, region, visible)
    st.scale = scale
    with _Timer("ensure_batches"):
        highlight_border = (
            _alpha_mul(colors["node_active"], 0.5) if (st.hovered_type_label or st.hovered_node) else None
        )
        _ensure_minimap_batches(
            st,
            mx,
            my,
            mw,
            mh,
            cx,
            cy,
            scale,
            tree_cx,
            tree_cy,
            ui_scale,
            master_alpha,
            show_borders,
            highlight_border,
        )

    # Draw minimap panel
    try:
        original_blend = gpu.state.blend_get()
    except Exception:
        original_blend = None
    gpu.state.blend_set("ALPHA")

    with _Timer("draw_background"):
        bg_color, panel_r = _draw_background(mx, my, mw, mh, colors, master_alpha)

    # Clip node/wire content to the minimap interior
    with _Timer("setup_scissor"):
        scissor_state = _setup_scissor(mx, my, mw, mh)
        scissor_active = scissor_state[0]

    # Editor View fill
    with _Timer("draw_view_fill"):
        _draw_view_fill(
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
            panel_r,
            master_alpha,
            ui_scale,
            visible,
        )

    # Content batches are baked in map-local space; place them with one
    # matrix transform (translate -> scale about the view pivot) instead of
    # rebuilding vertex data on pan/drag frames.
    origin = st.tree_data.get("origin") if st.tree_data else None
    content_k = 1.0
    piv_x = 0.0
    piv_y = 0.0
    if origin:
        batch_sb = st.batch_scale if st.batch_scale > 0.0 else scale
        content_k = scale / batch_sb
        piv_x = (tree_cx - origin[0]) * batch_sb
        piv_y = (tree_cy - origin[1]) * batch_sb
        content_mat = (
            Matrix.Translation((cx, cy, 0.0)) @ Matrix.Scale(content_k, 4) @ Matrix.Translation((-piv_x, -piv_y, 0.0))
        )

        gpu.matrix.push()
        try:
            gpu.matrix.multiply_matrix(content_mat)

            # Frame nodes
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

            # Link wires (baked batches; shadow underlay first, then colors)
            wire_batches = st.cached_wire_batches or []
            wire_shadow_batch = st.cached_wire_shadow_batch
            if getattr(settings, "show_wires", True) and (wire_shadow_batch or wire_batches):
                with _Timer("draw_wires"):
                    pill_shader = _get_batch_pill_shader()
                    pill_shader.bind()
                    pill_shader.uniform_float(
                        "ModelViewProjectionMatrix",
                        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
                    )
                    shadow_alpha = 0.35 * master_alpha
                    if wire_shadow_batch is not None and shadow_alpha > 0:
                        pill_shader.uniform_float("color", (0.0, 0.0, 0.0, shadow_alpha))
                        wire_shadow_batch.draw(pill_shader)
                    for wire_color, batch in wire_batches:
                        pill_shader.uniform_float("color", _srgb_to_linear(wire_color))
                        batch.draw(pill_shader)

            # Node fill backgrounds
            backdrops_batch = st.cached_backdrops_batch
            if backdrops_batch:
                with _Timer("draw_backdrops"):
                    fill_shader = _get_batch_rect_shader()
                    fill_shader.bind()
                    fill_shader.uniform_float(
                        "ModelViewProjectionMatrix",
                        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
                    )
                    backdrops_batch.draw(fill_shader)

            # Node borders
            borders_batch = st.cached_borders_batch
            if borders_batch:
                with _Timer("draw_borders"):
                    border_shader = _get_batch_rect_border_shader()
                    border_shader.bind()
                    border_shader.uniform_float(
                        "ModelViewProjectionMatrix",
                        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
                    )
                    borders_batch.draw(border_shader)

            # Group node underline markers (baked batches)
            marker_batches = st.cached_marker_batches or []
            if marker_batches:
                with _Timer("draw_group_markers"):
                    pill_shader = _get_batch_pill_shader()
                    pill_shader.bind()
                    pill_shader.uniform_float(
                        "ModelViewProjectionMatrix",
                        gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
                    )
                    for marker_color, batch in marker_batches:
                        pill_shader.uniform_float("color", _srgb_to_linear(marker_color))
                        batch.draw(pill_shader)

            # Socket indicator pills (single batch with per-vertex color)
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
        finally:
            gpu.matrix.pop()

    # Text labels — mapped manually so BLF never sees the content matrix
    cached_text = st.cached_text or []
    if cached_text and origin:
        with _Timer("draw_text"):
            gpu.state.blend_set("ALPHA")
            off_x = cx - content_k * piv_x
            off_y = cy - content_k * piv_y
            for font_id, text, lx, ly, text_color, font_size in cached_text:
                _draw_text_with_shadow(
                    font_id,
                    text,
                    round(content_k * lx + off_x),
                    round(content_k * ly + off_y),
                    text_color,
                    font_size,
                )
            gpu.state.blend_set("ALPHA")

    # Viewport overlay with cutout hole
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
            visible=visible,
        )

    # Minimap Scrollbars
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

    # Minimap buttons
    with _Timer("draw_buttons"):
        _draw_minimap_buttons(mx, my, mw, mh, padding, colors, ui_scale, master_alpha)

    # Edge resize handle pills
    with _Timer("draw_resize_handles"):
        _draw_resize_handles(mx, my, mw, mh, colors, master_alpha, ui_scale, corner, st)

    # Node count overlay text
    with _Timer("draw_node_count"):
        _draw_node_count(settings, content_count, mx, my, mw, colors, master_alpha, ui_scale)

    # Restore GPU state
    _teardown_scissor(scissor_state)
    try:
        gpu.state.blend_set(original_blend if original_blend else "NONE")
    except Exception:
        gpu.state.blend_set("NONE")

    # Interactive node-type list zone (drawn unclipped, on top of map content)
    try:
        gpu.state.blend_set("ALPHA")
        with _Timer("draw_type_list"):
            _draw_type_list(settings, st, mx, my, mh, padding, colors, master_alpha, ui_scale)
    finally:
        try:
            gpu.state.blend_set(original_blend if original_blend else "NONE")
        except Exception:
            gpu.state.blend_set("NONE")

    # Stop & dump profile stats after N frames
    _maybe_stop_profiler(st)
