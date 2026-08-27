"""Minimap rendering in the Node Editor."""

import logging
import time

import blf
import bpy
import gpu
from mathutils import Matrix

from .batch_build import _ensure_minimap_batches
from .gpu_draw import (
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
    MIN_MAP_HEIGHT,
    MIN_MAP_WIDTH,
    _expand_bounds_margin,
    _get_minimap_margins,
    _get_safe_bounds,
    _get_tree_snapshot,
    _get_ui_scale,
)
from .state import (
    _MINIMAP_BUTTONS,
    MinimapState,
    ResizeHandle,
    _minimap_window_operators,
    _registration_state,
    _state,
)
from .theme import (
    _alpha_mul,
    _get_node_editor_theme_colors,
    _srgb_to_linear,
)
from .transforms import (
    _clamp_pan_to_viewport,
    _get_map_content_rect,
    _get_minimap_transform,
    _get_visible_rect,
)
from .tree_compile import (
    _MOVE_REFRESH_MIN_INTERVAL,
    _apply_move_updates,
    _debounced_compile,
    _is_move_only_diff,
    _maybe_start_profiler,
    _maybe_stop_profiler,
    _Timer,
)
from .type_list import _draw_minimap_scrollbars, _draw_type_list, _step_list_width

logger = logging.getLogger(__package__)

# Variables
FONT_SIZE = 11
BTN_SIZE = 20
BTN_MARGIN = 2
FRAME_BTN_GAP = 0

BTN_HOVER_ALPHA = 0.015


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
        st.view.rect = (0.0, 0.0, 0.0, 0.0)
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
    handle = st.interaction.resize_active
    if not handle:
        return

    w_clamped = st.view.width_clamped
    h_clamped = st.view.height_clamped

    col_base = _alpha_mul(colors["text"], 0.5 * master_alpha)
    col_warn = _alpha_mul(colors["indicator"], master_alpha)
    thick = 3.0 * ui_scale
    margin = 6 * ui_scale

    match handle:
        case ResizeHandle.W:
            wx = mx + 2 * ui_scale if corner in ("TOP_RIGHT", "BOTTOM_RIGHT") else mx + mw - 2 * ui_scale - thick
            _draw_pill(wx, my + margin, thick, mh - 2 * margin, col_warn if w_clamped else col_base)
        case ResizeHandle.H:
            hy = my + 2 * ui_scale if corner in ("TOP_RIGHT", "TOP_LEFT") else my + mh - 2 * ui_scale - thick
            _draw_pill(mx + margin, hy, mw - 2 * margin, thick, col_warn if h_clamped else col_base)
        case ResizeHandle.C:
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

    # border_alpha_mul = 0.5 if st and st.interaction.pressed else 1.0

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
        if st and st.interaction.pressed:
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
    btn_bottoms = [rect[1] for rect in st.buttons.rects.values() if rect]
    if btn_bottoms and min(btn_bottoms) <= ty + font_size:
        return

    text_color = _alpha_mul(colors["text"], 0.85 * master_alpha)

    _draw_text_with_shadow(font_id, info_text, tx + pad, ty + pad, text_color, font_size)


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
    return [btn_id for btn_id, pref_attr in _MINIMAP_BUTTONS if getattr(settings, pref_attr, True)]


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
    size = BTN_SIZE * ui_scale
    margin = BTN_MARGIN * ui_scale
    gap = FRAME_BTN_GAP * ui_scale
    top_y = round(my + mh - padding - margin - size)
    stack_x = round(mx + mw - padding - margin - size)

    rects: dict[str, tuple[float, float, float]] = {}
    stack_index = 0
    for btn_id in visible_ids:
        if btn_id == "LIST":
            lx = round(mx + padding + margin)
            if st.list.width > 0:
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
    st.buttons.rects.clear()

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

    for btn_id, _pref_attr in _MINIMAP_BUTTONS:
        if btn_id not in rects:
            continue
        bx, by, size = rects[btn_id]
        if btn_id == "LIST":
            # Standalone capsule outside the shared stack
            _draw_pill(bx, by, size, size, bg_color)
            _draw_pill_border(bx, by, size, size, border_color, 0.5)
        hovered = st.buttons.hovered == btn_id
        ico_color = _alpha_mul(colors["text"], master_alpha * 0.7)
        if hovered:
            _draw_pill(bx + 1, by + 1, size - 2, size - 2, _alpha_mul(colors["text"], BTN_HOVER_ALPHA * master_alpha))
            ico_color = _alpha_mul(colors["text"], master_alpha)
        _BUTTON_ICONS[btn_id](bx, by, size, ico_color, ui_scale)
        st.buttons.rects[btn_id] = (bx, by, size, size)


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

        st.view.rect = (mx, my, mw, mh)
        st.view.tree_bounds = bounds
        st.view.margin = y_margin
        st.view.padding = padding

        # Reserve the type-list zone before computing the map transform so
        # node framing and panning never place tree content behind the list.
        with _Timer("type_list_width"):
            _step_list_width(st, settings, mw, ui_scale)

        _clamp_pan_to_viewport(space, region, st, visible)

    # Refresh tree data: pure position changes (node drags) patch the cached
    # tables immediately; anything else schedules a debounced full compile.
    old_fingerprint = st.cache.fingerprint
    if old_fingerprint != current_fingerprint:
        move_only = _is_move_only_diff(old_fingerprint, current_fingerprint)
        applied = False
        if move_only and (time.perf_counter() - st.cache.last_move_refresh) >= _MOVE_REFRESH_MIN_INTERVAL:
            with _Timer("apply_move_updates"):
                applied = _apply_move_updates(st, node_tree)
            if applied:
                st.cache.last_move_refresh = time.perf_counter()
                st.cache.pending_settle_flush = True
        if applied:
            st.cache.fingerprint = current_fingerprint
        # Always keep a settle timer armed: it flushes frozen wire/marker
        # batches (forced via pending_settle_flush) or runs the pending full
        # compile. Re-arm (push back) only when the fingerprint changed again
        # since arming; identical-fingerprint redraw streams (list animation,
        # hover) must leave the live timer alone so the settle event cannot be
        # starved by continuous redraws.
        delay = getattr(settings, "debounce_interval", 0.15)
        now = time.perf_counter()
        if st.cache.pending_timer is not None and st.cache.pending_fingerprint != current_fingerprint:
            if now < st.cache.pending_timer_deadline:
                try:
                    bpy.app.timers.unregister(st.cache.pending_timer)
                except ValueError:
                    pass
                st.cache.pending_timer = None
        if st.cache.pending_timer is None:

            def _settle_fire():
                return _debounced_compile(st, node_tree, colors, settings, master_alpha, ui_scale)

            # An expanding type list needs compiled type stats to measure its
            # target width; compile immediately instead of after the debounce.
            # List click actions also request an immediate compile so the visual
            # feedback is not delayed by the debounce interval.
            immediate = (st.list.anim_active and st.list.anim_target < 0) or st.cache.force_immediate
            interval = 0.0 if immediate else delay
            bpy.app.timers.register(_settle_fire, first_interval=interval)
            st.cache.pending_timer = _settle_fire
            st.cache.pending_timer_deadline = now + delay
            st.cache.pending_fingerprint = current_fingerprint
            st.cache.force_immediate = False

    # Build screen-space batches (cached; applies current zoom/pan via matrix)
    # When a structural preference changed, _batches_dirty forces a batch
    # rebuild using the existing tree_data (which still reflects the old
    # settings). The debounce timer will recompile tree_data on the next
    # event-loop iteration, producing a second rebuild with fresh data.
    if st.cache._batches_dirty:
        st.cache._batches_dirty = False
        st.cache.position_version += 1
    cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform(st, space, region, visible)
    st.view.scale = scale
    with _Timer("ensure_batches"):
        highlight_border = (
            _alpha_mul(colors["node_active"], 0.2)
            if (st.list.hovered_type_label or st.interaction.hovered_node)
            else None
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
    origin = st.cache.tree_data.get("origin") if st.cache.tree_data else None
    content_k = 1.0
    piv_x = 0.0
    piv_y = 0.0
    if origin:
        batch_sb = st.cache.batch_scale if st.cache.batch_scale > 0.0 else scale
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
            frames_fill_batch = st.cache.frames_fill_batch
            frames_border_batch = st.cache.frames_border_batch
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
            wire_batches = st.cache.wire_batches or []
            wire_shadow_batch = st.cache.wire_shadow_batch
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
            backdrops_batch = st.cache.backdrops_batch
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
            borders_batch = st.cache.borders_batch
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
            marker_batches = st.cache.marker_batches or []
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
            socket_batch = st.cache.socket_batch
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
    cached_text = st.cache.text or []
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
