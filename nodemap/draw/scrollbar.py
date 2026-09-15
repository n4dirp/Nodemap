"""Provide scrollbar track and thumb drawing for the minimap and type list."""

from typing import Any

import gpu

from ..core.constants import (
    SCROLLBAR_ALPHA,
    SCROLLBAR_INSET,
    SCROLLBAR_MIN_THUMB,
    SCROLLBAR_THICKNESS,
    SCROLLBAR_THICKNESS_HOVER,
)
from ..core.theme import _alpha_mul
from .gpu_draw import _draw_filled_rounded_rect, _draw_pill


def _get_scrollbar_style(ui_scale: float) -> tuple[int, int]:
    """Return the shared scrollbar `(thickness, inset)` scaled for the UI."""
    return max(2, int(SCROLLBAR_THICKNESS * ui_scale)), int(SCROLLBAR_INSET * ui_scale)


def _scrollbar_thickness(ui_scale: float, active: bool = False) -> int:
    """Return the scrollbar thumb thickness; expand while hovered or dragged."""
    thick, _ = _get_scrollbar_style(ui_scale)
    if not active:
        return thick
    return max(thick + 1, int(SCROLLBAR_THICKNESS_HOVER * ui_scale))


def _scrollbar_track_color(colors: dict, master_alpha: float, active: bool) -> tuple:
    """Return the scrollbar track fill, brightened while hovered or dragged."""
    inner = colors["scroll_inner"]
    alpha = (min(inner[3], 0.15) if not active else max(inner[3], 0.15)) * master_alpha
    return (inner[0], inner[1], inner[2], alpha)


def _draw_list_scrollbar_core(
    track_x: float,
    track_y: float,
    track_len: float,
    thumb_x: float,
    thumb_y: float,
    thumb_track_len: float,
    visible_frac: float,
    pos_frac: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    active: bool,
    horizontal: bool = False,
    mvp: Any = None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Draw a scrollbar track background and thumb, returning `(thumb_rect, track_rect)`.

    ``track_len`` sizes the background along the scroll direction; the thumb
    slides along ``(thumb_x, thumb_y, thumb_track_len)``.
    """
    gpu.state.blend_set("ALPHA")
    thick = _scrollbar_thickness(ui_scale, active)
    fill_color = _scrollbar_track_color(colors, master_alpha, active)

    if horizontal:
        _draw_filled_rounded_rect(track_x, track_y, track_len, thick, thick / 2.0, fill_color, mvp=mvp)
    else:
        _draw_filled_rounded_rect(track_x, track_y, thick, track_len, thick / 2.0, fill_color, mvp=mvp)

    return _draw_scrollbar_thumb(
        thumb_x,
        thumb_y,
        thumb_track_len,
        visible_frac,
        pos_frac,
        colors,
        master_alpha,
        ui_scale,
        horizontal=horizontal,
        active=active,
        mvp=mvp,
    )


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
    mvp: Any = None,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Draw a scrollbar thumb pill and return `(thumb_rect, track_rect)`.

    ``visible_frac`` sizes the thumb, ``pos_frac`` slides it along the track.
    """
    thick = _scrollbar_thickness(ui_scale, active)

    if not active:
        color = _alpha_mul(colors["scroll_item"], master_alpha * SCROLLBAR_ALPHA)
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

    min_thumb_len = int(SCROLLBAR_MIN_THUMB * ui_scale)
    thumb_len = max(min_thumb_len, int(track_len * visible_frac))
    offset = int((track_len - thumb_len) * min(max(pos_frac, 0.0), 1.0))

    if horizontal:
        _draw_pill(x + offset, y, thumb_len, thick, color, mvp=mvp)
        return (x + offset, y, thumb_len, thick), (x, y, track_len, thick)

    _draw_pill(x, y + offset, thick, thumb_len, color, mvp=mvp)
    return (x, y + offset, thick, thumb_len), (x, y, thick, track_len)


def _draw_minimap_scrollbars(
    map_x,
    map_y,
    map_w,
    map_h,
    padding,
    map_anchor_x,
    map_anchor_y,
    scale,
    tree_center_x,
    tree_center_y,
    content_bounds,
    colors,
    ui_scale,
    master_alpha,
    content_rect: tuple[float, float, float, float] | None = None,
    mvp: Any = None,
):
    """Draw horizontal/vertical minimap scrollbar thumbs when zoomed in.

    ``content_rect`` is the reduced node content area ``(left, bottom, width,
    height)`` already reserved for the type-list zone; both bars confine their
    tracks and thumb math to it. When ``None``, fall back to the full map inner
    rect.
    """
    if content_rect is None:
        inner_l = map_x + padding
        inner_b = map_y + padding
        inner_w = map_w - 2 * padding
        inner_h = map_h - 2 * padding
    else:
        inner_l, inner_b, inner_w, inner_h = content_rect
    inner_r = inner_l + inner_w
    inner_t = inner_b + inner_h

    bbox_l, bbox_b, bbox_r, bbox_t = content_bounds
    bbox_w = bbox_r - bbox_l
    bbox_h = bbox_t - bbox_b

    if bbox_w <= 0 or bbox_h <= 0:
        return

    tree_l = tree_center_x + (inner_l - map_anchor_x) / scale
    tree_r = tree_center_x + (inner_r - map_anchor_x) / scale
    tree_b = tree_center_y + (inner_b - map_anchor_y) / scale
    tree_t = tree_center_y + (inner_t - map_anchor_y) / scale

    v_left = max(bbox_l, min(bbox_r, tree_l))
    v_right = max(bbox_l, min(bbox_r, tree_r))
    v_bottom = max(bbox_b, min(bbox_t, tree_b))
    v_top = max(bbox_b, min(bbox_t, tree_t))

    visible_w = v_right - v_left
    visible_h = v_top - v_bottom

    if visible_w >= bbox_w and visible_h >= bbox_h:
        return

    bar_thickness, bar_offset = _get_scrollbar_style(ui_scale)

    if visible_w < bbox_w:
        frac_h = (v_left - bbox_l) / (bbox_w - visible_w)
        _draw_scrollbar_thumb(
            inner_l,
            map_y + bar_offset,
            inner_w,
            visible_w / bbox_w,
            frac_h,
            colors,
            master_alpha,
            ui_scale,
            horizontal=True,
            mvp=mvp,
        )

    if visible_h < bbox_h:
        frac_v = (v_bottom - bbox_b) / (bbox_h - visible_h)
        _draw_scrollbar_thumb(
            map_x + map_w - bar_offset - bar_thickness,
            inner_b,
            inner_h,
            visible_h / bbox_h,
            frac_v,
            colors,
            master_alpha,
            ui_scale,
            mvp=mvp,
        )
