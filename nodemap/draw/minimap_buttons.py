"""Provide layout and painting for the minimap button chrome."""

from dataclasses import dataclass
from typing import Any

from ..core.buttons import (
    BUTTON_ORDER,
    BUTTONS,
    ButtonKind,
    _cull_row_ids,
    _group_button_ids,
    _row_hover_geometry,
    _row_radii,
    _row_xs,
)
from ..core.constants import (
    BUTTON_HOVER_ALPHA,
    BUTTON_ICON_SIZE,
    BUTTON_SIZE,
    CONTENT_PADDING,
    ELEMENT_GAP,
)
from ..core.state import MinimapState, Rect
from ..core.theme import _alpha_mul
from ..geo.transforms import _get_map_content_rect
from .gpu_draw import (
    _draw_filled_rounded_rect,
    _draw_filled_rounded_rect_varying,
    _draw_rounded_rect_border,
    _draw_rounded_rect_border_varying_sides,
    _draw_tinted_image,
)
from .icon_textures import _get_button_texture


@dataclass
class ButtonLayout:
    """Store culled button hit rects and combined-row membership for one frame."""

    rects: dict[str, Rect]
    rows: list[list[str]]
    size: float
    slots: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class ButtonTheme:
    """Store resolved button colors and metrics for one frame."""

    radius: float
    fill_radius: float
    border_width: float
    tool_bg: tuple
    tool_border: tuple
    tool_pressed: tuple
    tool_text: tuple
    tool_text_selected: tuple
    regular_bg: tuple
    regular_border: tuple
    regular_selected: tuple
    regular_text: tuple
    regular_text_selected: tuple


def _paint_button_icon(
    button_id: str, x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None
) -> None:
    """Paint the tinted PNG glyph for a button id, centered in its rect."""
    texture = _get_button_texture(BUTTONS[button_id].icon)
    if texture is None:
        return
    draw_size = BUTTON_ICON_SIZE * ui_scale
    offset = (size - draw_size) / 2.0
    _draw_tinted_image(x + offset, y + offset, draw_size, texture, color, mvp=mvp)


def _row_geometry(
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    padding: float,
    ui_scale: float,
    list_placement: str,
    list_width: float,
) -> tuple[int, int, int]:
    """Return ``(top_y, size, drag_x)`` for the button row.

    The row sits on the top edge, except in the top-list placement where it
    sits just below the list strip. The drag handle anchors to the right
    padding edge. The button size snaps to whole pixels so shared row edges
    coincide exactly instead of drifting by a rounding fraction.
    """
    size = round(BUTTON_SIZE * ui_scale)
    if list_placement == "TOP" and list_width > 0:
        # Vertical layout: the row sits just below the top list strip,
        # overlapping the map content filling the bottom, like the left
        # placement.
        strip_h = min(list_width, map_h - 2 * CONTENT_PADDING * ui_scale)
        zone_bottom = map_y + map_h - CONTENT_PADDING * ui_scale - strip_h
        top_y = round(zone_bottom - ELEMENT_GAP * ui_scale - size)
    else:
        top_y = round(map_y + map_h - padding - size)
    drag_x = round(map_x + map_w - padding - size)
    return top_y, size, drag_x


def _layout_buttons(
    state: MinimapState,
    visible_ids: list[str],
    map_x: float,
    map_y: float,
    map_w: float,
    map_h: float,
    padding: float,
    ui_scale: float,
    settings,
) -> ButtonLayout:
    """Return culled 4-tuple hit rects for every visible button.

    The row hosts the whole chrome: the list toggle at the left; on the right
    the follow-view toggle, the frame row, the zoom pair, and the move-grip
    drag handle, each group separated by the same padding gap. Buttons within
    the frame row and the zoom pair share edges, so only their external
    corners round. Rows are culled progressively when they would spill past
    the left padding or collide with the list toggle. The drag handle is only
    laid out while interactive mode is on, which is the mode that enables
    repositioning the map.
    """
    top_y, size, drag_x = _row_geometry(
        map_x, map_y, map_w, map_h, padding, ui_scale, state.list.list_placement, state.list.list_width
    )
    gap = padding
    show_move = "DRAG" in visible_ids

    # Left chrome: the list toggle, sliding right of an open type-list zone.
    left_ids = _group_button_ids(visible_ids, "left")
    left_x: int | None = None
    left_limit = round(map_x + padding)
    if left_ids:
        left_x = left_limit
        if state.list.list_width > 0:
            left_x = max(left_x, round(_get_map_content_rect(state)[0]))
        left_right = left_x + len(left_ids) * size + (len(left_ids) - 1) * gap
        left_limit = round(left_right + gap)

    follow_ids = _group_button_ids(visible_ids, "view")
    frame_ids = _group_button_ids(visible_ids, "frame")
    zoom_ids = _group_button_ids(visible_ids, "zoom")

    # The right side fills leftwards from the drag handle, or from the right
    # padding edge when the move button is hidden.
    cursor = round(drag_x - gap - size) if show_move else drag_x

    # Zoom pair, then the frame row, then the follow toggle.
    zoom_kept = _cull_row_ids(zoom_ids, cursor, size, left_limit, tuple(zoom_ids))
    zoom_origins = _row_xs(cursor, len(zoom_kept), size)
    if zoom_kept:
        cursor = zoom_origins[0] - gap - size

    # Reserve room for the follow toggle left of the frame row so frame
    # culling never pushes it past the left chrome.
    follow_reserve = size + gap if follow_ids else 0.0
    frame_priority = tuple(button_id for button_id in BUTTON_ORDER if BUTTONS[button_id].group == "frame")
    frame_kept = _cull_row_ids(frame_ids, cursor, size, left_limit + follow_reserve, frame_priority)
    frame_origins = _row_xs(cursor, len(frame_kept), size)
    if frame_kept:
        cursor = frame_origins[0] - gap - size

    follow_id = follow_ids[0] if follow_ids and cursor >= left_limit else None

    rects: dict[str, Rect] = {}
    for button_index, button_id in enumerate(frame_kept):
        rects[button_id] = (frame_origins[button_index], top_y, size, size)
    for button_index, button_id in enumerate(zoom_kept):
        rects[button_id] = (zoom_origins[button_index], top_y, size, size)
    if follow_id is not None:
        rects[follow_id] = (round(cursor), top_y, size, size)

    if left_x is not None:
        for left_index, button_id in enumerate(left_ids):
            rects[button_id] = (round(left_x + left_index * (size + gap)), top_y, size, size)

    if show_move:
        rects["DRAG"] = (drag_x, top_y, size, size)

    rows = [row_ids for row_ids in (frame_kept, zoom_kept) if len(row_ids) >= 2]
    slots = {
        button_id: (button_index, len(row_ids)) for row_ids in rows for button_index, button_id in enumerate(row_ids)
    }
    return ButtonLayout(rects=rects, rows=rows, size=size, slots=slots)


def _resolve_button_theme(colors: dict, master_alpha: float, ui_scale: float) -> ButtonTheme:
    """Resolve themed button colors and metrics for one frame."""
    radius = colors["node_roundness"] * ui_scale
    return ButtonTheme(
        radius=radius,
        fill_radius=radius * 1.2,
        border_width=0.5 * ui_scale,
        tool_bg=_alpha_mul(colors["tool_inner"], master_alpha),
        tool_border=_alpha_mul(colors["tool_outline"], master_alpha),
        tool_pressed=_alpha_mul(colors["tool_selected"], master_alpha),
        tool_text=_alpha_mul(colors["tool_text"], master_alpha),
        tool_text_selected=_alpha_mul(colors["tool_text_selected"], master_alpha),
        regular_bg=_alpha_mul(colors["regular_inner"], master_alpha),
        regular_border=_alpha_mul(colors["regular_outline"], master_alpha),
        regular_selected=_alpha_mul(colors["regular_selected"], master_alpha),
        regular_text=_alpha_mul(colors["regular_text"], master_alpha),
        regular_text_selected=_alpha_mul(colors["regular_text_selected"], master_alpha),
    )


def _paint_buttons(
    layout: ButtonLayout,
    theme: ButtonTheme,
    state: MinimapState,
    settings,
    ui_scale: float,
    master_alpha: float,
    mvp: Any = None,
    only_id: str | None = None,
) -> None:
    """Paint buttons from a layout without touching hit rects.

    Each combined row (the frame row and the zoom pair) draws as one box: only
    the row's external corners round, inner corners meet square, and a shared
    seam is stroked once. Every other button stays standalone. Pass *only_id*
    to repaint a single standalone button (the pressed move grip above the
    moving overlay).
    """
    rects = layout.rects
    if only_id is not None:
        if only_id not in rects:
            return
        targets = [only_id]
    else:
        targets = [button_id for button_id in BUTTON_ORDER if button_id in rects]

    shadow_color = (0, 0, 0, 0.15 * master_alpha)
    if only_id is None:
        for row_ids in layout.rows:
            row_x, row_y = rects[row_ids[0]][0], rects[row_ids[0]][1]
            row_w = len(row_ids) * layout.size
            _draw_filled_rounded_rect(
                row_x + theme.radius,
                row_y - 1,
                row_w - 2 * theme.radius,
                1,
                0,
                shadow_color,
                mvp=mvp,
            )
            # Draw each button as its own box, edge-to-edge with no gap.
            # Only the external corners round, inner corners meet square; each
            # button's border is drawn on its own rect, so neighboring borders
            # coincide at the seam and every interior is equally inset.
            # Fills use the wider radius to keep anti-aliased edges smooth while
            # borders keep the base radius. Buttons that have a left neighbor
            # skip their left border stroke: two coincident strokes would stack
            # into a heavy 2px seam, so the seam line is emitted once by the
            # neighbor's right border only.
            for button_index, button_id in enumerate(row_ids):
                button_x, button_y, button_w, button_h = rects[button_id]
                fill_radii = _row_radii(button_index, len(row_ids), theme.fill_radius)
                border_radii = _row_radii(button_index, len(row_ids), theme.radius)
                _draw_filled_rounded_rect_varying(
                    button_x, button_y, button_w, button_h, fill_radii, theme.tool_bg, mvp=mvp
                )
                _draw_rounded_rect_border_varying_sides(
                    button_x,
                    button_y,
                    button_w,
                    button_h,
                    border_radii,
                    theme.tool_border,
                    theme.border_width,
                    skip_left=button_index > 0,
                    mvp=mvp,
                )

    for button_id in targets:
        if button_id in layout.slots:
            continue
        button_x, button_y, button_w = rects[button_id][:3]
        _draw_filled_rounded_rect(
            button_x + theme.radius,
            button_y - 1,
            button_w - 2 * theme.radius,
            1,
            0,
            shadow_color,
            mvp=mvp,
        )
    for button_id in targets:
        button_def = BUTTONS.get(button_id)
        button_x, button_y, button_w, button_h = rects[button_id]
        is_pressed = state.buttons.pressed_button_id == button_id
        is_hovered = (not is_pressed) and state.buttons.hovered_button_id == button_id
        standalone = button_id not in layout.slots
        if button_def is not None and button_def.kind == ButtonKind.REGULAR:
            toggled = bool(button_def.toggle_attr and getattr(settings, button_def.toggle_attr, False))
            box_fill = theme.regular_selected if toggled else theme.regular_bg
            box_border = theme.regular_border
            pressed_fill = theme.regular_selected
            icon_color = theme.regular_text_selected if (toggled or is_pressed) else theme.regular_text
        else:
            box_fill = theme.tool_bg
            box_border = theme.tool_border
            pressed_fill = theme.tool_pressed
            icon_color = theme.tool_text_selected if is_pressed else theme.tool_text
        if standalone:
            _draw_filled_rounded_rect(button_x, button_y, button_w, button_h, theme.fill_radius, box_fill, mvp=mvp)
            _draw_rounded_rect_border(
                button_x, button_y, button_w, button_h, theme.radius, box_border, theme.border_width, mvp=mvp
            )
        if is_pressed or is_hovered:
            fill_color = pressed_fill if is_pressed else (1, 1, 1, BUTTON_HOVER_ALPHA * master_alpha)
            if not standalone:
                # Only the row's external corners round, inner corners stay square.
                hover_radius = max(2.0, theme.fill_radius - 1)
                button_index, row_count = layout.slots[button_id]
                hover_x, hover_width, hover_radii = _row_hover_geometry(
                    button_x, button_w, button_index, row_count, hover_radius
                )
                _draw_filled_rounded_rect_varying(
                    hover_x, button_y + 1, hover_width, button_h - 2, hover_radii, fill_color, mvp=mvp
                )
            else:
                _draw_filled_rounded_rect(
                    button_x + 1,
                    button_y + 1,
                    button_w - 2,
                    button_h - 2,
                    max(2.0, theme.fill_radius - 1),
                    fill_color,
                    mvp=mvp,
                )
        _paint_button_icon(button_id, button_x, button_y, button_w, icon_color, ui_scale, mvp=mvp)
