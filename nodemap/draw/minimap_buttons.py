"""Provide layout and painting for the minimap button chrome."""

from dataclasses import dataclass
from typing import Any

from ..core.buttons import (
    BUTTON_ORDER,
    BUTTONS,
    ButtonKind,
    _cull_frame_ids,
    _row_hover_geometry,
    _row_radii,
    _row_xs,
)
from ..core.constants import (
    BUTTON_HOVER_ALPHA,
    BUTTON_MARGIN,
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
)


@dataclass
class ButtonLayout:
    """Store culled button hit rects and frame-row order for one frame."""

    rects: dict[str, Rect]
    frame_order: list[str]
    size: float
    combined: bool


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


def _paint_frame_all_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
    """Draw the frame-all corner brackets icon."""
    inset = 5 * ui_scale
    stroke_thickness = max(1, int(1.5 * ui_scale))
    arm_length = size * 0.15

    # Top-left bracket
    _draw_filled_rounded_rect(
        x + inset, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + inset, y + inset, stroke_thickness, arm_length, stroke_thickness * 0.5, color, mvp=mvp
    )
    # Top-right bracket
    _draw_filled_rounded_rect(
        x + size - inset - arm_length, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + size - inset - stroke_thickness,
        y + inset,
        stroke_thickness,
        arm_length,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    # Bottom-left bracket
    _draw_filled_rounded_rect(
        x + inset,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    _draw_filled_rounded_rect(
        x + inset, y + size - inset - arm_length, stroke_thickness, arm_length, stroke_thickness * 0.5, color, mvp=mvp
    )
    # Bottom-right bracket
    _draw_filled_rounded_rect(
        x + size - inset - arm_length,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    _draw_filled_rounded_rect(
        x + size - inset - stroke_thickness,
        y + size - inset - arm_length,
        stroke_thickness,
        arm_length,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )


def _paint_frame_view_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
    """Draw the frame-view viewport rectangle icon."""
    inset = 5 * ui_scale
    border_thickness = max(4, int(4.0 * ui_scale))
    _draw_rounded_rect_border(
        round(x + inset),
        round(y + inset),
        round(size - 2 * inset),
        round(size - 2 * inset),
        border_thickness,
        color,
        0.5 * ui_scale,
        mvp=mvp,
    )


def _paint_frame_selected_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
    """Draw the frame-selected rails and center box icon."""
    inset = 5 * ui_scale
    stroke_thickness = max(1, int(1.5 * ui_scale))
    arm_length = size * 0.15

    # Left/right rails connecting top and bottom corners
    _draw_filled_rounded_rect(
        x + inset, y + inset, stroke_thickness, size - 2 * inset, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + size - inset - stroke_thickness,
        y + inset,
        stroke_thickness,
        size - 2 * inset,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    # Corner arms
    _draw_filled_rounded_rect(
        x + inset, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + size - inset - arm_length, y + inset, arm_length, stroke_thickness, stroke_thickness * 0.5, color, mvp=mvp
    )
    _draw_filled_rounded_rect(
        x + inset,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )
    _draw_filled_rounded_rect(
        x + size - inset - arm_length,
        y + size - inset - stroke_thickness,
        arm_length,
        stroke_thickness,
        stroke_thickness * 0.5,
        color,
        mvp=mvp,
    )

    # Center box
    center_box_w = center_box_h = 2 * ui_scale
    center_box_x = x + (size - center_box_w) / 2
    center_box_y = y + (size - center_box_h) / 2
    _draw_filled_rounded_rect(center_box_x, center_box_y, center_box_w, center_box_h, 1.5 * ui_scale, color, mvp=mvp)


def _paint_list_toggle_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
    """Draw the list-toggle icon: three horizontal bars, or an X when active."""
    stroke_thickness = max(1, int(1.5 * ui_scale))
    bar_width = size * 0.5
    bar_gap = 2.0 * ui_scale
    bar_x = x + (size - bar_width) / 2
    bar_y = y + (size - (3 * stroke_thickness + 2 * bar_gap)) / 2 - 0.5

    for bar_index in range(3):
        _draw_filled_rounded_rect(
            bar_x,
            bar_y + bar_index * (stroke_thickness + bar_gap),
            bar_width,
            stroke_thickness,
            stroke_thickness * 0.5,
            color,
            mvp=mvp,
        )
    return


def _paint_grip_icon(x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None) -> None:
    """Draw a move grip icon: two rows of four dots (like a drag handle)."""
    dot_size = 1.0 * ui_scale
    gap = 2.0 * ui_scale
    column_gap = 2.0 * ui_scale
    group_w = 4 * dot_size + 3 * column_gap
    group_h = 2 * dot_size + gap
    group_x = round(x + (size - group_w) / 2)
    group_y = round(y + (size - group_h) / 2)
    for column in range(4):
        for row in range(2):
            _draw_filled_rounded_rect(
                round(group_x + column * (dot_size + column_gap)),
                round(group_y + row * (dot_size + gap)),
                dot_size,
                dot_size,
                dot_size / 2,
                color,
                mvp=mvp,
            )


_ICONS = {
    "ALL": _paint_frame_all_icon,
    "VIEW": _paint_frame_view_icon,
    "SELECTED": _paint_frame_selected_icon,
    "LIST": _paint_list_toggle_icon,
    "DRAG": _paint_grip_icon,
}

assert all(button_def.icon in _ICONS for button_def in BUTTONS.values()), "Button registry icon missing a painter"


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
    margin = BUTTON_MARGIN * ui_scale
    if list_placement == "TOP" and list_width > 0:
        # Vertical layout: the row sits just below the top list strip,
        # overlapping the map content filling the bottom, like the left
        # placement.
        strip_h = min(list_width, map_h - 2 * CONTENT_PADDING * ui_scale)
        zone_bottom = map_y + map_h - CONTENT_PADDING * ui_scale - strip_h
        top_y = round(zone_bottom - ELEMENT_GAP * ui_scale - size)
    else:
        top_y = round(map_y + map_h - padding - margin - size)
    drag_x = round(map_x + map_w - padding - margin - size)
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

    The row hosts the whole chrome: the list toggle at the left and the
    frame buttons (ALL/VIEW/SELECTED) leading into the move-grip drag handle
    at the right. When the move button is disabled the frame row extends to
    the right padding edge instead. Frame buttons are culled progressively
    when the row would spill past the left padding or collide with the list
    toggle. The drag handle is only laid out while interactive mode is on,
    which is the mode that enables repositioning the map.
    """
    top_y, size, drag_x = _row_geometry(
        map_x, map_y, map_w, map_h, padding, ui_scale, state.list.list_placement, state.list.list_width
    )
    gap = padding
    show_move = "DRAG" in visible_ids

    # Frame buttons sit left of the drag handle, or flush to the right
    # padding edge when the move button is hidden.
    frame_ids = [
        button_id for button_id in BUTTON_ORDER if BUTTONS[button_id].group == "frame" and button_id in visible_ids
    ]
    row_right_x = round(drag_x - gap - size) if show_move else drag_x

    # List toggle at the top-left, sliding right of an open type-list zone.
    list_x: float | None = None
    if "LIST" in visible_ids:
        list_x = round(map_x + padding + BUTTON_MARGIN * ui_scale)
        if state.list.list_width > 0:
            list_x = max(list_x, round(_get_map_content_rect(state)[0] + BUTTON_MARGIN * ui_scale))

    priority = tuple(button_id for button_id in BUTTON_ORDER if BUTTONS[button_id].group == "frame")
    kept = _cull_frame_ids(
        frame_ids,
        row_right_x,
        size,
        map_x + padding,
        list_x + size if list_x is not None else None,
        gap,
        priority,
    )

    rects: dict[str, Rect] = {}
    count = len(kept)
    row_origins = _row_xs(row_right_x, count, size)
    for button_index, button_id in enumerate(kept):
        rects[button_id] = (row_origins[button_index], top_y, size, size)

    if list_x is not None:
        rects["LIST"] = (list_x, top_y, size, size)

    if show_move:
        rects["DRAG"] = (drag_x, top_y, size, size)

    frame_order = sorted(kept, key=lambda button_id: rects[button_id][0])
    return ButtonLayout(rects=rects, frame_order=frame_order, size=size, combined=len(frame_order) >= 2)


def _resolve_button_theme(colors: dict, master_alpha: float, ui_scale: float) -> ButtonTheme:
    """Resolve themed button colors and metrics for one frame."""
    radius = colors["node_roundness"] * ui_scale
    return ButtonTheme(
        radius=radius,
        fill_radius=radius * 1.5,
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


def _paint_button_icon(
    button_id: str, x: float, y: float, size: float, color, ui_scale: float, mvp: Any = None
) -> None:
    """Paint the glyph for a button id at its rect origin."""
    _ICONS[BUTTONS[button_id].icon](x, y, size, color, ui_scale, mvp=mvp)


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

    Frame buttons draw as a horizontal row when two or more are shown, each
    as its own box sharing square inner corners; the list toggle and the
    move-grip handle stay standalone. Pass *only_id* to repaint a single
    standalone button (the pressed move grip above the moving overlay).
    """
    rects = layout.rects
    if only_id is not None:
        if only_id not in rects:
            return
        targets = [only_id]
    else:
        targets = [button_id for button_id in BUTTON_ORDER if button_id in rects]

    if only_id is None and layout.combined:
        # Draw each frame button as its own box, edge-to-edge with no gap.
        # Only the external corners round, inner corners meet square; each
        # button's border is drawn on its own rect, so neighboring borders
        # coincide at the seam and every interior is equally inset.
        # Fills use the wider radius to keep anti-aliased edges smooth while
        # borders keep the base radius. Buttons that have a left neighbor
        # skip their left border stroke: two coincident strokes would stack
        # into a heavy 2px seam, so the seam line is emitted once by the
        # neighbor's right border only.
        for button_index, button_id in enumerate(layout.frame_order):
            button_x, button_y, button_w, button_h = rects[button_id]
            fill_radii = _row_radii(button_index, len(layout.frame_order), theme.fill_radius)
            border_radii = _row_radii(button_index, len(layout.frame_order), theme.radius)
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

    order_index = {button_id: index for index, button_id in enumerate(layout.frame_order)}
    for button_id in targets:
        button_def = BUTTONS.get(button_id)
        button_x, button_y, button_w, button_h = rects[button_id]
        is_pressed = state.buttons.pressed_button_id == button_id
        is_hovered = (not is_pressed) and state.buttons.hovered_button_id == button_id
        standalone = (button_def.group != "frame" if button_def else True) or not layout.combined
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
                button_index = order_index.get(button_id, -1)
                hover_x, hover_width, hover_radii = _row_hover_geometry(
                    button_x, button_w, button_index, len(layout.frame_order), hover_radius
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
