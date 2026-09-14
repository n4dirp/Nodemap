"""Define the minimap button registry and pure button layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable


class ButtonKind(StrEnum):
    """Classify how a minimap button looks and what it does."""

    TOOL = "TOOL"
    REGULAR = "REGULAR"


class PressStyle(StrEnum):
    """Classify how a minimap button is activated."""

    CLICK = "CLICK"
    DRAG = "DRAG"


@dataclass(frozen=True)
class ButtonDef:
    """Describe one minimap button; static data only, no behavior callbacks."""

    id: str
    label: str
    kind: ButtonKind
    group: str
    press_style: PressStyle
    pref_attr: str
    icon: str
    action: str | None = None
    toggle_attr: str | None = None
    menu_group: str = "frame"
    list_animation: bool = False
    disable_when_follow_view: bool = False
    visible_when: Callable[[Any], bool] = lambda settings: True


def _not_follow_view(settings) -> bool:
    """Return True unless Follow View drives framing (hides Frame Selected)."""
    return not bool(getattr(settings, "use_follow_view", False))


BUTTONS: dict[str, ButtonDef] = {
    "SELECTED": ButtonDef(
        id="SELECTED",
        label="Frame Selected",
        kind=ButtonKind.TOOL,
        group="frame",
        press_style=PressStyle.CLICK,
        pref_attr="show_frame_selected_button",
        icon="SELECTED",
        action="SELECTED",
        disable_when_follow_view=True,
        visible_when=_not_follow_view,
    ),
    "VIEW": ButtonDef(
        id="VIEW",
        label="Frame View",
        kind=ButtonKind.TOOL,
        group="frame",
        press_style=PressStyle.CLICK,
        pref_attr="show_frame_view_button",
        icon="VIEW",
        action="VIEW",
    ),
    "ALL": ButtonDef(
        id="ALL",
        label="Frame All",
        kind=ButtonKind.TOOL,
        group="frame",
        press_style=PressStyle.CLICK,
        pref_attr="show_frame_all_button",
        icon="ALL",
        action="ALL",
    ),
    "LIST": ButtonDef(
        id="LIST",
        label="List Toggle",
        kind=ButtonKind.REGULAR,
        group="left",
        press_style=PressStyle.CLICK,
        pref_attr="show_list_toggle_button",
        icon="LIST",
        toggle_attr="show_type_list",
        menu_group="list",
        list_animation=True,
    ),
    "DRAG": ButtonDef(
        id="DRAG",
        label="Move Handle",
        kind=ButtonKind.REGULAR,
        group="right",
        press_style=PressStyle.DRAG,
        pref_attr="show_move_button",
        icon="DRAG",
        menu_group="move",
    ),
}

# Order defines the top-edge draw row (left to right), the cull priority when
# space is tight, and the deterministic paint order.
BUTTON_ORDER: tuple[str, ...] = ("SELECTED", "VIEW", "ALL", "LIST", "DRAG")

# Display order for the Buttons visibility submenu (frame entries
# most-used-first, unlike the draw row order above).
MENU_ORDER: tuple[str, ...] = ("LIST", "ALL", "VIEW", "SELECTED", "DRAG")

# Placement-zone order for grouped menus: list toggle, frame row, move handle.
GROUPS: tuple[str, ...] = ("left", "frame", "right")


def _visible_button_ids(settings) -> list[str]:
    """Return ids of enabled minimap buttons in draw order.

    Pure helper with no Blender dependency.
    """
    if not bool(getattr(settings, "use_interactive", False)):
        return []
    return [
        button_id
        for button_id in BUTTON_ORDER
        if getattr(settings, BUTTONS[button_id].pref_attr, True) and BUTTONS[button_id].visible_when(settings)
    ]


def _row_radii(index: int, count: int, radius: float) -> tuple[float, float, float, float]:
    """Return per-corner radii for a combined-row button (Blender align style).

    Only the row's external corners round; inner corners stay square. Order
    is top-left, top-right, bottom-right, bottom-left. Pure helper with no
    Blender dependency.
    """
    if index == 0:
        return (radius, 0.0, 0.0, radius)
    if index == count - 1:
        return (0.0, radius, radius, 0.0)
    return (0.0, 0.0, 0.0, 0.0)


def _cull_frame_ids(
    frame_ids: list[str],
    row_right_x: float,
    button_size: float,
    map_left_limit: float,
    list_right: float | None,
    gap: float,
    priority: tuple[str, ...],
) -> list[str]:
    """Return the frame ids that fit without overflowing or overlapping.

    When space is tight, hide buttons progressively in *priority* order.
    Pure helper with no Blender dependency.
    """
    # Work on a copy so culling never affects the caller's order.
    kept = list(frame_ids)
    while kept:
        count = len(kept)
        row_left_x = row_right_x - (count - 1) * button_size if count else row_right_x
        # 1) row would overflow left padding
        row_overflows_left = row_left_x < map_left_limit
        # 2) row would overlap the list toggle (with clearance)
        row_overlaps_left_chrome = list_right is not None and row_left_x < list_right + gap
        if not row_overflows_left and not row_overlaps_left_chrome:
            break
        # Hide next priority button that is still visible.
        to_hide: str | None = None
        for candidate in priority:
            if candidate in kept:
                to_hide = candidate
                break
        if to_hide is None:
            # Fallback: hide leftmost (first in current order).
            to_hide = kept[0]
        kept = [button_id for button_id in kept if button_id != to_hide]
    return kept
