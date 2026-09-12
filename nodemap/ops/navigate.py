"""Provide a modal operator for minimap interaction."""

import logging
import time

import blf
import bpy
from bpy.types import Area, Context, Event, Operator, Region, SpaceNodeEditor

from .. import __package__ as base_package
from ..core.constants import (
    DOCK_DWELL_MS,
    HANDLE_THICKNESS,
    INERTIA_MIN_SPEED,
    MAX_FRAME_ZOOM,
    MIN_FRAME_ZOOM,
    SCROLLBAR_HIT_PAD,
    SMOOTH_DAMP_STILL,
    TYPE_LIST_FONT_ID,
    WHEEL_ZOOM_IN,
    WHEEL_ZOOM_OUT,
)
from ..core.helpers import (
    _expand_bounds_margin,
    _get_area_and_region_under_mouse,
    _get_minimap_margins,
    _get_node_tree_bounds,
    _get_safe_bounds,
    _get_ui_scale,
    clamp_free_rect,
    get_addon_preferences,
    redraw_ui,
    start_list_width_animation,
)
from ..core.state import (
    MinimapState,
    ResizeHandle,
    _minimap_window_operators,
    _state,
)
from ..geo.framing import (
    _compute_frame_all_targets,
    _compute_frame_selected_targets,
    _compute_frame_to_bounds_targets,
    _frame_to_bounds,
    frame_all,
    frame_selected,
    frame_view,
)
from ..geo.transforms import (
    _clamp_pan_to_viewport,
    _compute_map_transform,
    _get_minimap_transform,
    _get_visible_rect,
)
from ..ui.menus import open_minimap_button_menu
from . import resize, selection
from .animations import AnimationController

logger = logging.getLogger(base_package)

# Printable characters accepted while typing a type-list search query.
# Key names match Blender's event enum (see rna_wm.cc): SEMI_COLON,
# ACCENT_GRAVE, LEFT_BRACKET, RIGHT_BRACKET, EQUAL.
_SEARCH_SPECIAL_KEYS: dict[str, str] = {
    "SPACE": " ",
    "PERIOD": ".",
    "COMMA": ",",
    "SLASH": "/",
    "SEMI_COLON": ";",
    "QUOTE": "'",
    "UNDERSCORE": "_",
    "MINUS": "-",
    "COLON": ":",
    "BACKSLASH": "\\",
    "LEFT_BRACKET": "[",
    "RIGHT_BRACKET": "]",
    "EQUAL": "=",
    "ACCENT_GRAVE": "`",
}

_SEARCH_DIGIT_KEYS: dict[str, str] = {
    "ZERO": "0",
    "ONE": "1",
    "TWO": "2",
    "THREE": "3",
    "FOUR": "4",
    "FIVE": "5",
    "SIX": "6",
    "SEVEN": "7",
    "EIGHT": "8",
    "NINE": "9",
}

_SEARCH_NUMPAD_KEYS: dict[str, str] = {
    "NUMPAD_0": "0",
    "NUMPAD_1": "1",
    "NUMPAD_2": "2",
    "NUMPAD_3": "3",
    "NUMPAD_4": "4",
    "NUMPAD_5": "5",
    "NUMPAD_6": "6",
    "NUMPAD_7": "7",
    "NUMPAD_8": "8",
    "NUMPAD_9": "9",
    "NUMPAD_PERIOD": ".",
    "NUMPAD_SLASH": "/",
    "NUMPAD_ASTERIX": "*",
    "NUMPAD_MINUS": "-",
    "NUMPAD_PLUS": "+",
}

_SEARCH_ACCEPTED: set[str] = set(" .,_;/:'-[]=`\\*+")


def _search_caret_at_x(query: str, click_x: int, text_start_x: float, font_size: int) -> int:
    """Return the search caret index nearest the click x within *query*.

    Uses the same font and text-left origin as the search-row drawing so the
    caret lands where the user clicked. Empty queries keep the caret at 0.
    """
    if not query:
        return 0
    blf.size(TYPE_LIST_FONT_ID, font_size)
    target = click_x - text_start_x
    return min(
        range(len(query) + 1),
        key=lambda i: abs(blf.dimensions(TYPE_LIST_FONT_ID, query[:i])[0] - target),
    )


def _event_char(event: Event) -> str | None:
    """Return the printable character for *event*, or None when not one.

    Blender reports character keys through ``event.type``: a single
    uppercase letter, a named digit (``ZERO``...``NINE``), a numpad key
    (``NUMPAD_0``...``NUMPAD_9``, ``NUMPAD_SLASH``, ...), or a named
    symbol key (``SPACE``, ``PERIOD``, ...). There is no ``event.char``
    attribute in Blender 5.2.
    """
    key_type = event.type
    if key_type in _SEARCH_NUMPAD_KEYS:
        return _SEARCH_NUMPAD_KEYS[key_type]
    if key_type in _SEARCH_DIGIT_KEYS:
        return _SEARCH_DIGIT_KEYS[key_type]
    if key_type in _SEARCH_SPECIAL_KEYS:
        return _SEARCH_SPECIAL_KEYS[key_type]
    if len(key_type) == 1 and key_type.isalpha():
        return key_type.lower()
    return None


# Mouse and frame-callback events that never get captured by the search box.
# While the box is focused these reach the mouse handlers, where the first
# press outside the search zone only blurs and is swallowed.
_SEARCH_PASSTHROUGH: set[str] = {
    "LEFTMOUSE",
    "RIGHTMOUSE",
    "MIDDLEMOUSE",
    "MOUSEMOVE",
    "MOUSEPAN",
    "MOUSEROTATE",
    "WHEELUPMOUSE",
    "WHEELDOWNMOUSE",
    "TIMER",
    "INBETWEEN_MOUSEMOVE",
}


def _is_search_keyboard_event(event: Event) -> bool:
    """Return True when *event* should be captured by the focused search box.

    Keyboard events are captured while the box is focused so they never reach
    the Node Editor; mouse events reach the normal handlers, where the first
    press outside the search zone blurs and is swallowed.
    """
    return event.type not in _SEARCH_PASSTHROUGH


def _is_in_minimap(region_x: int, region_y: int, state: MinimapState | None = None) -> bool:
    if state is None:
        state = _state()
    map_x, map_y, map_w, map_h = state.view.rect
    return map_x <= region_x <= map_x + map_w and map_y <= region_y <= map_y + map_h


def _region_to_tree(region_x: int, region_y: int, state: MinimapState | None = None) -> tuple[float, float] | None:
    if state is None:
        state = _state()
    if not state.view.rect or not state.view.tree_bounds:
        return None
    return _tree_from_region(region_x, region_y, _compute_map_transform(state))


def _tree_from_region(
    region_x: int, region_y: int, transform: tuple[float, float, float, float, float]
) -> tuple[float, float] | None:
    """Inverse-map a minimap pixel coordinate to tree space using a precomputed transform."""
    map_anchor_x, map_anchor_y, scale, tree_center_x, tree_center_y = transform
    if scale <= 0:
        return None
    return tree_center_x + (region_x - map_anchor_x) / scale, tree_center_y + (region_y - map_anchor_y) / scale


def _view_zoom_factors(space, region, visible: tuple[float, float, float, float] | None = None) -> tuple[float, float]:
    """Return pixels-per-tree-unit for each axis given the editor's visible rect."""
    if visible is None:
        visible = _get_visible_rect(space, region)
    if not visible:
        return 1.0, 1.0
    visible_w = max(visible[2] - visible[0], 1e-6)
    visible_h = max(visible[3] - visible[1], 1e-6)
    return region.width / visible_w, region.height / visible_h


def _frame_button_at(mouse_x: int, mouse_y: int, state: MinimapState) -> str | None:
    """Return the id of the frame button under the cursor, if any."""
    for frame_button_id, button_rect in state.buttons.rects.items():
        if button_rect:
            button_x, button_y, button_width, button_height = button_rect
            if button_x <= mouse_x <= button_x + button_width and button_y <= mouse_y <= button_y + button_height:
                return frame_button_id
    return None


def _in_list_zone(region_x: int, region_y: int, state: MinimapState) -> bool:
    """Return True when the cursor is over the type-list zone of the minimap."""
    if state.list.list_width <= 0 or not state.view.rect:
        return False
    zone_rect = state.list.list_zone_rect
    if not zone_rect:
        # Fallback for the first frame before the zone rect is recorded:
        # assume the legacy left-edge placement.
        map_x, map_y, _, map_h = state.view.rect
        hit_pad = HANDLE_THICKNESS * _get_ui_scale()
        zone_w = state.view.inner_padding + state.list.list_width
        zone_rect = (map_x + hit_pad, map_y + hit_pad, zone_w, map_h - 2 * hit_pad)
    zone_x, zone_y, zone_w, zone_h = zone_rect
    return zone_x <= region_x <= zone_x + zone_w and zone_y <= region_y <= zone_y + zone_h


def _list_row_at(region_x: int, region_y: int, state: MinimapState) -> str | None:
    """Return the type label of the type-list row under the cursor, if any."""
    for x, y, w, h, label in state.list.row_rects:
        if x <= region_x <= x + w and y <= region_y <= y + h:
            return label
    return None


def _list_child_at(region_x: int, region_y: int, state: MinimapState) -> tuple[str, str] | None:
    """Return ``(label, node_name)`` of the expanded child row under the cursor."""
    for x, y, w, h, label, node_name in state.list.node_rects:
        if x <= region_x <= x + w and y <= region_y <= y + h:
            return label, node_name
    return None


def _in_rect(region_x: int, region_y: int, rect: tuple[float, float, float, float]) -> bool:
    """Return True when the cursor falls inside the ``(x, y, w, h)`` rect."""
    rect_x, rect_y, rect_w, rect_h = rect
    return rect_x <= region_x <= rect_x + rect_w and rect_y <= region_y <= rect_y + rect_h


def _over_search_zone(region_x: int, region_y: int, state: MinimapState) -> bool:
    """Return True when the cursor is over the search box zone.

    The clear button lives inside the search pill, so a single
    ``search_rect`` hit test covers the box, the text, and the button.
    """
    search_rect = state.list.search_rect
    return bool(search_rect and _in_list_zone(region_x, region_y, state) and _in_rect(region_x, region_y, search_rect))


def _clear_search_stale_hover(state: MinimapState) -> None:
    """Clear outside hover state so no highlight lingers under the I-beam."""
    state.list.hovered_type_label = None
    state.list.hovered_list_row = None
    state.interaction.hovered_node_id = None
    state.buttons.hovered_button_id = None
    state.list.hovered_scrollbar = False
    state.interaction.hovered_handle = None
    state.buttons.pressed_button_id = None


def _list_scrollbar_hit(region_x: int, region_y: int, state: MinimapState) -> bool:
    """Return True when the cursor is over the type-list scrollbar gutter."""
    scrollbar_track = state.list.scrollbar_track
    if not scrollbar_track or state.list.scroll_max <= 0:
        return False
    x, y, w, h = scrollbar_track
    hit_pad = SCROLLBAR_HIT_PAD * _get_ui_scale()
    # The scrollbar sits inside the zone, so the right hit pad must not bleed
    # into the list/map divider band where the LIST resize handle owns hover.
    right = x + w + hit_pad
    zone_rect = state.list.list_zone_rect
    if zone_rect:
        right = min(right, zone_rect[0] + zone_rect[2])
    return x - hit_pad <= region_x <= right and y <= region_y <= y + h


def _apply_list_scroll_drag(mouse_x: int, mouse_y: int, grab: float, state: MinimapState) -> None:
    """Scroll the type list so the dragged thumb tracks the cursor.

    *grab* is the cursor-to-thumb-top distance captured at press; mapping the
    thumb top back to a track fraction keeps the grab point stable.
    """
    scrollbar_track = state.list.scrollbar_track
    scrollbar_thumb = state.list.scrollbar_thumb
    if not scrollbar_track or not scrollbar_thumb or state.list.scroll_max <= 0:
        return
    track_x, track_y, track_w, track_len = scrollbar_track
    thumb_length = scrollbar_thumb[3]
    track_span = max(track_len - thumb_length, 1.0)
    scroll_offset = min(max(mouse_y + grab - thumb_length - track_y, 0.0), track_span)
    state.list.scroll = (1.0 - scroll_offset / track_span) * state.list.scroll_max


_CURSOR_MAP: dict[ResizeHandle, str] = {
    ResizeHandle.LEFT: "MOVE_X",
    ResizeHandle.RIGHT: "MOVE_X",
    ResizeHandle.TOP: "MOVE_Y",
    ResizeHandle.BOTTOM: "MOVE_Y",
    ResizeHandle.TOP_LEFT: "SCROLL_XY",
    ResizeHandle.TOP_RIGHT: "SCROLL_XY",
    ResizeHandle.BOTTOM_LEFT: "SCROLL_XY",
    ResizeHandle.BOTTOM_RIGHT: "SCROLL_XY",
    ResizeHandle.LIST: "MOVE_X",
}


class NODEMAP_OT_toggle(Operator):
    """Display the minimap overlay."""

    bl_idname = "nodemap.toggle"
    bl_label = "Show Nodemap"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        state = _state()
        state.enabled = not state.enabled
        if not state.enabled:
            win = context.window
            if win:
                op = _minimap_window_operators.get(win.as_pointer())
                if op:
                    op._cancel_interaction(context)
        redraw_ui("NODE_EDITOR")
        return {"FINISHED"}


class NODEMAP_OT_restore_keymap(Operator):
    """Restore the default Nodemap keymap shortcut."""

    bl_idname = "nodemap.restore_keymap"
    bl_label = "Restore Default Shortcut"
    bl_options = {"INTERNAL"}

    @classmethod
    def poll(cls, context: Context) -> bool:
        """Return True when the toggle shortcut is missing and can be restored."""
        window_manager = context.window_manager
        user_keyconfig = window_manager.keyconfigs.user
        if not user_keyconfig:
            return False
        node_editor_keymap = user_keyconfig.keymaps.get("Node Editor")
        if node_editor_keymap:
            return node_editor_keymap.keymap_items.get("nodemap.toggle") is None
        return True

    def execute(self, context: Context) -> set[str]:
        window_manager = context.window_manager
        user_keyconfig = window_manager.keyconfigs.user
        node_editor_keymap = user_keyconfig.keymaps.get("Node Editor")
        if not node_editor_keymap:
            node_editor_keymap = user_keyconfig.keymaps.new(name="Node Editor", space_type="NODE_EDITOR")
        node_editor_keymap.keymap_items.new("nodemap.toggle", type="M", value="PRESS", ctrl=True)
        return {"FINISHED"}


def _try_animated_frame(context: Context, button_id: str) -> bool:
    """Run a frame action eased via the navigate modal. Return True when handled.

    Route standalone frame operators through the same animated dispatch the
    minimap buttons use. Return False when no animation applies so the caller
    falls back to the instant frame function.
    """
    window = context.window
    area = context.area
    if window is None or area is None or area.type != "NODE_EDITOR":
        return False
    op = _minimap_window_operators.get(window.as_pointer())
    if op is None or getattr(op, "_anim", None) is None:
        return False
    if not op._anim._animations_enabled(context):
        return False
    addon = get_addon_preferences(context)
    settings = addon.settings if addon else None
    if settings is None:
        return False
    space = area.spaces.active
    if space is None or getattr(space, "type", None) != "NODE_EDITOR":
        return False
    region = context.region
    if region is None or getattr(region, "type", None) != "WINDOW" or not hasattr(region, "view2d"):
        region = next((r for r in area.regions if r.type == "WINDOW"), None)
    if region is None:
        return False
    op._area = area
    op._region = region
    op._space = space
    op._state = _state(area.as_pointer())
    op._dispatch_frame_action(context, settings, button_id)
    op._redraw_ui()
    return True


class NODEMAP_OT_frame_all(Operator):
    """Reset the minimap view to show all nodes."""

    bl_idname = "nodemap.frame_all"
    bl_label = "Frame All"
    bl_description = "Reset the minimap view to show all nodes.\nShortcut: Home"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        if _try_animated_frame(context, "ALL"):
            return {"FINISHED"}
        frame_all()
        return {"FINISHED"}


class NODEMAP_OT_frame_selected(Operator):
    """Focus the minimap view on selected nodes."""

    bl_idname = "nodemap.frame_selected"
    bl_label = "Frame Selected"
    bl_description = "Focus the minimap view on selected nodes.\nShortcut: Numpad ."
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        if _try_animated_frame(context, "SELECTED"):
            return {"FINISHED"}
        frame_selected()
        return {"FINISHED"}


class NODEMAP_OT_frame_view(Operator):
    """Focus the minimap view on the current editor viewport."""

    bl_idname = "nodemap.frame_view"
    bl_label = "Frame View"
    bl_description = "Focus the minimap view on the current editor viewport.\nShortcut: End"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        if _try_animated_frame(context, "VIEW"):
            return {"FINISHED"}
        frame_view()
        return {"FINISHED"}


class NODEMAP_OT_navigate(Operator):
    """Navigate the Node Editor view via the minimap."""

    bl_idname = "nodemap.navigate"
    bl_label = "Nodemap Navigate"
    bl_options = {"INTERNAL"}

    _drag_start: tuple[int, int] | None = None
    _window_ptr: int = 0
    _dragging: bool = False
    _was_in_minimap: bool = False

    _drag_mode: str | None = None
    _click_action: str | None = None
    _click_extend: bool = False
    _click_toggle: bool = False

    _mmb_dragging: bool = False
    _mmb_drag_start: tuple[int, int] | None = None

    _mx: int = 0
    _my: int = 0
    _marquee_dragging: bool = False
    _state: MinimapState | None = None
    _area: Area | None = None
    _region: Region | None = None
    _space: SpaceNodeEditor | None = None

    _resize_handle: str | None = None
    _resize_start_mouse: tuple[int, int] | None = None
    _resize_start_values: tuple[int, int] | None = None
    _resize_start_offset: tuple[int, int] | None = None
    _list_width_dragging: bool = False
    _moving: bool = False
    _move_start_mouse: tuple[int, int] | None = None
    _move_start_offset: tuple[int, int] | None = None
    _snap_candidate: tuple[float, float, str] | None = None
    _snap_dwell: float = 0.0
    _snap_last_time: float = 0.0
    _list_width_start_x: int = 0
    _list_width_start_y: int = 0
    _list_width_start_px: int = 160
    _last_cursor: str = ""
    _pan_acc: list[float]
    _redirect_acc: list[float]
    _armed_button: str | None = None
    _list_row_pressed: str | None = None
    _list_child_pressed: tuple[str, str] | None = None
    _list_toggle_pressed: str | None = None
    _list_scroll_pressed: bool = False
    _list_scroll_grab: float = 0.0
    _list_search_pressed: bool = False
    _list_search_clear_pressed: bool = False
    _search_blur_consumed: bool = False
    _context_menu_button: str | None = None
    _list_mmb_dragging: bool = False
    _list_mmb_drag_start: tuple[int, int] | None = None
    _list_last_row_index: int = -1

    _anim: AnimationController

    def _override_ctx(self, context: Context):
        return context.temp_override(
            area=self._area,
            region=self._region,
            space_data=self._space,
        )

    def _redraw_ui(self) -> None:
        """Redraw only the Node Editor area this operator is interacting with."""
        area_ptr = self._area.as_pointer() if self._area else None
        redraw_ui("NODE_EDITOR", area_ptr)

    def _blur_search(self, context: Context, state: MinimapState) -> None:
        """Blur the search box without arming any click guard.

        Shared by keyboard blurs (Enter, Esc) which have no matching button
        release to swallow. Clears stale outside hover and restores the
        default cursor so no I-beam lingers after focus is lost.
        """
        state.list.search_focused = False
        state.list.search_cursor = 0
        _clear_search_stale_hover(state)
        if context.window:
            context.window.cursor_modal_set("DEFAULT")
        self._last_cursor = ""
        self._redraw_ui()

    def _blur_search_consume(self, context: Context, state: MinimapState) -> None:
        """Blur the search box and swallow the current outside click.

        Arms the release guard so the matching button release is consumed
        too.
        """
        self._blur_search(context, state)
        self._search_blur_consumed = True
        self._redraw_ui()

    def _consume_search_blur_press(self, context: Context, state: MinimapState) -> bool:
        """Blur on an outside press and swallow it. Return True when consumed."""
        if state.list.search_focused and not _over_search_zone(self._mouse_x, self._mouse_y, state):
            self._blur_search_consume(context, state)
            return True
        return False

    def modal(self, context: Context, event: Event) -> set[str]:
        if not context.window:
            return {"CANCELLED"}
        window_ptr = context.window.as_pointer()
        if _minimap_window_operators.get(window_ptr) is not self:
            return {"CANCELLED"}

        is_interactive = (
            self._dragging
            or self._mmb_dragging
            or self._list_mmb_dragging
            or self._marquee_dragging
            or self._resize_handle is not None
            or self._list_width_dragging
            or self._moving
            or self._drag_start is not None
            or self._list_scroll_pressed
            or self._search_blur_consumed
            or self._anim.anim_active
            or self._anim.inertia_active
            or self._anim.drag_active
            or self._anim.frame_anim_active
            or self._anim.editor_anim_active
        )

        if not is_interactive:
            under_mouse_area, under_mouse_region = _get_area_and_region_under_mouse(context, event)
            if not under_mouse_area or under_mouse_area.type != "NODE_EDITOR" or not under_mouse_region:
                if self._state and self._state.interaction.hovered_minimap:
                    self._state.interaction.hovered_minimap = False
                    self._redraw_ui()
                # The cursor left the Node Editor (e.g. over a popup, menu, or
                # header): drop any pending click/drag arming so a later
                # release outside the minimap cannot fire a stale gesture.
                # An actively held drag keeps its state so it can complete.
                if not (
                    self._dragging
                    or self._marquee_dragging
                    or self._moving
                    or self._mmb_dragging
                    or self._list_mmb_dragging
                    or self._list_scroll_pressed
                    or self._resize_handle is not None
                    or self._list_width_dragging
                ):
                    self._was_in_minimap = False
                    self._drag_start = None
                    self._reset_gesture()
                self._state = None
                self._area = None
                self._region = None
                self._space = None
                return {"PASS_THROUGH"}
            self._state = _state(under_mouse_area.as_pointer())
            self._area = under_mouse_area
            self._region = under_mouse_region
            self._space = under_mouse_area.spaces.active
            _clamp_pan_to_viewport(self._space, self._region, self._state)

        if not self._state or not self._state.enabled:
            return {"PASS_THROUGH"}

        if self._space and not self._space.overlay.show_overlays:
            if is_interactive:
                self._cancel_interaction(context)
            return {"PASS_THROUGH"}

        if self._region is not None:
            self._mouse_x = event.mouse_x - self._region.x
            self._mouse_y = event.mouse_y - self._region.y
        else:
            self._mouse_x = event.mouse_x
            self._mouse_y = event.mouse_y

        addon = get_addon_preferences(context)
        settings = addon.settings if addon else None
        if addon and not settings.use_interactive:
            return {"PASS_THROUGH"}

        state = self._state
        in_minimap = _is_in_minimap(self._mouse_x, self._mouse_y, state)
        if state.interaction.hovered_minimap != in_minimap:
            state.interaction.hovered_minimap = in_minimap
            self._redraw_ui()

        # Type-list search: while focused, swallow keyboard events so keystrokes
        # never leak into the Node Editor (rename, tab, etc.). Mouse events
        # reach the normal handlers, where the first press outside the search
        # zone only blurs and is swallowed.
        if state.list.search_focused and _is_search_keyboard_event(event):
            return self._handle_list_search(context, event)

        # Second Esc over the minimap clears the filter text. The first Esc
        # (handled above) only blurs and arms this; repeats of a held key are
        # swallowed without clearing so clearing takes a distinct press.
        if (
            event.type == "ESC"
            and event.value == "PRESS"
            and not state.list.search_focused
            and state.list.search_esc_armed
            and state.list.search_query
            and in_minimap
        ):
            if not event.is_repeat:
                state.list.search_query = ""
                state.list.search_cursor = 0
                state.list.search_esc_armed = False
                self._redraw_ui()
            return {"RUNNING_MODAL"}

        # Ctrl+F over the minimap: reveal the type list (if hidden) and focus
        # its filter field. Ignored when the filter bar is hidden so the key
        # passes through to the editor.
        if (
            event.type == "F"
            and event.value == "PRESS"
            and in_minimap
            and event.ctrl
            and not (event.shift or event.alt)
            and (settings is None or settings.show_search_bar)
        ):
            if not settings.show_type_list:
                settings.show_type_list = True
                start_list_width_animation(state, settings)
            state.list.search_focused = True
            state.list.search_cursor = len(state.list.search_query)
            state.list.search_esc_armed = False
            _clear_search_stale_hover(state)
            # Entering text-input mode: show the I-beam so the pointer
            # indicates typing, like clicking the filter field.
            context.window.cursor_modal_set("TEXT")
            self._last_cursor = "TEXT"
            self._redraw_ui()
            return {"RUNNING_MODAL"}

        match event.type:
            case "LEFTMOUSE":
                return self._handle_left_mouse(context, event)

            case "RIGHTMOUSE":
                return self._handle_right_mouse(context, event)

            case "MIDDLEMOUSE":
                if event.value == "PRESS" and in_minimap:
                    if self._consume_search_blur_press(context, state):
                        return {"RUNNING_MODAL"}
                    state.interaction.pressed = True
                    self._anim.cancel_smooth(context)
                    if _in_list_zone(self._mouse_x, self._mouse_y, state):
                        self._list_mmb_dragging = True
                        self._list_mmb_drag_start = (self._mouse_x, self._mouse_y)
                    else:
                        self._mmb_dragging = True
                        self._mmb_drag_start = (self._mouse_x, self._mouse_y)
                    return {"RUNNING_MODAL"}
                if event.value == "RELEASE" and self._search_blur_consumed:
                    self._search_blur_consumed = False
                    return {"RUNNING_MODAL"}
                if event.value == "RELEASE" and self._list_mmb_dragging:
                    state.interaction.pressed = False
                    self._redraw_ui()
                    self._list_mmb_dragging = False
                    self._list_mmb_drag_start = None
                    return {"RUNNING_MODAL"}
                if event.value == "RELEASE" and self._mmb_dragging:
                    state.interaction.pressed = False
                    self._redraw_ui()
                    self._mmb_dragging = False
                    self._mmb_drag_start = None
                    _clamp_pan_to_viewport(self._space, self._region, state)
                    if self._anim._animations_enabled(context):
                        speed = max(abs(self._anim.smooth_velocity[0]), abs(self._anim.smooth_velocity[1]))
                        if speed > INERTIA_MIN_SPEED:
                            self._anim.inertia_active = True
                            self._anim.inertia_mode = "PAN"
                            self._anim.create_timer(context)
                            self._redirect_acc = [0.0, 0.0]
                            return {"RUNNING_MODAL"}
                    self._anim.smooth_velocity = [0.0, 0.0]
                    self._redirect_acc = [0.0, 0.0]
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "MOUSEMOVE":
                return self._handle_mouse_move(context, event)

            case "WHEELUPMOUSE" | "WHEELDOWNMOUSE":
                return self._handle_wheel(context, event)

            case "HOME":
                if event.value == "PRESS" and in_minimap and not event.shift:
                    self._dispatch_frame_action(context, settings, "ALL")
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "END":
                if event.value == "PRESS" and in_minimap:
                    self._dispatch_frame_action(context, settings, "VIEW")
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "NUMPAD_PERIOD":
                if event.value == "PRESS" and in_minimap:
                    if _in_list_zone(self._mouse_x, self._mouse_y, state):
                        self._dispatch_frame_action(context, settings, "SELECTED", scope="LIST")
                    else:
                        self._dispatch_frame_action(context, settings, "SELECTED", scope="MAP")
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "T":
                if event.value == "PRESS" and in_minimap and not (event.ctrl or event.shift or event.alt):
                    settings.show_type_list = not settings.show_type_list
                    start_list_width_animation(state, settings)
                    self._redraw_ui()
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "UP_ARROW" | "DOWN_ARROW":
                if (
                    event.value == "PRESS"
                    and _in_list_zone(self._mouse_x, self._mouse_y, state)
                    and not (event.ctrl or event.shift or event.alt)
                    and settings is not None
                    and settings.show_type_list
                    and state.list.list_width > 0
                    and selection.handle_list_arrow(
                        self, context, state, settings, -1 if event.type == "UP_ARROW" else 1
                    )
                ):
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "LEFT_ARROW" | "RIGHT_ARROW":
                if (
                    event.value == "PRESS"
                    and _in_list_zone(self._mouse_x, self._mouse_y, state)
                    and not (event.ctrl or event.shift or event.alt)
                    and settings is not None
                    and settings.show_type_list
                    and state.list.list_width > 0
                    and selection.handle_list_expand(self, context, state, settings, expand=event.type == "RIGHT_ARROW")
                ):
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "A":
                if (
                    event.value == "PRESS"
                    and _in_list_zone(self._mouse_x, self._mouse_y, state)
                    and event.shift
                    and not (event.ctrl or event.alt)
                    and settings is not None
                    and settings.show_type_list
                    and state.list.list_width > 0
                    and selection.handle_list_toggle_all(self, context, state, settings)
                ):
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "TIMER":
                if self._anim.drag_active:
                    self._anim.apply_smooth_drag(context)
                    return {"RUNNING_MODAL"}
                if self._anim.inertia_active:
                    self._anim.apply_inertia(context)
                    return {"RUNNING_MODAL"}
                if self._anim.frame_anim_active:
                    self._anim.apply_frame_animation(context)
                    return {"RUNNING_MODAL"}
                if self._anim.editor_anim_active:
                    self._anim.apply_editor_animation(context)
                    return {"RUNNING_MODAL"}
                if self._anim.anim_active:
                    self._anim.apply_center_animation(context)
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}
            case _:
                return {"PASS_THROUGH"}

    def _minimap_event_context(self, context: Context):
        """Resolve the shared per-event values used by the event handlers.

        Return the minimap state, the add-on (if registered), its settings, and
        whether the cursor is currently over the minimap.
        """
        state = self._state
        addon = get_addon_preferences(context)
        settings = addon.settings if addon else None
        in_minimap = _is_in_minimap(self._mouse_x, self._mouse_y, state) if state else False
        return state, addon, settings, in_minimap

    def _resolve_gesture(self, event: Event, settings, side: str) -> tuple[str, str, bool, bool]:
        """Resolve modifier overrides into (drag, click, extend, toggle).

        An unmodified gesture uses the configured actions. Shift, Ctrl, and
        Alt are stateless overrides so every drag action stays reachable on
        either button: Shift frames a region, Ctrl pans the view, and Alt
        frames a region in the editor. Shift and Ctrl also turn a click into
        an extend or toggle select. Captured at press so the release finishes
        the gesture the user started.
        """
        if side == "LEFT":
            drag_action = settings.left_drag_action
            click_action = settings.left_click_action
        else:
            drag_action = settings.right_drag_action
            click_action = settings.right_click_action
        if event.shift:
            return "FRAME_RECT", "SELECT", True, False
        if event.ctrl:
            return "PAN", "SELECT", False, True
        if event.alt:
            return "FRAME_RECT_EDITOR", click_action, False, False
        return drag_action, click_action, False, False

    def _reset_gesture(self) -> None:
        """Clear the resolved gesture captured at press."""
        self._drag_mode = None
        self._click_action = None
        self._click_extend = False
        self._click_toggle = False

    def _run_click_action(self, context: Context, state: MinimapState) -> None:
        """Run the resolved click action for a no-drag release."""
        action = self._click_action
        if action is None:
            return
        if action in ("SELECT", "SELECT_PAN", "SELECT_FRAME"):
            state.request_immediate_compile()
            selection.handle_click_selection(
                self,
                context,
                state,
                frame=action == "SELECT_FRAME",
                extend=self._click_extend,
                toggle=self._click_toggle,
            )
        if action in ("PAN", "SELECT_PAN"):
            if self._drag_mode == "CENTER_PAN":
                # The press already centered on nearly the same point, so a
                # second center would only restart the animation.
                return
            self._center_view_on_mouse(context, self._mouse_x, self._mouse_y)
            state.interaction.pressed = False

    def _handle_list_search(self, context: Context, event: Event) -> set[str]:
        """Handle key input while the type-list search box is focused.

        Typed characters (letters, digits, and a few common separators) are
        inserted at the caret; Left/Right/Home/End move the caret, Backspace
        deletes before it and Delete after it. Enter blurs (keeping the
        filter); the first Esc blurs (keeping the filter) and arms clearing,
        so a second Esc over the minimap clears the query. Every other event
        is swallowed so it never reaches the Node Editor.
        """
        state = self._state
        if event.value != "PRESS":
            return {"RUNNING_MODAL"}

        query = state.list.search_query
        cursor = min(max(state.list.search_cursor, 0), len(query))
        key = event.type

        if key == "ESC":
            self._blur_search(context, state)
            state.list.search_esc_armed = bool(state.list.search_query)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if key in ("RET", "NUMPAD_ENTER"):
            self._blur_search(context, state)
            state.list.search_esc_armed = False
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if key == "LEFT_ARROW":
            if cursor > 0:
                state.list.search_cursor = cursor - 1
                self._redraw_ui()
            return {"RUNNING_MODAL"}
        if key == "RIGHT_ARROW":
            if cursor < len(query):
                state.list.search_cursor = cursor + 1
                self._redraw_ui()
            return {"RUNNING_MODAL"}
        if key == "HOME":
            if cursor != 0:
                state.list.search_cursor = 0
                self._redraw_ui()
            return {"RUNNING_MODAL"}
        if key == "END":
            if cursor != len(query):
                state.list.search_cursor = len(query)
                self._redraw_ui()
            return {"RUNNING_MODAL"}
        if key in ("BACK_SPACE", "BACKSPACE"):
            if cursor > 0:
                state.list.search_query = query[: cursor - 1] + query[cursor:]
                state.list.search_cursor = cursor - 1
                self._redraw_ui()
            return {"RUNNING_MODAL"}
        if key == "DEL":
            if cursor < len(query):
                state.list.search_query = query[:cursor] + query[cursor + 1 :]
                self._redraw_ui()
            return {"RUNNING_MODAL"}

        char = _event_char(event)
        if char is not None and (char.isalnum() or char in _SEARCH_ACCEPTED):
            state.list.search_query = (query[:cursor] + char + query[cursor:])[:64]
            state.list.search_cursor = min(cursor + 1, 64)
            self._redraw_ui()
        return {"RUNNING_MODAL"}

    def _handle_left_mouse(self, context: Context, event: Event) -> set[str]:
        state, addon, settings, in_minimap = self._minimap_event_context(context)
        # --- Release ---
        if event.value == "RELEASE":
            if self._search_blur_consumed:
                self._search_blur_consumed = False
                return {"RUNNING_MODAL"}
            if self._moving:
                self._moving = False
                self._move_start_mouse = None
                self._move_start_offset = None
                self._snap_candidate = None
                self._snap_dwell = 0.0
                state.view.moving = False
                state.view.snapped = False
                state.buttons.pressed_button_id = None
                state.cache.invalidate_batches_only()
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._list_width_dragging:
                self._list_width_dragging = False
                state.list.dragging_width = None
                state.list.width_clamped = False
                state.interaction.resize_active = None
                state.interaction.hovered_handle = None
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
                state.cache.invalidate_batches_only()
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if state.interaction.pressed:
                state.interaction.pressed = False
                self._redraw_ui()
            if self._list_scroll_pressed:
                self._list_scroll_pressed = False
                state.list.scrollbar_dragging = False
                self._list_scroll_grab = 0.0
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._list_search_clear_pressed:
                self._list_search_clear_pressed = False
                clear_rect = state.list.search_clear_rect
                if clear_rect and _in_rect(self._mouse_x, self._mouse_y, clear_rect):
                    state.list.search_query = ""
                    state.list.search_cursor = 0
                    state.list.search_focused = False
                    state.list.search_esc_armed = False
                    state.list.search_clear_rect = None
                    state.list.search_clear_hovered = False
                    self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._list_search_pressed:
                self._list_search_pressed = False
                search_rect = state.list.search_rect
                if (
                    search_rect
                    and _in_list_zone(self._mouse_x, self._mouse_y, state)
                    and _in_rect(self._mouse_x, self._mouse_y, search_rect)
                ):
                    if state.list.search_focused:
                        # Clicking the focused box again moves the caret to the
                        # click position (like a normal text field) instead of
                        # clearing the query.
                        state.list.search_cursor = _search_caret_at_x(
                            state.list.search_query,
                            self._mouse_x,
                            state.list.search_text_start_x,
                            settings.type_list_font_size,
                        )
                    else:
                        state.list.search_cursor = len(state.list.search_query)
                        state.list.search_focused = True
                        state.list.search_esc_armed = False
                        _clear_search_stale_hover(state)
                        if context.window:
                            context.window.cursor_modal_set("TEXT")
                        self._last_cursor = "TEXT"
                    self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._armed_button:
                # Clear the pressed fill and the hover that may now be stale: a
                # toggled button (e.g. the list) moves away from the cursor, so
                # the cached hover would linger until the next mouse move.
                state.buttons.pressed_button_id = None
                state.buttons.hovered_button_id = None
                self._activate_armed_button(context, settings)
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._list_child_pressed:
                # The selection already ran on press; just consume the release
                # so it does not fall through to the map click handler.
                self._list_child_pressed = None
                return {"RUNNING_MODAL"}
            if self._list_toggle_pressed:
                label = self._list_toggle_pressed
                self._list_toggle_pressed = None
                toggle = state.list.toggle_rects.get(label)
                if (
                    toggle
                    and _in_list_zone(self._mouse_x, self._mouse_y, state)
                    and _in_rect(self._mouse_x, self._mouse_y, toggle)
                ):
                    if label in state.list.expanded:
                        state.list.expanded.discard(label)
                    else:
                        state.list.expanded.add(label)
                    state.cache.list_key = None
                    state.request_immediate_compile()
                    self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._list_row_pressed:
                # The selection already ran on press; just consume the release
                # so it does not fall through to the map click handler.
                self._list_row_pressed = None
                return {"RUNNING_MODAL"}
            if self._resize_handle:
                self._resize_handle = None
                self._resize_start_mouse = None
                self._resize_start_values = None
                self._resize_start_offset = None
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
                state.view.width_clamped = False
                state.view.height_clamped = False
                state.interaction.hovered_handle = None
                state.interaction.resize_active = None
                state.cache.invalidate_batches_only()
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._marquee_dragging:
                self._marquee_dragging = False
                start = state.interaction.marquee_start
                end = state.interaction.marquee_end
                state.interaction.marquee_active = False
                state.interaction.marquee_start = None
                state.interaction.marquee_end = None
                framed = False
                if start is not None and end is not None:
                    if self._drag_mode == "FRAME_RECT_EDITOR":
                        framed = self._frame_marquee_rect_editor(context, state, start, end)
                    else:
                        framed = self._frame_marquee_rect(context, state, start, end)
                if framed:
                    self._was_in_minimap = False
                    self._drag_start = None
                    self._reset_gesture()
                    self._redraw_ui()
                    return {"RUNNING_MODAL"}
            if self._dragging:
                self._dragging = False
                self._drag_start = None
                self._was_in_minimap = False
                self._reset_gesture()
                if self._anim.drag_active:
                    self._pan_acc[0] += self._anim.drag_target[0]
                    self._pan_acc[1] += self._anim.drag_target[1]
                    self._anim.drag_target = [0.0, 0.0]
                    self._anim.drag_active = False
                if self._anim._animations_enabled(context):
                    speed = max(abs(self._anim.smooth_velocity[0]), abs(self._anim.smooth_velocity[1]))
                    if speed > INERTIA_MIN_SPEED:
                        self._anim.inertia_active = True
                        self._anim.inertia_mode = "VIEW"
                        if not self._anim.smooth_timer:
                            self._anim.create_timer(context)
                        return {"RUNNING_MODAL"}
                self._anim.smooth_velocity = [0.0, 0.0]
                pan_x = int(self._pan_acc[0])
                pan_y = int(self._pan_acc[1])
                self._pan_acc = [0.0, 0.0]
                if pan_x != 0 or pan_y != 0:
                    try:
                        with self._override_ctx(context):
                            bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                    except RuntimeError:
                        pass
                self._anim.destroy_timer(context)
                return {"RUNNING_MODAL"}
            if not self._dragging and self._was_in_minimap:
                # A click only acts when both press and release land inside
                # the minimap: a press swallowed elsewhere (e.g. by a popup)
                # must never let its release pan or select, and a release
                # outside belongs to the editor, so it passes through.
                self._was_in_minimap = False
                self._drag_start = None
                if self._click_action is not None and in_minimap:
                    self._run_click_action(context, state)
                    self._reset_gesture()
                    return {"RUNNING_MODAL"}
                self._reset_gesture()
                return {"PASS_THROUGH"}
            self._was_in_minimap = False
            self._drag_start = None
            self._reset_gesture()
            return {"PASS_THROUGH"}
        # --- Press ---
        # Synthetic follow-ups (CLICK) carry no new user intent: only PRESS
        # and DOUBLE_CLICK may arm a gesture, so a stray CLICK can never
        # leave a pending drag or click behind for a later release to fire.
        if event.value not in ("PRESS", "DOUBLE_CLICK"):
            self._was_in_minimap = False
            self._drag_start = None
            self._reset_gesture()
            return {"PASS_THROUGH"}
        # While the search box is focused, the first click outside the search
        # zone only blurs and is swallowed, so it never arms buttons, rows,
        # drags, or resizes. Clicking the box itself keeps focus (below).
        if self._consume_search_blur_press(context, state):
            return {"RUNNING_MODAL"}
        self._was_in_minimap = in_minimap
        if self._was_in_minimap:
            self._anim.cancel_smooth(context)
            # Move-grip drag: reposition the whole minimap freely.
            if resize.get_drag_handle(state, self._mouse_x, self._mouse_y):
                self._start_move(context, state, settings)
                return {"RUNNING_MODAL"}
            armed_button_id = _frame_button_at(self._mouse_x, self._mouse_y, state)
            if armed_button_id:
                self._armed_button = armed_button_id
                state.buttons.pressed_button_id = armed_button_id
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            # List/map divider — same style as outer resize borders, percent width.
            ui_scale = _get_ui_scale()
            divider_resize_handle = resize.get_list_divider_handle(state, self._mouse_x, self._mouse_y, ui_scale)
            if divider_resize_handle:
                self._list_width_dragging = True
                state.interaction.resize_active = divider_resize_handle
                self._redraw_ui()
                self._list_width_start_x = self._mouse_x
                self._list_width_start_y = self._mouse_y
                self._list_width_start_px = settings.type_list_width
                # The top strip is resized vertically.
                cursor = "MOVE_Y" if state.list.list_placement == "TOP" else _CURSOR_MAP[divider_resize_handle]
                context.window.cursor_modal_set(cursor)
                self._last_cursor = cursor
                return {"RUNNING_MODAL"}
            if _in_list_zone(self._mouse_x, self._mouse_y, state):
                clear_rect = state.list.search_clear_rect
                if clear_rect and _in_rect(self._mouse_x, self._mouse_y, clear_rect):
                    self._list_search_clear_pressed = True
                    return {"RUNNING_MODAL"}
                search_rect = state.list.search_rect
                if search_rect and _in_rect(self._mouse_x, self._mouse_y, search_rect):
                    self._list_search_pressed = True
                    return {"RUNNING_MODAL"}
                scrollbar_track = state.list.scrollbar_track
                scrollbar_thumb = state.list.scrollbar_thumb
                if _list_scrollbar_hit(self._mouse_x, self._mouse_y, state) and scrollbar_track and scrollbar_thumb:
                    thumb_bottom_y = scrollbar_thumb[1] + scrollbar_thumb[3]
                    if scrollbar_thumb[1] <= self._mouse_y <= thumb_bottom_y:
                        # Direct grab: keep the pressed point pinned to the cursor.
                        self._list_scroll_grab = thumb_bottom_y - self._mouse_y
                    else:
                        # Trough click pages one track-length toward the
                        # click, then continues as a drag from there.
                        # Note: y grows upward but larger list_scroll
                        # shifts content down, so a click above the
                        # thumb decreases the scroll.
                        if self._mouse_y > thumb_bottom_y:
                            state.list.scroll = max(state.list.scroll - scrollbar_track[3], 0.0)
                        else:
                            state.list.scroll = min(state.list.scroll + scrollbar_track[3], state.list.scroll_max)
                        track_span = max(scrollbar_track[3] - scrollbar_thumb[3], 1.0)
                        scroll_offset = min(track_span * (1.0 - state.list.scroll / state.list.scroll_max), track_span)
                        self._list_scroll_grab = scrollbar_track[1] + scroll_offset + scrollbar_thumb[3] - self._mouse_y
                    self._list_scroll_pressed = True
                    state.list.scrollbar_dragging = True
                    self._redraw_ui()
                    return {"RUNNING_MODAL"}
                child_row = _list_child_at(self._mouse_x, self._mouse_y, state)
                if child_row:
                    # The flag is only armed so the release is consumed by the
                    # left-mouse release handler instead of the map click.
                    self._list_child_pressed = child_row
                    label, node_name = child_row
                    state.request_immediate_compile()
                    if event.shift:
                        selection.apply_list_range(
                            self, context, state, ("child", label, node_name), self._list_last_row_index
                        )
                    elif event.ctrl:
                        selection.select_single_node(self, context, node_name, toggle=True)
                    else:
                        selection.select_single_node(self, context, node_name)
                        key = ("child", label, node_name)
                        self._list_last_row_index = state.list.visible_row_index_map.get(key, -1)
                        state.list.arrow_key = key
                else:
                    row_label = _list_row_at(self._mouse_x, self._mouse_y, state)
                    if row_label:
                        toggle_rect = state.list.toggle_rects.get(row_label)
                        if toggle_rect and _in_rect(self._mouse_x, self._mouse_y, toggle_rect):
                            self._list_toggle_pressed = row_label
                        else:
                            # The flag is only armed so the release is
                            # consumed by the left-mouse release handler
                            # instead of the map click.
                            self._list_row_pressed = row_label
                            state.request_immediate_compile()
                            if event.shift:
                                selection.apply_list_range(
                                    self, context, state, ("header", row_label), self._list_last_row_index
                                )
                            elif event.ctrl:
                                selection.select_type_nodes(self, context, row_label, toggle=True)
                            else:
                                selection.select_type_nodes(self, context, row_label)
                                key = ("header", row_label)
                                self._list_last_row_index = state.list.visible_row_index_map.get(key, -1)
                                state.list.arrow_key = key
                return {"RUNNING_MODAL"}
            if addon:
                resize_handle = self._get_handle_at(context, event)
                if resize_handle:
                    self._resize_handle = resize_handle
                    state.interaction.resize_active = resize_handle
                    self._redraw_ui()
                    self._resize_start_mouse = (self._mouse_x, self._mouse_y)
                    ui_scale = _get_ui_scale()
                    map_x, map_y, map_w, map_h = state.view.rect
                    self._resize_start_values = (
                        int(map_w / ui_scale),
                        int(map_h / ui_scale),
                    )
                    self._resize_start_offset = (settings.offset_x, settings.offset_y)
                    cursor = _CURSOR_MAP[resize_handle]
                    context.window.cursor_modal_set(cursor)
                    self._last_cursor = cursor
                    return {"RUNNING_MODAL"}
            drag_mode, click_action, extend, toggle = self._resolve_gesture(event, settings, "LEFT")
            self._drag_mode = drag_mode
            self._click_action = click_action
            self._click_extend = extend
            self._click_toggle = toggle
            if drag_mode in ("PAN", "CENTER_PAN"):
                self._drag_start = (self._mouse_x, self._mouse_y)
                if drag_mode == "CENTER_PAN":
                    self._center_view_on_mouse(context, self._mouse_x, self._mouse_y)
            else:
                self._marquee_dragging = True
                state.interaction.marquee_active = True
                state.interaction.marquee_start = (self._mouse_x, self._mouse_y)
                state.interaction.marquee_end = (self._mouse_x, self._mouse_y)
                self._redraw_ui()
            return {"RUNNING_MODAL"}
        else:
            self._was_in_minimap = False
            self._drag_start = None
            self._reset_gesture()
            return {"PASS_THROUGH"}

    def _handle_right_mouse(self, context: Context, event: Event) -> set[str]:
        state, addon, settings, in_minimap = self._minimap_event_context(context)
        # --- Release ---
        if event.value == "RELEASE":
            if self._search_blur_consumed:
                self._search_blur_consumed = False
                return {"RUNNING_MODAL"}
            context_menu_button = self._context_menu_button
            if context_menu_button is not None:
                self._context_menu_button = None
                if state is not None and _frame_button_at(self._mouse_x, self._mouse_y, state) == context_menu_button:
                    self._open_button_context_menu(context, context_menu_button)
                return {"RUNNING_MODAL"}
            if self._list_width_dragging:
                self._list_width_dragging = False
                state.list.dragging_width = None
                state.list.width_clamped = False
                state.interaction.resize_active = None
                state.interaction.hovered_handle = None
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
                state.cache.invalidate_batches_only()
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if state.interaction.pressed:
                state.interaction.pressed = False
                self._redraw_ui()
            if self._resize_handle:
                self._resize_handle = None
                self._resize_start_mouse = None
                self._resize_start_values = None
                self._resize_start_offset = None
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
                state.view.width_clamped = False
                state.view.height_clamped = False
                state.interaction.hovered_handle = None
                state.interaction.resize_active = None
                state.cache.invalidate_batches_only()
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._marquee_dragging:
                self._marquee_dragging = False
                start = state.interaction.marquee_start
                end = state.interaction.marquee_end
                state.interaction.marquee_active = False
                state.interaction.marquee_start = None
                state.interaction.marquee_end = None
                framed = False
                if start is not None and end is not None:
                    if self._drag_mode == "FRAME_RECT_EDITOR":
                        framed = self._frame_marquee_rect_editor(context, state, start, end)
                    else:
                        framed = self._frame_marquee_rect(context, state, start, end)
                if framed:
                    self._was_in_minimap = False
                    self._drag_start = None
                    self._reset_gesture()
                    self._redraw_ui()
                    return {"RUNNING_MODAL"}
            if self._dragging:
                self._dragging = False
                self._drag_start = None
                self._was_in_minimap = False
                self._reset_gesture()
                if self._anim.drag_active:
                    self._pan_acc[0] += self._anim.drag_target[0]
                    self._pan_acc[1] += self._anim.drag_target[1]
                    self._anim.drag_target = [0.0, 0.0]
                    self._anim.drag_active = False
                if self._anim._animations_enabled(context):
                    speed = max(abs(self._anim.smooth_velocity[0]), abs(self._anim.smooth_velocity[1]))
                    if speed > INERTIA_MIN_SPEED:
                        self._anim.inertia_active = True
                        self._anim.inertia_mode = "VIEW"
                        if not self._anim.smooth_timer:
                            self._anim.create_timer(context)
                        return {"RUNNING_MODAL"}
                self._anim.smooth_velocity = [0.0, 0.0]
                pan_x = int(self._pan_acc[0])
                pan_y = int(self._pan_acc[1])
                self._pan_acc = [0.0, 0.0]
                if pan_x != 0 or pan_y != 0:
                    try:
                        with self._override_ctx(context):
                            bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                    except RuntimeError:
                        pass
                self._anim.destroy_timer(context)
                return {"RUNNING_MODAL"}
            if not self._dragging and self._was_in_minimap:
                # A click only acts when both press and release land inside
                # the minimap: a press swallowed elsewhere (e.g. by a popup)
                # must never let its release pan or select, and a release
                # outside belongs to the editor, so it passes through.
                self._was_in_minimap = False
                self._drag_start = None
                if self._click_action is not None and in_minimap:
                    self._run_click_action(context, state)
                    self._reset_gesture()
                    return {"RUNNING_MODAL"}
                self._reset_gesture()
                return {"PASS_THROUGH"}
            self._was_in_minimap = False
            self._drag_start = None
            self._reset_gesture()
            return {"PASS_THROUGH"}
        # --- Press ---
        # Synthetic follow-ups (CLICK) carry no new user intent: only PRESS
        # and DOUBLE_CLICK may arm a gesture, so a stray CLICK can never
        # leave a pending drag or click behind for a later release to fire.
        if event.value not in ("PRESS", "DOUBLE_CLICK"):
            self._was_in_minimap = False
            self._drag_start = None
            self._reset_gesture()
            return {"PASS_THROUGH"}
        # Same blur-then-swallow rule as the left button: the first right-click
        # outside the focused search box only blurs instead of selecting rows.
        if self._consume_search_blur_press(context, state):
            return {"RUNNING_MODAL"}
        self._was_in_minimap = in_minimap
        if self._was_in_minimap:
            context_button_id = _frame_button_at(self._mouse_x, self._mouse_y, state)
            if context_button_id:
                self._reset_gesture()
                self._context_menu_button = context_button_id
                return {"RUNNING_MODAL"}
            self._anim.cancel_smooth(context)
            ui_scale = _get_ui_scale()
            divider_handle_r = resize.get_list_divider_handle(state, self._mouse_x, self._mouse_y, ui_scale)
            if divider_handle_r:
                self._list_width_dragging = True
                state.interaction.resize_active = divider_handle_r
                self._redraw_ui()
                self._list_width_start_x = self._mouse_x
                self._list_width_start_y = self._mouse_y
                self._list_width_start_px = settings.type_list_width
                # The top strip is resized vertically.
                cursor = "MOVE_Y" if state.list.list_placement == "TOP" else _CURSOR_MAP[divider_handle_r]
                context.window.cursor_modal_set(cursor)
                self._last_cursor = cursor
                return {"RUNNING_MODAL"}
            if _in_list_zone(self._mouse_x, self._mouse_y, state):
                if _list_scrollbar_hit(self._mouse_x, self._mouse_y, state):
                    # Scrollbar owns the press; no row selection or pan.
                    return {"RUNNING_MODAL"}
                child_row = _list_child_at(self._mouse_x, self._mouse_y, state)
                if child_row:
                    child_label, node_name = child_row
                    state.request_immediate_compile()
                    if event.shift:
                        selection.apply_list_range(
                            self, context, state, ("child", child_label, node_name), self._list_last_row_index
                        )
                    elif event.ctrl:
                        selection.select_single_node(self, context, node_name, toggle=True)
                    else:
                        selection.select_single_node(self, context, node_name)
                        key = ("child", child_label, node_name)
                        self._list_last_row_index = state.list.visible_row_index_map.get(key, -1)
                        state.list.arrow_key = key
                    if not (event.shift or event.ctrl):
                        if not self._anim.view_selected_animated(context):
                            try:
                                with self._override_ctx(context):
                                    bpy.ops.node.view_selected()
                            except RuntimeError:
                                pass
                    self._was_in_minimap = False
                    return {"RUNNING_MODAL"}
                row_label = _list_row_at(self._mouse_x, self._mouse_y, state)
                if row_label:
                    toggle_rect = state.list.toggle_rects.get(row_label)
                    if toggle_rect and _in_rect(self._mouse_x, self._mouse_y, toggle_rect):
                        if row_label in state.list.expanded:
                            state.list.expanded.discard(row_label)
                        else:
                            state.list.expanded.add(row_label)
                        state.cache.list_key = None
                        state.request_immediate_compile()
                        self._redraw_ui()
                        self._was_in_minimap = False
                        return {"RUNNING_MODAL"}
                    state.request_immediate_compile()
                    if event.shift:
                        selection.apply_list_range(
                            self, context, state, ("header", row_label), self._list_last_row_index
                        )
                    elif event.ctrl:
                        selection.select_type_nodes(self, context, row_label, toggle=True)
                    else:
                        selection.select_type_nodes(self, context, row_label)
                        key = ("header", row_label)
                        self._list_last_row_index = state.list.visible_row_index_map.get(key, -1)
                        state.list.arrow_key = key
                    if not (event.shift or event.ctrl):
                        if not self._anim.view_selected_animated(context):
                            try:
                                with self._override_ctx(context):
                                    bpy.ops.node.view_selected()
                            except RuntimeError:
                                pass
                self._was_in_minimap = False
                return {"RUNNING_MODAL"}
            if addon:
                resize_handle = self._get_handle_at(context, event)
                if resize_handle:
                    self._resize_handle = resize_handle
                    state.interaction.resize_active = resize_handle
                    self._redraw_ui()
                    self._resize_start_mouse = (self._mouse_x, self._mouse_y)
                    ui_scale = _get_ui_scale()
                    map_x, map_y, map_w, map_h = state.view.rect
                    self._resize_start_values = (
                        int(map_w / ui_scale),
                        int(map_h / ui_scale),
                    )
                    self._resize_start_offset = (settings.offset_x, settings.offset_y)
                    cursor = _CURSOR_MAP[resize_handle]
                    context.window.cursor_modal_set(cursor)
                    self._last_cursor = cursor
                    return {"RUNNING_MODAL"}
            drag_mode, click_action, extend, toggle = self._resolve_gesture(event, settings, "RIGHT")
            self._drag_mode = drag_mode
            self._click_action = click_action
            self._click_extend = extend
            self._click_toggle = toggle
            if drag_mode in ("PAN", "CENTER_PAN"):
                self._drag_start = (self._mouse_x, self._mouse_y)
                if drag_mode == "CENTER_PAN":
                    self._center_view_on_mouse(context, self._mouse_x, self._mouse_y)
            else:
                self._marquee_dragging = True
                state.interaction.marquee_active = True
                state.interaction.marquee_start = (self._mouse_x, self._mouse_y)
                state.interaction.marquee_end = (self._mouse_x, self._mouse_y)
                self._redraw_ui()
            return {"RUNNING_MODAL"}
        else:
            self._was_in_minimap = False
            self._drag_start = None
            self._reset_gesture()
            return {"PASS_THROUGH"}

    def _handle_mouse_move(self, context: Context, event: Event) -> set[str]:
        state, addon, settings, in_minimap = self._minimap_event_context(context)
        if self._moving:
            self._apply_move_drag(context, state, settings)
            return {"RUNNING_MODAL"}
        if self._list_width_dragging:
            resize.apply_list_width_drag(self, context)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if self._resize_handle:
            resize.resize_apply_delta(self, context, event)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if self._list_scroll_pressed and state.list.scrollbar_dragging:
            _apply_list_scroll_drag(self._mouse_x, self._mouse_y, self._list_scroll_grab, state)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if self._marquee_dragging:
            state.interaction.marquee_end = (self._mouse_x, self._mouse_y)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if not self._dragging and not self._mmb_dragging and not self._drag_start and not self._marquee_dragging:
            self._update_cursor(context, event)
        if (
            not self._dragging
            and not self._mmb_dragging
            and not self._marquee_dragging
            and not self._resize_handle
            and not self._list_width_dragging
        ):
            # Clear (X) button hover feedback; shown only while a query exists.
            # It stays live while the search box is focused since the button is
            # part of the search zone.
            clear_rect = state.list.search_clear_rect
            over_clear = bool(clear_rect and _in_rect(self._mouse_x, self._mouse_y, clear_rect))
            if state.list.search_clear_hovered != over_clear:
                state.list.search_clear_hovered = over_clear
                self._redraw_ui()
            if not state.list.search_focused:
                in_list = _in_list_zone(self._mouse_x, self._mouse_y, state)
                # The scrollbar gutter suppresses row hovers so the bar can
                # be approached without flashing the rows underneath.
                over_bar = (
                    in_list
                    and _list_scrollbar_hit(self._mouse_x, self._mouse_y, state)
                    and state.interaction.hovered_handle != ResizeHandle.LIST
                )
                if state.list.hovered_scrollbar != over_bar:
                    state.list.hovered_scrollbar = over_bar
                    self._redraw_ui()
                if over_bar or not in_list:
                    row_label = None
                else:
                    row_label = _list_row_at(self._mouse_x, self._mouse_y, state)
                child_hover = None
                if not over_bar and in_list:
                    child_hover = _list_child_at(self._mouse_x, self._mouse_y, state)
                if state.list.hovered_type_label != row_label:
                    state.list.hovered_type_label = row_label
                    self._redraw_ui()
                if state.list.hovered_list_row != child_hover:
                    state.list.hovered_list_row = child_hover
                    self._redraw_ui()
                new_hovered = None
                if in_list and child_hover is not None:
                    # Hovering a single child row highlights only that node's
                    # border on the minimap (not the whole type group).
                    new_hovered = child_hover[1]
                if state.interaction.hovered_node_id != new_hovered:
                    state.interaction.hovered_node_id = new_hovered
                    self._redraw_ui()
                old_btn = state.buttons.hovered_button_id
                new_btn = _frame_button_at(self._mouse_x, self._mouse_y, state) if in_minimap and not in_list else None
                # The list toggle slides horizontally while the type-list zone
                # width animates, so a hit-test during that window can land on the
                # button's transient position and leave a stale highlight once it
                # has moved away from the cursor. Drop only the LIST hover here;
                # the other buttons keep their normal hover.
                if new_btn == "LIST" and (state.list.anim_active or state.list.dragging_width is not None):
                    new_btn = None
                if old_btn != new_btn:
                    state.buttons.hovered_button_id = new_btn
                    self._redraw_ui()
        if self._list_mmb_dragging and self._list_mmb_drag_start:
            dy = self._mouse_y - self._list_mmb_drag_start[1]
            if abs(dy) > 0:
                state.list.scroll = min(
                    max(state.list.scroll - dy, 0.0),
                    state.list.scroll_max,
                )
                self._list_mmb_drag_start = (self._mouse_x, self._mouse_y)
                self._redraw_ui()
            return {"RUNNING_MODAL"}
        if self._mmb_dragging and self._mmb_drag_start:
            dx = self._mouse_x - self._mmb_drag_start[0]
            dy = self._mouse_y - self._mmb_drag_start[1]
            if abs(dx) <= 1 and abs(dy) <= 1:
                self._anim.smooth_velocity[0] *= SMOOTH_DAMP_STILL
                self._anim.smooth_velocity[1] *= SMOOTH_DAMP_STILL
            else:
                self._anim.smooth_velocity[0] = self._anim.smooth_velocity[0] * 0.6 + dx * 0.4
                self._anim.smooth_velocity[1] = self._anim.smooth_velocity[1] * 0.6 + dy * 0.4
            pan_before = state.view.pan
            state.view.pan = (state.view.pan[0] + dx, state.view.pan[1] + dy)
            _clamp_pan_to_viewport(self._space, self._region, state)
            rejected_x = dx - (state.view.pan[0] - pan_before[0])
            rejected_y = dy - (state.view.pan[1] - pan_before[1])
            if (rejected_x != 0 or rejected_y != 0) and settings and settings.use_follow_view:
                state.view.pan = (pan_before[0] + dx, pan_before[1] + dy)
                self._redirect_to_view2d(context, -dx, -dy)
            elif rejected_x != 0 or rejected_y != 0:
                self._redirect_to_view2d(context, -int(rejected_x), -int(rejected_y))
            self._mmb_drag_start = (self._mouse_x, self._mouse_y)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if self._drag_start is not None:
            dx = self._mouse_x - self._drag_start[0]
            dy = self._mouse_y - self._drag_start[1]
            if abs(dx) > 2 or abs(dy) > 2 or self._dragging:
                if not self._dragging and (
                    self._anim.anim_active or self._anim.frame_anim_active or self._anim.editor_anim_active
                ):
                    self._anim.cancel_smooth(context)
                self._dragging = True
                if self._was_in_minimap:
                    state.interaction.pressed = True
                    smooth = self._anim._animations_enabled(context)
                    self._pan_view(context, dx, dy, smooth)
                    self._drag_start = (self._mouse_x, self._mouse_y)
            return {"RUNNING_MODAL"}
        if in_minimap:
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def _handle_wheel(self, context: Context, event: Event) -> set[str]:
        state, addon, settings, in_minimap = self._minimap_event_context(context)
        if in_minimap and (event.ctrl or event.shift):
            return {"PASS_THROUGH"}
        if state.list.search_focused and not _over_search_zone(self._mouse_x, self._mouse_y, state):
            self._blur_search_consume(context, state)
            self._search_blur_consumed = False
            return {"RUNNING_MODAL"}
        if in_minimap and _in_list_zone(self._mouse_x, self._mouse_y, state):
            if state.list.search_focused:
                return {"RUNNING_MODAL"}
            direction = -1 if event.type == "WHEELUPMOUSE" else 1
            state.list.scroll = min(
                max(state.list.scroll + direction * state.list.row_height * 3, 0.0), state.list.scroll_max
            )
            over_bar = _list_scrollbar_hit(self._mouse_x, self._mouse_y, state)
            state.list.hovered_type_label = None if over_bar else _list_row_at(self._mouse_x, self._mouse_y, state)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if in_minimap:
            prefs = addon.settings if addon else None
            scroll_mode = prefs.scroll_wheel_mode if prefs else "MINIMAP"
            if event.alt:
                scroll_mode = "NODE_EDITOR" if scroll_mode == "MINIMAP" else "MINIMAP"

            if scroll_mode == "NODE_EDITOR":
                try:
                    zoom_factor = 0.05
                    with self._override_ctx(context):
                        if event.type == "WHEELUPMOUSE":
                            bpy.ops.view2d.zoom_in(zoomfacx=zoom_factor, zoomfacy=zoom_factor)
                        else:
                            bpy.ops.view2d.zoom_out(zoomfacx=-zoom_factor, zoomfacy=-zoom_factor)
                except RuntimeError:
                    pass
            else:
                zoom_delta = WHEEL_ZOOM_IN if event.type == "WHEELUPMOUSE" else WHEEL_ZOOM_OUT
                effective_zoom = state.view.user_zoom

                is_constrained = False
                if addon and addon.settings.use_follow_view:
                    if effective_zoom < state.view.anchor_zoom - 0.001:
                        is_constrained = True

                if is_constrained and event.type == "WHEELUPMOUSE":
                    try:
                        zoom_factor = 0.05
                        with self._override_ctx(context):
                            bpy.ops.view2d.zoom_in(zoomfacx=zoom_factor, zoomfacy=zoom_factor)
                    except RuntimeError:
                        pass
                else:
                    new_zoom = max(MIN_FRAME_ZOOM, min(effective_zoom * zoom_delta, MAX_FRAME_ZOOM))

                    transform = _get_minimap_transform(state)
                    tree_coord = _tree_from_region(self._mouse_x, self._mouse_y, transform)

                    if transform[2] > 0 and tree_coord is not None:
                        _, _, scale, tree_center_x, tree_center_y = transform
                        base_scale = scale / state.view.user_zoom
                        hit_tx, hit_ty = tree_coord
                        pan_x, pan_y = state.view.pan

                        pan_x_new = pan_x - (hit_tx - tree_center_x) * base_scale * (new_zoom - state.view.user_zoom)
                        pan_y_new = pan_y - (hit_ty - tree_center_y) * base_scale * (new_zoom - state.view.user_zoom)

                        state.view.anchor_zoom = new_zoom
                        state.view.user_zoom = new_zoom
                        state.view.pan = (pan_x_new, pan_y_new)
                        _clamp_pan_to_viewport(self._space, self._region, state)

            self._redraw_ui()
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def _open_button_context_menu(self, context: Context, button_id: str) -> None:
        """Open the toggle context menu tied to the right-clicked minimap button."""
        try:
            with self._override_ctx(context):
                open_minimap_button_menu(context, button_id)
        except RuntimeError:
            pass

    def _activate_armed_button(self, context: Context, settings) -> None:
        """Release the armed minimap button; run its action when still under the cursor."""
        button_id = self._armed_button
        self._armed_button = None
        state = self._state
        if not button_id or state is None:
            return
        rect = state.buttons.rects.get(button_id)
        if not rect:
            return
        button_x, button_y, button_width, button_height = rect
        if not (
            button_x <= self._mouse_x <= button_x + button_width
            and button_y <= self._mouse_y <= button_y + button_height
        ):
            return
        if button_id == "LIST":
            settings.show_type_list = not settings.show_type_list
            start_list_width_animation(state, settings)
            self._redraw_ui()
        self._dispatch_frame_action(context, settings, button_id)

    def _dispatch_frame_action(self, context: Context, settings, button_id: str, scope: str = "BOTH") -> None:
        """Run a frame action directly, or eased via animation when smooth pan applies.

        Shared by the minimap button release and the Home / End / Numpad shortcuts.
        The SELECTED action touches two spaces: the minimap view and the type
        list. *scope* selects which one runs: "MAP" frames only the nodes in
        the minimap, "LIST" focuses only the list on the active node, and
        "BOTH" keeps the combined behaviour for buttons and operators.
        """
        state = self._state
        if not state:
            return
        smooth = self._anim._animations_enabled(context)
        area_ptr = self._area.as_pointer() if self._area else 0
        match button_id:
            case "ALL":
                if smooth:
                    targets = _compute_frame_all_targets(self._space, self._region, area_ptr)
                    if targets:
                        self._anim.start_frame_animation(context, targets[0], [targets[1], targets[2]])
                else:
                    frame_all(self._space, self._region, area_ptr)
            case "VIEW":
                if smooth:
                    visible = _get_visible_rect(self._space, self._region)
                    if visible:
                        node_tree = self._space.edit_tree
                        if node_tree:
                            _, _, map_w, map_h = state.view.rect
                            state.view.tree_bounds = _expand_bounds_margin(
                                _get_node_tree_bounds(node_tree.nodes),
                                _get_ui_scale(),
                                map_h,
                                state.view.inner_padding,
                            )
                        targets = _compute_frame_to_bounds_targets(visible, area_ptr)
                        self._anim.start_frame_animation(context, targets[0], [targets[1], targets[2]])
                else:
                    frame_view(self._space, self._region, area_ptr)
            case "SELECTED":
                if scope in ("BOTH", "MAP"):
                    if smooth:
                        targets = _compute_frame_selected_targets(self._space, self._region, area_ptr)
                        if targets:
                            target_zoom = targets[0] if targets[0] is not None else state.view.user_zoom
                            self._anim.start_frame_animation(context, target_zoom, [targets[1], targets[2]])
                    else:
                        frame_selected(self._space, self._region, area_ptr)
                if scope in ("BOTH", "LIST"):
                    selection.focus_list_on_active_node(self, context)

    def _pan_view(self, context: Context, dx: int, dy: int, smooth: bool = False) -> None:
        state = self._state
        if not state:
            return
        visible = _get_visible_rect(self._space, self._region)
        if not visible:
            return
        _, _, scale, _, _ = _compute_map_transform(state)
        if scale <= 0:
            return
        view_zoom_x, view_zoom_y = _view_zoom_factors(self._space, self._region, visible)

        view_delta_x = (dx / scale) * view_zoom_x
        view_delta_y = (dy / scale) * view_zoom_y
        if abs(dx) <= 1 and abs(dy) <= 1:
            self._anim.smooth_velocity[0] *= 0.15
            self._anim.smooth_velocity[1] *= 0.15
        else:
            self._anim.smooth_velocity[0] = self._anim.smooth_velocity[0] * 0.6 + view_delta_x * 0.4
            self._anim.smooth_velocity[1] = self._anim.smooth_velocity[1] * 0.6 + view_delta_y * 0.4

        if smooth:
            self._anim.drag_target[0] += view_delta_x
            self._anim.drag_target[1] += view_delta_y
            if not self._anim.drag_active:
                self._anim.drag_active = True
                self._anim.create_timer(context)
            return

        self._pan_acc[0] += view_delta_x
        self._pan_acc[1] += view_delta_y
        pan_x = int(self._pan_acc[0])
        pan_y = int(self._pan_acc[1])
        self._pan_acc[0] -= pan_x
        self._pan_acc[1] -= pan_y

        if pan_x != 0 or pan_y != 0:
            try:
                pan_before = state.view.pan

                with self._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                _clamp_pan_to_viewport(self._space, self._region, state)

                clamp_dx = state.view.pan[0] - pan_before[0]
                clamp_dy = state.view.pan[1] - pan_before[1]

                if clamp_dx != 0 or clamp_dy != 0:
                    self._pan_acc[0] += (-clamp_dx / scale) * view_zoom_x
                    self._pan_acc[1] += (-clamp_dy / scale) * view_zoom_y

                    extra_pan_x = int(self._pan_acc[0])
                    extra_pan_y = int(self._pan_acc[1])
                    self._pan_acc[0] -= extra_pan_x
                    self._pan_acc[1] -= extra_pan_y

                    if extra_pan_x != 0 or extra_pan_y != 0:
                        with self._override_ctx(context):
                            bpy.ops.view2d.pan(deltax=extra_pan_x, deltay=extra_pan_y)
                        _clamp_pan_to_viewport(self._space, self._region, state)
            except RuntimeError:
                pass

    def _redirect_to_view2d(self, context: Context, dx: float, dy: float) -> None:
        state = self._state
        if not state:
            return
        _, _, scale, _, _ = _compute_map_transform(state)
        if scale <= 0:
            return
        visible = _get_visible_rect(self._space, self._region)
        if not visible:
            return
        view_zoom_x, view_zoom_y = _view_zoom_factors(self._space, self._region, visible)
        self._redirect_acc[0] += (dx / scale) * view_zoom_x
        self._redirect_acc[1] += (dy / scale) * view_zoom_y
        pan_x = int(self._redirect_acc[0])
        pan_y = int(self._redirect_acc[1])
        self._redirect_acc[0] -= pan_x
        self._redirect_acc[1] -= pan_y
        if pan_x != 0 or pan_y != 0:
            try:
                with self._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
            except RuntimeError:
                pass

    def _center_view_on_mouse(self, context: Context, mouse_x: int, mouse_y: int) -> None:
        state = self._state
        if not state:
            return
        tree_coord = _region_to_tree(mouse_x, mouse_y, state)
        if not tree_coord:
            return

        visible = _get_visible_rect(self._space, self._region)
        if not visible:
            return

        view_cx = (visible[0] + visible[2]) / 2.0
        view_cy = (visible[1] + visible[3]) / 2.0
        delta_tree_x = tree_coord[0] - view_cx
        delta_tree_y = tree_coord[1] - view_cy

        view_zoom_x, view_zoom_y = _view_zoom_factors(self._space, self._region, visible)

        pan_x = int(delta_tree_x * view_zoom_x)
        pan_y = int(delta_tree_y * view_zoom_y)
        if pan_x == 0 and pan_y == 0:
            return

        state.interaction.pressed = True
        if self._anim._animations_enabled(context):
            self._anim.start_center_animation(context, float(pan_x), float(pan_y), visible)
        else:
            try:
                with self._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                _clamp_pan_to_viewport(self._space, self._region, state)
            except RuntimeError:
                pass

    def _marquee_tree_bounds(
        self, state: MinimapState, start: tuple[int, int], end: tuple[int, int]
    ) -> tuple[float, float, float, float] | None:
        """Return tree-space bounds for a marquee rect in region pixels, or None."""
        if abs(end[0] - start[0]) < 2 or abs(end[1] - start[1]) < 2:
            return None
        transform = _compute_map_transform(state)
        tree_start = _tree_from_region(start[0], start[1], transform)
        tree_end = _tree_from_region(end[0], end[1], transform)
        if tree_start is None or tree_end is None:
            return None
        return (
            min(tree_start[0], tree_end[0]),
            min(tree_start[1], tree_end[1]),
            max(tree_start[0], tree_end[0]),
            max(tree_start[1], tree_end[1]),
        )

    def _frame_marquee_rect(
        self, context: Context, state: MinimapState, start: tuple[int, int], end: tuple[int, int]
    ) -> bool:
        """Frame the tree-space area covered by a marquee rect in the minimap.

        Return True when a non-degenerate rect produced a frame; False for a
        near-zero rect so the caller can fall back to the click action.
        """
        bounds = self._marquee_tree_bounds(state, start, end)
        if bounds is None:
            return False
        area_ptr = self._area.as_pointer() if self._area else 0
        if self._anim._animations_enabled(context):
            zoom, pan_x, pan_y = _compute_frame_to_bounds_targets(bounds, area_ptr)
            self._anim.start_frame_animation(context, zoom, [pan_x, pan_y])
        else:
            _frame_to_bounds(bounds, area_ptr)
        return True

    def _frame_marquee_rect_editor(
        self, context: Context, state: MinimapState, start: tuple[int, int], end: tuple[int, int]
    ) -> bool:
        """Frame the tree-space area covered by a marquee rect in the editor.

        Return True when a non-degenerate rect produced a frame; False for a
        near-zero rect so the caller can fall back to the click action.
        """
        bounds = self._marquee_tree_bounds(state, start, end)
        if bounds is None:
            return False
        target = [bounds[0], bounds[1], bounds[2], bounds[3]]
        for _ in range(100):
            visible = _get_visible_rect(self._space, self._region)
            if not visible or self._anim._editor_view_close(visible, target):
                break
            self._anim._correct_editor_view(context, target)
        return True

    def _cancel_interaction(self, context: Context) -> None:
        self._anim.cancel_smooth(context)
        if self._dragging or self._drag_start is not None:
            self._dragging = False
            self._drag_start = None
            self._anim.drag_active = False
            self._anim.drag_target = [0.0, 0.0]
        if self._mmb_dragging:
            self._mmb_dragging = False
            self._mmb_drag_start = None
        if self._list_mmb_dragging:
            self._list_mmb_dragging = False
            self._list_mmb_drag_start = None
        if self._marquee_dragging:
            self._marquee_dragging = False
            state = self._state
            if state:
                state.interaction.marquee_active = False
                state.interaction.marquee_start = None
                state.interaction.marquee_end = None
        if self._resize_handle:
            self._resize_handle = None
            self._resize_start_mouse = None
            self._resize_start_values = None
            state = self._state
            if state:
                state.view.width_clamped = False
                state.view.height_clamped = False
                state.interaction.hovered_handle = None
                state.interaction.resize_active = None
        if self._list_width_dragging:
            self._list_width_dragging = False
            state = self._state
            if state:
                state.list.dragging_width = None
                state.list.width_clamped = False
                state.interaction.hovered_handle = None
                state.interaction.resize_active = None
                state.cache.invalidate_batches_only()
        if self._moving:
            self._moving = False
            self._move_start_mouse = None
            self._move_start_offset = None
            self._snap_candidate = None
            self._snap_dwell = 0.0
            self._snap_last_time = 0.0
            state = self._state
            if state:
                state.view.moving = False
                state.view.snapped = False
        context.window.cursor_modal_set("DEFAULT")
        self._last_cursor = ""
        self._armed_button = None
        self._list_row_pressed = None
        self._list_child_pressed = None
        self._list_toggle_pressed = None
        self._list_scroll_pressed = False
        self._list_scroll_grab = 0.0
        self._list_search_pressed = False
        self._list_search_clear_pressed = False
        self._search_blur_consumed = False
        self._reset_gesture()
        state = self._state
        if state:
            state.buttons.hovered_button_id = None
            state.buttons.pressed_button_id = None
            state.list.hovered_type_label = None
            state.interaction.hovered_node_id = None
            state.list.hovered_scrollbar = False
            state.list.scrollbar_dragging = False
            state.list.search_cursor = 0
            state.list.search_focused = False
            state.list.search_esc_armed = False
            if state.interaction.pressed:
                state.interaction.pressed = False
        self._redraw_ui()

    def _update_cursor(self, context: Context, event: Event) -> None:
        state = self._state
        if not state or not state.view.rect:
            return
        in_minimap = _is_in_minimap(self._mouse_x, self._mouse_y, state)
        if not in_minimap:
            if self._last_cursor:
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
            old_handle = state.interaction.hovered_handle
            state.interaction.hovered_handle = None
            if old_handle:
                self._redraw_ui()
            return
        # While the search box is focused the minimap is in text-input mode:
        # lock the I-beam and stop hover feedback (divider, handles, buttons)
        # from swapping the cursor out from under the user.
        if state.list.search_focused:
            if self._last_cursor != "TEXT":
                context.window.cursor_modal_set("TEXT")
                self._last_cursor = "TEXT"
            return
        # Divider takes precedence over outer borders and list hover.
        ui_scale = _get_ui_scale()
        divider = resize.get_list_divider_handle(state, self._mouse_x, self._mouse_y, ui_scale)
        if divider:
            old_handle = state.interaction.hovered_handle
            state.interaction.hovered_handle = divider
            if divider != old_handle:
                self._redraw_ui()
            if state.list.width_clamped:
                cursor = "HAND"
            elif state.list.list_placement == "TOP":
                cursor = "MOVE_Y"
            else:
                cursor = _CURSOR_MAP.get(divider, "MOVE_X")
            if cursor != self._last_cursor:
                context.window.cursor_modal_set(cursor)
                self._last_cursor = cursor
            return
        handle = self._get_handle_at(context, event)
        old_handle = state.interaction.hovered_handle
        state.interaction.hovered_handle = handle
        if handle != old_handle:
            self._redraw_ui()
        if _in_list_zone(self._mouse_x, self._mouse_y, state):
            # While the search box is focused the list is in text-input mode, so
            # the I-beam signals "typing here"; otherwise the list body resets
            # to the default arrow.
            cursor = "TEXT" if state.list.search_focused else "DEFAULT"
            if cursor != self._last_cursor:
                context.window.cursor_modal_set(cursor)
                self._last_cursor = cursor
            return
        is_clamped = handle and (state.view.width_clamped or state.view.height_clamped)
        cursor = "HAND" if is_clamped else _CURSOR_MAP.get(handle, "DEFAULT")
        if cursor != self._last_cursor:
            context.window.cursor_modal_set(cursor)
            self._last_cursor = cursor

    def _start_move(self, context: Context, state: MinimapState, settings) -> None:
        """Begin dragging the whole minimap from the move-grip button.

        Preserves the current screen position by converting it into a FREE
        offset, then lets the map follow the cursor freely until it snaps to a
        border or corner.
        """
        ui_scale = _get_ui_scale()
        sx, sy, ex, ey = _get_safe_bounds(self._area, self._region)
        x_margin, y_margin, margin = _get_minimap_margins(self._space, "FREE", ui_scale)
        map_x, map_y, _map_w, _map_h = state.view.rect
        offset_x = int(round((map_x - (sx + x_margin)) / ui_scale))
        offset_y = int(round((map_y - (sy + y_margin)) / ui_scale))
        self._moving = True
        self._move_start_mouse = (self._mouse_x, self._mouse_y)
        self._move_start_offset = (offset_x, offset_y)
        self._snap_candidate = None
        self._snap_dwell = 0.0
        self._snap_last_time = time.perf_counter()
        from ..core.state import suppress_update_callbacks

        with suppress_update_callbacks():
            settings.dock_mode = "FREE"
            settings.offset_x = offset_x
            settings.offset_y = offset_y
        state.view.moving = True
        state.view.snapped = False
        state.buttons.pressed_button_id = "DRAG"
        self._redraw_ui()

    def _apply_move_drag(self, context: Context, state: MinimapState, settings) -> None:
        """Update the FREE offset from the cursor delta and snap to borders."""
        ui_scale = _get_ui_scale()
        sx, sy, ex, ey = _get_safe_bounds(self._area, self._region)
        x_margin, y_margin, margin = _get_minimap_margins(self._space, "FREE", ui_scale)
        dx = (self._mouse_x - self._move_start_mouse[0]) / ui_scale
        dy = (self._mouse_y - self._move_start_mouse[1]) / ui_scale
        offset_x = int(round(self._move_start_offset[0] + dx))
        offset_y = int(round(self._move_start_offset[1] + dy))

        from ..core.state import suppress_update_callbacks

        with suppress_update_callbacks():
            settings.dock_mode = "FREE"
            settings.offset_x = offset_x
            settings.offset_y = offset_y

        # Recompute the live rect from the offset so snapping does not lag one
        # frame behind the cursor, then clamp the origin to the same insets the
        # drawing pass applies. A map pushed past an edge therefore pins itself
        # to the border (where it can snap) instead of shrinking away.
        safe_width = ex - sx
        safe_height = ey - sy
        map_w = min(settings.minimap_width * ui_scale, (safe_width - x_margin) * settings.max_width_percent / 100.0)
        map_h = min(
            settings.minimap_height * ui_scale, (safe_height - y_margin - margin) * settings.max_height_percent / 100.0
        )
        map_x = sx + x_margin + offset_x * ui_scale
        map_y = sy + y_margin + offset_y * ui_scale
        safe = (sx, sy, ex, ey)
        map_x, map_y = clamp_free_rect(map_x, map_y, map_w, map_h, safe, x_margin, y_margin, margin)
        cand = (
            resize._nearest_dock(map_x, map_y, map_w, map_h, safe, x_margin, y_margin, margin, ui_scale)
            if settings.use_snap_to_borders
            else None
        )

        now = time.perf_counter()
        if cand and cand == self._snap_candidate:
            self._snap_dwell += (now - self._snap_last_time) * 1000.0
        else:
            self._snap_candidate = cand
            self._snap_dwell = 0.0
        self._snap_last_time = now

        snapped = cand is not None and self._snap_dwell >= DOCK_DWELL_MS
        if snapped:
            if settings.current_position != cand[2]:
                with suppress_update_callbacks():
                    if cand[2] == "FREE":
                        settings.dock_mode = "FREE"
                    elif cand[2] in ("TOP_LEFT", "TOP_RIGHT", "BOTTOM_LEFT", "BOTTOM_RIGHT"):
                        settings.dock_mode = "CORNER"
                        settings.corner_position = cand[2]
                    else:
                        settings.dock_mode = "EDGE"
                        settings.edge_position = cand[2]
                # Re-anchor the drag at the latching dock so releasing the snap
                # continues from the snapped position instead of leaping back to
                # where the drag originally started.
                self._move_start_offset = (
                    int(round((cand[0] - (sx + x_margin)) / ui_scale)),
                    int(round((cand[1] - (sy + y_margin)) / ui_scale)),
                )
                self._move_start_mouse = (self._mouse_x, self._mouse_y)
        elif settings.current_position != "FREE":
            # Not (yet) snapped: revert to the free offset position.
            with suppress_update_callbacks():
                settings.dock_mode = "FREE"
        state.view.snapped = snapped
        self._redraw_ui()

    def _get_handle_at(self, context: Context, event: Event) -> str | None:
        state = self._state
        if not state:
            return None
        addon = get_addon_preferences(context)
        if not addon:
            return None
        corner = addon.settings.current_position
        ui_scale = _get_ui_scale()
        return resize.get_resize_handle(
            state, corner, self._mouse_x, self._mouse_y, ui_scale, self._space, self._area, self._region
        )

    def invoke(self, context: Context, _event: Event) -> set[str]:
        if context.area.type != "NODE_EDITOR":
            logger.debug("invoke: cancelled — area type is %s", context.area.type)
            return {"CANCELLED"}
        self._window_ptr = context.window.as_pointer()
        self._pan_acc = [0.0, 0.0]
        self._redirect_acc = [0.0, 0.0]
        self._armed_button = None
        self._list_row_pressed = None
        self._list_scroll_pressed = False
        self._list_scroll_grab = 0.0
        self._list_search_pressed = False
        self._list_search_clear_pressed = False
        self._search_blur_consumed = False
        self._context_menu_button = None
        self._reset_gesture()
        self._list_last_row_index = -1
        self._list_width_dragging = False
        self._list_width_start_x = 0
        self._list_width_start_px = 160
        self._moving = False
        self._move_start_mouse = None
        self._move_start_offset = None
        self._snap_candidate = None
        self._snap_dwell = 0.0
        self._snap_last_time = 0.0
        self._anim = AnimationController(self)
        _minimap_window_operators[self._window_ptr] = self
        context.window_manager.modal_handler_add(self)
        ops_keys = list(_minimap_window_operators.keys())
        logger.debug("invoke: RUNNING_MODAL for window %d, ops=%s", self._window_ptr, ops_keys)
        return {"RUNNING_MODAL"}

    def cancel(self, context: Context) -> None:
        logger.debug("cancel: window %d ops_before=%s", self._window_ptr, list(_minimap_window_operators.keys()))
        if self._window_ptr in _minimap_window_operators:
            del _minimap_window_operators[self._window_ptr]
        logger.debug("cancel: ops_after=%s", list(_minimap_window_operators.keys()))
        self._anim.destroy_timer(context)
        if self._state is not None:
            self._state.view.width_clamped = False
            self._state.view.height_clamped = False
            self._state.view.moving = False
            self._state.view.snapped = False
            self._state.interaction.hovered_handle = None
            self._state.interaction.resize_active = None
            self._state.buttons.hovered_button_id = None
            self._state.buttons.pressed_button_id = None
            self._state.list.hovered_type_label = None
            self._state.list.hovered_list_row = None
            self._state.interaction.hovered_node_id = None
            self._state.list.hovered_scrollbar = False
            self._state.list.scrollbar_dragging = False
            self._state.list.search_esc_armed = False
            self._state.interaction.marquee_active = False
            self._state.interaction.marquee_start = None
            self._state.interaction.marquee_end = None
        self._list_row_pressed = None
        self._list_child_pressed = None
        self._list_toggle_pressed = None
        self._list_scroll_pressed = False
        self._list_scroll_grab = 0.0
        self._list_search_pressed = False
        self._list_search_clear_pressed = False
        self._search_blur_consumed = False
        self._context_menu_button = None
        self._reset_gesture()
        self._list_last_row_index = -1
        self._list_width_dragging = False
        self._marquee_dragging = False
        self._moving = False
        self._move_start_mouse = None
        self._move_start_offset = None
        self._snap_candidate = None
        self._snap_dwell = 0.0
        self._snap_last_time = 0.0


class NODEMAP_OT_open_preferences(Operator):
    bl_idname = "nodemap.open_preferences"
    bl_label = "Open Preferences"
    bl_description = "Open the add-on preferences panel"

    def execute(self, context):
        bpy.ops.screen.userpref_show()
        context.preferences.active_section = "ADDONS"
        context.window_manager.addon_search = "Nodemap"
        return {"FINISHED"}


classes = (
    NODEMAP_OT_toggle,
    NODEMAP_OT_restore_keymap,
    NODEMAP_OT_frame_all,
    NODEMAP_OT_frame_selected,
    NODEMAP_OT_frame_view,
    NODEMAP_OT_navigate,
    NODEMAP_OT_open_preferences,
)
