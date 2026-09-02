"""Provide a modal operator for minimap interaction."""

import logging

import bpy
from bpy.types import Area, Context, Event, Operator, Region, SpaceNodeEditor

from .. import __package__ as base_package
from ..core.constants import HANDLE_THICKNESS, SCROLLBAR_HIT_PAD
from ..core.helpers import (
    _expand_bounds_margin,
    _get_area_and_region_under_mouse,
    _get_node_tree_bounds,
    _get_ui_scale,
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
from . import resize, selection
from .animations import AnimationController

logger = logging.getLogger(base_package)


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
    map_x, map_y, _, map_h = state.view.rect
    hit_pad = HANDLE_THICKNESS * _get_ui_scale()
    zone_left = map_x + hit_pad
    zone_right = map_x + state.view.inner_padding + state.list.list_width
    zone_rect = state.list.list_zone_rect
    if zone_rect:
        _, zone_y, _, zone_h = zone_rect
    else:
        zone_y = map_y + hit_pad
        zone_h = map_h - 2 * hit_pad
    return zone_left <= region_x <= zone_right and zone_y <= region_y <= zone_y + zone_h


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
    ResizeHandle.W: "MOVE_X",
    ResizeHandle.H: "MOVE_Y",
    ResizeHandle.C: "SCROLL_XY",
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


class NODEMAP_OT_frame_all(Operator):
    """Reset the minimap view to show all nodes."""

    bl_idname = "nodemap.frame_all"
    bl_label = "Frame All"
    bl_description = "Reset the minimap view to show all nodes.\nShortcut: Home"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        frame_all()
        return {"FINISHED"}


class NODEMAP_OT_frame_selected(Operator):
    """Focus the minimap view on selected nodes."""

    bl_idname = "nodemap.frame_selected"
    bl_label = "Frame Selected"
    bl_description = "Focus the minimap view on selected nodes.\nShortcut: Numpad ."
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        frame_selected()
        return {"FINISHED"}


class NODEMAP_OT_frame_view(Operator):
    """Focus the minimap view on the current editor viewport."""

    bl_idname = "nodemap.frame_view"
    bl_label = "Frame View"
    bl_description = "Focus the minimap view on the current editor viewport.\nShortcut: Shift+Home"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
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

    _mmb_dragging: bool = False
    _mmb_drag_start: tuple[int, int] | None = None

    _mx: int = 0
    _my: int = 0
    _state: MinimapState | None = None
    _area: Area | None = None
    _region: Region | None = None
    _space: SpaceNodeEditor | None = None

    _resize_handle: str | None = None
    _resize_start_mouse: tuple[int, int] | None = None
    _resize_start_values: tuple[int, int] | None = None
    _list_width_dragging: bool = False
    _list_width_start_x: int = 0
    _list_width_start_pct: int = 0
    _list_width_start_map_w: float = 0.0
    _last_cursor: str = ""
    _pan_acc: list[float]
    _redirect_acc: list[float]
    _armed_button: str | None = None
    _list_row_pressed: str | None = None
    _list_child_pressed: tuple[str, str] | None = None
    _list_toggle_pressed: str | None = None
    _list_scroll_pressed: bool = False
    _list_scroll_grab: float = 0.0
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
            or self._resize_handle is not None
            or self._list_width_dragging
            or self._drag_start is not None
            or self._list_scroll_pressed
            or self._anim.anim_active
            or self._anim.inertia_active
            or self._anim.drag_active
            or self._anim.frame_anim_active
            or self._anim.editor_anim_active
        )

        if not is_interactive:
            under_mouse_area, under_mouse_region = _get_area_and_region_under_mouse(context, event)
            if not under_mouse_area or under_mouse_area.type != "NODE_EDITOR" or not under_mouse_region:
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
        if addon and not settings.interactive:
            return {"PASS_THROUGH"}

        state = self._state
        in_minimap = _is_in_minimap(self._mouse_x, self._mouse_y, state)

        match event.type:
            case "LEFTMOUSE":
                return self._handle_left_mouse(context, event)

            case "RIGHTMOUSE":
                return self._handle_right_mouse(context, event)

            case "MIDDLEMOUSE":
                if event.value == "PRESS" and in_minimap:
                    state.interaction.pressed = True
                    self._anim.cancel_smooth(context)
                    if _in_list_zone(self._mouse_x, self._mouse_y, state):
                        self._list_mmb_dragging = True
                        self._list_mmb_drag_start = (self._mouse_x, self._mouse_y)
                    else:
                        self._mmb_dragging = True
                        self._mmb_drag_start = (self._mouse_x, self._mouse_y)
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
                    if settings and self._anim._animations_enabled(settings, context):
                        speed = max(abs(self._anim.smooth_velocity[0]), abs(self._anim.smooth_velocity[1]))
                        if speed > 2.0:
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
                if event.value == "PRESS" and in_minimap:
                    self._dispatch_frame_action(context, settings, "VIEW" if event.shift else "ALL")
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "NUMPAD_PERIOD":
                if event.value == "PRESS" and in_minimap:
                    self._dispatch_frame_action(context, settings, "SELECTED")
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

    def _handle_left_mouse(self, context: Context, event: Event) -> set[str]:
        state, addon, settings, in_minimap = self._minimap_event_context(context)
        # --- Release ---
        if event.value == "RELEASE":
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
            if self._armed_button:
                self._activate_armed_button(context, settings)
                return {"RUNNING_MODAL"}
            if self._list_child_pressed:
                label, node_name = self._list_child_pressed
                self._list_child_pressed = None
                still_over = _list_child_at(self._mouse_x, self._mouse_y, state) == (label, node_name)
                if _in_list_zone(self._mouse_x, self._mouse_y, state) and still_over:
                    state.cache.force_immediate = True
                    if event.shift:
                        selection.apply_list_range(self, context, state, ("child", label, node_name))
                    elif event.ctrl:
                        selection.select_single_node(self, context, node_name, toggle=True)
                    else:
                        selection.select_single_node(self, context, node_name)
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
                    state.cache.force_immediate = True
                    self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._list_row_pressed:
                label = self._list_row_pressed
                self._list_row_pressed = None
                if (
                    _in_list_zone(self._mouse_x, self._mouse_y, state)
                    and _list_row_at(self._mouse_x, self._mouse_y, state) == label
                ):
                    state.cache.force_immediate = True
                    if event.shift:
                        selection.apply_list_range(self, context, state, ("header", label))
                    elif event.ctrl:
                        selection.select_type_nodes(self, context, label, toggle=True)
                    else:
                        selection.select_type_nodes(self, context, label)
                return {"RUNNING_MODAL"}
            if self._resize_handle:
                self._resize_handle = None
                self._resize_start_mouse = None
                self._resize_start_values = None
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
                state.view.width_clamped = False
                state.view.height_clamped = False
                state.interaction.hovered_handle = None
                state.interaction.resize_active = None
                state.cache.invalidate_batches_only()
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._dragging:
                self._dragging = False
                self._drag_start = None
                if self._anim.drag_active:
                    self._pan_acc[0] += self._anim.drag_target[0]
                    self._pan_acc[1] += self._anim.drag_target[1]
                    self._anim.drag_target = [0.0, 0.0]
                    self._anim.drag_active = False
                if settings and self._anim._animations_enabled(settings, context):
                    speed = max(abs(self._anim.smooth_velocity[0]), abs(self._anim.smooth_velocity[1]))
                    if speed > 2.0:
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
                if settings and settings.left_click_action in ("SELECT", "SELECT_PAN", "SELECT_FRAME"):
                    state.cache.force_immediate = True
                    selection.handle_click_selection(
                        self, context, event, state, frame=settings.left_click_action == "SELECT_FRAME"
                    )
                self._was_in_minimap = False
                self._drag_start = None
                return {"RUNNING_MODAL"}
            self._was_in_minimap = False
            self._drag_start = None
            return {"PASS_THROUGH"}
        # --- Press ---
        self._was_in_minimap = in_minimap
        if self._was_in_minimap:
            self._anim.cancel_smooth(context)
            armed_button_id = _frame_button_at(self._mouse_x, self._mouse_y, state)
            if armed_button_id:
                self._armed_button = armed_button_id
                return {"RUNNING_MODAL"}
            # List/map divider — same style as outer resize borders, percent width.
            ui_scale = _get_ui_scale()
            divider_resize_handle = resize.get_list_divider_handle(state, self._mouse_x, self._mouse_y, ui_scale)
            if divider_resize_handle:
                self._list_width_dragging = True
                state.interaction.resize_active = divider_resize_handle
                self._redraw_ui()
                self._list_width_start_x = self._mouse_x
                self._list_width_start_pct = settings.type_list_width_pct
                map_x, map_y, map_w, map_h = state.view.rect
                self._list_width_start_map_w = map_w
                cursor = _CURSOR_MAP[divider_resize_handle]
                context.window.cursor_modal_set(cursor)
                self._last_cursor = cursor
                return {"RUNNING_MODAL"}
            if _in_list_zone(self._mouse_x, self._mouse_y, state):
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
                    self._list_child_pressed = child_row
                    if not event.shift and not event.ctrl:
                        child_label, child_node_name = child_row
                        key = ("child", child_label, child_node_name)
                        self._list_last_row_index = state.list.visible_row_index_map.get(key, -1)
                else:
                    row_label = _list_row_at(self._mouse_x, self._mouse_y, state)
                    if row_label:
                        toggle_rect = state.list.toggle_rects.get(row_label)
                        if toggle_rect and _in_rect(self._mouse_x, self._mouse_y, toggle_rect):
                            self._list_toggle_pressed = row_label
                        else:
                            self._list_row_pressed = row_label
                            if not event.shift and not event.ctrl:
                                key = ("header", row_label)
                                self._list_last_row_index = state.list.visible_row_index_map.get(key, -1)
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
                    cursor = _CURSOR_MAP[resize_handle]
                    context.window.cursor_modal_set(cursor)
                    self._last_cursor = cursor
                    return {"RUNNING_MODAL"}
            if settings and settings.left_click_action in ("PAN", "SELECT_PAN"):
                self._drag_start = (self._mouse_x, self._mouse_y)
                self._center_view_on_mouse(context, self._mouse_x, self._mouse_y)
            return {"RUNNING_MODAL"}
        else:
            self._drag_start = None
            return {"PASS_THROUGH"}

    def _handle_right_mouse(self, context: Context, event: Event) -> set[str]:
        state, addon, settings, in_minimap = self._minimap_event_context(context)
        # --- Release ---
        if event.value == "RELEASE":
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
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
                state.view.width_clamped = False
                state.view.height_clamped = False
                state.interaction.hovered_handle = None
                state.interaction.resize_active = None
                state.cache.invalidate_batches_only()
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._dragging:
                self._dragging = False
                self._drag_start = None
                if self._anim.drag_active:
                    self._pan_acc[0] += self._anim.drag_target[0]
                    self._pan_acc[1] += self._anim.drag_target[1]
                    self._anim.drag_target = [0.0, 0.0]
                    self._anim.drag_active = False
                if settings and self._anim._animations_enabled(settings, context):
                    speed = max(abs(self._anim.smooth_velocity[0]), abs(self._anim.smooth_velocity[1]))
                    if speed > 2.0:
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
            self._was_in_minimap = False
            self._drag_start = None
            return {"PASS_THROUGH"}
        # --- Press ---
        self._was_in_minimap = in_minimap
        if self._was_in_minimap:
            self._anim.cancel_smooth(context)
            ui_scale = _get_ui_scale()
            divider_handle_r = resize.get_list_divider_handle(state, self._mouse_x, self._mouse_y, ui_scale)
            if divider_handle_r:
                self._list_width_dragging = True
                state.interaction.resize_active = divider_handle_r
                self._redraw_ui()
                self._list_width_start_x = self._mouse_x
                self._list_width_start_pct = settings.type_list_width_pct if settings else 35
                map_x, map_y, map_w, map_h = state.view.rect
                self._list_width_start_map_w = map_w
                cursor = _CURSOR_MAP[divider_handle_r]
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
                    state.cache.force_immediate = True
                    if event.shift:
                        selection.apply_list_range(self, context, state, ("child", child_label, node_name))
                    elif event.ctrl:
                        selection.select_single_node(self, context, node_name, toggle=True)
                    else:
                        selection.select_single_node(self, context, node_name)
                        key = ("child", child_label, node_name)
                        self._list_last_row_index = state.list.visible_row_index_map.get(key, -1)
                    if not (event.shift or event.ctrl):
                        if not self._anim.view_selected_animated(context, settings):
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
                        state.cache.force_immediate = True
                        self._redraw_ui()
                        self._was_in_minimap = False
                        return {"RUNNING_MODAL"}
                    state.cache.force_immediate = True
                    if event.shift:
                        selection.apply_list_range(self, context, state, ("header", row_label))
                    elif event.ctrl:
                        selection.select_type_nodes(self, context, row_label, toggle=True)
                    else:
                        selection.select_type_nodes(self, context, row_label)
                        key = ("header", row_label)
                        self._list_last_row_index = state.list.visible_row_index_map.get(key, -1)
                    if not (event.shift or event.ctrl):
                        if not self._anim.view_selected_animated(context, settings):
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
                    cursor = _CURSOR_MAP[resize_handle]
                    context.window.cursor_modal_set(cursor)
                    self._last_cursor = cursor
                    return {"RUNNING_MODAL"}
            if settings and settings.right_click_action in ("SELECT", "SELECT_PAN", "SELECT_FRAME"):
                state.cache.force_immediate = True
                selection.handle_click_selection(
                    self, context, event, state, frame=settings.right_click_action == "SELECT_FRAME"
                )
            if settings and settings.right_click_action in ("PAN", "SELECT_PAN"):
                self._drag_start = (self._mouse_x, self._mouse_y)
                self._center_view_on_mouse(context, self._mouse_x, self._mouse_y)
            self._was_in_minimap = False
            return {"RUNNING_MODAL"}
        else:
            self._drag_start = None
            return {"PASS_THROUGH"}

    def _handle_mouse_move(self, context: Context, event: Event) -> set[str]:
        state, addon, settings, in_minimap = self._minimap_event_context(context)
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
        if not self._dragging and not self._mmb_dragging and not self._drag_start:
            self._update_cursor(context, event)
        if not self._dragging and not self._mmb_dragging and not self._resize_handle and not self._list_width_dragging:
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
            row_label = None if over_bar else (_list_row_at(self._mouse_x, self._mouse_y, state) if in_list else None)
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
                self._anim.smooth_velocity[0] *= 0.15
                self._anim.smooth_velocity[1] *= 0.15
            else:
                self._anim.smooth_velocity[0] = self._anim.smooth_velocity[0] * 0.6 + dx * 0.4
                self._anim.smooth_velocity[1] = self._anim.smooth_velocity[1] * 0.6 + dy * 0.4
            pan_before = state.view.pan
            state.view.pan = (state.view.pan[0] + dx, state.view.pan[1] + dy)
            _clamp_pan_to_viewport(self._space, self._region, state)
            rejected_x = dx - (state.view.pan[0] - pan_before[0])
            rejected_y = dy - (state.view.pan[1] - pan_before[1])
            if (rejected_x != 0 or rejected_y != 0) and settings and settings.follow_view:
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
                    smooth = settings and self._anim._animations_enabled(settings, context, default=False)
                    self._pan_view(context, dx, dy, smooth)
                    self._drag_start = (self._mouse_x, self._mouse_y)
            return {"RUNNING_MODAL"}
        if in_minimap:
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def _handle_wheel(self, context: Context, event: Event) -> set[str]:
        state, addon, settings, in_minimap = self._minimap_event_context(context)
        if in_minimap and _in_list_zone(self._mouse_x, self._mouse_y, state):
            direction = -1 if event.type == "WHEELUPMOUSE" else 1
            state.list.scroll = min(
                max(state.list.scroll + direction * state.list.row_height * 3, 0.0), state.list.scroll_max
            )
            over_bar = _list_scrollbar_hit(self._mouse_x, self._mouse_y, state)
            state.list.hovered_type_label = None if over_bar else _list_row_at(self._mouse_x, self._mouse_y, state)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if in_minimap:
            if event.ctrl or event.shift:
                visible = _get_visible_rect(self._space, self._region)
                if visible:
                    ui_scale = _get_ui_scale()
                    vw = (visible[2] - visible[0]) * ui_scale
                    vh = (visible[3] - visible[1]) * ui_scale
                    scroll_factor = 0.05
                    direction = 1 if event.type == "WHEELUPMOUSE" else -1
                    pan_x = int(vw * scroll_factor * -direction) if event.ctrl else 0
                    pan_y = int(vh * scroll_factor * direction) if event.shift else 0
                    try:
                        with self._override_ctx(context):
                            bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                    except RuntimeError:
                        pass
                self._redraw_ui()
                return {"RUNNING_MODAL"}

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
                zoom_delta = 1.15 if event.type == "WHEELUPMOUSE" else 0.85
                effective_zoom = state.view.user_zoom

                is_constrained = False
                if addon and addon.settings.follow_view:
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
                    new_zoom = max(0.1, min(effective_zoom * zoom_delta, 20.0))

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
            if settings:
                settings.show_type_list = not settings.show_type_list
                start_list_width_animation(state, settings)
                self._redraw_ui()
            return
        self._dispatch_frame_action(context, settings, button_id)

    def _dispatch_frame_action(self, context: Context, settings, button_id: str) -> None:
        """Run a frame action directly, or eased via animation when smooth pan applies.

        Shared by the minimap button release and the Home / Numpad shortcuts.
        """
        state = self._state
        if not state:
            return
        smooth = bool(settings) and self._anim._animations_enabled(settings, context)
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
                        fill = settings.frame_view_fill
                        targets = _compute_frame_to_bounds_targets(visible, fill, area_ptr)
                        self._anim.start_frame_animation(context, targets[0], [targets[1], targets[2]])
                else:
                    frame_view(self._space, self._region, area_ptr)
            case "SELECTED":
                if smooth:
                    targets = _compute_frame_selected_targets(self._space, self._region, area_ptr)
                    if targets:
                        target_zoom = targets[0] if targets[0] is not None else state.view.user_zoom
                        self._anim.start_frame_animation(context, target_zoom, [targets[1], targets[2]])
                else:
                    frame_selected(self._space, self._region, area_ptr)

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
        addon = get_addon_preferences(context)
        settings = addon.settings if addon else None
        if settings and self._anim._animations_enabled(settings, context):
            self._anim.anim_target = [float(pan_x), float(pan_y)]
            self._anim.anim_applied = [0.0, 0.0]
            self._anim.anim_progress = 0.0
            self._anim.anim_acc = [0.0, 0.0]
            self._anim.anim_active = True
            self._anim.create_timer(context)
        else:
            try:
                with self._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                _clamp_pan_to_viewport(self._space, self._region, state)
            except RuntimeError:
                pass

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
        context.window.cursor_modal_set("DEFAULT")
        self._last_cursor = ""
        self._armed_button = None
        self._list_row_pressed = None
        self._list_child_pressed = None
        self._list_toggle_pressed = None
        self._list_scroll_pressed = False
        self._list_scroll_grab = 0.0
        state = self._state
        if state:
            state.buttons.hovered_button_id = None
            state.list.hovered_type_label = None
            state.interaction.hovered_node_id = None
            state.list.hovered_scrollbar = False
            state.list.scrollbar_dragging = False
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
        # Divider takes precedence over outer borders and list hover.
        ui_scale = _get_ui_scale()
        divider = resize.get_list_divider_handle(state, self._mouse_x, self._mouse_y, ui_scale)
        if divider:
            old_handle = state.interaction.hovered_handle
            state.interaction.hovered_handle = divider
            if divider != old_handle:
                self._redraw_ui()
            cursor = "HAND" if state.list.width_clamped else _CURSOR_MAP.get(divider, "MOVE_X")
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
            if self._last_cursor:
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
            return
        is_clamped = handle and (state.view.width_clamped or state.view.height_clamped)
        cursor = "HAND" if is_clamped else _CURSOR_MAP.get(handle, "DEFAULT")
        if cursor != self._last_cursor:
            context.window.cursor_modal_set(cursor)
            self._last_cursor = cursor

    def _get_handle_at(self, context: Context, event: Event) -> str | None:
        state = self._state
        if not state:
            return None
        addon = get_addon_preferences(context)
        if not addon:
            return None
        corner = addon.settings.position
        ui_scale = _get_ui_scale()
        return resize.get_resize_handle(state, corner, self._mouse_x, self._mouse_y, ui_scale)

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
        self._list_last_row_index = -1
        self._list_width_dragging = False
        self._list_width_start_x = 0
        self._list_width_start_pct = 35
        self._list_width_start_map_w = 0.0
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
            self._state.interaction.hovered_handle = None
            self._state.interaction.resize_active = None
            self._state.buttons.hovered_button_id = None
            self._state.list.hovered_type_label = None
            self._state.list.hovered_list_row = None
            self._state.interaction.hovered_node_id = None
            self._state.list.hovered_scrollbar = False
            self._state.list.scrollbar_dragging = False
        self._list_row_pressed = None
        self._list_child_pressed = None
        self._list_toggle_pressed = None
        self._list_scroll_pressed = False
        self._list_scroll_grab = 0.0
        self._list_last_row_index = -1
        self._list_width_dragging = False


class NODEMAP_OT_open_preferences(Operator):
    bl_idname = "nodemap.open_pref"
    bl_label = "Open Preferences"
    bl_description = "Open the add-on preferences panel"

    def execute(self, context):
        bpy.ops.screen.userpref_show()
        context.preferences.active_section = "ADDONS"
        context.window_manager.addon_search = "Nodemap"
        try:
            import addon_utils

            module = __package__
            mod = addon_utils.addons_fake_modules.get(module)

            if mod is not None:
                bl_info = addon_utils.module_bl_info(mod)

                if not bl_info["show_expanded"]:
                    bpy.ops.preferences.addon_expand(module=module)

        except RuntimeError as e:
            self.report({"WARNING"}, f"Could not expand addon: {e}")

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
