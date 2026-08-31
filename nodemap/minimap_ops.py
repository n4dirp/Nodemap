"""Modal operator for minimap interaction."""

import logging

import bpy
from bpy.types import Area, Context, Event, Operator, Region, SpaceNodeEditor

from .framing import (
    _compute_editor_frame_selected_targets,
    _compute_frame_all_targets,
    _compute_frame_selected_targets,
    _compute_frame_to_bounds_targets,
    frame_all,
    frame_selected,
    frame_view,
)
from .helpers import (
    _HANDLE_THICKNESS,
    _SCROLLBAR_HIT_PAD,
    _TYPE_LIST_MAX_WIDTH_PCT,
    _TYPE_LIST_MIN_WIDTH,
    MIN_MAP_HEIGHT,
    MIN_MAP_WIDTH,
    _expand_bounds_margin,
    _find_node_at,
    _get_area_and_region_under_mouse,
    _get_minimap_margins,
    _get_node_dims,
    _get_node_tree_bounds,
    _get_safe_bounds,
    _get_ui_scale,
    redraw_ui,
    start_list_width_animation,
)
from .state import (
    MinimapState,
    ResizeHandle,
    _minimap_window_operators,
    _state,
)
from .transforms import (
    _clamp_pan_to_viewport,
    _compute_map_transform,
    _get_minimap_transform,
    _get_visible_rect,
)

logger = logging.getLogger(__package__)


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
    hit_pad = _HANDLE_THICKNESS * _get_ui_scale()
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
    hit_pad = _SCROLLBAR_HIT_PAD * _get_ui_scale()
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


def _get_list_divider_handle(state: MinimapState, region_x: int, region_y: int, ui_scale: float) -> ResizeHandle | None:
    """Return ``LIST`` when the cursor is over the divider between list and map.

    The divider spans the gap (``6*scale``) between the list zone's right edge
    and the map content's left edge and uses the same hit thickness as the
    outer resize borders.
    """
    if state.list.list_width <= 0 or not state.list.list_zone_rect or not state.view.rect:
        return None
    zone_x, zone_y, zone_w, zone_h = state.list.list_zone_rect
    # Hit zone starts at the zone's right edge (never reaching into the
    # scrollbar) and extends right into the map gutter for reachability.
    zone_right_edge = zone_x + zone_w
    hit_half_width = (_HANDLE_THICKNESS - 1) * ui_scale
    if zone_right_edge <= region_x <= zone_right_edge + hit_half_width and zone_y <= region_y <= zone_y + zone_h:
        return ResizeHandle.LIST
    return None


def _get_resize_handle(
    state: MinimapState, corner: str, region_x: int, region_y: int, ui_scale: float
) -> ResizeHandle | None:
    map_x, map_y, map_w, map_h = state.view.rect
    if map_w <= 0 or map_h <= 0:
        return None
    half_w = _HANDLE_THICKNESS * ui_scale

    def is_near_edge(value, target):
        return target - half_w <= value <= target + half_w

    match corner:
        case "TOP_RIGHT":
            on_left = map_x <= region_x <= map_x + half_w
            on_bottom = map_y <= region_y <= map_y + half_w
            if on_left and on_bottom:
                return ResizeHandle.C
            if on_left:
                return ResizeHandle.W
            if on_bottom:
                return ResizeHandle.H
        case "TOP_LEFT":
            on_right = map_x + map_w - half_w <= region_x <= map_x + map_w
            on_bottom = map_y <= region_y <= map_y + half_w
            if on_right and on_bottom:
                return ResizeHandle.C
            if on_right:
                return ResizeHandle.W
            if on_bottom:
                return ResizeHandle.H
        case "BOTTOM_RIGHT":
            on_left = map_x <= region_x <= map_x + half_w
            on_top = map_y + map_h - half_w <= region_y <= map_y + map_h
            if on_left and on_top:
                return ResizeHandle.C
            if on_left:
                return ResizeHandle.W
            if on_top:
                return ResizeHandle.H
        case "BOTTOM_LEFT":
            on_right = map_x + map_w - half_w <= region_x <= map_x + map_w
            on_top = map_y + map_h - half_w <= region_y <= map_y + map_h
            if on_right and on_top:
                return ResizeHandle.C
            if on_right:
                return ResizeHandle.W
            if on_top:
                return ResizeHandle.H
        case _:
            return None


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

    _smooth_timer: str | None = None
    _inertia_active: bool = False
    _inertia_mode: str | None = None
    _smooth_velocity: list[float]
    _anim_active: bool = False
    _anim_target: list[float]
    _anim_applied: list[float]
    _anim_progress: float
    _anim_acc: list[float]
    _drag_target: list[float]
    _drag_active: bool = False
    _frame_anim_active: bool = False
    _frame_anim_start_zoom: float = 1.0
    _frame_anim_start_pan: list[float]
    _frame_anim_target_zoom: float = 1.0
    _frame_anim_target_pan: list[float]
    _editor_anim_active: bool = False
    _editor_anim_progress: float = 0.0
    _editor_anim_start_rect: list[float]
    _editor_anim_target_rect: list[float]

    def _animations_enabled(self, settings, context, default=True) -> bool:
        if context.preferences.view.use_reduce_motion:
            return False
        if settings is None:
            return default
        return settings.animations

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
            or self._anim_active
            or self._inertia_active
            or self._drag_active
            or self._frame_anim_active
            or self._editor_anim_active
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

        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
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
                    self._cancel_smooth(context)
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
                    if settings and self._animations_enabled(settings, context):
                        speed = max(abs(self._smooth_velocity[0]), abs(self._smooth_velocity[1]))
                        if speed > 2.0:
                            self._inertia_active = True
                            self._inertia_mode = "PAN"
                            self._create_timer(context)
                            self._redirect_acc = [0.0, 0.0]
                            return {"RUNNING_MODAL"}
                    self._smooth_velocity = [0.0, 0.0]
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
                if self._drag_active:
                    self._apply_smooth_drag(context)
                    return {"RUNNING_MODAL"}
                if self._inertia_active:
                    self._apply_inertia(context)
                    return {"RUNNING_MODAL"}
                if self._frame_anim_active:
                    self._apply_frame_animation(context)
                    return {"RUNNING_MODAL"}
                if self._editor_anim_active:
                    self._apply_editor_animation(context)
                    return {"RUNNING_MODAL"}
                if self._anim_active:
                    self._apply_center_animation(context)
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}
            case _:
                return {"PASS_THROUGH"}

    def _minimap_event_context(self, context: Context):
        """Resolve the shared per-event values used by the event handlers.

        Returns the minimap state, the add-on (if registered), its settings, and
        whether the cursor is currently over the minimap.
        """
        state = self._state
        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
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
                        self._apply_list_range(context, state, ("child", label, node_name))
                    elif event.ctrl:
                        self._select_single_node(context, node_name, toggle=True)
                    else:
                        self._select_single_node(context, node_name)
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
                        self._apply_list_range(context, state, ("header", label))
                    elif event.ctrl:
                        self._select_type_nodes(context, label, toggle=True)
                    else:
                        self._select_type_nodes(context, label)
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
                if self._drag_active:
                    self._pan_acc[0] += self._drag_target[0]
                    self._pan_acc[1] += self._drag_target[1]
                    self._drag_target = [0.0, 0.0]
                    self._drag_active = False
                if settings and self._animations_enabled(settings, context):
                    speed = max(abs(self._smooth_velocity[0]), abs(self._smooth_velocity[1]))
                    if speed > 2.0:
                        self._inertia_active = True
                        self._inertia_mode = "VIEW"
                        if not self._smooth_timer:
                            self._create_timer(context)
                        return {"RUNNING_MODAL"}
                self._smooth_velocity = [0.0, 0.0]
                pan_x = int(self._pan_acc[0])
                pan_y = int(self._pan_acc[1])
                self._pan_acc = [0.0, 0.0]
                if pan_x != 0 or pan_y != 0:
                    try:
                        with self._override_ctx(context):
                            bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                    except RuntimeError:
                        pass
                self._destroy_timer(context)
                return {"RUNNING_MODAL"}
            if not self._dragging and self._was_in_minimap:
                if settings and settings.left_click_action in ("SELECT", "SELECT_PAN", "SELECT_FRAME"):
                    state.cache.force_immediate = True
                    self._handle_click_selection(
                        context, event, state, frame=settings.left_click_action == "SELECT_FRAME"
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
            self._cancel_smooth(context)
            armed_button_id = _frame_button_at(self._mouse_x, self._mouse_y, state)
            if armed_button_id:
                self._armed_button = armed_button_id
                return {"RUNNING_MODAL"}
            # List/map divider — same style as outer resize borders, percent width.
            ui_scale = _get_ui_scale()
            divider_resize_handle = _get_list_divider_handle(state, self._mouse_x, self._mouse_y, ui_scale)
            if divider_resize_handle:
                self._list_width_dragging = True
                state.interaction.resize_active = divider_resize_handle
                self._redraw_ui()
                self._list_width_start_x = self._mouse_x
                self._list_width_start_pct = settings.type_list_width_pct if settings else 35
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
                        try:
                            self._list_last_row_index = state.list.visible_row_keys.index(key)
                        except ValueError:
                            self._list_last_row_index = -1
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
                                try:
                                    self._list_last_row_index = state.list.visible_row_keys.index(key)
                                except ValueError:
                                    self._list_last_row_index = -1
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
                if self._drag_active:
                    self._pan_acc[0] += self._drag_target[0]
                    self._pan_acc[1] += self._drag_target[1]
                    self._drag_target = [0.0, 0.0]
                    self._drag_active = False
                if settings and self._animations_enabled(settings, context):
                    speed = max(abs(self._smooth_velocity[0]), abs(self._smooth_velocity[1]))
                    if speed > 2.0:
                        self._inertia_active = True
                        self._inertia_mode = "VIEW"
                        if not self._smooth_timer:
                            self._create_timer(context)
                        return {"RUNNING_MODAL"}
                self._smooth_velocity = [0.0, 0.0]
                pan_x = int(self._pan_acc[0])
                pan_y = int(self._pan_acc[1])
                self._pan_acc = [0.0, 0.0]
                if pan_x != 0 or pan_y != 0:
                    try:
                        with self._override_ctx(context):
                            bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                    except RuntimeError:
                        pass
                self._destroy_timer(context)
                return {"RUNNING_MODAL"}
            self._was_in_minimap = False
            self._drag_start = None
            return {"PASS_THROUGH"}
        # --- Press ---
        self._was_in_minimap = in_minimap
        if self._was_in_minimap:
            self._cancel_smooth(context)
            ui_scale = _get_ui_scale()
            divider_handle_r = _get_list_divider_handle(state, self._mouse_x, self._mouse_y, ui_scale)
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
                        self._apply_list_range(context, state, ("child", child_label, node_name))
                    elif event.ctrl:
                        self._select_single_node(context, node_name, toggle=True)
                    else:
                        self._select_single_node(context, node_name)
                        key = ("child", child_label, node_name)
                        try:
                            self._list_last_row_index = state.list.visible_row_keys.index(key)
                        except ValueError:
                            self._list_last_row_index = -1
                    if not (event.shift or event.ctrl):
                        if not self._view_selected_animated(context, settings):
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
                        self._apply_list_range(context, state, ("header", row_label))
                    elif event.ctrl:
                        self._select_type_nodes(context, row_label, toggle=True)
                    else:
                        self._select_type_nodes(context, row_label)
                        key = ("header", row_label)
                        try:
                            self._list_last_row_index = state.list.visible_row_keys.index(key)
                        except ValueError:
                            self._list_last_row_index = -1
                    if not (event.shift or event.ctrl):
                        if not self._view_selected_animated(context, settings):
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
                self._handle_click_selection(context, event, state, frame=settings.right_click_action == "SELECT_FRAME")
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
            self._apply_list_width_drag(context)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if self._resize_handle:
            self._resize_apply_delta(context, event)
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
                self._smooth_velocity[0] *= 0.15
                self._smooth_velocity[1] *= 0.15
            else:
                self._smooth_velocity[0] = self._smooth_velocity[0] * 0.6 + dx * 0.4
                self._smooth_velocity[1] = self._smooth_velocity[1] * 0.6 + dy * 0.4
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
                if not self._dragging and (self._anim_active or self._frame_anim_active or self._editor_anim_active):
                    self._cancel_smooth(context)
                self._dragging = True
                if self._was_in_minimap:
                    state.interaction.pressed = True
                    smooth = settings and self._animations_enabled(settings, context, default=False)
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

            prefs = addon.preferences.settings if addon else None
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
                if addon and addon.preferences.settings.follow_view:
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

    def _node_select_location(self, node) -> tuple[float, float]:
        """Tree-space coordinate to emulate a click on *node*.

        Native Blender only selects frame nodes when clicked on their
        header/border, so frames are redirected to their header. Other nodes
        reuse the minimap hit-test dims so collapsed or never-drawn nodes
        probe inside their actual drawn bounds.
        """
        if node.type == "FRAME":
            return node.location_absolute.x + 15, node.location_absolute.y - 15
        w, h = _get_node_dims(node)
        return node.location_absolute.x + w / 2.0, node.location_absolute.y - h / 2.0

    def _node_fallback_location(self, node) -> tuple[float, float]:
        """Tree-space point near the node's top-left interior edge.

        Falls inside a collapsed node's header strip whatever its drawn label
        width, covering center probes that miss due to stale dimensions.
        """
        x, y = node.location_absolute.x, node.location_absolute.y
        if node.type == "FRAME":
            return x + 15, y - 15
        return x + 10.0, y - 10.0

    def _project_tree_to_region(self, tree_x: float, tree_y: float) -> tuple[int, int] | None:
        """Project a tree-space point to editor region pixels via view2d.

        View2d coordinates are tree coordinates scaled by the UI scale factor
        (mirroring ``_get_visible_rect``), so the point is scaled here first.
        """
        view2d = self._region.view2d if self._region else None
        if not view2d:
            return None
        ui = _get_ui_scale()
        pt = view2d.view_to_region(tree_x * ui, tree_y * ui, clip=False)
        if not pt:
            return None
        return int(pt[0]), int(pt[1])

    def _select_node_via_operator(self, context: Context, node, extend: bool, deselect_all: bool) -> bool:
        """Select *node* via the native ``node.select`` operator.

        Projects candidate tree positions into the editor's region coordinates
        and passes those to ``bpy.ops.node.select``, emulating a standard UI
        click. This avoids the NodeTree "modified" tag that Python property
        assignment (``node.select = True``) triggers, which forces a full EEVEE
        material rebuild. Probes run from the node center to its header edge
        and each pick is verified against ``node.select``, so a silent miss
        retries instead of reporting success; returns False only when no probe
        selects the node so callers can fall back to the property API.
        """
        if not self._region or not self._region.view2d:
            return False
        probes = [self._node_select_location(node)]
        fallback = self._node_fallback_location(node)
        if fallback != probes[0]:
            probes.append(fallback)

        tree_nodes = getattr(getattr(node, "id_data", None), "nodes", None)
        for probe_index, (probe_tx, probe_ty) in enumerate(probes):
            projected = self._project_tree_to_region(probe_tx, probe_ty)
            if projected is None:
                continue
            proj_x, proj_y = projected
            keep = None
            if probe_index and extend and tree_nodes:
                keep = {n.name for n in tree_nodes if n.select}
            kwargs: dict = {"extend": extend}
            if bpy.app.version >= (3, 0, 0):
                kwargs["location"] = (proj_x, proj_y)
                kwargs["deselect_all"] = deselect_all
            else:
                kwargs["mouse_x"] = proj_x
                kwargs["mouse_y"] = proj_y
            try:
                with self._override_ctx(context):
                    bpy.ops.node.select(**kwargs)
            except Exception as e:
                logger.debug("Failed to select via operator: %s", e)
                continue
            try:
                if not node.select:
                    continue
            except ReferenceError:
                return False
            if keep is not None and tree_nodes:
                # Release neighbors an overlapping retry probe picked up unintentionally.
                for other in tree_nodes:
                    if other.select and other.name != node.name and other.name not in keep:
                        other.select = False
            return True
        return False

    def _handle_click_selection(self, context: Context, event: Event, state: dict, frame: bool = False) -> None:
        space = self._space
        if not space or space.type != "NODE_EDITOR":
            return
        node_tree = space.edit_tree
        if not node_tree or not node_tree.nodes:
            return

        tree_coord = _region_to_tree(self._mouse_x, self._mouse_y, state)
        if tree_coord is None:
            return

        node = _find_node_at(node_tree.nodes, tree_coord[0], tree_coord[1])
        if node:
            if not self._select_node_via_operator(context, node, extend=event.shift, deselect_all=not event.shift):
                # Fallback for API changes (may trigger EEVEE compile)
                if event.shift:
                    node.select = not node.select
                    if node.select:
                        node_tree.nodes.active = node
                else:
                    for n in node_tree.nodes:
                        n.select = False
                    node.select = True
                    node_tree.nodes.active = node

            if frame:
                addon = context.preferences.addons.get(__package__)
                settings = addon.preferences.settings if addon else None
                if not (settings and self._view_selected_animated(context, settings)):
                    try:
                        with self._override_ctx(context):
                            bpy.ops.node.view_selected()
                    except RuntimeError:
                        pass

        state.list.hovered_list_row = None
        state.interaction.hovered_node_id = None
        self._redraw_ui()

    def _apply_list_range(self, context: Context, state: MinimapState, target_key: tuple) -> None:
        """Select all visible rows between the last-clicked and *target_key*.

        Replaces the current selection with the contiguous range, matching
        standard file-explorer Shift-click behaviour.  The anchor
        (``_list_last_row_index``) is **not** moved — it stays at the last
        plain-clicked row so repeated Shift-clicks expand from the same origin.
        """
        keys = state.list.visible_row_keys
        try:
            target_idx = keys.index(target_key)
        except ValueError:
            return
        last = self._list_last_row_index
        if last < 0 or last >= len(keys):
            lo, hi = target_idx, target_idx
        else:
            lo, hi = min(last, target_idx), max(last, target_idx)

        # Deselect everything first so the range *replaces* the selection.
        space = self._space
        node_tree = space.edit_tree if space else None
        if not node_tree:
            return
        try:
            with self._override_ctx(context):
                bpy.ops.node.select_all(action="DESELECT")
        except RuntimeError:
            pass

        for key_idx in range(lo, hi + 1):
            key = keys[key_idx]
            if key[0] == "header":
                self._select_type_nodes(context, key[1], extend=True)
            elif key[0] == "child":
                node = node_tree.nodes.get(key[2])
                if node:
                    if not self._select_node_via_operator(context, node, extend=True, deselect_all=False):
                        node.select = True
        self._redraw_ui()

    def _select_type_nodes(self, context: Context, label: str, extend: bool = False, toggle: bool = False) -> None:
        """Select all editor nodes whose compiled type label matches *label*.

        When *extend* is True the current selection is preserved and the
        matching nodes are added.  When *toggle* is True the behaviour
        depends on the current state: if every matching node is already
        selected they are all deselected, otherwise they are all selected.
        """
        space = self._space
        state = self._state
        if not space or space.type != "NODE_EDITOR" or not state:
            return
        node_tree = space.edit_tree
        if not node_tree:
            return
        type_nodes = (state.cache.tree_data or {}).get("type_nodes") or {}
        names = type_nodes.get(label)
        if not names:
            return

        if toggle:
            all_sel = all((node_tree.nodes.get(n) is not None and node_tree.nodes[n].select) for n in names)
            if all_sel:
                # Deselect only this type group, preserving other selections.
                for name in names:
                    node = node_tree.nodes.get(name)
                    if node and node.select:
                        node.select = False
                self._redraw_ui()
                return
            else:
                deselect = False
                extend = True
        else:
            deselect = not extend

        if deselect:
            try:
                with self._override_ctx(context):
                    bpy.ops.node.select_all(action="DESELECT")
            except RuntimeError:
                pass
        # After the upfront deselect the selection is empty, so every
        # addition must use extend to accumulate the whole group. Passing
        # extend=False would replace the previous node on each iteration
        # and leave only the last one selected (and framed).
        node_extend = extend or deselect
        for name in names:
            node = node_tree.nodes.get(name)
            if node:
                # Native operator keeps selection/additive state and sets the
                # active node without tagging the NodeTree for an EEVEE rebuild.
                if not self._select_node_via_operator(context, node, extend=node_extend, deselect_all=False):
                    node.select = True
        self._redraw_ui()

    def _select_single_node(self, context: Context, node_name: str, extend: bool = False, toggle: bool = False) -> None:
        """Select only the editor node whose compiled name matches *node_name*.

        When *extend* is True the current selection is preserved and the
        node is added.  When *toggle* is True the node's selection state
        is flipped instead of replaced.
        """
        space = self._space
        state = self._state
        if not space or space.type != "NODE_EDITOR" or not state:
            return
        node_tree = space.edit_tree
        if not node_tree:
            return
        node = node_tree.nodes.get(node_name)
        if not node:
            return
        if toggle:
            node.select = not node.select
            if node.select:
                node_tree.nodes.active = node
        else:
            if not extend:
                try:
                    with self._override_ctx(context):
                        bpy.ops.node.select_all(action="DESELECT")
                except RuntimeError:
                    pass
            if not self._select_node_via_operator(context, node, extend=extend, deselect_all=False):
                node.select = True
                node_tree.nodes.active = node
        self._redraw_ui()

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
        smooth = bool(settings) and self._animations_enabled(settings, context)
        area_ptr = self._area.as_pointer() if self._area else 0
        match button_id:
            case "ALL":
                if smooth:
                    targets = _compute_frame_all_targets(self._space, self._region, area_ptr)
                    if targets:
                        self._start_frame_animation(context, targets[0], [targets[1], targets[2]])
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
                        self._start_frame_animation(context, targets[0], [targets[1], targets[2]])
                else:
                    frame_view(self._space, self._region, area_ptr)
            case "SELECTED":
                if smooth:
                    targets = _compute_frame_selected_targets(self._space, self._region, area_ptr)
                    if targets:
                        target_zoom = targets[0] if targets[0] is not None else state.view.user_zoom
                        self._start_frame_animation(context, target_zoom, [targets[1], targets[2]])
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
            self._smooth_velocity[0] *= 0.15
            self._smooth_velocity[1] *= 0.15
        else:
            self._smooth_velocity[0] = self._smooth_velocity[0] * 0.6 + view_delta_x * 0.4
            self._smooth_velocity[1] = self._smooth_velocity[1] * 0.6 + view_delta_y * 0.4

        if smooth:
            self._drag_target[0] += view_delta_x
            self._drag_target[1] += view_delta_y
            if not self._drag_active:
                self._drag_active = True
                self._create_timer(context)
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
        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
        if settings and self._animations_enabled(settings, context):
            self._anim_target = [float(pan_x), float(pan_y)]
            self._anim_applied = [0.0, 0.0]
            self._anim_progress = 0.0
            self._anim_acc = [0.0, 0.0]
            self._anim_active = True
            self._create_timer(context)
        else:
            try:
                with self._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                _clamp_pan_to_viewport(self._space, self._region, state)
            except RuntimeError:
                pass

    def _create_timer(self, context: Context) -> None:
        if self._smooth_timer:
            return
        self._smooth_timer = context.window_manager.event_timer_add(1 / 60, window=context.window)

    def _destroy_timer(self, context: Context) -> None:
        if self._smooth_timer:
            try:
                context.window_manager.event_timer_remove(self._smooth_timer)
            except (RuntimeError, ValueError):
                pass
            self._smooth_timer = None

    def _cancel_interaction(self, context: Context) -> None:
        self._cancel_smooth(context)
        if self._dragging or self._drag_start is not None:
            self._dragging = False
            self._drag_start = None
            self._drag_active = False
            self._drag_target = [0.0, 0.0]
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

    def _cancel_smooth(self, context: Context) -> None:
        if self._inertia_active:
            self._inertia_active = False
            self._inertia_mode = None
            self._smooth_velocity = [0.0, 0.0]
            self._destroy_timer(context)
        if self._anim_active:
            if self._anim_applied[0] != self._anim_target[0] or self._anim_applied[1] != self._anim_target[1]:
                remaining_x = self._anim_target[0] - self._anim_applied[0]
                remaining_y = self._anim_target[1] - self._anim_applied[1]
                if abs(remaining_x) >= 0.5 or abs(remaining_y) >= 0.5:
                    try:
                        with self._override_ctx(context):
                            bpy.ops.view2d.pan(deltax=int(remaining_x), deltay=int(remaining_y))
                    except RuntimeError:
                        pass
            self._anim_active = False
            self._destroy_timer(context)
        if self._frame_anim_active:
            state = self._state
            if state:
                state.view.anchor_zoom = self._frame_anim_target_zoom
                state.view.user_zoom = self._frame_anim_target_zoom
                state.view.pan = (self._frame_anim_target_pan[0], self._frame_anim_target_pan[1])
                _clamp_pan_to_viewport(self._space, self._region, state)
            self._frame_anim_active = False
            self._frame_anim_progress = 0.0
            self._destroy_timer(context)
        if self._editor_anim_active:
            self._cancel_editor_animation(context)

    def _apply_inertia(self, context: Context) -> None:
        decay = 0.92
        self._smooth_velocity[0] *= decay
        self._smooth_velocity[1] *= decay
        speed = max(abs(self._smooth_velocity[0]), abs(self._smooth_velocity[1]))
        if speed < 0.5:
            self._inertia_active = False
            self._inertia_mode = None
            self._destroy_timer(context)
            return
        if self._inertia_mode == "PAN":
            state = self._state
            if state:
                self._pan_acc[0] += self._smooth_velocity[0]
                self._pan_acc[1] += self._smooth_velocity[1]
                dx = int(self._pan_acc[0])
                dy = int(self._pan_acc[1])
                self._pan_acc[0] -= dx
                self._pan_acc[1] -= dy
                if dx != 0 or dy != 0:
                    state.view.pan = (state.view.pan[0] + dx, state.view.pan[1] + dy)
                    _clamp_pan_to_viewport(self._space, self._region, state)
        elif self._inertia_mode == "VIEW":
            self._pan_acc[0] += self._smooth_velocity[0]
            self._pan_acc[1] += self._smooth_velocity[1]
            dx = int(self._pan_acc[0])
            dy = int(self._pan_acc[1])
            self._pan_acc[0] -= dx
            self._pan_acc[1] -= dy
            if dx != 0 or dy != 0:
                try:
                    with self._override_ctx(context):
                        bpy.ops.view2d.pan(deltax=dx, deltay=dy)
                except RuntimeError:
                    pass
                _clamp_pan_to_viewport(self._space, self._region, self._state)
        self._redraw_ui()

    def _apply_smooth_drag(self, context: Context) -> None:
        if not self._drag_active:
            return
        magnitude = (self._drag_target[0] ** 2 + self._drag_target[1] ** 2) ** 0.5
        raw = magnitude / 200.0
        follow = 0.25 + raw * 0.55
        follow = min(follow, 0.8)
        max_move = 120.0 + magnitude * 0.15
        max_move = min(max_move, 800.0)
        dx = self._drag_target[0] * follow
        dy = self._drag_target[1] * follow
        dx = max(min(dx, max_move), -max_move)
        dy = max(min(dy, max_move), -max_move)
        self._pan_acc[0] += dx
        self._pan_acc[1] += dy
        self._drag_target[0] -= dx
        self._drag_target[1] -= dy
        pan_x = int(self._pan_acc[0])
        pan_y = int(self._pan_acc[1])
        self._pan_acc[0] -= pan_x
        self._pan_acc[1] -= pan_y
        if pan_x != 0 or pan_y != 0:
            try:
                with self._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
            except RuntimeError:
                pass
            _clamp_pan_to_viewport(self._space, self._region, self._state)
        if not self._dragging:
            self._drag_active = False
        self._redraw_ui()

    def _apply_center_animation(self, context: Context) -> None:
        if not self._anim_active:
            return
        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
        speed = settings.pan_speed if settings else "MEDIUM"
        frames = {"FAST": 10, "MEDIUM": 20}.get(speed, 24)
        self._anim_progress += 1 / frames
        if self._anim_progress >= 1.0:
            remaining_x = self._anim_target[0] - self._anim_applied[0]
            remaining_y = self._anim_target[1] - self._anim_applied[1]
            if abs(remaining_x) >= 0.5 or abs(remaining_y) >= 0.5:
                try:
                    with self._override_ctx(context):
                        bpy.ops.view2d.pan(deltax=int(remaining_x), deltay=int(remaining_y))
                except RuntimeError:
                    pass
            self._anim_active = False
            self._destroy_timer(context)
            return
        eased = 1.0 - (1.0 - self._anim_progress) ** 3
        desired_x = self._anim_target[0] * eased
        desired_y = self._anim_target[1] * eased
        delta_x = desired_x - self._anim_applied[0]
        delta_y = desired_y - self._anim_applied[1]
        self._anim_applied[0] += delta_x
        self._anim_applied[1] += delta_y
        self._anim_acc[0] += delta_x
        self._anim_acc[1] += delta_y
        dx = int(self._anim_acc[0])
        dy = int(self._anim_acc[1])
        self._anim_acc[0] -= dx
        self._anim_acc[1] -= dy
        if dx != 0 or dy != 0:
            try:
                with self._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=dx, deltay=dy)
            except RuntimeError:
                pass
        self._redraw_ui()

    def _start_frame_animation(self, context: Context, target_zoom: float, target_pan: list[float]) -> None:
        state = self._state
        if not state:
            return
        if self._frame_anim_active:
            self._frame_anim_active = False
        self._frame_anim_progress = 0.0
        self._frame_anim_start_zoom = state.view.user_zoom
        self._frame_anim_start_pan = [state.view.pan[0], state.view.pan[1]]
        self._frame_anim_target_zoom = target_zoom
        self._frame_anim_target_pan = [target_pan[0], target_pan[1]]
        self._frame_anim_active = True
        self._create_timer(context)

    def _apply_frame_animation(self, context: Context) -> None:
        if not self._frame_anim_active:
            return
        state = self._state
        if not state:
            self._frame_anim_active = False
            self._destroy_timer(context)
            return
        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
        speed = settings.pan_speed if settings else "MEDIUM"
        frames = {"FAST": 10, "MEDIUM": 20}.get(speed, 24)
        progress = self._frame_anim_progress + 1 / frames
        self._frame_anim_progress = progress
        if progress >= 1.0:
            state.view.anchor_zoom = self._frame_anim_target_zoom
            state.view.user_zoom = self._frame_anim_target_zoom
            state.view.pan = (self._frame_anim_target_pan[0], self._frame_anim_target_pan[1])
            _clamp_pan_to_viewport(self._space, self._region, state)
            self._frame_anim_active = False
            self._frame_anim_progress = 0.0
            self._destroy_timer(context)
            self._redraw_ui()
            return
        eased = 1.0 - (1.0 - progress) ** 3
        state.view.user_zoom = (
            self._frame_anim_start_zoom + (self._frame_anim_target_zoom - self._frame_anim_start_zoom) * eased
        )
        state.view.anchor_zoom = state.view.user_zoom
        state.view.pan = (
            self._frame_anim_start_pan[0] + (self._frame_anim_target_pan[0] - self._frame_anim_start_pan[0]) * eased,
            self._frame_anim_start_pan[1] + (self._frame_anim_target_pan[1] - self._frame_anim_start_pan[1]) * eased,
        )
        _clamp_pan_to_viewport(self._space, self._region, state)
        self._redraw_ui()

    def _view_selected_animated(self, context: Context, settings) -> bool:
        """Ease the editor viewport onto the selected nodes; True when started.

        Falls back to False so callers can run the instant ``node.view_selected``
        operator when smooth pan is disabled or nothing is selected.
        """
        if not self._animations_enabled(settings, context):
            return False
        targets = _compute_editor_frame_selected_targets(self._space, self._region)
        if targets is None:
            return False
        self._start_editor_animation(context, list(targets))
        return True

    def _start_editor_animation(self, context: Context, target_rect: list[float]) -> None:
        """Begin animating the editor viewport toward the target tree-space rect."""
        visible = _get_visible_rect(self._space, self._region)
        if not visible:
            return
        # Already framed: skip the animation entirely so re-running focus
        # selected on the same node does nothing instead of micro-jittering.
        if self._editor_view_close(visible, target_rect):
            return
        self._editor_anim_progress = 0.0
        self._editor_anim_start_rect = [visible[0], visible[1], visible[2], visible[3]]
        self._editor_anim_target_rect = target_rect
        self._editor_anim_active = True
        self._create_timer(context)

    def _apply_editor_animation(self, context: Context) -> None:
        if not self._editor_anim_active:
            return
        if not self._space or not self._region or not self._state:
            self._editor_anim_active = False
            self._destroy_timer(context)
            return
        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
        speed = settings.pan_speed if settings else "MEDIUM"
        frames = {"FAST": 10, "MEDIUM": 20}.get(speed, 24)
        progress = self._editor_anim_progress + 1 / frames
        if progress >= 1.0:
            self._correct_editor_view(context, self._editor_anim_target_rect)
            self._editor_anim_active = False
            self._editor_anim_progress = 0.0
            self._destroy_timer(context)
            self._redraw_ui()
            return
        self._editor_anim_progress = progress
        eased = 1.0 - (1.0 - progress) ** 3
        desired = [
            start + (target - start) * eased
            for start, target in zip(self._editor_anim_start_rect, self._editor_anim_target_rect)
        ]
        self._correct_editor_view(context, desired)
        self._redraw_ui()

    def _editor_view_close(self, visible: tuple[float, float, float, float], target: list[float]) -> bool:
        """Return True when the editor viewport already frames *target*.

        Uses the same tolerances as ``_correct_editor_view`` stop conditions
        (0.5% size, sub-pixel center) so a node that is already framed is left
        untouched instead of being nudged every animation frame.
        """
        if not self._region:
            return False
        cur_w = max(visible[2] - visible[0], 1e-6)
        cur_h = max(visible[3] - visible[1], 1e-6)
        des_w = max(target[2] - target[0], 1e-6)
        des_h = max(target[3] - target[1], 1e-6)
        ratio = min(max(des_w / cur_w, des_h / cur_h), 1e6)
        if ratio < 1.0 - 0.005 or ratio > 1.0 + 0.005:
            return False
        vzx = self._region.width / cur_w
        vzy = self._region.height / cur_h
        dcx = (target[0] + target[2] - visible[0] - visible[2]) / 2
        dcy = (target[1] + target[3] - visible[1] - visible[3]) / 2
        return abs(dcx * vzx) <= 0.5 and abs(dcy * vzy) <= 0.5

    def _correct_editor_view(self, context: Context, desired: list[float]) -> None:
        """Nudge the editor view2d one monotonic step toward the desired rect.

        Issues at most a single zoom step and a single rounded pan per call so
        the animation can never overshoot and oscillate between frames. The
        view is re-read live each call, keeping the correction idempotent once
        it is within tolerance.
        """
        space = self._space
        region = self._region
        if not space or not region:
            return

        current = _get_visible_rect(space, region)
        if not current:
            return

        cur_w = max(current[2] - current[0], 1e-6)
        cur_h = max(current[3] - current[1], 1e-6)
        des_w = max(desired[2] - desired[0], 1e-6)
        des_h = max(desired[3] - desired[1], 1e-6)
        ratio = min(max(des_w / cur_w, des_h / cur_h), 1e6)
        if ratio < 1.0 - 0.005:
            fac = min((1.0 - ratio) / 2.0, 0.4)
            try:
                with self._override_ctx(context):
                    bpy.ops.view2d.zoom_in(zoomfacx=fac, zoomfacy=fac)
            except RuntimeError:
                pass
        elif ratio > 1.0 + 0.005:
            fac = max((1.0 / ratio - 1.0) / 2.0, -0.4)
            try:
                with self._override_ctx(context):
                    bpy.ops.view2d.zoom_out(zoomfacx=fac, zoomfacy=fac)
            except RuntimeError:
                pass

        current = _get_visible_rect(space, region)
        if not current:
            return

        view_zoom_x, view_zoom_y = _view_zoom_factors(space, region, current)
        dcx = (desired[0] + desired[2] - current[0] - current[2]) / 2
        dcy = (desired[1] + desired[3] - current[1] - current[3]) / 2
        pan_x = int(round(dcx * view_zoom_x))
        pan_y = int(round(dcy * view_zoom_y))
        if pan_x != 0 or pan_y != 0:
            try:
                with self._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
            except RuntimeError:
                pass

    def _cancel_editor_animation(self, context: Context) -> None:
        """Snap the editor viewport to the animation target and stop stepping."""
        if not self._editor_anim_active:
            return
        self._editor_anim_active = False
        self._editor_anim_progress = 0.0
        if self._space and self._region:
            self._correct_editor_view(context, self._editor_anim_target_rect)
        self._destroy_timer(context)

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
        divider = _get_list_divider_handle(state, self._mouse_x, self._mouse_y, ui_scale)
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
        addon = context.preferences.addons.get(__package__)
        if not addon:
            return None
        corner = addon.preferences.settings.position
        ui_scale = _get_ui_scale()
        return _get_resize_handle(state, corner, self._mouse_x, self._mouse_y, ui_scale)

    def _resize_apply_delta(self, context: Context, event: Event) -> None:
        addon = context.preferences.addons.get(__package__)
        if not addon:
            return
        settings = addon.preferences.settings
        if not self._resize_start_values:
            return
        w0, h0 = self._resize_start_values
        dx = self._mouse_x - self._resize_start_mouse[0]
        dy = self._mouse_y - self._resize_start_mouse[1]
        corner = settings.position

        ui_scale = _get_ui_scale()
        sx, sy, ex, ey = _get_safe_bounds(self._area, self._region)
        x_margin, y_margin, margin = _get_minimap_margins(self._space, corner, ui_scale)

        safe_w = ex - sx
        safe_h = ey - sy
        max_width_pct = settings.max_width_pct / 100.0
        max_height_pct = settings.max_height_pct / 100.0
        max_w = max(MIN_MAP_WIDTH, int((safe_w - 2 * x_margin) * max_width_pct))
        max_h = max(MIN_MAP_HEIGHT, int((safe_h - y_margin - margin) * max_height_pct))

        # Suppress property update callbacks during drag to avoid clearing
        # tree_data, which causes a one-frame flash while recompiling.
        from . import preferences as _pref_mod

        _pref_mod._suppress_update = True
        try:
            if self._resize_handle in (ResizeHandle.W, ResizeHandle.C):
                if corner in ("TOP_RIGHT", "BOTTOM_RIGHT"):
                    new_w = max(MIN_MAP_WIDTH, min(max_w, int(w0 - dx / ui_scale)))
                else:
                    new_w = max(MIN_MAP_WIDTH, min(max_w, int(w0 + dx / ui_scale)))
                settings.minimap_width = new_w

            if self._resize_handle in (ResizeHandle.H, ResizeHandle.C):
                if corner in ("TOP_RIGHT", "TOP_LEFT"):
                    new_h = max(MIN_MAP_HEIGHT, min(max_h, int(h0 - dy / ui_scale)))
                else:
                    new_h = max(MIN_MAP_HEIGHT, min(max_h, int(h0 + dy / ui_scale)))
                settings.minimap_height = new_h
        finally:
            _pref_mod._suppress_update = False

        state = self._state
        if not state:
            return
        state.interaction.hovered_handle = self._resize_handle
        state.view.width_clamped = settings.minimap_width >= max_w or settings.minimap_width <= MIN_MAP_WIDTH
        state.view.height_clamped = settings.minimap_height >= max_h or settings.minimap_height <= MIN_MAP_HEIGHT

    def _apply_list_width_drag(self, context: Context) -> None:
        """Update the type-list percent width from the current mouse delta."""
        state = self._state
        addon = context.preferences.addons.get(__package__)
        if not state or not addon:
            return
        settings = addon.preferences.settings
        if self._list_width_start_map_w <= 0:
            return
        dx = self._mouse_x - self._list_width_start_x
        map_w = self._list_width_start_map_w
        ui_scale = _get_ui_scale()
        min_w = _TYPE_LIST_MIN_WIDTH * ui_scale
        max_w = map_w * _TYPE_LIST_MAX_WIDTH_PCT
        start_w = map_w * (self._list_width_start_pct / 100.0)
        start_w = min(max(start_w, min_w), max_w)
        new_w = min(max(start_w + dx, min_w), max_w)
        new_pct = int(round(new_w / max(map_w, 1.0) * 100.0))
        new_pct = min(max(new_pct, 15), 50)
        from . import preferences as _pref_mod

        _pref_mod._suppress_update = True
        try:
            settings.type_list_width_pct = new_pct
        finally:
            _pref_mod._suppress_update = False
        # Preserve framing so the same world rect stays centered in the
        # reduced/expanded available width (100→75 keeps same relative pos).
        old_w = state.list.list_width
        if abs(new_w - old_w) >= 0.5:
            from .transforms import _preserve_view_for_list_width

            _preserve_view_for_list_width(state, old_w, new_w, ui_scale)
        # Drive the zone width live (per-pixel) so the pill tracks the cursor
        # without the integer-percent quantization or a one-frame zone lag.
        state.list.dragging_width = new_w
        state.list.list_width = new_w
        state.list.width_clamped = new_w <= min_w + 0.5 or new_w >= max_w - 0.5
        # Keep state hover in sync so the pill draws during drag.
        state.interaction.hovered_handle = ResizeHandle.LIST
        state.interaction.resize_active = ResizeHandle.LIST

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
        self._smooth_timer = None
        self._inertia_active = False
        self._inertia_mode = None
        self._smooth_velocity = [0.0, 0.0]
        self._anim_active = False
        self._anim_target = [0.0, 0.0]
        self._anim_applied = [0.0, 0.0]
        self._anim_progress = 0.0
        self._anim_acc = [0.0, 0.0]
        self._drag_target = [0.0, 0.0]
        self._drag_active = False
        self._frame_anim_active = False
        self._frame_anim_start_zoom = 1.0
        self._frame_anim_start_pan = [0.0, 0.0]
        self._frame_anim_target_zoom = 1.0
        self._frame_anim_target_pan = [0.0, 0.0]
        self._frame_anim_progress = 0.0
        self._editor_anim_active = False
        self._editor_anim_progress = 0.0
        self._editor_anim_start_rect = [0.0, 0.0, 0.0, 0.0]
        self._editor_anim_target_rect = [0.0, 0.0, 0.0, 0.0]
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
        self._destroy_timer(context)
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
        bpy.context.preferences.active_section = "ADDONS"

        window_manager = context.window_manager
        window_manager.addon_search = "Nodemap"

        try:
            bpy.ops.preferences.addon_expand(module="render_commander")
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
