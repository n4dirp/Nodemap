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


def _is_in_minimap(region_x: int, region_y: int, st: MinimapState | None = None) -> bool:
    if st is None:
        st = _state()
    mx, my, mw, mh = st.view.rect
    return mx <= region_x <= mx + mw and my <= region_y <= my + mh


def _region_to_tree(region_x: int, region_y: int, st: MinimapState | None = None) -> tuple[float, float] | None:
    if st is None:
        st = _state()
    if not st.view.rect or not st.view.tree_bounds:
        return None
    return _tree_from_region(region_x, region_y, _compute_map_transform(st))


def _tree_from_region(
    region_x: int, region_y: int, t: tuple[float, float, float, float, float]
) -> tuple[float, float] | None:
    """Inverse-map a minimap pixel coordinate to tree space using a precomputed transform."""
    cx, cy, scale, tree_cx, tree_cy = t
    if scale <= 0:
        return None
    return tree_cx + (region_x - cx) / scale, tree_cy + (region_y - cy) / scale


def _view_zoom_factors(space, region, visible: tuple[float, float, float, float] | None = None) -> tuple[float, float]:
    """Return pixels-per-tree-unit for each axis given the editor's visible rect."""
    if visible is None:
        visible = _get_visible_rect(space, region)
    if not visible:
        return 1.0, 1.0
    vw = max(visible[2] - visible[0], 1e-6)
    vh = max(visible[3] - visible[1], 1e-6)
    return region.width / vw, region.height / vh


def _frame_btn_at(mx: int, my: int, st: MinimapState) -> str | None:
    """Return the id of the frame button under the cursor, if any."""
    for btn_id, rect in st.buttons.rects.items():
        if rect:
            bx, by, bw, bh = rect
            if bx <= mx <= bx + bw and by <= my <= by + bh:
                return btn_id
    return None


def _in_list_zone(region_x: int, region_y: int, st: MinimapState) -> bool:
    """Return True when the cursor is over the type-list zone of the minimap."""
    if st.list.width <= 0 or not st.view.rect:
        return False
    mx, my, _, mh = st.view.rect
    pad = _HANDLE_THICKNESS * _get_ui_scale()
    zone_left = mx + pad
    zone_right = mx + st.view.padding + st.list.width
    zone_rect = st.list.zone_rect
    if zone_rect:
        _, zone_y, _, zone_h = zone_rect
    else:
        zone_y = my + pad
        zone_h = mh - 2 * pad
    return zone_left <= region_x <= zone_right and zone_y <= region_y <= zone_y + zone_h


def _list_row_at(region_x: int, region_y: int, st: MinimapState) -> str | None:
    """Return the type label of the type-list row under the cursor, if any."""
    for x, y, w, h, label in st.list.row_rects:
        if x <= region_x <= x + w and y <= region_y <= y + h:
            return label
    return None


def _list_child_at(region_x: int, region_y: int, st: MinimapState) -> tuple[str, str] | None:
    """Return ``(label, node_name)`` of the expanded child row under the cursor."""
    for x, y, w, h, label, node_name in st.list.node_rects:
        if x <= region_x <= x + w and y <= region_y <= y + h:
            return label, node_name
    return None


def _in_rect(region_x: int, region_y: int, rect: tuple[float, float, float, float]) -> bool:
    """Return True when the cursor falls inside the ``(x, y, w, h)`` rect."""
    x, y, w, h = rect
    return x <= region_x <= x + w and y <= region_y <= y + h


def _list_scrollbar_hit(region_x: int, region_y: int, st: MinimapState) -> bool:
    """Return True when the cursor is over the type-list scrollbar gutter."""
    track = st.list.scrollbar_track
    if not track or st.list.scroll_max <= 0:
        return False
    x, y, w, h = track
    pad = _SCROLLBAR_HIT_PAD * _get_ui_scale()
    return x - pad <= region_x <= x + w + pad and y <= region_y <= y + h


def _apply_list_scroll_drag(mx: int, my: int, grab: float, st: MinimapState) -> None:
    """Scroll the type list so the dragged thumb tracks the cursor.

    *grab* is the cursor-to-thumb-top distance captured at press; mapping the
    thumb top back to a track fraction keeps the grab point stable.
    """
    track = st.list.scrollbar_track
    thumb = st.list.scrollbar_thumb
    if not track or not thumb or st.list.scroll_max <= 0:
        return
    _tx, ty, _tw, tl = track
    thumb_len = thumb[3]
    span = max(tl - thumb_len, 1.0)
    offset = min(max(my + grab - thumb_len - ty, 0.0), span)
    st.list.scroll = (1.0 - offset / span) * st.list.scroll_max


_CURSOR_MAP: dict[ResizeHandle, str] = {
    ResizeHandle.W: "MOVE_X",
    ResizeHandle.H: "MOVE_Y",
    ResizeHandle.C: "SCROLL_XY",
}


def _get_resize_handle(st: MinimapState, corner: str, rx: int, ry: int, ui_scale: float) -> ResizeHandle | None:
    mx, my, mw, mh = st.view.rect
    if mw <= 0 or mh <= 0:
        return None
    hw = _HANDLE_THICKNESS * ui_scale

    def near(v, target):
        return target - hw <= v <= target + hw

    match corner:
        case "TOP_RIGHT":
            on_left = mx <= rx <= mx + hw
            on_bottom = my <= ry <= my + hw
            if on_left and on_bottom:
                return ResizeHandle.C
            if on_left:
                return ResizeHandle.W
            if on_bottom:
                return ResizeHandle.H
        case "TOP_LEFT":
            on_right = mx + mw - hw <= rx <= mx + mw
            on_bottom = my <= ry <= my + hw
            if on_right and on_bottom:
                return ResizeHandle.C
            if on_right:
                return ResizeHandle.W
            if on_bottom:
                return ResizeHandle.H
        case "BOTTOM_RIGHT":
            on_left = mx <= rx <= mx + hw
            on_top = my + mh - hw <= ry <= my + mh
            if on_left and on_top:
                return ResizeHandle.C
            if on_left:
                return ResizeHandle.W
            if on_top:
                return ResizeHandle.H
        case "BOTTOM_LEFT":
            on_right = mx + mw - hw <= rx <= mx + mw
            on_top = my + mh - hw <= ry <= my + mh
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
        st = _state()
        st.enabled = not st.enabled
        if not st.enabled:
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
        wm = context.window_manager
        kc = wm.keyconfigs.user
        if not kc:
            return False
        km = kc.keymaps.get("Node Editor")
        if km:
            return km.keymap_items.get("nodemap.toggle") is None
        return True

    def execute(self, context: Context) -> set[str]:
        wm = context.window_manager
        kc = wm.keyconfigs.user
        km = kc.keymaps.get("Node Editor")
        if not km:
            km = kc.keymaps.new(name="Node Editor", space_type="NODE_EDITOR")
        km.keymap_items.new("nodemap.toggle", type="M", value="PRESS", ctrl=True)
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
    _st: MinimapState | None = None
    _area: Area | None = None
    _region: Region | None = None
    _space: SpaceNodeEditor | None = None

    _resize_handle: str | None = None
    _resize_start_mouse: tuple[int, int] | None = None
    _resize_start_values: tuple[int, int] | None = None
    _last_cursor: str = ""
    _pan_acc: list[float]
    _redirect_acc: list[float]
    _armed_btn: str | None = None
    _list_row_pressed: str | None = None
    _list_child_pressed: tuple[str, str] | None = None
    _list_toggle_pressed: str | None = None
    _list_scroll_pressed: bool = False
    _list_scroll_grab: float = 0.0
    _list_mmb_dragging: bool = False
    _list_mmb_drag_start: tuple[int, int] | None = None

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
        return getattr(settings, "animations", default)

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
        win_ptr = context.window.as_pointer()
        if _minimap_window_operators.get(win_ptr) is not self:
            return {"CANCELLED"}

        _is_interacting = (
            self._dragging
            or self._mmb_dragging
            or self._list_mmb_dragging
            or self._resize_handle is not None
            or self._drag_start is not None
            or self._list_scroll_pressed
            or self._anim_active
            or self._inertia_active
            or self._drag_active
            or self._frame_anim_active
            or self._editor_anim_active
        )

        if not _is_interacting:
            area, region = _get_area_and_region_under_mouse(context, event)
            if not area or area.type != "NODE_EDITOR" or not region:
                self._st = None
                self._area = None
                self._region = None
                self._space = None
                return {"PASS_THROUGH"}
            self._st = _state(area.as_pointer())
            self._area = area
            self._region = region
            self._space = area.spaces.active
            _clamp_pan_to_viewport(self._space, self._region, self._st)

        if not self._st or not self._st.enabled:
            return {"PASS_THROUGH"}

        if self._space and not self._space.overlay.show_overlays:
            if _is_interacting:
                self._cancel_interaction(context)
            return {"PASS_THROUGH"}

        if self._region is not None:
            self._mx = event.mouse_x - self._region.x
            self._my = event.mouse_y - self._region.y
        else:
            self._mx = event.mouse_x
            self._my = event.mouse_y

        addon = context.preferences.addons.get(__package__)
        if addon and not getattr(addon.preferences.settings, "interactive", True):
            return {"PASS_THROUGH"}
        settings = addon.preferences.settings if addon else None

        st = self._st
        in_minimap = _is_in_minimap(self._mx, self._my, st)

        match event.type:
            case "LEFTMOUSE":
                return self._handle_left_mouse(context, event)

            case "RIGHTMOUSE":
                return self._handle_right_mouse(context, event)

            case "MIDDLEMOUSE":
                if event.value == "PRESS" and in_minimap:
                    st.interaction.pressed = True
                    self._cancel_smooth(context)
                    if _in_list_zone(self._mx, self._my, st):
                        self._list_mmb_dragging = True
                        self._list_mmb_drag_start = (self._mx, self._my)
                    else:
                        self._mmb_dragging = True
                        self._mmb_drag_start = (self._mx, self._my)
                    return {"RUNNING_MODAL"}
                if event.value == "RELEASE" and self._list_mmb_dragging:
                    st.interaction.pressed = False
                    self._redraw_ui()
                    self._list_mmb_dragging = False
                    self._list_mmb_drag_start = None
                    return {"RUNNING_MODAL"}
                if event.value == "RELEASE" and self._mmb_dragging:
                    st.interaction.pressed = False
                    self._redraw_ui()
                    self._mmb_dragging = False
                    self._mmb_drag_start = None
                    _clamp_pan_to_viewport(self._space, self._region, st)
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
        st = self._st
        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
        in_minimap = _is_in_minimap(self._mx, self._my, st) if st else False
        return st, addon, settings, in_minimap

    def _handle_left_mouse(self, context: Context, event: Event) -> set[str]:
        st, addon, settings, in_minimap = self._minimap_event_context(context)
        # --- Release ---
        if event.value == "RELEASE":
            if st.interaction.pressed:
                st.interaction.pressed = False
                self._redraw_ui()
            if self._list_scroll_pressed:
                self._list_scroll_pressed = False
                st.list.scrollbar_dragging = False
                self._list_scroll_grab = 0.0
                self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._armed_btn:
                self._activate_armed_button(context, settings)
                return {"RUNNING_MODAL"}
            if self._list_child_pressed:
                label, node_name = self._list_child_pressed
                self._list_child_pressed = None
                still_over = _list_child_at(self._mx, self._my, st) == (label, node_name)
                if _in_list_zone(self._mx, self._my, st) and still_over:
                    st.cache.force_immediate = True
                    self._select_single_node(context, node_name)
                return {"RUNNING_MODAL"}
            if self._list_toggle_pressed:
                label = self._list_toggle_pressed
                self._list_toggle_pressed = None
                toggle = st.list.toggle_rects.get(label)
                if toggle and _in_list_zone(self._mx, self._my, st) and _in_rect(self._mx, self._my, toggle):
                    if label in st.list.expanded:
                        st.list.expanded.discard(label)
                    else:
                        st.list.expanded.add(label)
                    st.cache.list_key = None
                    st.cache.force_immediate = True
                    self._redraw_ui()
                return {"RUNNING_MODAL"}
            if self._list_row_pressed:
                label = self._list_row_pressed
                self._list_row_pressed = None
                if _in_list_zone(self._mx, self._my, st) and _list_row_at(self._mx, self._my, st) == label:
                    st.cache.force_immediate = True
                    self._select_type_nodes(context, label)
                return {"RUNNING_MODAL"}
            if self._resize_handle:
                self._resize_handle = None
                self._resize_start_mouse = None
                self._resize_start_values = None
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
                st.view.width_clamped = False
                st.view.height_clamped = False
                st.interaction.hovered_handle = None
                st.interaction.resize_active = None
                st.cache.invalidate_batches_only()
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
                    st.cache.force_immediate = True
                    self._handle_click_selection(context, event, st, frame=settings.left_click_action == "SELECT_FRAME")
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
            armed = _frame_btn_at(self._mx, self._my, st)
            if armed:
                self._armed_btn = armed
                return {"RUNNING_MODAL"}
            if _in_list_zone(self._mx, self._my, st):
                track = st.list.scrollbar_track
                thumb = st.list.scrollbar_thumb
                if _list_scrollbar_hit(self._mx, self._my, st) and track and thumb:
                    thumb_top = thumb[1] + thumb[3]
                    if thumb[1] <= self._my <= thumb_top:
                        # Direct grab: keep the pressed point pinned to the cursor.
                        self._list_scroll_grab = thumb_top - self._my
                    else:
                        # Trough click pages one track-length toward the
                        # click, then continues as a drag from there.
                        # Note: y grows upward but larger list_scroll
                        # shifts content down, so a click above the
                        # thumb decreases the scroll.
                        if self._my > thumb_top:
                            st.list.scroll = max(st.list.scroll - track[3], 0.0)
                        else:
                            st.list.scroll = min(st.list.scroll + track[3], st.list.scroll_max)
                        span = max(track[3] - thumb[3], 1.0)
                        offset = min(span * (1.0 - st.list.scroll / st.list.scroll_max), span)
                        self._list_scroll_grab = track[1] + offset + thumb[3] - self._my
                    self._list_scroll_pressed = True
                    st.list.scrollbar_dragging = True
                    self._redraw_ui()
                    return {"RUNNING_MODAL"}
                child = _list_child_at(self._mx, self._my, st)
                if child:
                    self._list_child_pressed = child
                else:
                    label = _list_row_at(self._mx, self._my, st)
                    if label:
                        toggle = st.list.toggle_rects.get(label)
                        if toggle and _in_rect(self._mx, self._my, toggle):
                            self._list_toggle_pressed = label
                        else:
                            self._list_row_pressed = label
                return {"RUNNING_MODAL"}
            if addon:
                handle = self._get_handle_at(context, event)
                if handle:
                    self._resize_handle = handle
                    st.interaction.resize_active = handle
                    self._redraw_ui()
                    self._resize_start_mouse = (self._mx, self._my)
                    _ui_scale = _get_ui_scale()
                    rmx, rmy, rmw, rmh = st.view.rect
                    self._resize_start_values = (
                        int(rmw / _ui_scale),
                        int(rmh / _ui_scale),
                    )
                    cursor = _CURSOR_MAP[handle]
                    context.window.cursor_modal_set(cursor)
                    self._last_cursor = cursor
                    return {"RUNNING_MODAL"}
            if settings and settings.left_click_action in ("PAN", "SELECT_PAN"):
                self._drag_start = (self._mx, self._my)
                self._center_view_on_mouse(context, self._mx, self._my)
            return {"RUNNING_MODAL"}
        else:
            self._drag_start = None
            return {"PASS_THROUGH"}

    def _handle_right_mouse(self, context: Context, event: Event) -> set[str]:
        st, addon, settings, in_minimap = self._minimap_event_context(context)
        # --- Release ---
        if event.value == "RELEASE":
            if st.interaction.pressed:
                st.interaction.pressed = False
                self._redraw_ui()
            if self._resize_handle:
                self._resize_handle = None
                self._resize_start_mouse = None
                self._resize_start_values = None
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
                st.view.width_clamped = False
                st.view.height_clamped = False
                st.interaction.hovered_handle = None
                st.interaction.resize_active = None
                st.cache.invalidate_batches_only()
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
            if _in_list_zone(self._mx, self._my, st):
                if _list_scrollbar_hit(self._mx, self._my, st):
                    # Scrollbar owns the press; no row selection or pan.
                    return {"RUNNING_MODAL"}
                child = _list_child_at(self._mx, self._my, st)
                if child:
                    _label, node_name = child
                    st.cache.force_immediate = True
                    self._select_single_node(context, node_name)
                    if not self._view_selected_animated(context, settings):
                        try:
                            with self._override_ctx(context):
                                bpy.ops.node.view_selected()
                        except RuntimeError:
                            pass
                    self._was_in_minimap = False
                    return {"RUNNING_MODAL"}
                label = _list_row_at(self._mx, self._my, st)
                if label:
                    toggle = st.list.toggle_rects.get(label)
                    if toggle and _in_rect(self._mx, self._my, toggle):
                        if label in st.list.expanded:
                            st.list.expanded.discard(label)
                        else:
                            st.list.expanded.add(label)
                        st.cache.list_key = None
                        st.cache.force_immediate = True
                        self._redraw_ui()
                        self._was_in_minimap = False
                        return {"RUNNING_MODAL"}
                    st.cache.force_immediate = True
                    self._select_type_nodes(context, label)
                    if not self._view_selected_animated(context, settings):
                        try:
                            with self._override_ctx(context):
                                bpy.ops.node.view_selected()
                        except RuntimeError:
                            pass
                self._was_in_minimap = False
                return {"RUNNING_MODAL"}
            if addon:
                handle = self._get_handle_at(context, event)
                if handle:
                    self._resize_handle = handle
                    st.interaction.resize_active = handle
                    self._redraw_ui()
                    self._resize_start_mouse = (self._mx, self._my)
                    _ui_scale = _get_ui_scale()
                    rmx, rmy, rmw, rmh = st.view.rect
                    self._resize_start_values = (
                        int(rmw / _ui_scale),
                        int(rmh / _ui_scale),
                    )
                    cursor = _CURSOR_MAP[handle]
                    context.window.cursor_modal_set(cursor)
                    self._last_cursor = cursor
                    return {"RUNNING_MODAL"}
            if settings and settings.right_click_action in ("SELECT", "SELECT_PAN", "SELECT_FRAME"):
                st.cache.force_immediate = True
                self._handle_click_selection(context, event, st, frame=settings.right_click_action == "SELECT_FRAME")
            if settings and settings.right_click_action in ("PAN", "SELECT_PAN"):
                self._drag_start = (self._mx, self._my)
                self._center_view_on_mouse(context, self._mx, self._my)
            self._was_in_minimap = False
            return {"RUNNING_MODAL"}
        else:
            self._drag_start = None
            return {"PASS_THROUGH"}

    def _handle_mouse_move(self, context: Context, event: Event) -> set[str]:
        st, addon, settings, in_minimap = self._minimap_event_context(context)
        if self._resize_handle:
            self._resize_apply_delta(context, event)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if self._list_scroll_pressed and st.list.scrollbar_dragging:
            _apply_list_scroll_drag(self._mx, self._my, self._list_scroll_grab, st)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if not self._dragging and not self._mmb_dragging and not self._drag_start:
            self._update_cursor(context, event)
        if not self._dragging and not self._mmb_dragging and not self._resize_handle:
            in_list = _in_list_zone(self._mx, self._my, st)
            # The scrollbar gutter suppresses row hovers so the bar can
            # be approached without flashing the rows underneath.
            over_bar = in_list and _list_scrollbar_hit(self._mx, self._my, st)
            if st.list.hovered_scrollbar != over_bar:
                st.list.hovered_scrollbar = over_bar
                self._redraw_ui()
            row_label = None if over_bar else (_list_row_at(self._mx, self._my, st) if in_list else None)
            child_hover = None if over_bar else (_list_child_at(self._mx, self._my, st) if in_list else None)
            if st.list.hovered_type_label != row_label:
                st.list.hovered_type_label = row_label
                self._redraw_ui()
            if st.list.hovered_node != child_hover:
                st.list.hovered_node = child_hover
                self._redraw_ui()
            new_hovered = None
            if in_list and child_hover is not None:
                # Hovering a single child row highlights only that node's
                # border on the minimap (not the whole type group).
                new_hovered = child_hover[1]
            if st.interaction.hovered_node != new_hovered:
                st.interaction.hovered_node = new_hovered
                self._redraw_ui()
            old_btn = st.buttons.hovered
            new_btn = _frame_btn_at(self._mx, self._my, st) if in_minimap and not in_list else None
            if old_btn != new_btn:
                st.buttons.hovered = new_btn
                self._redraw_ui()
        if self._list_mmb_dragging and self._list_mmb_drag_start:
            dy = self._my - self._list_mmb_drag_start[1]
            if abs(dy) > 0:
                st.list.scroll = min(
                    max(st.list.scroll - dy, 0.0),
                    st.list.scroll_max,
                )
                self._list_mmb_drag_start = (self._mx, self._my)
                self._redraw_ui()
            return {"RUNNING_MODAL"}
        if self._mmb_dragging and self._mmb_drag_start:
            dx = self._mx - self._mmb_drag_start[0]
            dy = self._my - self._mmb_drag_start[1]
            if abs(dx) <= 1 and abs(dy) <= 1:
                self._smooth_velocity[0] *= 0.15
                self._smooth_velocity[1] *= 0.15
            else:
                self._smooth_velocity[0] = self._smooth_velocity[0] * 0.6 + dx * 0.4
                self._smooth_velocity[1] = self._smooth_velocity[1] * 0.6 + dy * 0.4
            pan_before = st.view.pan
            st.view.pan = (st.view.pan[0] + dx, st.view.pan[1] + dy)
            _clamp_pan_to_viewport(self._space, self._region, st)
            rejected_x = dx - (st.view.pan[0] - pan_before[0])
            rejected_y = dy - (st.view.pan[1] - pan_before[1])
            if (rejected_x != 0 or rejected_y != 0) and getattr(settings, "follow_view", False):
                st.view.pan = (pan_before[0] + dx, pan_before[1] + dy)
                self._redirect_to_view2d(context, -dx, -dy)
            elif rejected_x != 0 or rejected_y != 0:
                self._redirect_to_view2d(context, -int(rejected_x), -int(rejected_y))
            self._mmb_drag_start = (self._mx, self._my)
            self._redraw_ui()
            return {"RUNNING_MODAL"}
        if self._drag_start is not None:
            dx = self._mx - self._drag_start[0]
            dy = self._my - self._drag_start[1]
            if abs(dx) > 2 or abs(dy) > 2 or self._dragging:
                if not self._dragging and (self._anim_active or self._frame_anim_active or self._editor_anim_active):
                    self._cancel_smooth(context)
                self._dragging = True
                if self._was_in_minimap:
                    st.interaction.pressed = True
                    smooth = settings and self._animations_enabled(settings, context, default=False)
                    self._pan_view(context, dx, dy, smooth)
                    self._drag_start = (self._mx, self._my)
            return {"RUNNING_MODAL"}
        if in_minimap:
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}

    def _handle_wheel(self, context: Context, event: Event) -> set[str]:
        st, addon, settings, in_minimap = self._minimap_event_context(context)
        if in_minimap and _in_list_zone(self._mx, self._my, st):
            direction = -1 if event.type == "WHEELUPMOUSE" else 1
            st.list.scroll = min(max(st.list.scroll + direction * st.list.row_height * 3, 0.0), st.list.scroll_max)
            over_bar = _list_scrollbar_hit(self._mx, self._my, st)
            st.list.hovered_type_label = None if over_bar else _list_row_at(self._mx, self._my, st)
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
            scroll_mode = getattr(prefs, "scroll_wheel_mode", "MINIMAP") if prefs else "MINIMAP"
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
                effective_zoom = st.view.zoom

                is_constrained = False
                if addon and getattr(addon.preferences.settings, "follow_view", False):
                    if effective_zoom < st.view.base_zoom - 0.001:
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

                    transform = _get_minimap_transform(st)
                    tree_coord = _tree_from_region(self._mx, self._my, transform)

                    if transform[2] > 0 and tree_coord is not None:
                        _, _, scale, tree_cx, tree_cy = transform
                        base_scale = scale / st.view.zoom
                        tx, ty = tree_coord
                        pan_x, pan_y = st.view.pan

                        pan_x_new = pan_x - (tx - tree_cx) * base_scale * (new_zoom - st.view.zoom)
                        pan_y_new = pan_y - (ty - tree_cy) * base_scale * (new_zoom - st.view.zoom)

                        st.view.base_zoom = new_zoom
                        st.view.zoom = new_zoom
                        st.view.pan = (pan_x_new, pan_y_new)
                        _clamp_pan_to_viewport(self._space, self._region, st)

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
        vr = self._region.view2d if self._region else None
        if not vr:
            return None
        ui = _get_ui_scale()
        pt = vr.view_to_region(tree_x * ui, tree_y * ui, clip=False)
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
        for probe_index, (tx, ty) in enumerate(probes):
            projected = self._project_tree_to_region(tx, ty)
            if projected is None:
                continue
            rx, ry = projected
            keep = None
            if probe_index and extend and tree_nodes:
                keep = {n.name for n in tree_nodes if n.select}
            kwargs: dict = {"extend": extend}
            if bpy.app.version >= (3, 0, 0):
                kwargs["location"] = (rx, ry)
                kwargs["deselect_all"] = deselect_all
            else:
                kwargs["mouse_x"] = rx
                kwargs["mouse_y"] = ry
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

    def _handle_click_selection(self, context: Context, event: Event, st: dict, frame: bool = False) -> None:
        space = self._space
        if not space or space.type != "NODE_EDITOR":
            return
        node_tree = space.edit_tree
        if not node_tree or not node_tree.nodes:
            return

        tree_coord = _region_to_tree(self._mx, self._my, st)
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

        st.list.hovered_node = None
        st.interaction.hovered_node = None
        self._redraw_ui()

    def _select_type_nodes(self, context: Context, label: str) -> None:
        """Select all editor nodes whose compiled type label matches *label*."""
        space = self._space
        st = self._st
        if not space or space.type != "NODE_EDITOR" or not st:
            return
        node_tree = space.edit_tree
        if not node_tree:
            return
        type_nodes = (st.cache.tree_data or {}).get("type_nodes") or {}
        names = type_nodes.get(label)
        if not names:
            return

        try:
            with self._override_ctx(context):
                bpy.ops.node.select_all(action="DESELECT")
        except RuntimeError:
            pass
        for name in names:
            node = node_tree.nodes.get(name)
            if node:
                # Native operator keeps selection/additive state and sets the
                # active node without tagging the NodeTree for an EEVEE rebuild.
                if not self._select_node_via_operator(context, node, extend=True, deselect_all=False):
                    node.select = True
        self._redraw_ui()

    def _select_single_node(self, context: Context, node_name: str) -> None:
        """Select only the editor node whose compiled name matches *node_name*."""
        space = self._space
        st = self._st
        if not space or space.type != "NODE_EDITOR" or not st:
            return
        node_tree = space.edit_tree
        if not node_tree:
            return
        node = node_tree.nodes.get(node_name)
        if not node:
            return
        try:
            with self._override_ctx(context):
                bpy.ops.node.select_all(action="DESELECT")
        except RuntimeError:
            pass
        if not self._select_node_via_operator(context, node, extend=False, deselect_all=False):
            node.select = True
            node_tree.nodes.active = node
        self._redraw_ui()

    def _activate_armed_button(self, context: Context, settings) -> None:
        """Release the armed minimap button; run its action when still under the cursor."""
        btn_id = self._armed_btn
        self._armed_btn = None
        st = self._st
        if not btn_id or st is None:
            return
        rect = st.buttons.rects.get(btn_id)
        if not rect:
            return
        bx, by, bw, bh = rect
        if not (bx <= self._mx <= bx + bw and by <= self._my <= by + bh):
            return
        if btn_id == "LIST":
            if settings:
                settings.show_type_list = not settings.show_type_list
                start_list_width_animation(st, settings)
                self._redraw_ui()
            return
        self._dispatch_frame_action(context, settings, btn_id)

    def _dispatch_frame_action(self, context: Context, settings, btn_id: str) -> None:
        """Run a frame action directly, or eased via animation when smooth pan applies.

        Shared by the minimap button release and the Home / Numpad shortcuts.
        """
        st = self._st
        if not st:
            return
        smooth = bool(settings) and self._animations_enabled(settings, context)
        area_ptr = self._area.as_pointer() if self._area else 0
        match btn_id:
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
                            _, _, mw, mh = st.view.rect
                            st.view.tree_bounds = _expand_bounds_margin(
                                _get_node_tree_bounds(node_tree.nodes),
                                _get_ui_scale(),
                                mh,
                                st.view.padding,
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
                        target_zoom = targets[0] if targets[0] is not None else st.view.zoom
                        self._start_frame_animation(context, target_zoom, [targets[1], targets[2]])
                else:
                    frame_selected(self._space, self._region, area_ptr)

    def _pan_view(self, context: Context, dx: int, dy: int, smooth: bool = False) -> None:
        st = self._st
        if not st:
            return
        visible = _get_visible_rect(self._space, self._region)
        if not visible:
            return
        _, _, scale, _, _ = _compute_map_transform(st)
        if scale <= 0:
            return
        view_zoom_x, view_zoom_y = _view_zoom_factors(self._space, self._region, visible)

        vx = (dx / scale) * view_zoom_x
        vy = (dy / scale) * view_zoom_y
        if abs(dx) <= 1 and abs(dy) <= 1:
            self._smooth_velocity[0] *= 0.15
            self._smooth_velocity[1] *= 0.15
        else:
            self._smooth_velocity[0] = self._smooth_velocity[0] * 0.6 + vx * 0.4
            self._smooth_velocity[1] = self._smooth_velocity[1] * 0.6 + vy * 0.4

        if smooth:
            self._drag_target[0] += vx
            self._drag_target[1] += vy
            if not self._drag_active:
                self._drag_active = True
                self._create_timer(context)
            return

        self._pan_acc[0] += vx
        self._pan_acc[1] += vy
        pan_x = int(self._pan_acc[0])
        pan_y = int(self._pan_acc[1])
        self._pan_acc[0] -= pan_x
        self._pan_acc[1] -= pan_y

        if pan_x != 0 or pan_y != 0:
            try:
                pan_before = st.view.pan

                with self._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                _clamp_pan_to_viewport(self._space, self._region, st)

                clamp_dx = st.view.pan[0] - pan_before[0]
                clamp_dy = st.view.pan[1] - pan_before[1]

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
                        _clamp_pan_to_viewport(self._space, self._region, st)
            except RuntimeError:
                pass

    def _redirect_to_view2d(self, context: Context, dx: float, dy: float) -> None:
        st = self._st
        if not st:
            return
        _, _, scale, _, _ = _compute_map_transform(st)
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

    def _center_view_on_mouse(self, context: Context, mx: int, my: int) -> None:
        st = self._st
        if not st:
            return
        tree_coord = _region_to_tree(mx, my, st)
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

        st.interaction.pressed = True
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
                _clamp_pan_to_viewport(self._space, self._region, st)
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
            st = self._st
            if st:
                st.view.width_clamped = False
                st.view.height_clamped = False
                st.interaction.hovered_handle = None
                st.interaction.resize_active = None
        context.window.cursor_modal_set("DEFAULT")
        self._last_cursor = ""
        self._armed_btn = None
        self._list_row_pressed = None
        self._list_child_pressed = None
        self._list_toggle_pressed = None
        self._list_scroll_pressed = False
        self._list_scroll_grab = 0.0
        st = self._st
        if st:
            st.buttons.hovered = None
            st.list.hovered_type_label = None
            st.interaction.hovered_node = None
            st.list.hovered_scrollbar = False
            st.list.scrollbar_dragging = False
            if st.interaction.pressed:
                st.interaction.pressed = False
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
            st = self._st
            if st:
                st.view.base_zoom = self._frame_anim_target_zoom
                st.view.zoom = self._frame_anim_target_zoom
                st.view.pan = (self._frame_anim_target_pan[0], self._frame_anim_target_pan[1])
                _clamp_pan_to_viewport(self._space, self._region, st)
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
            st = self._st
            if st:
                self._pan_acc[0] += self._smooth_velocity[0]
                self._pan_acc[1] += self._smooth_velocity[1]
                dx = int(self._pan_acc[0])
                dy = int(self._pan_acc[1])
                self._pan_acc[0] -= dx
                self._pan_acc[1] -= dy
                if dx != 0 or dy != 0:
                    st.view.pan = (st.view.pan[0] + dx, st.view.pan[1] + dy)
                    _clamp_pan_to_viewport(self._space, self._region, st)
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
                _clamp_pan_to_viewport(self._space, self._region, self._st)
        self._redraw_ui()

    def _apply_smooth_drag(self, context: Context) -> None:
        if not self._drag_active:
            return
        magnitude = (self._drag_target[0] ** 2 + self._drag_target[1] ** 2) ** 0.5
        raw = magnitude / 200.0
        follow = 0.25 + raw * 0.55
        follow = min(follow, 0.8)
        _MAX_MOVE = 120.0 + magnitude * 0.15
        _MAX_MOVE = min(_MAX_MOVE, 800.0)
        dx = self._drag_target[0] * follow
        dy = self._drag_target[1] * follow
        dx = max(min(dx, _MAX_MOVE), -_MAX_MOVE)
        dy = max(min(dy, _MAX_MOVE), -_MAX_MOVE)
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
            _clamp_pan_to_viewport(self._space, self._region, self._st)
        if not self._dragging:
            self._drag_active = False
        self._redraw_ui()

    def _apply_center_animation(self, context: Context) -> None:
        if not self._anim_active:
            return
        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
        speed = getattr(settings, "pan_speed", "MEDIUM")
        frames = {"FAST": 10, "MEDIUM": 20, "SLOW": 30}.get(speed, 24)
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
        st = self._st
        if not st:
            return
        if self._frame_anim_active:
            self._frame_anim_active = False
        self._frame_anim_progress = 0.0
        self._frame_anim_start_zoom = st.view.zoom
        self._frame_anim_start_pan = [st.view.pan[0], st.view.pan[1]]
        self._frame_anim_target_zoom = target_zoom
        self._frame_anim_target_pan = [target_pan[0], target_pan[1]]
        self._frame_anim_active = True
        self._create_timer(context)

    def _apply_frame_animation(self, context: Context) -> None:
        if not self._frame_anim_active:
            return
        st = self._st
        if not st:
            self._frame_anim_active = False
            self._destroy_timer(context)
            return
        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
        speed = getattr(settings, "pan_speed", "MEDIUM")
        frames = {"FAST": 10, "MEDIUM": 20, "SLOW": 30}.get(speed, 24)
        progress = self._frame_anim_progress + 1 / frames
        self._frame_anim_progress = progress
        if progress >= 1.0:
            st.view.base_zoom = self._frame_anim_target_zoom
            st.view.zoom = self._frame_anim_target_zoom
            st.view.pan = (self._frame_anim_target_pan[0], self._frame_anim_target_pan[1])
            _clamp_pan_to_viewport(self._space, self._region, st)
            self._frame_anim_active = False
            self._frame_anim_progress = 0.0
            self._destroy_timer(context)
            self._redraw_ui()
            return
        eased = 1.0 - (1.0 - progress) ** 3
        st.view.zoom = (
            self._frame_anim_start_zoom + (self._frame_anim_target_zoom - self._frame_anim_start_zoom) * eased
        )
        st.view.base_zoom = st.view.zoom
        st.view.pan = (
            self._frame_anim_start_pan[0] + (self._frame_anim_target_pan[0] - self._frame_anim_start_pan[0]) * eased,
            self._frame_anim_start_pan[1] + (self._frame_anim_target_pan[1] - self._frame_anim_start_pan[1]) * eased,
        )
        _clamp_pan_to_viewport(self._space, self._region, st)
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
        if not self._space or not self._region or not self._st:
            self._editor_anim_active = False
            self._destroy_timer(context)
            return
        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
        speed = getattr(settings, "pan_speed", "MEDIUM")
        frames = {"FAST": 10, "MEDIUM": 20, "SLOW": 30}.get(speed, 24)
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
        st = self._st
        if not st or not st.view.rect:
            return
        in_minimap = _is_in_minimap(self._mx, self._my, st)
        if not in_minimap:
            if self._last_cursor:
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
            old_handle = st.interaction.hovered_handle
            st.interaction.hovered_handle = None
            if old_handle:
                self._redraw_ui()
            return
        handle = self._get_handle_at(context, event)
        old_handle = st.interaction.hovered_handle
        st.interaction.hovered_handle = handle
        if handle != old_handle:
            self._redraw_ui()
        if _in_list_zone(self._mx, self._my, st):
            if self._last_cursor:
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
            return
        is_clamped = handle and (st.view.width_clamped or st.view.height_clamped)
        cursor = "HAND" if is_clamped else _CURSOR_MAP.get(handle, "DEFAULT")
        if cursor != self._last_cursor:
            context.window.cursor_modal_set(cursor)
            self._last_cursor = cursor

    def _get_handle_at(self, context: Context, event: Event) -> str | None:
        st = self._st
        if not st:
            return None
        addon = context.preferences.addons.get(__package__)
        if not addon:
            return None
        corner = getattr(addon.preferences.settings, "position", "TOP_RIGHT")
        ui_scale = _get_ui_scale()
        return _get_resize_handle(st, corner, self._mx, self._my, ui_scale)

    def _resize_apply_delta(self, context: Context, event: Event) -> None:
        addon = context.preferences.addons.get(__package__)
        if not addon:
            return
        settings = addon.preferences.settings
        if not self._resize_start_values:
            return
        w0, h0 = self._resize_start_values
        dx = self._mx - self._resize_start_mouse[0]
        dy = self._my - self._resize_start_mouse[1]
        corner = getattr(settings, "position", "TOP_RIGHT")

        ui_scale = _get_ui_scale()
        sx, sy, ex, ey = _get_safe_bounds(self._area, self._region)
        x_margin, y_margin, margin = _get_minimap_margins(self._space, corner, ui_scale)

        safe_w = ex - sx
        safe_h = ey - sy
        max_mw_pct = getattr(settings, "max_width_pct", 50) / 100.0
        max_mh_pct = getattr(settings, "max_height_pct", 50) / 100.0
        max_w = max(MIN_MAP_WIDTH, int((safe_w - 2 * x_margin) * max_mw_pct))
        max_h = max(MIN_MAP_HEIGHT, int((safe_h - y_margin - margin) * max_mh_pct))

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

        st = self._st
        if not st:
            return
        st.interaction.hovered_handle = self._resize_handle
        st.view.width_clamped = settings.minimap_width >= max_w or settings.minimap_width <= MIN_MAP_WIDTH
        st.view.height_clamped = settings.minimap_height >= max_h or settings.minimap_height <= MIN_MAP_HEIGHT

    def invoke(self, context: Context, _event: Event) -> set[str]:
        if context.area.type != "NODE_EDITOR":
            logger.debug("invoke: cancelled — area type is %s", context.area.type)
            return {"CANCELLED"}
        self._window_ptr = context.window.as_pointer()
        self._pan_acc = [0.0, 0.0]
        self._redirect_acc = [0.0, 0.0]
        self._armed_btn = None
        self._list_row_pressed = None
        self._list_scroll_pressed = False
        self._list_scroll_grab = 0.0
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
        if self._st is not None:
            self._st.view.width_clamped = False
            self._st.view.height_clamped = False
            self._st.interaction.hovered_handle = None
            self._st.interaction.resize_active = None
            self._st.buttons.hovered = None
            self._st.list.hovered_type_label = None
            self._st.list.hovered_node = None
            self._st.interaction.hovered_node = None
            self._st.list.hovered_scrollbar = False
            self._st.list.scrollbar_dragging = False
        self._list_row_pressed = None
        self._list_child_pressed = None
        self._list_toggle_pressed = None
        self._list_scroll_pressed = False
        self._list_scroll_grab = 0.0


class NODEMAP_OT_OpenPreferences(Operator):
    bl_idname = "nodemap.open_pref"
    bl_label = "Open Preferences"
    bl_description = "Open the add-on preferences panel"

    def execute(self, context):
        bpy.ops.screen.userpref_show()
        bpy.context.preferences.active_section = "ADDONS"

        wm = context.window_manager
        wm.addon_search = "Nodemap"

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
    NODEMAP_OT_OpenPreferences,
)
