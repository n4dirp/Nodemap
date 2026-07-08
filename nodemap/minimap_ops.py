"""Modal operator for minimap interaction."""

import logging

import bpy
from bpy.types import Area, Context, Event, Operator, Region, SpaceNodeEditor

from .helpers import (
    _clamp_pan_to_viewport,
    _find_node_at,
    _get_area_and_region_under_mouse,
    _get_minimap_transform,
    _get_safe_bounds,
    _get_ui_scale,
    _get_visible_rect,
    _minimap_window_operators,
    _state,
    frame_all,
    frame_selected,
    frame_view,
    redraw_ui,
)

logger = logging.getLogger(__package__)


def _is_in_minimap(region_x: int, region_y: int, st: dict | None = None) -> bool:
    if st is None:
        st = _state()
    mx, my, mw, mh = st.get("rect", (0, 0, 0, 0))
    return mx <= region_x <= mx + mw and my <= region_y <= my + mh


def _region_to_tree(region_x: int, region_y: int, st: dict | None = None) -> tuple[float, float] | None:
    if st is None:
        st = _state()
    if not st.get("rect"):
        return None
    cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform(st)
    if scale <= 0:
        return None
    tx = tree_cx + (region_x - cx) / scale
    ty = tree_cy + (region_y - cy) / scale
    return tx, ty


_HANDLE_THICKNESS = 6

_CURSOR_MAP: dict[str, str] = {
    "W": "MOVE_X",
    "H": "MOVE_Y",
    "C": "SCROLL_XY",
}


def _get_resize_handle(st: dict, corner: str, rx: int, ry: int, ui_scale: float) -> str | None:
    mx, my, mw, mh = st.get("rect", (0, 0, 0, 0))
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
                return "C"
            if on_left:
                return "W"
            if on_bottom:
                return "H"
        case "TOP_LEFT":
            on_right = mx + mw - hw <= rx <= mx + mw
            on_bottom = my <= ry <= my + hw
            if on_right and on_bottom:
                return "C"
            if on_right:
                return "W"
            if on_bottom:
                return "H"
        case "BOTTOM_RIGHT":
            on_left = mx <= rx <= mx + hw
            on_top = my + mh - hw <= ry <= my + mh
            if on_left and on_top:
                return "C"
            if on_left:
                return "W"
            if on_top:
                return "H"
        case "BOTTOM_LEFT":
            on_right = mx + mw - hw <= rx <= mx + mw
            on_top = my + mh - hw <= ry <= my + mh
            if on_right and on_top:
                return "C"
            if on_right:
                return "W"
            if on_top:
                return "H"
        case _:
            return None


class NODEMAP_OT_toggle(Operator):
    """Display the minimap overlay."""

    bl_idname = "nodemap.toggle"
    bl_label = "Show Nodemap"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        st = _state()
        st["enabled"] = not st.get("enabled", True)
        redraw_ui("NODE_EDITOR")
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
    _st: dict | None = None
    _area: Area | None = None
    _region: Region | None = None
    _space: SpaceNodeEditor | None = None

    _resize_handle: str | None = None
    _resize_start_mouse: tuple[int, int] | None = None
    _resize_start_values: tuple[int, int] | None = None
    _last_cursor: str = ""
    _pan_acc: list[float]
    _redirect_acc: list[float]
    _frame_all_armed: bool = False

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

    def _override_ctx(self, context: Context):
        return context.temp_override(
            area=self._area,
            region=self._region,
            space_data=self._space,
        )

    def modal(self, context: Context, event: Event) -> set[str]:
        if not context.window:
            return {"CANCELLED"}
        win_ptr = context.window.as_pointer()
        if _minimap_window_operators.get(win_ptr) is not self:
            return {"CANCELLED"}

        _is_interacting = (
            self._dragging or self._mmb_dragging or self._resize_handle is not None or self._drag_start is not None
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

        if not self._st or not self._st.get("enabled", True):
            return {"PASS_THROUGH"}

        if self._region is not None:
            self._mx = event.mouse_x - self._region.x
            self._my = event.mouse_y - self._region.y
        else:
            self._mx = event.mouse_x
            self._my = event.mouse_y
        logger.log(5, "modal: event %s value=%s", event.type, event.value)

        addon = context.preferences.addons.get(__package__)
        if addon and not getattr(addon.preferences.settings, "interactive", True):
            return {"PASS_THROUGH"}
        settings = addon.preferences.settings if addon else None

        st = self._st
        in_minimap = _is_in_minimap(self._mx, self._my, st)

        match event.type:
            case "LEFTMOUSE":
                # --- Release ---
                if event.value == "RELEASE":
                    if self._frame_all_armed:
                        self._frame_all_armed = False
                        btn = st.get("frame_all_btn")
                        if btn:
                            bx, by, bw, bh = btn
                            mx = self._mx
                            my = self._my
                            if bx <= mx <= bx + bw and by <= my <= by + bh:
                                frame_all(self._space, self._region, self._area.as_pointer())
                        return {"RUNNING_MODAL"}
                    if self._resize_handle:
                        self._resize_handle = None
                        self._resize_start_mouse = None
                        self._resize_start_values = None
                        context.window.cursor_modal_set("DEFAULT")
                        self._last_cursor = ""
                        st["width_clamped"] = False
                        st["height_clamped"] = False
                        st["hovered_handle"] = None
                        st["resize_active"] = None
                        redraw_ui("NODE_EDITOR")
                        return {"RUNNING_MODAL"}
                    if self._dragging:
                        self._dragging = False
                        self._drag_start = None
                        if self._drag_active:
                            self._pan_acc[0] += self._drag_target[0]
                            self._pan_acc[1] += self._drag_target[1]
                            self._drag_target = [0.0, 0.0]
                            self._drag_active = False
                        if settings and getattr(settings, "smooth_pan", True):
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
                        if settings and settings.left_click_action in ("SELECT", "PAN_SELECT"):
                            self._handle_click_selection(context, event, st)
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
                    btn = st.get("frame_all_btn")
                    if btn:
                        bx, by, bw, bh = btn
                        mx = self._mx
                        my = self._my
                        if bx <= mx <= bx + bw and by <= my <= by + bh:
                            self._frame_all_armed = True
                            return {"RUNNING_MODAL"}
                    if addon:
                        handle = self._get_handle_at(context, event)
                        if handle:
                            self._resize_handle = handle
                            st["resize_active"] = handle
                            redraw_ui("NODE_EDITOR")
                            self._resize_start_mouse = (self._mx, self._my)
                            self._resize_start_values = (
                                settings.minimap_width,
                                settings.minimap_height,
                            )
                            cursor = _CURSOR_MAP[handle]
                            context.window.cursor_modal_set(cursor)
                            self._last_cursor = cursor
                            return {"RUNNING_MODAL"}
                    self._drag_start = (self._mx, self._my)
                    if settings and settings.left_click_action in ("PAN", "PAN_SELECT"):
                        self._center_view_on_mouse(context, self._mx, self._my)
                    return {"RUNNING_MODAL"}
                else:
                    self._drag_start = None
                    return {"PASS_THROUGH"}

            case "RIGHTMOUSE":
                # --- Release ---
                if event.value == "RELEASE":
                    if self._resize_handle:
                        self._resize_handle = None
                        self._resize_start_mouse = None
                        self._resize_start_values = None
                        context.window.cursor_modal_set("DEFAULT")
                        self._last_cursor = ""
                        st["width_clamped"] = False
                        st["height_clamped"] = False
                        st["hovered_handle"] = None
                        st["resize_active"] = None
                        redraw_ui("NODE_EDITOR")
                        return {"RUNNING_MODAL"}
                    if self._dragging:
                        self._dragging = False
                        self._drag_start = None
                        if self._drag_active:
                            self._pan_acc[0] += self._drag_target[0]
                            self._pan_acc[1] += self._drag_target[1]
                            self._drag_target = [0.0, 0.0]
                            self._drag_active = False
                        if settings and getattr(settings, "smooth_pan", True):
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
                        if settings and settings.right_click_action in ("SELECT", "PAN_SELECT"):
                            self._handle_click_selection(context, event, st)
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
                    if addon:
                        handle = self._get_handle_at(context, event)
                        if handle:
                            self._resize_handle = handle
                            st["resize_active"] = handle
                            redraw_ui("NODE_EDITOR")
                            self._resize_start_mouse = (self._mx, self._my)
                            self._resize_start_values = (
                                settings.minimap_width,
                                settings.minimap_height,
                            )
                            cursor = _CURSOR_MAP[handle]
                            context.window.cursor_modal_set(cursor)
                            self._last_cursor = cursor
                            return {"RUNNING_MODAL"}
                    self._drag_start = (self._mx, self._my)
                    if settings and settings.right_click_action in ("PAN", "PAN_SELECT"):
                        self._center_view_on_mouse(context, self._mx, self._my)
                    return {"RUNNING_MODAL"}
                else:
                    self._drag_start = None
                    return {"PASS_THROUGH"}

            case "MIDDLEMOUSE":
                if event.value == "PRESS" and in_minimap:
                    self._cancel_smooth(context)
                    self._mmb_dragging = True
                    self._mmb_drag_start = (self._mx, self._my)
                    return {"RUNNING_MODAL"}
                if event.value == "RELEASE" and self._mmb_dragging:
                    self._mmb_dragging = False
                    self._mmb_drag_start = None
                    if settings and getattr(settings, "smooth_pan", True):
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
                if self._resize_handle:
                    self._resize_apply_delta(context, event)
                    redraw_ui("NODE_EDITOR")
                    return {"RUNNING_MODAL"}
                if not self._dragging and not self._mmb_dragging and not self._drag_start:
                    self._update_cursor(context, event)
                if not self._dragging and not self._mmb_dragging and not self._resize_handle:
                    old_hovered = st.get("hovered_node")
                    new_hovered = None
                    if in_minimap:
                        tree_coord = _region_to_tree(self._mx, self._my, st)
                        if tree_coord and self._space and self._space.edit_tree:
                            hovered = _find_node_at(self._space.edit_tree.nodes, tree_coord[0], tree_coord[1])
                            if hovered:
                                new_hovered = hovered.name
                    if old_hovered != new_hovered:
                        st["hovered_node"] = new_hovered
                        redraw_ui("NODE_EDITOR")
                if self._mmb_dragging and self._mmb_drag_start:
                    dx = self._mx - self._mmb_drag_start[0]
                    dy = self._my - self._mmb_drag_start[1]
                    if abs(dx) <= 1 and abs(dy) <= 1:
                        self._smooth_velocity[0] *= 0.15
                        self._smooth_velocity[1] *= 0.15
                    else:
                        self._smooth_velocity[0] = self._smooth_velocity[0] * 0.6 + dx * 0.4
                        self._smooth_velocity[1] = self._smooth_velocity[1] * 0.6 + dy * 0.4
                    pan_before = st["pan"][0], st["pan"][1]
                    st["pan"][0] += dx
                    st["pan"][1] += dy
                    _clamp_pan_to_viewport(self._space, self._region, st)
                    rejected_x = dx - (st["pan"][0] - pan_before[0])
                    rejected_y = dy - (st["pan"][1] - pan_before[1])
                    if (rejected_x != 0 or rejected_y != 0) and getattr(settings, "follow_view", False):
                        st["pan"][0] = pan_before[0] + dx
                        st["pan"][1] = pan_before[1] + dy
                        self._redirect_to_view2d(context, -dx, -dy)
                    elif rejected_x != 0 or rejected_y != 0:
                        self._redirect_to_view2d(context, -int(rejected_x), -int(rejected_y))
                    self._mmb_drag_start = (self._mx, self._my)
                    redraw_ui("NODE_EDITOR")
                    return {"RUNNING_MODAL"}
                if self._drag_start is not None:
                    dx = self._mx - self._drag_start[0]
                    dy = self._my - self._drag_start[1]
                    if abs(dx) > 2 or abs(dy) > 2 or self._dragging:
                        if not self._dragging and self._anim_active:
                            self._cancel_smooth(context)
                        self._dragging = True
                        if self._was_in_minimap:
                            smooth = settings and getattr(settings, "smooth_pan", False)
                            self._pan_view(context, dx, dy, smooth)
                            self._drag_start = (self._mx, self._my)
                    return {"RUNNING_MODAL"}
                if in_minimap:
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "WHEELUPMOUSE" | "WHEELDOWNMOUSE":
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
                        redraw_ui("NODE_EDITOR")
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
                        effective_zoom = st.get("zoom", 1.0)

                        is_constrained = False
                        if addon and getattr(addon.preferences.settings, "follow_view", False):
                            if effective_zoom < st.get("base_zoom", 1.0) - 0.001:
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

                            cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform(st)
                            tree_coord = _region_to_tree(self._mx, self._my, st)

                            if scale > 0 and tree_coord is not None:
                                tx, ty = tree_coord
                                base_scale = scale / effective_zoom
                                pan_x, pan_y = st.get("pan", [0.0, 0.0])

                                pan_x_new = pan_x - (tx - tree_cx) * base_scale * (new_zoom - effective_zoom)
                                pan_y_new = pan_y - (ty - tree_cy) * base_scale * (new_zoom - effective_zoom)

                                st["base_zoom"] = new_zoom
                                st["zoom"] = new_zoom
                                st["pan"] = [pan_x_new, pan_y_new]
                                _clamp_pan_to_viewport(self._space, self._region, st)

                    redraw_ui("NODE_EDITOR")
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "HOME":
                if event.value == "PRESS" and in_minimap:
                    frame_all(self._space, self._region, self._area.as_pointer())
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "NUMPAD_PERIOD":
                if event.value == "PRESS" and in_minimap:
                    frame_selected(self._space, self._region, self._area.as_pointer())
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "TIMER":
                if self._drag_active:
                    self._apply_smooth_drag(context)
                    return {"RUNNING_MODAL"}
                if self._inertia_active:
                    self._apply_inertia(context)
                    return {"RUNNING_MODAL"}
                if self._anim_active:
                    self._apply_center_animation(context)
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case _:
                return {"PASS_THROUGH"}

    def _handle_click_selection(self, context: Context, event: Event, st: dict) -> None:
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
            with self._override_ctx(context):
                bpy.ops.node.select_all(action="DESELECT")
            node.select = True
            node_tree.nodes.active = node

            addon = context.preferences.addons.get(__package__)
            if addon and getattr(addon.preferences.settings, "auto_frame_selected", True):
                try:
                    with self._override_ctx(context):
                        bpy.ops.node.view_selected()
                except RuntimeError:
                    pass

        st["hovered_node"] = None
        redraw_ui("NODE_EDITOR")

    def _pan_view(self, context: Context, dx: int, dy: int, smooth: bool = False) -> None:
        st = self._st
        if not st:
            return
        rect = st.get("rect", (0, 0, 100, 100))
        bounds = st.get("tree_bounds", (0, 0, 100, 100))
        padding = st.get("padding", 6 * _get_ui_scale())
        mx, my, mw, mh = rect
        inner_w = max(mw - 2 * padding, 1.0)
        inner_h = max(mh - 2 * padding, 1.0)
        bbox_w = max(bounds[2] - bounds[0], 1.0)
        bbox_h = max(bounds[3] - bounds[1], 1.0)
        base_scale = min(inner_w / bbox_w, inner_h / bbox_h)
        scale = base_scale * st.get("zoom", 1.0)
        if scale <= 0:
            return
        visible = _get_visible_rect(self._space, self._region)

        if visible:
            vw_rect = visible[2] - visible[0]
            vh_rect = visible[3] - visible[1]
            view_zoom_x = self._region.width / vw_rect if vw_rect > 0 else 1.0
            view_zoom_y = self._region.height / vh_rect if vh_rect > 0 else 1.0

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
                    pan_before = st["pan"][0], st["pan"][1]

                    with self._override_ctx(context):
                        bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                    _clamp_pan_to_viewport(self._space, self._region, st)

                    clamp_dx = st["pan"][0] - pan_before[0]
                    clamp_dy = st["pan"][1] - pan_before[1]

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
        rect = st.get("rect", (0, 0, 100, 100))
        bounds = st.get("tree_bounds", (0, 0, 100, 100))
        padding = st.get("padding", 6 * _get_ui_scale())
        mx, my, mw, mh = rect
        inner_w = max(mw - 2 * padding, 1.0)
        inner_h = max(mh - 2 * padding, 1.0)
        bbox_w = max(bounds[2] - bounds[0], 1.0)
        bbox_h = max(bounds[3] - bounds[1], 1.0)
        base_scale = min(inner_w / bbox_w, inner_h / bbox_h)
        scale = base_scale * st.get("zoom", 1.0)
        if scale <= 0:
            return
        visible = _get_visible_rect(self._space, self._region)
        if not visible:
            return
        vw = visible[2] - visible[0]
        vh = visible[3] - visible[1]
        view_zoom_x = self._region.width / vw if vw > 0 else 1.0
        view_zoom_y = self._region.height / vh if vh > 0 else 1.0
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

        vw = visible[2] - visible[0]
        vh = visible[3] - visible[1]
        view_zoom_x = self._region.width / vw if vw > 0 else 1.0
        view_zoom_y = self._region.height / vh if vh > 0 else 1.0

        pan_x = int(delta_tree_x * view_zoom_x)
        pan_y = int(delta_tree_y * view_zoom_y)
        if pan_x == 0 and pan_y == 0:
            return

        addon = context.preferences.addons.get(__package__)
        settings = addon.preferences.settings if addon else None
        if settings and getattr(settings, "smooth_pan", True):
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
                    st["pan"][0] += dx
                    st["pan"][1] += dy
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
        redraw_ui("NODE_EDITOR")

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
        redraw_ui("NODE_EDITOR")

    def _apply_center_animation(self, context: Context) -> None:
        if not self._anim_active:
            return
        self._anim_progress += 1 / 24
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
        redraw_ui("NODE_EDITOR")

    def _update_cursor(self, context: Context, event: Event) -> None:
        st = self._st
        if not st or not st.get("rect"):
            return
        in_minimap = _is_in_minimap(self._mx, self._my, st)
        if not in_minimap:
            if self._last_cursor:
                context.window.cursor_modal_set("DEFAULT")
                self._last_cursor = ""
            old_handle = st.get("hovered_handle")
            st["hovered_handle"] = None
            if old_handle:
                redraw_ui("NODE_EDITOR")
            return
        handle = self._get_handle_at(context, event)
        old_handle = st.get("hovered_handle")
        st["hovered_handle"] = handle
        if handle != old_handle:
            redraw_ui("NODE_EDITOR")
        is_clamped = handle and (st.get("width_clamped") or st.get("height_clamped"))
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
        sx, sy, ex, ey = _get_safe_bounds(self._area, self._region, self._space, corner)
        max_w = max(64, int(ex - sx - 10 * ui_scale))
        max_h = max(64, int(ey - sy - 35 * ui_scale))

        if self._resize_handle in ("W", "C"):
            if corner in ("TOP_RIGHT", "BOTTOM_RIGHT"):
                new_w = max(64, min(max_w, int(w0 - dx / ui_scale)))
            else:
                new_w = max(64, min(max_w, int(w0 + dx / ui_scale)))
            settings.minimap_width = new_w

        if self._resize_handle in ("H", "C"):
            if corner in ("TOP_RIGHT", "TOP_LEFT"):
                new_h = max(64, min(max_h, int(h0 - dy / ui_scale)))
            else:
                new_h = max(64, min(max_h, int(h0 + dy / ui_scale)))
            settings.minimap_height = new_h

        # Detect percentage clamp for visual feedback
        safe_w = ex - sx
        safe_h = ey - sy
        max_mw_pct = getattr(settings, "max_width_pct", 30) / 100.0
        max_mh_pct = getattr(settings, "max_height_pct", 40) / 100.0
        st = self._st
        if not st:
            return
        st["hovered_handle"] = self._resize_handle
        st["width_clamped"] = settings.minimap_width * ui_scale > safe_w * max_mw_pct
        st["height_clamped"] = settings.minimap_height * ui_scale > safe_h * max_mh_pct

    def invoke(self, context: Context, _event: Event) -> set[str]:
        if context.area.type != "NODE_EDITOR":
            logger.debug("invoke: cancelled — area type is %s", context.area.type)
            return {"CANCELLED"}
        self._window_ptr = context.window.as_pointer()
        self._pan_acc = [0.0, 0.0]
        self._redirect_acc = [0.0, 0.0]
        self._frame_all_armed = False
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
            self._st["width_clamped"] = False
            self._st["height_clamped"] = False
            self._st["hovered_handle"] = None
            self._st["resize_active"] = None


class NODEMAP_OT_frame_selected(Operator):
    """Focus the minimap view on selected nodes."""

    bl_idname = "nodemap.frame_selected"
    bl_label = "Frame Selected"
    bl_description = "Focus the minimap view on selected nodes"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        frame_selected()
        return {"FINISHED"}


class NODEMAP_OT_frame_view(Operator):
    """Focus the minimap view on the current editor viewport."""

    bl_idname = "nodemap.frame_view"
    bl_label = "Frame View"
    bl_description = "Focus the minimap view on the current editor viewport"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        frame_view()
        return {"FINISHED"}


classes = (
    NODEMAP_OT_toggle,
    NODEMAP_OT_frame_all,
    NODEMAP_OT_frame_selected,
    NODEMAP_OT_frame_view,
    NODEMAP_OT_navigate,
)
