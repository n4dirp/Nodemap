"""Modal operator for minimap interaction."""

import logging

import bpy
from bpy.types import Context, Event, Operator

from .helpers import (
    _clamp_pan_to_viewport,
    _find_node_at,
    _get_minimap_transform,
    _get_safe_bounds,
    _get_ui_scale,
    _get_visible_rect,
    _state,
    frame_all,
    redraw_ui,
)

logger = logging.getLogger(__package__)


def _is_in_minimap(region_x: int, region_y: int) -> bool:
    st = _state()
    mx, my, mw, mh = st.get("rect", (0, 0, 0, 0))
    return mx <= region_x <= mx + mw and my <= region_y <= my + mh


def _region_to_tree(region_x: int, region_y: int) -> tuple[float, float] | None:
    if not _state().get("rect"):
        return None
    cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform()
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
    _area_ptr: int = 0
    _dragging: bool = False
    _was_in_minimap: bool = False

    _mmb_dragging: bool = False
    _mmb_drag_start: tuple[int, int] | None = None

    _resize_handle: str | None = None
    _resize_start_mouse: tuple[int, int] | None = None
    _resize_start_values: tuple[int, int] | None = None
    _last_cursor: str = ""
    _pan_acc: list[float]
    _redirect_acc: list[float]
    _frame_all_armed: bool = False

    def modal(self, context: Context, event: Event) -> set[str]:
        if not context.area:
            st = _state(self._area_ptr)
            st["modal_active"] = False
            st["modal_area_ptr"] = 0
            return {"CANCELLED"}
        st = _state()
        area_ptr_now = context.area.as_pointer()
        modal_ptr = st.get("modal_area_ptr", 0)
        if modal_ptr != area_ptr_now:
            st = _state(self._area_ptr)
            st["modal_active"] = False
            st["modal_area_ptr"] = 0
            return {"CANCELLED"}
        if not st.get("enabled", True):
            return {"PASS_THROUGH"}

        addon = context.preferences.addons.get(__package__)
        if addon and not getattr(addon.preferences.settings, "interactive", True):
            return {"PASS_THROUGH"}
        settings = addon.preferences.settings if addon else None

        in_minimap = _is_in_minimap(event.mouse_region_x, event.mouse_region_y)

        match event.type:
            case "LEFTMOUSE":
                # --- Release ---
                if event.value == "RELEASE":
                    if self._frame_all_armed:
                        self._frame_all_armed = False
                        btn = st.get("frame_all_btn")
                        if btn:
                            bx, by, bw, bh = btn
                            mx = event.mouse_region_x
                            my = event.mouse_region_y
                            if bx <= mx <= bx + bw and by <= my <= by + bh:
                                frame_all()
                        return {"RUNNING_MODAL"}
                    if self._resize_handle:
                        self._resize_handle = None
                        self._resize_start_mouse = None
                        self._resize_start_values = None
                        context.window.cursor_modal_set("DEFAULT")
                        self._last_cursor = ""
                        st = _state()
                        st["width_clamped"] = False
                        st["height_clamped"] = False
                        st["hovered_handle"] = None
                        st["resize_active"] = None
                        redraw_ui("NODE_EDITOR")
                        return {"RUNNING_MODAL"}
                    if self._dragging:
                        self._dragging = False
                        self._drag_start = None
                        self._pan_acc = [0.0, 0.0]
                        return {"RUNNING_MODAL"}
                    if not self._dragging and self._was_in_minimap:
                        if settings and settings.left_click_action in ("SELECT", "PAN_SELECT"):
                            self._handle_click_selection(context, event)
                        self._was_in_minimap = False
                        self._drag_start = None
                        return {"RUNNING_MODAL"}
                    self._was_in_minimap = False
                    self._drag_start = None
                    return {"PASS_THROUGH"}
                # --- Press ---
                self._was_in_minimap = in_minimap
                if self._was_in_minimap:
                    btn = st.get("frame_all_btn")
                    if btn:
                        bx, by, bw, bh = btn
                        mx = event.mouse_region_x
                        my = event.mouse_region_y
                        if bx <= mx <= bx + bw and by <= my <= by + bh:
                            self._frame_all_armed = True
                            return {"RUNNING_MODAL"}
                    if addon:
                        handle = self._get_handle_at(context, event)
                        if handle:
                            self._resize_handle = handle
                            st["resize_active"] = handle
                            redraw_ui("NODE_EDITOR")
                            self._resize_start_mouse = (event.mouse_region_x, event.mouse_region_y)
                            self._resize_start_values = (
                                settings.minimap_width,
                                settings.minimap_height,
                            )
                            cursor = _CURSOR_MAP[handle]
                            context.window.cursor_modal_set(cursor)
                            self._last_cursor = cursor
                            return {"RUNNING_MODAL"}
                    self._drag_start = (event.mouse_region_x, event.mouse_region_y)
                    if settings and settings.left_click_action in ("PAN", "PAN_SELECT"):
                        self._center_view_on_mouse(context, event.mouse_region_x, event.mouse_region_y)
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
                        st = _state()
                        st["width_clamped"] = False
                        st["height_clamped"] = False
                        st["hovered_handle"] = None
                        st["resize_active"] = None
                        redraw_ui("NODE_EDITOR")
                        return {"RUNNING_MODAL"}
                    if self._dragging:
                        self._dragging = False
                        self._drag_start = None
                        self._pan_acc = [0.0, 0.0]
                        return {"RUNNING_MODAL"}
                    if not self._dragging and self._was_in_minimap:
                        if settings and settings.right_click_action in ("SELECT", "PAN_SELECT"):
                            self._handle_click_selection(context, event)
                        self._was_in_minimap = False
                        self._drag_start = None
                        return {"RUNNING_MODAL"}
                    self._was_in_minimap = False
                    self._drag_start = None
                    return {"PASS_THROUGH"}
                # --- Press ---
                self._was_in_minimap = in_minimap
                if self._was_in_minimap:
                    if addon:
                        handle = self._get_handle_at(context, event)
                        if handle:
                            self._resize_handle = handle
                            st["resize_active"] = handle
                            redraw_ui("NODE_EDITOR")
                            self._resize_start_mouse = (event.mouse_region_x, event.mouse_region_y)
                            self._resize_start_values = (
                                settings.minimap_width,
                                settings.minimap_height,
                            )
                            cursor = _CURSOR_MAP[handle]
                            context.window.cursor_modal_set(cursor)
                            self._last_cursor = cursor
                            return {"RUNNING_MODAL"}
                    self._drag_start = (event.mouse_region_x, event.mouse_region_y)
                    if settings and settings.right_click_action in ("PAN", "PAN_SELECT"):
                        self._center_view_on_mouse(context, event.mouse_region_x, event.mouse_region_y)
                    return {"RUNNING_MODAL"}
                else:
                    self._drag_start = None
                    return {"PASS_THROUGH"}

            case "MIDDLEMOUSE":
                if event.value == "PRESS" and in_minimap:
                    self._mmb_dragging = True
                    self._mmb_drag_start = (event.mouse_region_x, event.mouse_region_y)
                    return {"RUNNING_MODAL"}
                if event.value == "RELEASE" and self._mmb_dragging:
                    self._mmb_dragging = False
                    self._mmb_drag_start = None
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
                        tree_coord = _region_to_tree(event.mouse_region_x, event.mouse_region_y)
                        if tree_coord and context.space_data.edit_tree:
                            hovered = _find_node_at(context.space_data.edit_tree.nodes, tree_coord[0], tree_coord[1])
                            if hovered:
                                new_hovered = hovered.name
                    if old_hovered != new_hovered:
                        st["hovered_node"] = new_hovered
                        redraw_ui("NODE_EDITOR")
                if self._mmb_dragging and self._mmb_drag_start:
                    dx = event.mouse_region_x - self._mmb_drag_start[0]
                    dy = event.mouse_region_y - self._mmb_drag_start[1]
                    pan_before = st["pan"][0], st["pan"][1]
                    st["pan"][0] += dx
                    st["pan"][1] += dy
                    _clamp_pan_to_viewport(context.space_data, context.region, st)
                    rejected_x = dx - (st["pan"][0] - pan_before[0])
                    rejected_y = dy - (st["pan"][1] - pan_before[1])
                    if (rejected_x != 0 or rejected_y != 0) and getattr(settings, "follow_view", False):
                        st["pan"][0] = pan_before[0] + dx
                        st["pan"][1] = pan_before[1] + dy
                        self._redirect_to_view2d(context, -dx, -dy)
                    elif rejected_x != 0 or rejected_y != 0:
                        self._redirect_to_view2d(context, -int(rejected_x), -int(rejected_y))
                    self._mmb_drag_start = (event.mouse_region_x, event.mouse_region_y)
                    redraw_ui("NODE_EDITOR")
                    return {"RUNNING_MODAL"}
                if self._drag_start is not None:
                    dx = event.mouse_region_x - self._drag_start[0]
                    dy = event.mouse_region_y - self._drag_start[1]
                    if abs(dx) > 2 or abs(dy) > 2 or self._dragging:
                        self._dragging = True
                        if self._was_in_minimap:
                            self._pan_view(context, dx, dy)
                            self._drag_start = (event.mouse_region_x, event.mouse_region_y)
                    return {"RUNNING_MODAL"}
                if in_minimap:
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "WHEELUPMOUSE" | "WHEELDOWNMOUSE":
                if in_minimap:
                    if event.ctrl or event.shift:
                        space = context.space_data
                        region = context.region
                        visible = _get_visible_rect(space, region)
                        if visible:
                            vw = visible[2] - visible[0]
                            vh = visible[3] - visible[1]
                            scroll_factor = 0.05
                            direction = 1 if event.type == "WHEELUPMOUSE" else -1
                            pan_x = int(vw * scroll_factor * -direction) if event.ctrl else 0
                            pan_y = int(vh * scroll_factor * direction) if event.shift else 0
                            try:
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
                            if event.type == "WHEELUPMOUSE":
                                bpy.ops.view2d.zoom_in(zoomfacx=zoom_factor, zoomfacy=zoom_factor)
                            else:
                                bpy.ops.view2d.zoom_out(zoomfacx=-zoom_factor, zoomfacy=-zoom_factor)
                        except RuntimeError:
                            pass
                    else:
                        zoom_delta = 1.15 if event.type == "WHEELUPMOUSE" else 0.85
                        # Base the manual scroll jump off the effective visual zoom rather than
                        # the invisible stored preference to ensure a smooth transition out of the auto-clamp
                        effective_zoom = st.get("zoom", 1.0)

                        is_constrained = False
                        if addon and getattr(addon.preferences.settings, "follow_view", False):
                            # If effective zoom is strictly less than base_zoom, it means the viewport
                            # indicator hit the minimap boundary and forced an auto-zoom out.
                            if effective_zoom < st.get("base_zoom", 1.0) - 0.001:
                                is_constrained = True

                        # Intercept scroll UP if constrained and zoom the Node Editor instead
                        if is_constrained and event.type == "WHEELUPMOUSE":
                            try:
                                zoom_factor = 0.05
                                bpy.ops.view2d.zoom_in(zoomfacx=zoom_factor, zoomfacy=zoom_factor)
                            except RuntimeError:
                                pass
                        else:
                            new_zoom = max(0.1, min(effective_zoom * zoom_delta, 20.0))

                            cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform()
                            tree_coord = _region_to_tree(event.mouse_region_x, event.mouse_region_y)

                            if scale > 0 and tree_coord is not None:
                                tx, ty = tree_coord
                                base_scale = scale / effective_zoom
                                pan_x, pan_y = st.get("pan", [0.0, 0.0])

                                pan_x_new = pan_x - (tx - tree_cx) * base_scale * (new_zoom - effective_zoom)
                                pan_y_new = pan_y - (ty - tree_cy) * base_scale * (new_zoom - effective_zoom)

                                st["base_zoom"] = new_zoom
                                st["zoom"] = new_zoom
                                st["pan"] = [pan_x_new, pan_y_new]
                                _clamp_pan_to_viewport(context.space_data, context.region, st)

                    redraw_ui("NODE_EDITOR")
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "HOME":
                if event.value == "PRESS" and in_minimap:
                    frame_all()
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case _:
                return {"PASS_THROUGH"}

    def _handle_click_selection(self, context: Context, event: Event) -> None:
        space = context.space_data
        if not space or space.type != "NODE_EDITOR":
            return
        node_tree = space.edit_tree
        if not node_tree or not node_tree.nodes:
            return

        tree_coord = _region_to_tree(event.mouse_region_x, event.mouse_region_y)
        if tree_coord is None:
            return

        node = _find_node_at(node_tree.nodes, tree_coord[0], tree_coord[1])
        if node:
            bpy.ops.node.select_all(action="DESELECT")
            node.select = True

            addon = context.preferences.addons.get(__package__)
            if addon and getattr(addon.preferences.settings, "auto_frame_selected", True):
                try:
                    bpy.ops.node.view_selected()
                except RuntimeError:
                    pass

        _state()["hovered_node"] = None
        redraw_ui("NODE_EDITOR")

    def _pan_view(self, context: Context, dx: int, dy: int) -> None:
        st = _state()
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
        space = context.space_data
        region = context.region
        visible = _get_visible_rect(space, region)

        if visible:
            vw_rect = visible[2] - visible[0]
            vh_rect = visible[3] - visible[1]
            view_zoom_x = region.width / vw_rect if vw_rect > 0 else 1.0
            view_zoom_y = region.height / vh_rect if vh_rect > 0 else 1.0

            self._pan_acc[0] += (dx / scale) * view_zoom_x
            self._pan_acc[1] += (dy / scale) * view_zoom_y
            pan_x = int(self._pan_acc[0])
            pan_y = int(self._pan_acc[1])
            self._pan_acc[0] -= pan_x
            self._pan_acc[1] -= pan_y

            if pan_x != 0 or pan_y != 0:
                try:
                    st = _state()
                    pan_before = st["pan"][0], st["pan"][1]

                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                    _clamp_pan_to_viewport(space, region, st)

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
                            bpy.ops.view2d.pan(deltax=extra_pan_x, deltay=extra_pan_y)
                            _clamp_pan_to_viewport(space, region, st)
                except RuntimeError:
                    pass

    def _redirect_to_view2d(self, context: Context, dx: float, dy: float) -> None:
        st = _state()
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
        space = context.space_data
        region = context.region
        visible = _get_visible_rect(space, region)
        if not visible:
            return
        vw = visible[2] - visible[0]
        vh = visible[3] - visible[1]
        view_zoom_x = region.width / vw if vw > 0 else 1.0
        view_zoom_y = region.height / vh if vh > 0 else 1.0
        self._redirect_acc[0] += (dx / scale) * view_zoom_x
        self._redirect_acc[1] += (dy / scale) * view_zoom_y
        pan_x = int(self._redirect_acc[0])
        pan_y = int(self._redirect_acc[1])
        self._redirect_acc[0] -= pan_x
        self._redirect_acc[1] -= pan_y
        if pan_x != 0 or pan_y != 0:
            try:
                bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
            except RuntimeError:
                pass

    def _center_view_on_mouse(self, context: Context, mx: int, my: int) -> None:
        tree_coord = _region_to_tree(mx, my)
        if not tree_coord:
            return

        space = context.space_data
        region = context.region
        visible = _get_visible_rect(space, region)

        if visible:
            view_cx = (visible[0] + visible[2]) / 2.0
            view_cy = (visible[1] + visible[3]) / 2.0
            delta_tree_x = tree_coord[0] - view_cx
            delta_tree_y = tree_coord[1] - view_cy

            vw = visible[2] - visible[0]
            vh = visible[3] - visible[1]
            view_zoom_x = region.width / vw if vw > 0 else 1.0
            view_zoom_y = region.height / vh if vh > 0 else 1.0

            pan_x = int(delta_tree_x * view_zoom_x)
            pan_y = int(delta_tree_y * view_zoom_y)
            if pan_x != 0 or pan_y != 0:
                try:
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
                    _clamp_pan_to_viewport(space, region, _state())
                except RuntimeError:
                    pass

    def _update_cursor(self, context: Context, event: Event) -> None:
        st = _state()
        if not st.get("rect"):
            return
        in_minimap = _is_in_minimap(event.mouse_region_x, event.mouse_region_y)
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
        st = _state()
        addon = context.preferences.addons.get(__package__)
        if not addon:
            return None
        corner = getattr(addon.preferences.settings, "position", "TOP_RIGHT")
        ui_scale = _get_ui_scale()
        return _get_resize_handle(st, corner, event.mouse_region_x, event.mouse_region_y, ui_scale)

    def _resize_apply_delta(self, context: Context, event: Event) -> None:
        addon = context.preferences.addons.get(__package__)
        if not addon:
            return
        settings = addon.preferences.settings
        if not self._resize_start_values:
            return
        w0, h0 = self._resize_start_values
        dx = event.mouse_region_x - self._resize_start_mouse[0]
        dy = event.mouse_region_y - self._resize_start_mouse[1]
        corner = getattr(settings, "position", "TOP_RIGHT")

        ui_scale = _get_ui_scale()
        sx, sy, ex, ey = _get_safe_bounds(context.area, context.region, context.space_data, corner)
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
        st = _state()
        st["hovered_handle"] = self._resize_handle
        st["width_clamped"] = settings.minimap_width * ui_scale > safe_w * max_mw_pct
        st["height_clamped"] = settings.minimap_height * ui_scale > safe_h * max_mh_pct

    def invoke(self, context: Context, _event: Event) -> set[str]:
        if context.area.type != "NODE_EDITOR":
            return {"CANCELLED"}
        self._area_ptr = context.area.as_pointer()
        self._pan_acc = [0.0, 0.0]
        self._redirect_acc = [0.0, 0.0]
        self._frame_all_armed = False
        st = _state(self._area_ptr)
        st["modal_active"] = True
        st["modal_area_ptr"] = self._area_ptr
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, _context: Context) -> None:
        st = _state(self._area_ptr)
        st["modal_active"] = False
        st["modal_area_ptr"] = 0
        st["width_clamped"] = False
        st["height_clamped"] = False
        st["hovered_handle"] = None
        st["resize_active"] = None


classes = (
    NODEMAP_OT_toggle,
    NODEMAP_OT_frame_all,
    NODEMAP_OT_navigate,
)
