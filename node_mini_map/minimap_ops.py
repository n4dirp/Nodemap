"""Modal operator for minimap interaction."""

import logging

import bpy
from bpy.types import Context, Event, Operator

from .helpers import (
    _find_node_at,
    _get_minimap_transform,
    _get_safe_bounds,
    _get_ui_scale,
    _get_visible_rect,
    _state,
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

    if corner == "TOP_RIGHT":
        on_left = mx <= rx <= mx + hw
        on_bottom = my <= ry <= my + hw
        if on_left and on_bottom:
            return "C"
        if on_left:
            return "W"
        if on_bottom:
            return "H"
    elif corner == "TOP_LEFT":
        on_right = mx + mw - hw <= rx <= mx + mw
        on_bottom = my <= ry <= my + hw
        if on_right and on_bottom:
            return "C"
        if on_right:
            return "W"
        if on_bottom:
            return "H"
    elif corner == "BOTTOM_RIGHT":
        on_left = mx <= rx <= mx + hw
        on_top = my + mh - hw <= ry <= my + mh
        if on_left and on_top:
            return "C"
        if on_left:
            return "W"
        if on_top:
            return "H"
    elif corner == "BOTTOM_LEFT":
        on_right = mx + mw - hw <= rx <= mx + mw
        on_top = my + mh - hw <= ry <= my + mh
        if on_right and on_top:
            return "C"
        if on_right:
            return "W"
        if on_top:
            return "H"
    return None


class NODES_MINIMAP_OT_toggle(Operator):
    """Toggle the minimap overlay on and off."""

    bl_idname = "node_mini_map.toggle"
    bl_label = "Toggle Minimap"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        st = _state()
        st["enabled"] = not st.get("enabled", True)
        redraw_ui("NODE_EDITOR")
        return {"FINISHED"}


class NODES_MINIMAP_OT_navigate(Operator):
    """Navigate the Node Editor view via the minimap."""

    bl_idname = "node_mini_map.navigate"
    bl_label = "Minimap Navigate"
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

        in_minimap = _is_in_minimap(event.mouse_region_x, event.mouse_region_y)

        match event.type:
            case "LEFTMOUSE" | "RIGHTMOUSE":
                # --- Release ---
                if event.value == "RELEASE":
                    if self._resize_handle:
                        self._resize_handle = None
                        self._resize_start_mouse = None
                        self._resize_start_values = None
                        context.window.cursor_modal_set("DEFAULT")
                        self._last_cursor = ""
                        return {"RUNNING_MODAL"}
                    if self._dragging:
                        self._dragging = False
                        self._drag_start = None
                        return {"RUNNING_MODAL"}
                    if not self._dragging and self._was_in_minimap:
                        self._handle_click_selection(context, event)
                        self._was_in_minimap = False
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
                            self._resize_start_mouse = (event.mouse_region_x, event.mouse_region_y)
                            self._resize_start_values = (
                                addon.preferences.settings.minimap_width,
                                addon.preferences.settings.minimap_height,
                            )
                            cursor = _CURSOR_MAP[handle]
                            context.window.cursor_modal_set(cursor)
                            self._last_cursor = cursor
                            return {"RUNNING_MODAL"}
                    self._drag_start = (event.mouse_region_x, event.mouse_region_y)
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
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "MOUSEMOVE":
                if self._resize_handle:
                    self._resize_apply_delta(context, event)
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
                    st["pan"][0] += dx
                    st["pan"][1] += dy
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
                return {"PASS_THROUGH"}

            case "WHEELUPMOUSE" | "WHEELDOWNMOUSE":
                if in_minimap:
                    prefs = addon.preferences.settings if addon else None
                    scroll_mode = getattr(prefs, "scroll_wheel_mode", "MINIMAP") if prefs else "MINIMAP"
                    if scroll_mode == "NODE_EDITOR":
                        try:
                            factor = 0.1
                            if event.type == "WHEELUPMOUSE":
                                bpy.ops.view2d.zoom_in(zoomfacx=factor, zoomfacy=factor)
                            else:
                                bpy.ops.view2d.zoom_out(zoomfacx=-factor, zoomfacy=-factor)
                        except RuntimeError:
                            pass
                    else:
                        zoom_delta = 1.15 if event.type == "WHEELUPMOUSE" else 0.85
                        old_zoom = st.get("zoom", 1.0)
                        new_zoom = max(0.1, min(old_zoom * zoom_delta, 20.0))

                        cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform()
                        tree_coord = _region_to_tree(event.mouse_region_x, event.mouse_region_y)

                        if scale > 0 and tree_coord is not None:
                            tx, ty = tree_coord
                            base_scale = scale / old_zoom
                            pan_x, pan_y = st.get("pan", [0.0, 0.0])

                            pan_x_new = pan_x - (tx - tree_cx) * base_scale * (new_zoom - old_zoom)
                            pan_y_new = pan_y - (ty - tree_cy) * base_scale * (new_zoom - old_zoom)

                            st["zoom"] = new_zoom
                            st["pan"] = [pan_x_new, pan_y_new]
                    redraw_ui("NODE_EDITOR")
                    return {"RUNNING_MODAL"}
                return {"PASS_THROUGH"}

            case "HOME":
                if event.value == "PRESS" and in_minimap:
                    st["zoom"] = 1.0
                    st["pan"] = [0.0, 0.0]
                    redraw_ui("NODE_EDITOR")
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
        scale = _state().get("scale", 1.0)
        if scale <= 0:
            return
        space = context.space_data
        region = context.region
        visible = _get_visible_rect(space, region)

        if visible:
            vw = visible[2] - visible[0]
            vh = visible[3] - visible[1]
            view_zoom_x = region.width / vw if vw > 0 else 1.0
            view_zoom_y = region.height / vh if vh > 0 else 1.0

            pan_x = int((dx / scale) * view_zoom_x)
            pan_y = int((dy / scale) * view_zoom_y)
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
            return
        handle = self._get_handle_at(context, event)
        cursor = _CURSOR_MAP.get(handle, "DEFAULT")
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
        max_w = max(64, min(800, int(ex - sx - 10 * ui_scale)))
        max_h = max(64, min(600, int(ey - sy - 35 * ui_scale)))

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

        redraw_ui("NODE_EDITOR")

    def invoke(self, context: Context, _event: Event) -> set[str]:
        if context.area.type != "NODE_EDITOR":
            return {"CANCELLED"}
        self._area_ptr = context.area.as_pointer()
        st = _state(self._area_ptr)
        st["modal_active"] = True
        st["modal_area_ptr"] = self._area_ptr
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, _context: Context) -> None:
        st = _state(self._area_ptr)
        st["modal_active"] = False
        st["modal_area_ptr"] = 0


classes = (
    NODES_MINIMAP_OT_toggle,
    NODES_MINIMAP_OT_navigate,
)
