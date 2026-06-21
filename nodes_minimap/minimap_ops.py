"""Modal operator for minimap interaction."""

import logging

import bpy
from bpy.types import Context, Event, Operator

from .helpers import _find_node_at, _get_minimap_transform, _get_visible_rect, _state, redraw_ui

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


class NODES_MINIMAP_OT_toggle(Operator):
    """Toggle the minimap overlay on and off."""

    bl_idname = "nodes_minimap.toggle"
    bl_label = "Toggle Minimap"
    bl_options = {"INTERNAL"}

    def execute(self, context: Context) -> set[str]:
        prefs = context.preferences.addons.get(__package__)
        if not prefs:
            return {"CANCELLED"}
        settings = prefs.preferences.settings
        settings.enabled = not settings.enabled
        redraw_ui("NODE_EDITOR")
        return {"FINISHED"}


class NODES_MINIMAP_OT_navigate(Operator):
    """Navigate the Node Editor view via the minimap."""

    bl_idname = "nodes_minimap.navigate"
    bl_label = "Minimap Navigate"
    bl_options = {"INTERNAL"}

    _drag_start: tuple[int, int] | None = None
    _area_ptr: int = 0
    _dragging: bool = False
    _was_in_minimap: bool = False

    _mmb_dragging: bool = False
    _mmb_drag_start: tuple[int, int] | None = None

    def modal(self, context: Context, event: Event) -> set[str]:
        if not context.area:
            return {"CANCELLED"}
        if _state().get("modal_area_ptr", 0) != context.area.as_pointer():
            return {"CANCELLED"}

        addon = context.preferences.addons.get(__package__)
        if addon and not getattr(addon.preferences.settings, "interactive", True):
            return {"PASS_THROUGH"}

        in_minimap = _is_in_minimap(event.mouse_region_x, event.mouse_region_y)
        st = _state()

        # 1. Capture exact Hover State for accurate node highlighting
        if event.type == "MOUSEMOVE" and not self._dragging and not self._mmb_dragging:
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

        # 2. Minimap Internal Layout Panning (Middle Mouse Drag)
        if event.type == "MIDDLEMOUSE":
            if event.value == "PRESS" and in_minimap:
                self._mmb_dragging = True
                self._mmb_drag_start = (event.mouse_region_x, event.mouse_region_y)
                return {"RUNNING_MODAL"}
            elif event.value == "RELEASE" and self._mmb_dragging:
                self._mmb_dragging = False
                self._mmb_drag_start = None
                return {"RUNNING_MODAL"}

        if event.type == "MOUSEMOVE" and self._mmb_dragging and self._mmb_drag_start:
            dx = event.mouse_region_x - self._mmb_drag_start[0]
            dy = event.mouse_region_y - self._mmb_drag_start[1]
            st["pan"][0] += dx
            st["pan"][1] += dy
            self._mmb_drag_start = (event.mouse_region_x, event.mouse_region_y)
            redraw_ui("NODE_EDITOR")
            return {"RUNNING_MODAL"}

        # 3. Minimap Internal Layout Zooming (Scroll Wheel)
        if event.type in {"WHEELUPMOUSE", "WHEELDOWNMOUSE"}:
            if in_minimap:
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

        # Frame all nodes in minimap on HOME key
        if event.type == "HOME" and event.value == "PRESS" and in_minimap:
            st["zoom"] = 1.0
            st["pan"] = [0.0, 0.0]
            redraw_ui("NODE_EDITOR")
            return {"RUNNING_MODAL"}

        # 4. Main viewport Camera Action / Click Node Selecting (Left / Right Mouse)
        if event.type in {"LEFTMOUSE", "RIGHTMOUSE"}:
            if event.value == "RELEASE":
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

            if event.value == "PRESS":
                self._was_in_minimap = in_minimap
                if self._was_in_minimap:
                    self._drag_start = (event.mouse_region_x, event.mouse_region_y)
                    self._center_view_on_mouse(context, event.mouse_region_x, event.mouse_region_y)
                    return {"RUNNING_MODAL"}
                else:
                    self._drag_start = None
                    return {"PASS_THROUGH"}

        if event.type == "MOUSEMOVE" and self._drag_start is not None:
            dx = event.mouse_region_x - self._drag_start[0]
            dy = event.mouse_region_y - self._drag_start[1]
            if abs(dx) > 2 or abs(dy) > 2 or self._dragging:
                self._dragging = True
                if self._was_in_minimap:
                    self._pan_view(context, dx, dy)
                    self._drag_start = (event.mouse_region_x, event.mouse_region_y)
            return {"RUNNING_MODAL"}

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
