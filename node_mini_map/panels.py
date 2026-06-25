"""Node Editor minimap popover in the topbar."""

from bpy.types import Panel

from .helpers import _state


class NODES_MINIMAP_PT_popup(Panel):
    bl_label = "Network Viewer"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "HEADER"
    bl_ui_units_x = 12

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "NODE_EDITOR"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        overlay = context.space_data.overlay
        layout.enabled = overlay.show_overlays
        prefs = context.preferences.addons.get(__package__).preferences
        settings = prefs.settings

        layout.label(text="Network Viewer")
        layout.prop(settings, "show_by_default")

        header, body = layout.panel("appearance", default_closed=False)
        header.label(text="Appearance")
        if body:
            body.use_property_split = True
            body.use_property_decorate = False

            col = body.column(align=True)
            col.prop(settings, "position", text="Alignment", expand=False)

            col = body.column(align=True)
            col.prop(settings, "minimap_width", text="Size X")
            col.prop(settings, "minimap_height", text="Y")

            col = body.column(align=True)
            col.prop(settings, "max_width_pct", text="Max Region X")
            col.prop(settings, "max_height_pct", text="Y")

            body.prop(settings, "opacity", text="Opacity", slider=True)

            row = body.row(align=True, heading="Labels")
            row.prop(settings, "show_names", text="")
            sub = row.row(align=True)
            sub.active = settings.show_names
            sub.prop(settings, "node_label_mode", text="")

            col = body.column(align=True, heading="Show")
            col.prop(settings, "colored_nodes", text="Node Colors")
            col.prop(settings, "show_wires", text="Node Wires")
            sub = col.row(align=True)
            sub.active = settings.show_wires
            sub.prop(settings, "show_wire_color", text="Wire Colors")
            col.prop(settings, "show_node_count", text="Total Count")

        header, body = layout.panel("behavior", default_closed=False)
        header.label(text="Behavior")
        if body:
            body.use_property_split = True
            body.use_property_decorate = False

            col = body.column()
            col.prop(settings, "interactive", text="Interactive Minimap")
            sub = col.column()
            sub.active = settings.interactive
            sub.prop(settings, "left_click_action", text="Left Click")
            sub.prop(settings, "right_click_action", text="Right Click")
            sub.prop(settings, "scroll_wheel_mode", text="Scroll Wheel")
            sub.prop(settings, "auto_frame_selected", text="Auto Frame Selected")


def draw_minimap_header_button(self, context):
    if context.area.type != "NODE_EDITOR":
        return
    layout = self.layout
    overlay = context.space_data.overlay
    st = _state()

    row = layout.row(align=True)
    row.active = overlay.show_overlays
    row.operator("node_mini_map.toggle", text="", depress=st.get("enabled", True), icon="META_PLANE")
    row.popover(panel="NODES_MINIMAP_PT_popup", text="")


classes = (NODES_MINIMAP_PT_popup,)
