"""Nodemap popover in the topbar."""

from bpy.types import Panel

from .helpers import _state


class NODEMAP_PT_popup(Panel):
    bl_label = "Nodemap Options"
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

        layout.label(text="Nodemap")

        header, body = layout.panel("NODEMAP_PT_operators", default_closed=False)
        header.label(text="Frame")
        if body:
            row = body.row(align=True)
            row.operator("nodemap.frame_all", text="All")
            row.operator("nodemap.frame_view", text="View")
            row.operator("nodemap.frame_selected", text="Selected")

            col = body.column(heading="View", align=True)
            col.prop(settings, "frame_view_fill", text="Fill Minimap")

        header, body = layout.panel("NODEMAP_PT_appearance", default_closed=False)
        header.label(text="Appearance")
        if body:
            col = body.column(heading="Show", align=True)
            col.prop(settings, "show_node_count", text="Count")
            row = col.row(align=True)
            row.active = settings.interactive
            row.prop(settings, "show_frame_all_btn", text="Frame All")
            col.prop(settings, "show_frames", text="Frames")

            row = body.row(heading="Links", align=True)
            row.prop(settings, "show_socket_indicators", text="Sockets")
            row.prop(settings, "show_wires", text="Wires")

            body.separator()
            col = body.column(heading="Labels")
            row = col.row(align=True, heading="")
            row.prop(settings, "show_names", text="")
            sub = row.row(align=True)
            sub.active = settings.show_names
            sub.prop(settings, "node_label_mode", expand=True)
            sub = col.row(align=True)
            sub.active = settings.show_frames
            sub.prop(settings, "show_frame_labels", text="Frame Labels")

            header, sub_body = body.panel("NODEMAP_PT_layout", default_closed=True)
            header.label(text="Layout")
            if sub_body:
                sub_body.prop(settings, "position", text="Position")

                col = sub_body.column(align=True)
                col.prop(settings, "minimap_width", text="Size X")
                col.prop(settings, "minimap_height", text="Y")

                col = sub_body.column(align=True)
                col.prop(settings, "max_width_pct", text="Max Region X")
                col.prop(settings, "max_height_pct", text="Y")

            header, body = body.panel("NODEMAP_PT_theme", default_closed=True)
            header.label(text="Theme")
            if body:
                row = body.row(heading="Colored", align=True)
                row.prop(settings, "colored_nodes", text="Nodes")
                sub = row.row(align=True)
                sub.active = settings.show_wires | settings.show_socket_indicators
                sub.prop(settings, "show_wire_color", text="Wires")

                row = body.row(heading="Background", align=True)
                row.prop(settings, "custom_bg_color", text="")
                sub = row.row(align=True)
                sub.active = settings.custom_bg_color
                sub.prop(settings, "bg_color", text="")

                body.prop(settings, "opacity", text="Opacity", slider=True)

        header, body = layout.panel("NODEMAP_PT_behavior", default_closed=False)
        header.label(text="Behavior")
        if body:
            col = body.column(heading="Minimap", align=True)
            col.prop(settings, "show_by_default", text="Show in New Editors")

            body.prop(settings, "follow_view", text="Follow View")

            sub_header, sub_body = body.panel("NODEMAP_PT_interactive", default_closed=True)
            sub_header.use_property_split = False
            sub_header.use_property_decorate = False
            sub_header.prop(settings, "interactive", text="Interactive")

            if sub_body:
                sub_body.active = settings.interactive

                col = sub_body.column()
                col.prop(settings, "left_click_action", text="Left Click")
                col.prop(settings, "right_click_action", text="Right Click")

                if {"SELECT", "PAN_SELECT"} & {
                    settings.left_click_action,
                    settings.right_click_action,
                }:
                    sub_body.prop(settings, "auto_frame_selected", text="Auto Frame Selected")

                sub_body.row().prop(settings, "scroll_wheel_mode", text="Scroll Zoom", expand=True)


def draw_minimap_header_button(self, context):
    if context.area.type != "NODE_EDITOR":
        return
    layout = self.layout
    overlay = context.space_data.overlay
    st = _state()

    row = layout.row(align=True)
    row.active = overlay.show_overlays
    row.operator("nodemap.toggle", text="", depress=st.get("enabled", True), icon="META_PLANE")
    row.popover(panel="NODEMAP_PT_popup", text="")


classes = (NODEMAP_PT_popup,)
