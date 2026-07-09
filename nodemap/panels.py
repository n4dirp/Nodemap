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

        row = layout.row(align=True)
        row.operator("nodemap.frame_all", text="Frame All")
        row.operator("nodemap.frame_view", text="Frame View")

        header, body = layout.panel("NODEMAP_PT_appearance", default_closed=False)
        header.label(text="Appearance")
        if body:
            col = body.column(heading="Show", align=True)
            row = col.row(align=True)
            row.active = settings.interactive
            row.prop(settings, "show_frame_all_btn", text="Frame All")
            col.prop(settings, "show_frames", text="Frames")
            col.prop(settings, "show_node_count", text="Node Count")

            row = body.row(heading="Connections", align=True)
            row.prop(settings, "show_socket_indicators", text="Sockets")
            row.prop(settings, "show_wires", text="Wires")

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
                col = body.column()

                row = col.row(heading="Color", align=True)
                row.prop(settings, "colored_nodes", text="Nodes")
                sub = row.row(align=True)
                sub.active = settings.show_wires | settings.show_socket_indicators
                sub.prop(settings, "show_wire_color", text="Wires")

                row = col.row(heading="Overlay", align=True)
                row.prop(settings, "show_viewport_overlay", text="")
                sub = row.row(align=True)
                sub.active = settings.show_viewport_overlay
                sub.prop(settings, "viewport_overlay_color", text="")

                row = col.row(heading="Background", align=True)
                row.prop(settings, "custom_bg_color", text="")
                sub = row.row(align=True)
                sub.active = settings.custom_bg_color
                sub.prop(settings, "bg_color", text="")

                col.prop(settings, "opacity", text="Opacity", slider=True)

        header, body = layout.panel("NODEMAP_PT_behavior", default_closed=True)
        header.label(text="Behavior")
        if body:
            col = body.column(heading="Minimap", align=True)
            col.prop(settings, "show_by_default", text="Show in New Editors")

            body.prop(settings, "follow_view", text="Follow View")

            col = body.column(heading="Frame View", align=True)
            col.prop(settings, "frame_view_fill", text="Fill")

            sub_header, sub_body = body.panel("NODEMAP_PT_interactive", default_closed=False)
            sub_header.use_property_split = False
            sub_header.use_property_decorate = False
            sub_header.prop(settings, "interactive", text="Map Navigation")

            if sub_body:
                sub_body.active = settings.interactive

                col = sub_body.column()
                col.prop(settings, "left_click_action", text="Left Click")
                col.prop(settings, "right_click_action", text="Right Click")
                col.prop(settings, "scroll_wheel_mode", text="Scroll Wheel")

                if {"SELECT", "PAN_SELECT"} & {
                    settings.left_click_action,
                    settings.right_click_action,
                }:
                    col.prop(settings, "auto_frame_selected", text="Auto Frame Selected")

                col.prop(settings, "smooth_pan", text="Smooth Pan")
                if settings.smooth_pan:
                    col.prop(settings, "pan_speed", text="Pan Speed")


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
