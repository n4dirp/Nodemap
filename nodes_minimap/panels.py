"""Node Editor header button and minimap popover panel."""

from bpy.types import Panel


def draw_minimap_header_button(self, context):
    if context.area.type != "NODE_EDITOR":
        return
    layout = self.layout
    prefs = context.preferences.addons.get(__package__).preferences

    row = layout.row(align=True)
    row.operator(
        "nodes_minimap.toggle",
        text="",
        icon="META_PLANE",
        depress=getattr(prefs.settings, "enabled", True),
    )
    row.popover("NODES_MINIMAP_PT_popup", text="")


class NODES_MINIMAP_PT_popup(Panel):
    bl_label = "Minimap Settings"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "WINDOW"
    bl_ui_units_x = 10

    def draw(self, context):
        layout = self.layout
        prefs = context.preferences.addons.get(__package__).preferences
        settings = prefs.settings

        layout.label(text="Minimap")

        head, body = layout.panel("NODES_MINIMAP_PT_interface", default_closed=False)
        head.label(text="Interface")
        if body:
            body.prop(settings, "position", text="", expand=False)

            col = body.column(align=True)
            col.prop(settings, "minimap_width", text="Size X")
            col.prop(settings, "minimap_height", text="Size Y")

            body.prop(settings, "opacity", text="Opacity", slider=True)

        head, body = layout.panel("NODES_MINIMAP_PT_behavior", default_closed=False)
        head.label(text="Behavior")
        if body:
            body.prop(settings, "show_node_count")
            body.prop(settings, "auto_frame_selected")
            body.prop(settings, "interactive")
