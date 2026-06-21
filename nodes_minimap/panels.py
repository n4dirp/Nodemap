"""Node Editor minimap settings inside the native Overlays popover."""

from .helpers import _state


def draw_minimap_overlay_settings(self, context):
    if context.area.type != "NODE_EDITOR":
        return
    layout = self.layout
    prefs = context.preferences.addons.get(__package__).preferences
    settings = prefs.settings
    st = _state()

    layout.separator()
    col = layout.column()

    head, body = col.panel("NODES_MINIMAP_PT_settings", default_closed=True)
    row = head.row()
    row.operator(
        "nodes_minimap.toggle",
        text="Minimap",
        depress=st.get("enabled", True),
    )

    if body:
        body.label(text="Alignment")
        body.prop(settings, "position", text="", expand=False)

        col = body.column(align=True)
        col.prop(settings, "minimap_width", text="Size X")
        col.prop(settings, "minimap_height", text="Size Y")

        body.prop(settings, "opacity", text="Opacity", slider=True)

        body.separator()
        col = body.column(align=True)
        col.prop(settings, "show_node_count")
        col.prop(settings, "show_node_initials")
        col.prop(settings, "auto_frame_selected")
        col.prop(settings, "interactive")
