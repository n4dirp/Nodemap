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
        col = body.column(align=True)
        col.label(text="Alignment")
        col.prop(settings, "position", text="", expand=False)

        col = body.column(align=True)
        col.prop(settings, "minimap_width", text="Size X")
        col.prop(settings, "minimap_height", text="Size Y")

        body.prop(settings, "opacity", text="Opacity", slider=True)

        col = body.column(align=True)
        col.prop(settings, "colored_nodes", text="Colored Nodes")
        col.prop(settings, "show_node_initials", text="Name Initials")
        col.prop(settings, "show_node_count", text="Node Count")
        col.separator()
        col.prop(settings, "interactive", text="Interactive Minimap")
        sub = col.column(align=True)
        sub.active = settings.interactive
        sub.prop(settings, "auto_frame_selected", text="Auto Frame Selected")
