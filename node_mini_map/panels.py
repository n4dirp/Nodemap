"""Node Editor minimap settings in the sidebar View tab."""

from bpy.types import Panel

from .helpers import _state


class NODES_MINIMAP_PT_panel(Panel):
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Mini Map"
    bl_label = "Node Mini Map"

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "NODE_EDITOR"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        overlay = context.space_data.overlay
        layout.enabled = overlay.show_overlays

        st = _state()

        layout.operator("node_mini_map.toggle", text="Mini Map", depress=st.get("enabled", True), icon="META_PLANE")


class NODES_MINIMAP_PT_appearance(Panel):
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Mini Map"
    bl_parent_id = "NODES_MINIMAP_PT_panel"
    bl_label = "Appearance"

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

        col = layout.column(align=True)
        col.prop(settings, "position", text="Alignment", expand=False)

        col = layout.column(align=True)
        col.prop(settings, "minimap_width", text="Size X")
        col.prop(settings, "minimap_height", text="Y")

        layout.prop(settings, "opacity", text="Opacity", slider=True)

        row = layout.row(align=True, heading="Node Labels")
        row.prop(settings, "show_names", text="")
        sub = row.row(align=True)
        sub.active = settings.show_names
        sub.prop(settings, "node_label_mode", text="")

        col = layout.column(align=True, heading="Show")
        col.prop(settings, "colored_nodes", text="Colored Nodes")
        col.prop(settings, "show_wires", text="Node Wires")
        col.prop(settings, "show_node_count", text="Total Count")


class NODES_MINIMAP_PT_behavior(Panel):
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "UI"
    bl_category = "Mini Map"
    bl_parent_id = "NODES_MINIMAP_PT_panel"
    bl_label = "Behavior"

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

        col = layout.column(align=True)
        col.prop(settings, "interactive", text="Interactive Mini Map")
        sub = col.column(align=True)
        sub.active = settings.interactive
        sub.prop(settings, "auto_frame_selected", text="Auto Frame Selected")
        sub.prop(settings, "scroll_wheel_mode", text="Scroll Wheel")


classes = (
    NODES_MINIMAP_PT_panel,
    NODES_MINIMAP_PT_appearance,
    NODES_MINIMAP_PT_behavior,
)
