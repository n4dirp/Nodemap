"""Nodemap popover in the topbar."""

from bl_ui.utils import PresetPanel
from bpy.types import Panel

from .presets import PRESET_SUBDIR
from .state import _state


class NODEMAP_PT_popup(Panel):
    bl_label = "Nodemap Options"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "HEADER"
    bl_ui_units_x = 14

    @classmethod
    def poll(cls, context):
        return context.space_data.type == "NODE_EDITOR"

    def draw(self, context):
        layout = self.layout
        prefs = context.preferences.addons.get(__package__).preferences
        settings = prefs.settings
        st = _state()
        layout.active = st.enabled

        row = layout.row()
        row.label(text="Nodemap")

        sub = row.row(align=True)
        sub.alignment = "RIGHT"
        NODEMAP_PT_presets.draw_panel_header(sub)
        sub.separator()

        sub.operator("nodemap.open_pref", text="", icon="PREFERENCES", emboss=False)

        row = layout.row(align=True)
        row.operator("nodemap.frame_all", text="Frame All")
        row.operator("nodemap.frame_view", text="Frame View")
        if not settings.follow_view:
            row.operator("nodemap.frame_selected", text="Frame Selected")

        header, body = layout.panel("NODEMAP_PT_layout", default_closed=False)
        header.label(text="Layout")
        if body:
            row = body.row()
            row.label(text="Position")
            grid = row.grid_flow(
                row_major=True,
                columns=2,
                even_columns=True,
                even_rows=True,
                align=True,
            )
            for item in settings.bl_rna.properties["position"].enum_items:
                grid.prop_enum(settings, "position", item.identifier)

            col = body.column()
            col.label(text="Display")
            grid = body.grid_flow(
                row_major=True,
                columns=3,
                even_columns=True,
                even_rows=True,
                align=True,
            )

            grid.prop(settings, "show_node_borders", text="Borders")
            grid.prop(settings, "show_node_count", text="Count")
            grid.prop(settings, "show_frames", text="Frames")
            grid.prop(settings, "show_socket_indicators", text="Sockets")
            grid.prop(settings, "show_type_list", text="Type List")
            grid.prop(settings, "show_wires", text="Wires")

            body.separator()
            col = body.column()
            row = col.row()
            row.prop(settings, "show_names", text="Node Labels")
            sub = row.row()
            sub.active = settings.show_names
            sub.prop(settings, "node_label_mode", expand=True)
            if settings.show_frames:
                col.prop(settings, "show_frame_labels", text="Frame Labels")

            if settings.interactive:
                body.separator()
                col = body.column()
                col.label(text="Buttons")

                row = col.row()

                col = row.column()
                col.prop(settings, "show_frame_all_btn", text="Frame All")
                col.prop(settings, "show_frame_view_btn", text="Frame View")
                col.prop(settings, "show_frame_selected_btn", text="Frame Selected")

                row.prop(settings, "show_list_toggle_btn", text="Toggle List")

        header, body = layout.panel("NODEMAP_PT_theme", default_closed=True)
        header.label(text="Theme")
        if body:
            col = body.column()
            col.prop(settings, "opacity", text="Panel Opacity", slider=True)

            row = col.row(align=True)
            row.prop(settings, "viewport_fill_rect", text="View Highlight")
            sub = row.row(align=True)
            sub.active = settings.viewport_fill_rect
            sub.prop(settings, "viewport_fill_color", text="")

            row = col.row(align=True)
            row.prop(settings, "show_viewport_overlay", text="View Dimming")
            sub = row.row(align=True)
            sub.active = settings.show_viewport_overlay
            sub.prop(settings, "viewport_overlay_color", text="")

            row = col.row(align=True)
            row.prop(settings, "custom_bg_color", text="Custom Backdrop")
            sub = row.row(align=True)
            sub.active = settings.custom_bg_color
            sub.prop(settings, "bg_color", text="")

            row = col.row(align=True)
            row.prop(settings, "custom_text_color", text="Custom Text Color")
            sub = row.row(align=True)
            sub.active = settings.custom_text_color
            sub.prop(settings, "text_color", text="")

            row = col.row(align=True)
            row.prop(settings, "show_text_shadow", text="Text Shadows")

            row = col.row()
            row.prop(settings, "colored_nodes", text="Node Colors")
            sub = row.row()
            sub.active = settings.show_wires | settings.show_socket_indicators
            sub.prop(settings, "show_wire_color", text="Wire Colors")

        header, body = layout.panel("NODEMAP_PT_options", default_closed=False)
        header.label(text="Options")
        if body:
            flow = body.grid_flow(columns=2)
            flow.prop(settings, "interactive", text="Map Navigation")
            flow.prop(settings, "follow_view", text="Follow View")


def draw_minimap_header_button(self, context):
    if context.area.type != "NODE_EDITOR":
        return
    layout = self.layout
    snode = context.space_data
    overlay = context.space_data.overlay
    st = _state()

    row = layout.row(align=True)
    row.active = snode.node_tree is not None and overlay.show_overlays
    row.operator("nodemap.toggle", text="", depress=st.enabled, icon="META_PLANE")
    row.popover(panel="NODEMAP_PT_popup", text="")


class NODEMAP_PT_presets(PresetPanel, Panel):
    bl_label = "Nodemap Presets"
    preset_subdir = PRESET_SUBDIR
    preset_operator = "script.execute_preset"
    preset_add_operator = "nodemap.preset_add"


classes = (NODEMAP_PT_presets, NODEMAP_PT_popup)
