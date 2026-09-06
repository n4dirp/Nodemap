"""Provide a Nodemap popover in the top bar.

Provide the Node Editor header button and the ``NODEMAP_PT_popup`` popover
that exposes minimap settings, frame actions, and preset switching.
"""

from bl_ui.utils import PresetPanel
from bpy.types import Panel

from ..core.helpers import get_addon_preferences
from ..core.icons import _icon_id
from ..core.state import _state
from .presets import PRESET_SUBDIR


class NODEMAP_PT_popup(Panel):
    """Expose minimap display and interaction settings."""

    bl_label = "Nodemap Options"
    bl_space_type = "NODE_EDITOR"
    bl_region_type = "HEADER"
    bl_ui_units_x = 12

    @classmethod
    def poll(cls, context):
        """Return True only inside the Node Editor."""
        return context.space_data.type == "NODE_EDITOR"

    def draw(self, context):
        """Draw popover sections for framing, object visibility, and options."""
        layout = self.layout
        prefs = get_addon_preferences(context)
        if prefs is None:
            return
        settings = prefs.settings
        minimap_state = _state()
        layout.active = minimap_state.enabled

        row = layout.row()
        row.label(text="Nodemap")

        sub = row.row(align=True)
        sub.alignment = "RIGHT"
        NODEMAP_PT_presets.draw_panel_header(sub)
        sub.separator()

        sub.operator("nodemap.open_pref", text="", icon="PREFERENCES", emboss=False)

        header, body = layout.panel("NODEMAP_PT_frame", default_closed=False)
        header.label(text="Frame")
        if body:
            row = body.row(align=True)
            row.operator("nodemap.frame_all", text="All")
            row.operator("nodemap.frame_view", text="View")
            if not settings.follow_view:
                row.operator("nodemap.frame_selected", text="Selected")

        header, body = layout.panel("NODEMAP_PT_layout", default_closed=False)
        header.label(text="Objects")
        if body:
            row = body.row()
            col1 = row.column()
            col1.prop(settings, "show_frames", text="Frames")
            col1.prop(settings, "show_node_outline", text="Node Outline")
            col1.prop(settings, "show_reroutes", text="Reroutes")
            col1.prop(settings, "show_socket_indicators", text="Sockets")
            col1.prop(settings, "show_node_count", text="Total Count")

            if settings.interactive:
                col1.separator()
                col = col1.column(align=True)
                col.prop(settings, "show_type_list", text="Type List")
                sub = col.row()
                sub.active = settings.show_type_list
                sub.prop(settings, "show_search_bar", text="Filter Bar")

            col2 = row.column()
            col = col2.column(align=True)
            col.prop(settings, "show_wires", text="Wires")
            sub = col.row()
            sub.active = settings.show_wires
            sub.prop(settings, "show_dashed_wires", text="Dashed Wires")

            col2.separator()
            col = col2.column(align=True)
            col.prop(settings, "show_node_labels", text="Node Labels")
            sub = col.row()
            sub.active = settings.show_node_labels
            sub.prop(settings, "compact_node_labels", text="Compact Labels")
            sub = col.row()
            sub.active = settings.show_frames
            sub.prop(settings, "show_frame_labels", text="Frame Labels")

            if settings.interactive:
                header, body = layout.panel("NODEMAP_PT_buttons", default_closed=False)
                header.label(text="Buttons")
                if body:
                    col = body.column()
                    row = col.row()
                    col = row.column(align=True)
                    col.prop(settings, "show_frame_all_button", text="Frame All")
                    col.prop(settings, "show_frame_view_button", text="Frame View")
                    if not settings.follow_view:
                        col.prop(settings, "show_frame_selected_button", text="Frame Selected")

                    col = row.column()
                    col.prop(settings, "show_list_toggle_button", text="List Toggle")
                    col.prop(settings, "show_move_button", text="Move Handle")

        header, body = layout.panel("NODEMAP_PT_theme", default_closed=True)
        header.label(text="Theme")
        if body:
            col = body.column()
            col.prop(settings, "opacity", text="Opacity")

            row = col.row(align=True)
            row.prop(settings, "use_custom_viewport_fill", text="View Highlight")
            sub = row.row(align=True)
            sub.active = settings.use_custom_viewport_fill
            sub.prop(settings, "viewport_fill_color", text="")

            row = col.row(align=True)
            row.prop(settings, "use_custom_background", text="Background")
            sub = row.row(align=True)
            sub.active = settings.use_custom_background
            sub.prop(settings, "background_color", text="")

            row = col.row()
            row.prop(settings, "show_node_colors", text="Node Colors")
            sub = row.row()
            sub.active = settings.show_wires
            sub.prop(settings, "show_wire_color", text="Wire Colors")

        header, body = layout.panel("NODEMAP_PT_options", default_closed=True)
        header.label(text="Options")
        if body:
            flow = body.grid_flow(columns=2)
            flow.prop(settings, "interactive", text="Interactive Map")
            flow.prop(settings, "follow_view", text="Follow View")


def draw_minimap_header_button(self, context):
    """Append the minimap toggle and popover button to the Node Editor header."""
    if context.area.type != "NODE_EDITOR":
        return
    layout = self.layout
    space_node_editor = context.space_data
    overlay = context.space_data.overlay
    minimap_state = _state()

    row = layout.row(align=True)
    row.active = space_node_editor.node_tree is not None and overlay.show_overlays
    row.operator("nodemap.toggle", text="", depress=minimap_state.enabled, icon_value=_icon_id("NODEMAP_ICON"))
    row.popover(panel="NODEMAP_PT_popup", text="")


class NODEMAP_PT_presets(PresetPanel, Panel):
    """Manage presets for saving and applying minimap configurations."""

    bl_label = "Nodemap Presets"
    preset_subdir = PRESET_SUBDIR
    preset_operator = "script.execute_preset"
    preset_add_operator = "nodemap.preset_add"


classes = (NODEMAP_PT_presets, NODEMAP_PT_popup)
