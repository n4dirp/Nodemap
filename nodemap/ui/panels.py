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
    bl_ui_units_x = 13

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

        header, body = layout.panel("NODEMAP_PT_frame_operators", default_closed=False)
        header.label(text="Map Frame")
        if body:
            row = body.row(align=True)
            row.operator("nodemap.frame_all", text="All")
            row.operator("nodemap.frame_view", text="View")
            if not settings.follow_view:
                row.operator("nodemap.frame_selected", text="Selected")

        header, root_body = layout.panel("NODEMAP_PT_layout", default_closed=False)
        header.label(text="Objects")
        if root_body:
            flow = root_body.grid_flow(row_major=True, columns=2, even_columns=False, even_rows=False, align=True)
            flow.prop(settings, "show_frames", text="Frames")
            flow.prop(settings, "show_node_colors", text="Node Colors")
            flow.prop(settings, "show_node_outline", text="Outlines")
            flow.prop(settings, "show_reroutes", text="Reroutes")
            flow.prop(settings, "show_socket_indicators", text="Sockets")
            flow.prop(settings, "show_node_count", text="Total Count")

        header, body = root_body.panel("NODEMAP_PT_labels", default_closed=False)
        header.label(text="Labels")
        if body:
            flow = body.grid_flow(row_major=False, columns=2, even_columns=False, even_rows=False, align=True)
            flow.prop(settings, "show_node_labels", text="Node Labels")
            flow.prop(settings, "compact_node_labels", text="Compact")
            flow.prop(settings, "show_frame_labels", text="Frame Labels")

        header, body = root_body.panel("NODEMAP_PT_wires", default_closed=True)
        header.prop(settings, "show_wires", text="Wires")
        if body:
            body.active = settings.show_wires
            flow = body.grid_flow(row_major=True, columns=2, even_columns=False, even_rows=False, align=False)
            flow.prop(settings, "show_dashed_wires", text="Dashed Wires")
            flow.prop(settings, "highlight_selected_wires", text="Highlight Selections")
            flow.prop(settings, "show_wire_color", text="Wire Colors")

            row = flow.row()
            row.prop(settings, "use_custom_noodle_curving", text="")
            sub = row.row(align=True)
            sub.active = settings.use_custom_noodle_curving
            sub.prop(settings, "noodle_curving", text="Curving", slider=True)
            flow.prop(settings, "wire_opacity", text="Opacity", slider=True)
            flow.prop(settings, "wire_thickness", text="Thickness", slider=True)

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
            row.prop(settings, "show_viewport_overlay", text="View Dimming")
            sub = row.row(align=True)
            sub.active = settings.show_viewport_overlay
            sub.prop(settings, "viewport_overlay_color", text="")

            row = col.row(align=True)
            row.prop(settings, "use_custom_background", text="Background")
            sub = row.row(align=True)
            sub.active = settings.use_custom_background
            sub.prop(settings, "background_color", text="")

            row = col.row(align=True)
            row.prop(settings, "use_custom_text", text="Text Color")
            sub = row.row(align=True)
            sub.active = settings.use_custom_text
            sub.prop(settings, "text_color", text="")

        header, root_body = layout.panel("NODEMAP_PT_options", default_closed=True)
        # header.label(text="Interaction")
        header.prop(settings, "interactive", text="Interactive Map")
        header.prop(settings, "follow_view", text="Follow View")

        if root_body:
            # flow = root_body.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=True, align=True)
            # sub = flow.row(align=True)
            # sub.active = settings.interactive
            root_body.prop(settings, "use_animations", text="Animations")

            root_body.active = settings.interactive
            header, body = root_body.panel("NODEMAP_PT_type_list", default_closed=True)
            header.prop(settings, "show_type_list", text="Type List")
            if body:
                body.active = settings.show_type_list
                flow = body.grid_flow(row_major=True, columns=2, even_columns=True, even_rows=False, align=True)
                flow.prop(settings, "show_search_bar", text="Filter Bar")
                sub = flow.row(align=True)
                sub.active = settings.show_node_colors
                sub.prop(settings, "show_type_colors", text="Type Colors")

            header, body = root_body.panel("NODEMAP_PT_buttons", default_closed=False)
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
