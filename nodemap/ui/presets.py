"""Provide global presets for the minimap popup settings."""

from bl_operators.presets import AddPresetBase
from bpy.types import Operator

PRESET_SUBDIR = "nodemap"

# Captures the add-on id at preset-save time.
ADDON_ID_DEFINE_SNIPPET = (
    "addon_id = next((ext.module for ext in "
    "bpy.context.preferences.addons "
    "if ext.module.endswith('nodemap')), "
    "'nodemap')"
)


class NODEMAP_OT_preset(AddPresetBase, Operator):
    """Save or remove a minimap preset for the current settings."""

    bl_idname = "nodemap.preset_add"
    bl_label = "Add Nodemap Preset"
    bl_description = "Add or remove a preset"
    preset_menu = "NODEMAP_PT_presets"
    preset_defines = [
        ADDON_ID_DEFINE_SNIPPET,
        "settings = bpy.context.preferences.addons[addon_id].preferences.settings",
    ]
    preset_values = [
        # Layout.
        "settings.dock_mode",
        "settings.corner_position",
        "settings.edge_position",
        "settings.offset_x",
        "settings.offset_y",
        "settings.use_snap_to_borders",
        "settings.show_node_count",
        "settings.show_frames",
        "settings.show_node_outline",
        "settings.show_socket_indicators",
        "settings.show_reroutes",
        "settings.show_wires",
        "settings.show_type_list",
        "settings.show_type_colors",
        "settings.type_list_sort",
        "settings.type_list_font_size",
        "settings.type_list_width_percent",
        "settings.show_frame_all_button",
        "settings.show_frame_view_button",
        "settings.show_frame_selected_button",
        "settings.show_list_toggle_button",
        "settings.show_move_button",
        "settings.show_node_labels",
        "settings.compact_node_labels",
        "settings.show_frame_labels",
        # Theme.
        "settings.opacity",
        "settings.viewport_fill_rect",
        "settings.viewport_fill_color",
        "settings.show_viewport_overlay",
        "settings.viewport_overlay_color",
        "settings.use_custom_background",
        "settings.background_color",
        "settings.use_custom_text",
        "settings.text_color",
        "settings.show_text_shadow",
        "settings.show_node_colors",
        "settings.show_wire_color",
        "settings.use_custom_noodle_curving",
        "settings.noodle_curving",
        "settings.wire_thickness",
        "settings.wire_opacity",
        # Options.
        "settings.interactive",
        "settings.follow_view",
    ]
    preset_subdir = PRESET_SUBDIR


classes = (NODEMAP_OT_preset,)
