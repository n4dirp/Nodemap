"""Global presets for the nodemap popup settings."""

from bl_operators.presets import AddPresetBase
from bpy.types import Operator

PRESET_SUBDIR = "nodemap"

ADDON_ID_DEFINE = (
    "addon_id = next((ext.module for ext in "
    "bpy.context.preferences.addons "
    "if ext.module.endswith('nodemap')), "
    "'nodemap')"
)


class NODEMAP_OT_preset(AddPresetBase, Operator):
    bl_idname = "nodemap.preset_add"
    bl_label = "Add Nodemap Preset"
    bl_description = "Add or remove a preset"
    preset_menu = "NODEMAP_PT_presets"
    preset_defines = [
        ADDON_ID_DEFINE,
        "settings = bpy.context.preferences.addons[addon_id].preferences.settings",
    ]
    preset_values = [
        # Layout
        "settings.position",
        "settings.show_node_count",
        "settings.show_frames",
        "settings.show_node_borders",
        "settings.show_socket_indicators",
        "settings.show_wires",
        "settings.show_type_list",
        "settings.type_list_sort",
        "settings.show_frame_all_btn",
        "settings.show_frame_view_btn",
        "settings.show_frame_selected_btn",
        "settings.show_list_toggle_btn",
        "settings.show_names",
        "settings.node_label_mode",
        "settings.show_frame_labels",
        # Theme
        "settings.opacity",
        "settings.viewport_fill_rect",
        "settings.viewport_fill_color",
        "settings.show_viewport_overlay",
        "settings.viewport_overlay_color",
        "settings.custom_bg_color",
        "settings.bg_color",
        "settings.colored_nodes",
        "settings.show_wire_color",
        # Options
        "settings.interactive",
        "settings.follow_view",
    ]
    preset_subdir = PRESET_SUBDIR


classes = (NODEMAP_OT_preset,)
