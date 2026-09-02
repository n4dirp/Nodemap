"""Provide the minimal preset for the minimap."""

import bpy

addon_id = next((ext.module for ext in bpy.context.preferences.addons if ext.module.endswith("nodemap")), "nodemap")
settings = bpy.context.preferences.addons[addon_id].preferences.settings

settings.show_node_count = False
settings.show_frames = True
settings.show_node_outline = True
settings.show_socket_indicators = False
settings.show_wires = False
settings.show_type_list = False
settings.type_list_sort = "NAME"
settings.show_frame_all_btn = False
settings.show_frame_view_btn = False
settings.show_frame_selected_btn = False
settings.show_list_toggle_btn = False
settings.show_move_btn = False
settings.show_node_labels = False
settings.compact_node_labels = True
settings.show_frame_labels = False
settings.opacity = 0.95
settings.viewport_fill_rect = True
settings.show_viewport_overlay = True
settings.viewport_overlay_color = (0.05, 0.05, 0.05, 0.25)
settings.custom_bg_color = False
settings.show_node_colors = False
settings.show_wire_color = False
settings.interactive = True
settings.follow_view = False
