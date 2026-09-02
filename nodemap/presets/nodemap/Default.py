"""Provide the default preset for the minimap."""

import bpy

addon_id = next((ext.module for ext in bpy.context.preferences.addons if ext.module.endswith("nodemap")), "nodemap")
settings = bpy.context.preferences.addons[addon_id].preferences.settings

settings.animations = True
settings.show_node_count = True
settings.show_frames = True
settings.show_node_outline = True
settings.show_socket_indicators = True
settings.show_wires = True
settings.use_custom_wire_curvature = False
settings.highlight_selected_wires = True
settings.show_type_list = False
settings.type_list_sort = "NAME"
settings.show_frame_all_btn = True
settings.show_frame_view_btn = True
settings.show_frame_selected_btn = True
settings.show_list_toggle_btn = True
settings.show_move_btn = True
settings.show_node_labels = True
settings.compact_node_labels = True
settings.show_frame_labels = True
settings.opacity = 0.95
settings.viewport_fill_rect = True
settings.viewport_fill_color = (0.278, 0.447, 0.702, 1.0)
settings.show_viewport_overlay = True
settings.viewport_overlay_color = (0.1, 0.1, 0.1, 0.3)
settings.custom_bg_color = False
settings.show_node_colors = True
settings.show_wire_color = True
settings.interactive = True
settings.follow_view = False
