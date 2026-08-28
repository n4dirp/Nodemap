import bpy

addon_id = next((ext.module for ext in bpy.context.preferences.addons if ext.module.endswith("nodemap")), "nodemap")
settings = bpy.context.preferences.addons[addon_id].preferences.settings

settings.position = "BOTTOM_RIGHT"
settings.show_node_count = True
settings.show_frames = True
settings.show_node_borders = True
settings.show_socket_indicators = True
settings.show_wires = True
settings.show_type_list = False
settings.type_list_sort = "NAME"
settings.show_frame_all_btn = True
settings.show_frame_view_btn = True
settings.show_frame_selected_btn = True
settings.show_list_toggle_btn = True
settings.show_names = False
settings.compact_node_labels = True
settings.show_frame_labels = False
settings.opacity = 0.949999988079071
settings.viewport_fill_rect = True
settings.viewport_fill_color = (1.0, 1.0, 1.0, 0.05000000074505806)
settings.show_viewport_overlay = True
settings.viewport_overlay_color = (0.0, 0.0, 0.0, 0.5)
settings.custom_bg_color = False
settings.bg_color = (0.44999998807907104, 0.44999998807907104, 0.44999998807907104, 0.949999988079071)
settings.colored_nodes = True
settings.show_wire_color = True
settings.interactive = True
settings.follow_view = False
