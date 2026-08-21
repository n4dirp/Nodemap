import bpy

addon_id = next((ext.module for ext in bpy.context.preferences.addons if ext.module.endswith("nodemap")), "nodemap")
settings = bpy.context.preferences.addons[addon_id].preferences.settings

settings.position = "BOTTOM_RIGHT"
settings.show_node_count = False
settings.show_frames = True
settings.show_node_borders = False
settings.show_socket_indicators = False
settings.show_wires = False
settings.show_type_list = False
settings.type_list_sort = "NAME"
settings.show_frame_all_btn = False
settings.show_frame_view_btn = False
settings.show_frame_selected_btn = False
settings.show_list_toggle_btn = False
settings.show_names = False
settings.node_label_mode = "COMPACT"
settings.show_frame_labels = False
settings.opacity = 0.800000011920929
settings.viewport_fill_rect = False
settings.viewport_fill_color = (1.0, 1.0, 1.0, 0.05000000074505806)
settings.show_viewport_overlay = True
settings.viewport_overlay_color = (0.0, 0.0, 0.0, 0.5)
settings.custom_bg_color = False
settings.bg_color = (0.44999998807907104, 0.44999998807907104, 0.44999998807907104, 0.949999988079071)
settings.colored_nodes = False
settings.show_wire_color = False
settings.interactive = True
settings.follow_view = False
