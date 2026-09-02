"""Provide the performance preset for the minimap."""

import bpy

addon_id = next((ext.module for ext in bpy.context.preferences.addons if ext.module.endswith("nodemap")), "nodemap")
settings = bpy.context.preferences.addons[addon_id].preferences.settings

settings.animations = False
settings.show_text_shadows = False
settings.use_custom_wire_curvature = True
settings.wire_curvature = 0
settings.debounce_interval = 0.150
