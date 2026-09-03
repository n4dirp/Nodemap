"""Provide the performance preset for the minimap."""

import bpy

addon_id = next((ext.module for ext in bpy.context.preferences.addons if ext.module.endswith("nodemap")), "nodemap")
settings = bpy.context.preferences.addons[addon_id].preferences.settings

settings.use_animations = False
settings.show_text_shadows = False
settings.use_custom_noodle_curving = True
settings.noodle_curving = 0
settings.debounce_delay = 0.150
