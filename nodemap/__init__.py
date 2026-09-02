"""Register the Nodemap entry point.

Register add-on classes, draw handlers, header UI, keymaps, and preset
paths. The POST_PIXEL handler in the Node Editor drives minimap rendering,
while the modal operator auto-starts per window when interactive mode is on.
"""

# Nodemap - Blender Extension.
# Minimap overlay for the Node Editor.
# SPDX-License-Identifier: GPL-3.0-or-later

import logging
import os

import bpy
from bpy.types import SpaceNodeEditor

from .core.state import _ensure_area_states, _minimap_window_operators, _registration_state
from .draw.overlay import draw_minimap
from .ops.navigate import classes as operator_classes
from .ui.panels import classes as panel_classes
from .ui.panels import draw_minimap_header_button
from .ui.preferences import _update_logger_from_prefs
from .ui.preferences import classes as prefs_classes
from .ui.presets import classes as preset_classes

ADDON_DIR = os.path.dirname(__file__)

logger = logging.getLogger(__package__)
logger.propagate = False
logger.addHandler(logging.NullHandler())

classes = (
    *prefs_classes,
    *preset_classes,
    *operator_classes,
    *panel_classes,
)

_draw_handler = None
addon_keymap_bindings: list[tuple[bpy.types.KeyMap, bpy.types.KeyMapItem]] = []


def _register_keymaps():
    """Register the Ctrl+M toggle shortcut in the add-on keyconfig."""
    window_manager = bpy.context.window_manager
    addon_keyconfig = window_manager.keyconfigs.addon
    if not addon_keyconfig:
        return
    node_editor_keymap = addon_keyconfig.keymaps.new(name="Node Editor", space_type="NODE_EDITOR")
    toggle_keymap_item = node_editor_keymap.keymap_items.new("nodemap.toggle", type="M", value="PRESS", ctrl=True)
    addon_keymap_bindings.append((node_editor_keymap, toggle_keymap_item))


def _unregister_keymaps():
    """Remove all keymap items registered by this add-on."""
    for node_editor_keymap, toggle_keymap_item in addon_keymap_bindings:
        node_editor_keymap.keymap_items.remove(toggle_keymap_item)
    addon_keymap_bindings.clear()


def register():
    """Register classes, draw handler, header button, and keymaps."""
    for cls in classes:
        bpy.utils.register_class(cls)

    _update_logger_from_prefs()

    global _draw_handler
    _draw_handler = SpaceNodeEditor.draw_handler_add(
        draw_minimap,
        (),
        "WINDOW",
        "POST_PIXEL",
    )

    bpy.types.NODE_HT_header.append(draw_minimap_header_button)
    logger.debug("Register complete, calling _ensure_area_states()")
    _ensure_area_states()
    logger.debug("_ensure_area_states() done")

    _registration_state["done"] = True
    logger.debug("Registration fully complete (_registration_done=True)")

    _register_keymaps()

    bpy.utils.register_preset_path(ADDON_DIR)


def unregister():
    """Unregister preset paths, keymaps, draw handler, and classes."""

    bpy.utils.unregister_preset_path(ADDON_DIR)

    _unregister_keymaps()

    global _draw_handler
    if _draw_handler is not None:
        try:
            SpaceNodeEditor.draw_handler_remove(_draw_handler, "WINDOW")
        except (ValueError, RuntimeError):
            pass
        _draw_handler = None

    try:
        bpy.types.NODE_HT_header.remove(draw_minimap_header_button)
    except (ValueError, RuntimeError):
        pass

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass

    _minimap_window_operators.clear()
    _registration_state["done"] = False
