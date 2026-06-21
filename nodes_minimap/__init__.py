# Nodes Minimap - Blender Extension
# Minimap overlay for the Node Editor
# SPDX-License-Identifier: GPL-3.0-or-later

import logging

import bpy
from bpy.types import SpaceNodeEditor

from .minimap_draw import draw_minimap
from .minimap_ops import NODES_MINIMAP_OT_navigate, NODES_MINIMAP_OT_toggle
from .panels import NODES_MINIMAP_PT_popup, draw_minimap_header_button
from .preferences import _update_logger_from_prefs
from .preferences import classes as prefs_classes

logger = logging.getLogger(__package__)
logger.propagate = False
logger.addHandler(logging.NullHandler())

classes = (
    *prefs_classes,
    NODES_MINIMAP_OT_navigate,
    NODES_MINIMAP_OT_toggle,
    NODES_MINIMAP_PT_popup,
)

_draw_handler = None
_modal_operator = None


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    global _draw_handler, _modal_operator
    _draw_handler = SpaceNodeEditor.draw_handler_add(
        draw_minimap,
        (),
        "WINDOW",
        "POST_PIXEL",
    )

    bpy.types.NODE_HT_header.append(draw_minimap_header_button)

    _update_logger_from_prefs()

    logger.info("Nodes Minimap registered")


def unregister():
    global _draw_handler, _modal_operator
    if _draw_handler is not None:
        try:
            SpaceNodeEditor.draw_handler_remove(_draw_handler, "WINDOW")
        except (ValueError, RuntimeError):
            pass
        _draw_handler = None

    bpy.types.NODE_HT_header.remove(draw_minimap_header_button)

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass

    logger.info("Nodes Minimap unregistered")
