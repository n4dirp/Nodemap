# Node Mini Map - Blender Extension
# Mini Map overlay for the Node Editor
# SPDX-License-Identifier: GPL-3.0-or-later

import logging

import bpy
from bpy.types import SpaceNodeEditor

from .minimap_draw import draw_minimap
from .minimap_ops import classes as operator_classes
from .panels import classes as panel_classes
from .preferences import _update_logger_from_prefs
from .preferences import classes as prefs_classes

logger = logging.getLogger(__package__)
logger.propagate = False
logger.addHandler(logging.NullHandler())

classes = (
    *prefs_classes,
    *operator_classes,
    *panel_classes,
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

    _update_logger_from_prefs()


def unregister():
    global _draw_handler, _modal_operator
    if _draw_handler is not None:
        try:
            SpaceNodeEditor.draw_handler_remove(_draw_handler, "WINDOW")
        except (ValueError, RuntimeError):
            pass
        _draw_handler = None

    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass
