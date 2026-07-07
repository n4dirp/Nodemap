# Nodemap - Blender Extension
# Minimap overlay for the Node Editor
# SPDX-License-Identifier: GPL-3.0-or-later

import logging

import bpy
from bpy.types import SpaceNodeEditor

from .helpers import _ensure_area_states, _minimap_window_operators, _registration_state
from .minimap_draw import draw_minimap
from .minimap_ops import classes as operator_classes
from .panels import classes as panel_classes
from .panels import draw_minimap_header_button
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

    _update_logger_from_prefs()

    global _draw_handler, _modal_operator
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


def unregister():
    global _draw_handler, _modal_operator
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
