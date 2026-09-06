"""Manage custom preview icons for Nodemap UI.

Own a ``previews`` collection that is populated on registration and cleared
on unregistration. Uses a mutable dict so submodules can resolve icons by
name without import-order coupling.
"""

import os

import bpy
from bpy.utils import previews

_previews = None
_icons: dict[str, bpy.types.ImagePreview] = {}


def _addon_icon_dir():
    """Return the addon root directory containing icon assets."""
    return os.path.dirname(os.path.dirname(__file__))


def _load_icons():
    """Load custom icons from the addon directory into the registry."""
    global _previews
    if _previews is not None:
        return
    _previews = previews.new()
    icon_path = os.path.join(_addon_icon_dir(), "icons", "nodemap.png")
    _icons["NODEMAP_ICON"] = _previews.load("NODEMAP_ICON", icon_path, "IMAGE")


def _unload_icons():
    """Release custom icon previews and clear the registry."""
    global _previews
    _icons.clear()
    if _previews is not None:
        previews.remove(_previews)
        _previews = None


def _icon_id(name: str) -> int:
    """Return the icon id for the named preview, or 0 if missing."""
    preview = _icons.get(name)
    return preview.icon_id if preview is not None else 0
