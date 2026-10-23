"""Cache GPU textures for Nodemap button icons.

Load the white-alpha PNGs in ``nodemap/icons/`` once and expose them as GPU
textures for the tinted-image draw in ``gpu_draw``. Textures resolve lazily
on first paint so a missing file never breaks drawing; callers draw nothing
when a texture is unavailable.
"""

import logging
import os

import bpy
import gpu

from .. import __package__ as base_package

logger = logging.getLogger(base_package)

BUTTON_FILES: dict[str, str] = {
    "ALL": "nodemap_frame_all.png",
    "VIEW": "nodemap_frame_view.png",
    "SELECTED": "nodemap_frame_sel.png",
    "LIST": "nodemap_type_list.png",
    "FOLLOW_VIEW": "nodemap_follow_view.png",
    "DRAG": "blender_icon_grip.png",
}

SEARCH_FILE = "blender_icon_viewzoom.png"
CLEAR_FILE = "blender_icon_x.png"

EXPAND_FILE = "blender_icon_rightarrow.png"
COLLAPSE_FILE = "blender_icon_downarrow_hlt.png"

_textures: dict[str, gpu.types.GPUTexture] = {}
_owned_images: set[str] = set()


def _icon_path(filename: str) -> str:
    """Return the absolute path of an icon asset."""
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "icons", filename)


def _load_texture(filename: str):
    """Load *filename* as a GPU texture, or return None when unavailable."""
    path = _icon_path(filename)
    if not os.path.isfile(path):
        logger.debug("icon_textures: missing icon file %s", filename)
        return None
    known_pointers = {image.as_pointer() for image in bpy.data.images}
    try:
        image = bpy.data.images.load(path, check_existing=True)
    except (RuntimeError, OSError) as error:
        logger.debug("icon_textures: cannot load %s: %s", filename, error)
        return None
    try:
        texture = gpu.texture.from_image(image)
    except Exception as error:
        logger.debug("icon_textures: cannot create texture for %s: %s", filename, error)
        if image.as_pointer() not in known_pointers:
            try:
                bpy.data.images.remove(image)
            except (RuntimeError, ReferenceError):
                pass
        return None
    if image.as_pointer() not in known_pointers:
        _owned_images.add(image.name)
    return texture


def _texture_for(filename: str):
    """Return the cached GPU texture for *filename*, loading it on first use."""
    texture = _textures.get(filename)
    if texture is None:
        texture = _load_texture(filename)
        if texture is None:
            return None
        _textures[filename] = texture
    return texture


def _get_button_texture(icon: str):
    """Return the cached GPU texture for a button icon, or None."""
    filename = BUTTON_FILES.get(icon)
    if filename is None:
        return None
    return _texture_for(filename)


def _get_search_texture():
    """Return the cached GPU texture for the search icon, or None."""
    return _texture_for(SEARCH_FILE)


def _get_clear_texture():
    """Return the cached GPU texture for the search clear icon, or None."""
    return _texture_for(CLEAR_FILE)


def _get_chevron_texture(expanded: bool):
    """Return the cached GPU texture for a chevron state, or None."""
    return _texture_for(COLLAPSE_FILE if expanded else EXPAND_FILE)


def _clear_icon_textures() -> None:
    """Drop cached icon textures and remove images loaded by this module."""
    _textures.clear()
    while _owned_images:
        name = _owned_images.pop()
        image = bpy.data.images.get(name)
        if image is None or image.users > 0 or image.use_fake_user:
            continue
        try:
            bpy.data.images.remove(image)
        except (RuntimeError, ReferenceError):
            pass
