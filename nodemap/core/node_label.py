"""Mirror Blender's node header titles for minimap labels.

Blender draws the node header from ``bke::node_label``: the custom label wins, otherwise the node type's
label function shows the linked data or the active operation, otherwise the static type name is used.
``node.name`` is only the unique identifier and never the title.
"""

from __future__ import annotations

_MISSING_DATA_BLOCK = "Missing Data-Block"

_IMAGE_LABEL_TYPES = frozenset(
    {
        "ShaderNodeTexImage",
        "ShaderNodeTexEnvironment",
        "CompositorNodeImage",
        "TextureNodeImage",
    }
)

_MASK_LABEL_TYPES = frozenset({"CompositorNodeMask"})

_OPERATION_LABEL_PROPS: dict[str, str] = {
    "ShaderNodeMath": "operation",
    "ShaderNodeVectorMath": "operation",
    "TextureNodeMath": "operation",
    "FunctionNodeIntegerMath": "operation",
    "FunctionNodeBooleanMath": "operation",
    "FunctionNodeCompare": "operation",
    "FunctionNodeFloatToInt": "rounding_mode",
    "ShaderNodeMixRGB": "blend_type",
    "TextureNodeMixRGB": "blend_type",
}


def _enum_ui_name(node: object, prop_name: str) -> str:
    """Return the UI name of an enum property value, or "" when it cannot be resolved."""
    value = getattr(node, prop_name, None)
    if not isinstance(value, str) or not value:
        return ""
    props = getattr(getattr(node, "bl_rna", None), "properties", None)
    if props is None:
        return ""
    try:
        items = props[prop_name].enum_items
    except (KeyError, AttributeError, TypeError):
        return ""
    try:
        item_name = items[value].name
        return str(item_name) if item_name else ""
    except (KeyError, AttributeError, TypeError):
        pass
    for item in items:
        if getattr(item, "identifier", None) == value:
            return str(getattr(item, "name", "") or "")
    return ""


def _effective_node_label(node: object) -> str:
    """Return the header title Blender shows for a node.

    Mirror ``bke::node_label`` for the common cases: custom label first, then the linked data-block
    or operation name, then the static type name.
    """
    custom = getattr(node, "label", "") or ""
    if isinstance(custom, str) and custom:
        return custom
    bl_idname = getattr(node, "bl_idname", "") or ""
    if getattr(node, "type", "") == "GROUP" or str(bl_idname).endswith("NodeGroup"):
        tree = getattr(node, "node_tree", None)
        tree_name = getattr(tree, "name", "") if tree is not None else ""
        if isinstance(tree_name, str) and tree_name:
            return tree_name
        return _MISSING_DATA_BLOCK
    if bl_idname in _IMAGE_LABEL_TYPES:
        image = getattr(node, "image", None)
        image_name = getattr(image, "name", "") if image is not None else ""
        if isinstance(image_name, str) and image_name:
            return image_name
    elif bl_idname in _MASK_LABEL_TYPES:
        mask = getattr(node, "mask", None)
        mask_name = getattr(mask, "name", "") if mask is not None else ""
        if isinstance(mask_name, str) and mask_name:
            return mask_name
    elif bl_idname == "ShaderNodeMix":
        if getattr(node, "data_type", "") == "RGBA":
            blend_name = _enum_ui_name(node, "blend_type")
            if blend_name:
                return blend_name
    elif bl_idname in _OPERATION_LABEL_PROPS:
        operation_name = _enum_ui_name(node, _OPERATION_LABEL_PROPS[bl_idname])
        if operation_name:
            return operation_name
    bl_label = getattr(node, "bl_label", "") or ""
    return bl_label if isinstance(bl_label, str) else ""


def _label_fingerprint_parts(node: object) -> tuple:
    """Return the raw label inputs identifying a node's header title.

    Use stored property values instead of resolved UI names so the per-frame tree snapshot stays cheap;
    any edit that changes the displayed title changes these parts.
    """
    parts = [getattr(node, "name", "") or "", getattr(node, "label", "") or ""]
    tree = getattr(node, "node_tree", None)
    if tree is not None:
        parts.append(getattr(tree, "name", "") or "")
    elif getattr(node, "type", "") == "GROUP":
        parts.append("")
    image = getattr(node, "image", None)
    if image is not None:
        parts.append(getattr(image, "name", "") or "")
    mask = getattr(node, "mask", None)
    if mask is not None:
        parts.append(getattr(mask, "name", "") or "")
    bl_idname = getattr(node, "bl_idname", "") or ""
    if bl_idname == "ShaderNodeMix":
        parts.append(getattr(node, "data_type", "") or "")
        parts.append(getattr(node, "blend_type", "") or "")
    elif bl_idname in _OPERATION_LABEL_PROPS:
        parts.append(getattr(node, _OPERATION_LABEL_PROPS[bl_idname], "") or "")
    return tuple(parts)
