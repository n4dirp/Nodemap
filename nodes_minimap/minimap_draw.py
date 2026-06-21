"""Minimap rendering in the Node Editor."""

import logging
import math

import blf
import bpy
import gpu
from mathutils import Matrix

from .gpu_draw import (
    _draw_filled_rounded_rect,
    _draw_pill,
    _draw_rounded_rect_border,
    _draw_text_with_shadow,
)
from .helpers import (
    _get_minimap_transform,
    _get_node_color,
    _get_node_dims,
    _get_node_editor_theme_colors,
    _get_node_tree_bounds,
    _get_safe_bounds,
    _get_ui_scale,
    _get_visible_rect,
    _state,
)

logger = logging.getLogger(__package__)

_modal_invoked_for_area: set[int] = set()


def draw_minimap():
    context = bpy.context
    space = context.space_data
    region = context.region

    if space.type != "NODE_EDITOR":
        return

    addon = context.preferences.addons.get(__package__)
    if not addon:
        return
    settings = addon.preferences.settings
    if not getattr(settings, "enabled", True):
        return

    st = _state()

    area_ptr = context.area.as_pointer()
    if getattr(settings, "interactive", True):
        if area_ptr not in _modal_invoked_for_area:
            _modal_invoked_for_area.add(area_ptr)
            try:
                bpy.ops.nodes_minimap.navigate("INVOKE_DEFAULT")
            except RuntimeError:
                pass

    node_tree = space.edit_tree
    if not node_tree or not node_tree.nodes or len(node_tree.nodes) == 0:
        return

    nodes = node_tree.nodes
    bounds = _get_node_tree_bounds(nodes)

    bbox_w = bounds[2] - bounds[0]
    bbox_h = bounds[3] - bounds[1]
    if bbox_w <= 0 or bbox_h <= 0:
        return

    ui_scale = _get_ui_scale()
    colors = _get_node_editor_theme_colors()

    mw = getattr(settings, "minimap_width", 200) * ui_scale
    mh = getattr(settings, "minimap_height", 200) * ui_scale
    margin = 35 * ui_scale
    padding = 6 * ui_scale
    corner = getattr(settings, "position", "TOP_RIGHT")
    bg_opacity = getattr(settings, "opacity", 0.85)

    sx, sy, ex, ey = _get_safe_bounds(context.area, region, space, corner)
    if corner == "TOP_RIGHT":
        mx = ex - mw - 10
        my = ey - mh - margin
    elif corner == "TOP_LEFT":
        mx = sx + 10
        my = ey - mh - margin
    elif corner == "BOTTOM_RIGHT":
        mx = ex - mw - 10
        my = sy + margin
    else:
        mx = sx + 10
        my = sy + margin

    st["rect"] = (mx, my, mw, mh)
    st["tree_bounds"] = bounds
    st["margin"] = margin
    st["padding"] = padding

    gpu.state.blend_set("ALPHA")

    bg_color = colors["bg"][:3] + (bg_opacity,)
    panel_r = colors.get("panel_roundness", 4.0)
    _draw_filled_rounded_rect(mx, my, mw, mh, panel_r, bg_color)
    _draw_rounded_rect_border(mx, my, mw, mh, panel_r, colors["bg_border"], 0.5)

    # Calculate transformation incorporating Pan & Internal zoom levels
    cx, cy, scale, tree_cx, tree_cy = _get_minimap_transform()
    st["scale"] = scale  # Save standard scale multiplier for operations map

    # Prevent map internals bleeding outside background when user adjusts zoom
    _scissor_active = False
    try:
        gpu.state.scissor_test_set(True)
        gpu.state.scissor_set(int(mx + 1), int(my + 1), int(mw - 2), int(mh - 2))
        _scissor_active = True
    except Exception:
        pass

    hovered_node_name = st.get("hovered_node")

    frames = [n for n in nodes if n.type == "FRAME"]
    regular_nodes = [n for n in nodes if n.type != "FRAME"]

    # 1. Draw Transparent Layout Frame Nodes First
    for node in frames:
        w, h = _get_node_dims(node)
        nx = cx + (node.location_absolute.x - tree_cx) * scale
        ny = cy + (node.location_absolute.y - h - tree_cy) * scale
        nw_s = max(w * scale, 1.0)
        nh_s = max(h * scale, 1.0)

        is_hovered = node.name == hovered_node_name
        alpha = 0.6 if is_hovered else 0.5

        frame_color = _get_node_color(node, colors.get("frame_node", colors["node"]))
        bg_frame = (frame_color[0], frame_color[1], frame_color[2], alpha)
        _draw_filled_rounded_rect(nx, ny, nw_s, nh_s, 2.0 * ui_scale, bg_frame)

        border_col = colors["indicator"] if is_hovered else frame_color
        _draw_rounded_rect_border(nx, ny, nw_s, nh_s, 2.0 * ui_scale, border_col, 0.5)

    # 2. Draw Connection Wires
    _draw_wires(nodes, tree_cx, tree_cy, scale, cx, cy, colors)

    # 3. Draw Regular Nodes & Accurate Hover Highlight Mapping
    for node in regular_nodes:
        if node.hide:
            w, h = 100.0, 30.0
        else:
            w, h = _get_node_dims(node)

        nx = cx + (node.location_absolute.x - tree_cx) * scale
        ny = cy + (node.location_absolute.y - h - tree_cy) * scale
        nw_s = max(w * scale, 1.0)
        nh_s = max(h * scale, 1.0)

        node_r = colors.get("node_roundness", 2.0) * ui_scale
        min_dim = 3.0 * ui_scale
        is_hovered = node.name == hovered_node_name

        fill_color = _get_node_color(node, colors["node"])
        if is_hovered:
            fill_color = (
                min(fill_color[0] * 1.35, 1.0),
                min(fill_color[1] * 1.35, 1.0),
                min(fill_color[2] * 1.35, 1.0),
                fill_color[3],
            )

        if nw_s < min_dim or nh_s < min_dim:
            _draw_filled_rounded_rect(nx, ny, max(nw_s, min_dim), max(nh_s, min_dim), node_r, fill_color)
        else:
            _draw_filled_rounded_rect(nx, ny, nw_s, nh_s, node_r, fill_color)
            border_w = 1.5 * ui_scale if (node.select or is_hovered) else 0.5 * ui_scale
            border_c = colors["indicator"] if (node.select or is_hovered) else colors["node_border"]
            if node.mute:
                border_c = (border_c[0], border_c[1], border_c[2], border_c[3] * 0.35)
            _draw_rounded_rect_border(nx, ny, nw_s, nh_s, node_r, border_c, border_w)
            if node.mute:
                muted_overlay = (bg_color[0], bg_color[1], bg_color[2], 0.4)
                _draw_filled_rounded_rect(nx, ny, nw_s, nh_s, node_r, muted_overlay)

    # 4. Draw Active Main Viewport Interactive Panning Box
    visible = _get_visible_rect(space, region)
    if visible:
        vx = cx + (visible[0] - tree_cx) * scale
        vy = cy + (visible[1] - tree_cy) * scale
        vw = max((visible[2] - visible[0]) * scale, 1.0)
        vh = max((visible[3] - visible[1]) * scale, 1.0)

        # Clamp visible rect to minimap bounds
        v_left = max(vx, mx)
        v_bottom = max(vy, my)
        v_right = min(vx + vw, mx + mw)
        v_top = min(vy + vh, my + mh)

        overlay = (0.0, 0.0, 0.0, 0.35)

        if v_left > mx:
            _draw_filled_rounded_rect(mx, my, v_left - mx, mh, 0, overlay)
        if v_right < mx + mw:
            _draw_filled_rounded_rect(v_right, my, (mx + mw) - v_right, mh, 0, overlay)
        if v_bottom > my:
            _draw_filled_rounded_rect(v_left, my, v_right - v_left, v_bottom - my, 0, overlay)
        if v_top < my + mh:
            _draw_filled_rounded_rect(v_left, v_top, v_right - v_left, (my + mh) - v_top, 0, overlay)

        _draw_rounded_rect_border(vx, vy, vw, vh, 2.0, colors["node_outline"], 0.5 * ui_scale)

    if _scissor_active:
        try:
            gpu.state.scissor_test_set(False)
        except Exception:
            pass

    gpu.state.blend_set("NONE")

    # Node Count respects space overlay mode and drops scale footprint
    if getattr(settings, "show_node_count", True):
        show_overlays = getattr(space, "show_overlays", True)
        info_text = f"{len(nodes)} Nodes"
        font_id = 0

        font_size = int(12 * ui_scale) if show_overlays else int(9 * ui_scale)
        blf.size(font_id, font_size)
        text_w, _ = blf.dimensions(font_id, info_text)

        tx = mx + (mw - text_w) / 2
        ty = my + (8 * ui_scale if show_overlays else 4 * ui_scale)
        _draw_text_with_shadow(font_id, info_text, tx, ty, colors["text"], font_size)


def _draw_wires(nodes, tree_cx, tree_cy, scale, cx, cy, colors):
    wire_color = colors["wire"]
    thickness = max(1.5, 2.0 * scale)

    for node in nodes:
        if node.type == "FRAME" or not getattr(node, "outputs", None):
            continue

        w1, h1 = _get_node_dims(node)

        for output in node.outputs:
            if not getattr(output, "is_linked", False) or not getattr(output, "links", None):
                continue
            for link in output.links:
                to_node = link.to_node
                if to_node and to_node.name in nodes.keys() and to_node.type != "FRAME":
                    w2, h2 = _get_node_dims(to_node)

                    x1 = cx + (node.location_absolute.x + w1 / 2 - tree_cx) * scale
                    y1 = cy + (node.location_absolute.y - h1 / 2 - tree_cy) * scale
                    x2 = cx + (to_node.location_absolute.x + w2 / 2 - tree_cx) * scale
                    y2 = cy + (to_node.location_absolute.y - h2 / 2 - tree_cy) * scale

                    dx = x2 - x1
                    dy = y2 - y1
                    length = math.sqrt(dx * dx + dy * dy)
                    if length < 0.5:
                        continue

                    angle = math.atan2(dy, dx)
                    mx = (x1 + x2) / 2
                    my = (y1 + y2) / 2

                    gpu.matrix.push()
                    gpu.matrix.translate((mx, my, 0))
                    gpu.matrix.multiply_matrix(Matrix.Rotation(angle, 4, "Z"))
                    _draw_pill(-length / 2, -thickness / 2, length, thickness, wire_color)
                    gpu.matrix.pop()
