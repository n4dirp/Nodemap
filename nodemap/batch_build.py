"""GPU batch building for minimap content."""

import logging
import math
from typing import Any

import blf
from gpu_extras.batch import batch_for_shader

from .gpu_draw import (
    _build_noodle_batch,
    _build_pill_batch,
    _get_batch_rect_border_shader,
    _get_batch_rect_shader,
)
from .helpers import _get_node_label_lines
from .state import MinimapState
from .theme import _alpha_mul, _srgb_to_linear

logger = logging.getLogger(__package__)

_MIN_SOCKET_SCALE = 0.15

# Rebuild cached batches when the map scale drifts this far from the baked
# scale (relative); only radius/thickness/font buckets depend on it.
# 0.015 keeps radius error under ~0.03 px at minimap scale while avoiding
# per-frame rebuilds during smooth zoom animations (frame_all / frame_view),
# where 0.002 caused a rebuild every ~0.2% zoom change.
_SCALE_REBUILD_REL = 0.015
# Force a batch rebuild when the per-frame anchor drifts this far from the
# bake-time anchor; bounds how stale rect culling may become (px).
_BATCH_DRIFT_PX = 256.0
_CULL_MARGIN_PX = _BATCH_DRIFT_PX + 32.0


def _create_quad_indices(n: int) -> list[tuple[int, int, int]]:
    """Helper to populate triangular indices sequentially for quad batches."""
    indices = []
    for i in range(n):
        base = i * 4
        indices.append((base, base + 1, base + 2))
        indices.append((base + 2, base + 3, base))
    return indices


def _compute_frame_depths(node_infos: list[dict]) -> dict[int, int]:
    """Return ``{frame ptr: nesting depth}`` for every frame node.

    Depth 0 is the outermost level; a frame's depth counts how many other
    frames strictly contain its rectangle (edge-inclusive). Frames are few
    enough that the quadratic containment scan is trivial. Computed from live
    node rects here (not tree_compile) so position-only drag patches never
    leave stale nesting levels.
    """
    frames = [
        (
            info["ptr"],
            info["tree_x"],
            info["tree_y"],
            info["tree_w"],
            info["tree_h"],
        )
        for info in node_infos
        if info["is_frame"]
    ]
    depths: dict[int, int] = {}
    for ptr, x0, y0, w0, h0 in frames:
        depth = 0
        for cptr, cx0, cy0, cw0, ch0 in frames:
            if cptr == ptr or (cw0, ch0) == (w0, h0):
                continue
            if cx0 <= x0 and cy0 <= y0 and cx0 + cw0 >= x0 + w0 and cy0 + ch0 >= y0 + h0:
                depth += 1
        depths[ptr] = depth
    return depths


def _resolve_frame_label_layout(entries: list[dict], ui_scale: float) -> list[dict]:
    """Slide colliding frame labels apart, dropping labels that cannot fit.

    Outer frames (lower nesting depth, then higher on screen) place first; a
    label overlapping an already placed one is pushed down just below the
    blocker with a one-pixel clearance gap. A label that would need more than
    half its frame's baked height of travel to clear its blockers is dropped
    so deep-nesting labels yield to the outer frames instead of drifting away.
    Returns only the entries that keep collision-free baked-space rects, which
    the uniform content matrix preserves at any zoom.
    """
    if not entries:
        return []
    gap = 1.0 * ui_scale
    half_gap = gap * 0.5
    placed: list[tuple[float, float, float, float]] = []
    resolved: list[dict] = []
    # A label never needs more passes than the number of distinct blocker
    # bottoms it can collide with; the cap guards against degenerate layouts
    # (near-coincident frame edges) where sub-pixel shifts would otherwise
    # keep the loop alive for millions of iterations and hang the draw handler.
    max_passes = max(1, len(entries) * 2)
    for entry in sorted(entries, key=lambda e: (e["depth"], -e["rect"][1])):
        x, y, w, h = entry["rect"]
        budget = entry["frame_h"] * 0.5
        total_shift = 0.0
        dropped = False
        passes = 0
        while passes < max_passes:
            passes += 1
            lowest_bottom = None
            for px, py, pw, ph in placed:
                if x + w + half_gap <= px - half_gap:
                    continue
                if px + pw + half_gap <= x - half_gap:
                    continue
                if y + h + half_gap <= py - half_gap:
                    continue
                if py + ph + half_gap <= y - half_gap:
                    continue
                if lowest_bottom is None or py < lowest_bottom:
                    lowest_bottom = py
            if lowest_bottom is None:
                break
            new_y = lowest_bottom - h - gap
            shift = y - new_y
            # Drop when a pass makes no real progress: the label would sit
            # exactly at a blocker bottom it can no longer clear.
            if shift <= 1e-6:
                dropped = True
                break
            if total_shift + shift > budget:
                dropped = True
                break
            y = new_y
            total_shift += shift
        if passes >= max_passes:
            dropped = True
        if dropped:
            continue
        entry["rect"] = (x, y, w, h)
        placed.append(entry["rect"])
        resolved.append(entry)
    return resolved


def _ensure_minimap_batches(
    minimap_state: MinimapState,
    map_x,
    map_y,
    map_w,
    map_h,
    map_anchor_x,
    map_anchor_y,
    scale,
    tree_center_x,
    tree_center_y,
    ui_scale,
    master_alpha,
    show_borders,
    highlight_border=None,
    wire_curvature: int = 5,
    wire_thickness: float = 1.0,
):
    """Bake content batches in map-local space, rebuilding only when stale.

    Vertex data is stored relative to ``tree_data["origin"]`` at the bake-time
    scale, so pan/drag frames only need the matrix transform applied by the
    caller (see draw_minimap). Rebuilds happen when tree positions change,
    the scale drifts past the bucket width (radius/thickness/font buckets),
    styling keys change, or the anchor drifts too far for culling to stay
    conservative. When *highlight_border* is an RGBA color, nodes whose type
    matches ``minimap_state.list.hovered_type_label`` (or whose name matches
    ``minimap_state.interaction.hovered_node_id``) get a separate outside
    outline instead of recolouring their own border.
    """
    tree_data = minimap_state.cache.tree_data
    if tree_data is None:
        return
    origin = tree_data.get("origin")
    if not origin:
        return

    key = (
        minimap_state.cache.position_version,
        round(ui_scale, 3),
        show_borders,
        minimap_state.list.hovered_type_label,
        minimap_state.interaction.hovered_node_id,
        bool(highlight_border),
    )
    bake_scale = minimap_state.cache.batch_scale
    anchor_x, anchor_y = minimap_state.cache.batch_anchor
    # A settle bump changes only tree_data_version, so wire/marker freshness
    # must gate the early return too — otherwise wires stay frozen at their
    # pre-drag positions until an unrelated rebuild trigger fires.
    wire_key = (minimap_state.cache.tree_version, round(ui_scale, 3), int(wire_curvature), round(wire_thickness, 3))
    wires_fresh = wire_key == minimap_state.cache.wire_key and minimap_state.cache.wire_scale == bake_scale
    if (
        key == minimap_state.cache.batch_key
        and wires_fresh
        and bake_scale > 0.0
        and abs(scale - bake_scale) <= _SCALE_REBUILD_REL * max(bake_scale, 1e-6)
        and abs(map_anchor_x - anchor_x) <= _BATCH_DRIFT_PX
        and abs(map_anchor_y - anchor_y) <= _BATCH_DRIFT_PX
    ):
        return

    origin_x, origin_y = origin
    # Sticky bake scale: adopt the live scale only when the drift budget is
    # exceeded, so fill and wire generations always share one bake scale
    # (and thus one content-matrix factor) between bucket crossings.
    prev_bake_scale = minimap_state.cache.batch_scale
    if prev_bake_scale > 0.0 and abs(scale - prev_bake_scale) <= _SCALE_REBUILD_REL * max(prev_bake_scale, 1e-6):
        bake_scale = prev_bake_scale
    else:
        bake_scale = scale

    font_id = 0
    min_dim = 3.0 * ui_scale
    node_infos = tree_data["node_infos"]
    frame_depths = _compute_frame_depths(node_infos)
    frame_label_entries: list[dict] = []
    hovered_type = minimap_state.list.hovered_type_label
    hovered_node_name = minimap_state.interaction.hovered_node_id
    highlight_outline = None
    if (hovered_type or hovered_node_name) and highlight_border is not None:
        highlight_outline = _srgb_to_linear(_alpha_mul(highlight_border, master_alpha))
    highlight_margin = 2.0 * ui_scale
    highlight_line_width = 2.0

    # Cull window in baked space: the map interior plus slack for anchor
    # drift between rebuilds (nodes outside never reach the GPU batches).
    pivot_baked_x = (tree_center_x - origin_x) * bake_scale
    pivot_baked_y = (tree_center_y - origin_y) * bake_scale
    cull_left = map_x - _CULL_MARGIN_PX - map_anchor_x + pivot_baked_x
    cull_right = map_x + map_w + _CULL_MARGIN_PX - map_anchor_x + pivot_baked_x
    cull_bottom = map_y - _CULL_MARGIN_PX - map_anchor_y + pivot_baked_y
    cull_top = map_y + map_h + _CULL_MARGIN_PX - map_anchor_y + pivot_baked_y

    all_pos_fill = []
    all_uv_fill = []
    all_half_size_fill = []
    all_radius_fill = []
    all_color_fill = []

    all_pos_border = []
    all_uv_border = []
    all_half_size_border = []
    all_radius_border = []
    all_color_border = []
    all_line_width_border = []

    highlight_pos_border = []
    highlight_uv_border = []
    highlight_half_size_border = []
    highlight_radius_border = []
    highlight_color_border = []
    highlight_line_width_border = []

    frame_pos_fill = []
    frame_uv_fill = []
    frame_half_size_fill = []
    frame_radius_fill = []
    frame_color_fill = []

    frame_pos_border = []
    frame_uv_border = []
    frame_half_size_border = []
    frame_radius_border = []
    frame_color_border = []
    frame_line_width_border = []

    node_labels = []

    for info in node_infos:
        node_w_raw = info["tree_w"] * bake_scale
        node_h_raw = info["tree_h"] * bake_scale
        bx = (info["tree_x"] - origin_x) * bake_scale
        by = (info["tree_y"] - origin_y) * bake_scale
        node_w = max(node_w_raw, 1.0)
        node_h = max(node_h_raw, 1.0)
        is_frame = info["is_frame"]

        # Cull nodes whose quads cannot intersect the minimap interior
        if bx >= cull_right or bx + node_w <= cull_left or by >= cull_top or by + node_h <= cull_bottom:
            continue

        if is_frame:
            node_r = info["node_r_base"] * ui_scale * 1.6
        else:
            node_r = info["node_r_base"] * ui_scale * (bake_scale * 2)

        is_tiny = (node_w < min_dim or node_h < min_dim) and not is_frame

        border_color = info["border_color"]
        border_w = info["border_w"]
        # The normal node border is left as-is (active/selection styling);
        # list hover instead gets a separate outside outline (see below).
        if highlight_outline:
            if hovered_type is not None and info.get("type_label") == hovered_type:
                is_hovered = True
            elif hovered_node_name is not None and info.get("name") == hovered_node_name:
                is_hovered = True
            else:
                is_hovered = False
        else:
            is_hovered = False

        # Borders always emit vertices regardless of on-screen size so they
        # stay visible at any zoom (hover and normal alike); the SDF shader
        # clamps the line width for tiny nodes.
        draw_border = show_borders

        if is_tiny:
            node_w_final = max(node_w, min_dim)
            node_h_final = max(node_h, min_dim)
            half_w = node_w_final / 2
            half_h = node_h_final / 2
            all_pos_fill.extend(
                [
                    (bx, by, 0.0),
                    (bx + node_w_final, by, 0.0),
                    (bx + node_w_final, by + node_h_final, 0.0),
                    (bx, by + node_h_final, 0.0),
                ]
            )
            all_uv_fill.extend([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
            all_half_size_fill.extend([(half_w, half_h)] * 4)
            all_radius_fill.extend([node_r] * 4)
            all_color_fill.extend([info["fill_color"]] * 4)

            if draw_border:
                all_pos_border.extend(
                    [
                        (bx, by, 0.0),
                        (bx + node_w_final, by, 0.0),
                        (bx + node_w_final, by + node_h_final, 0.0),
                        (bx, by + node_h_final, 0.0),
                    ]
                )
                all_uv_border.extend([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
                all_half_size_border.extend([(half_w, half_h)] * 4)
                all_radius_border.extend([node_r] * 4)
                all_color_border.extend([border_color] * 4)
                all_line_width_border.extend([border_w] * 4)

            if is_hovered:
                outline_w = node_w_final + highlight_margin * 2
                outline_h = node_h_final + highlight_margin * 2
                outline_half_w = outline_w / 2
                outline_half_h = outline_h / 2
                outline_x = bx - highlight_margin
                outline_y = by - highlight_margin
                highlight_pos_border.extend(
                    [
                        (outline_x, outline_y, 0.0),
                        (outline_x + outline_w, outline_y, 0.0),
                        (outline_x + outline_w, outline_y + outline_h, 0.0),
                        (outline_x, outline_y + outline_h, 0.0),
                    ]
                )
                highlight_uv_border.extend(
                    [
                        (-outline_half_w, -outline_half_h),
                        (outline_half_w, -outline_half_h),
                        (outline_half_w, outline_half_h),
                        (-outline_half_w, outline_half_h),
                    ]
                )
                highlight_half_size_border.extend([(outline_half_w, outline_half_h)] * 4)
                highlight_radius_border.extend([node_r] * 4)
                highlight_color_border.extend([highlight_outline] * 4)
                highlight_line_width_border.extend([highlight_line_width] * 4)
        else:
            half_w = node_w / 2
            half_h = node_h / 2

            pos_fill = frame_pos_fill if is_frame else all_pos_fill
            uv_fill = frame_uv_fill if is_frame else all_uv_fill
            hs_fill = frame_half_size_fill if is_frame else all_half_size_fill
            rad_fill = frame_radius_fill if is_frame else all_radius_fill
            col_fill = frame_color_fill if is_frame else all_color_fill
            pos_fill.extend(
                [
                    (bx, by, 0.0),
                    (bx + node_w, by, 0.0),
                    (bx + node_w, by + node_h, 0.0),
                    (bx, by + node_h, 0.0),
                ]
            )
            uv_fill.extend([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
            hs_fill.extend([(half_w, half_h)] * 4)
            rad_fill.extend([node_r] * 4)
            col_fill.extend([info["fill_color"]] * 4)

            if draw_border:
                pb = frame_pos_border if is_frame else all_pos_border
                ub = frame_uv_border if is_frame else all_uv_border
                hsb = frame_half_size_border if is_frame else all_half_size_border
                rb = frame_radius_border if is_frame else all_radius_border
                cb = frame_color_border if is_frame else all_color_border
                lwb = frame_line_width_border if is_frame else all_line_width_border
                pb.extend(
                    [
                        (bx, by, 0.0),
                        (bx + node_w, by, 0.0),
                        (bx + node_w, by + node_h, 0.0),
                        (bx, by + node_h, 0.0),
                    ]
                )
                ub.extend([(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)])
                hsb.extend([(half_w, half_h)] * 4)
                rb.extend([node_r] * 4)
                cb.extend([border_color] * 4)
                lwb.extend([border_w] * 4)

            if is_hovered:
                outline_w = node_w + highlight_margin * 2
                outline_h = node_h + highlight_margin * 2
                outline_half_w = outline_w / 2
                outline_half_h = outline_h / 2
                outline_x = bx - highlight_margin
                outline_y = by - highlight_margin
                highlight_pos_border.extend(
                    [
                        (outline_x, outline_y, 0.0),
                        (outline_x + outline_w, outline_y, 0.0),
                        (outline_x + outline_w, outline_y + outline_h, 0.0),
                        (outline_x, outline_y + outline_h, 0.0),
                    ]
                )
                highlight_uv_border.extend(
                    [
                        (-outline_half_w, -outline_half_h),
                        (outline_half_w, -outline_half_h),
                        (outline_half_w, outline_half_h),
                        (-outline_half_w, outline_half_h),
                    ]
                )
                highlight_half_size_border.extend([(outline_half_w, outline_half_h)] * 4)
                highlight_radius_border.extend([node_r] * 4)
                highlight_color_border.extend([highlight_outline] * 4)
                highlight_line_width_border.extend([highlight_line_width] * 4)

            if is_frame:
                # Zoom gate evaluated live per bake (user_zoom drives scale),
                # so labels appear/disappear at the threshold without waiting
                # for a tree recompile.
                if not minimap_state.view.user_zoom >= 0.8:
                    continue
                frame_label = info.get("frame_label")
                if frame_label:
                    text, text_color, bg_label_color = frame_label
                    label_font_size = max(6, min(11, int(11 * ui_scale * bake_scale * 8)))
                    blf.size(font_id, label_font_size)
                    text_w, text_h = blf.dimensions(font_id, text)
                    label_pad = 2 * ui_scale
                    label_w = text_w + 2 * label_pad
                    label_h = text_h + 2 * label_pad
                    frame_label_entries.append(
                        {
                            "text": text,
                            "text_color": text_color,
                            "bg_label_color": bg_label_color,
                            "font_size": label_font_size,
                            "text_w": text_w,
                            "text_h": text_h,
                            "pad": label_pad,
                            "node_r": node_r,
                            "frame_h": node_h,
                            "depth": frame_depths.get(info["ptr"], 0),
                            "rect": (
                                bx + (node_w - label_w) / 2,
                                by + node_h + 3 * ui_scale - label_pad,
                                label_w,
                                label_h,
                            ),
                        }
                    )
            else:
                label_type = info.get("node_label_type")
                label_text = info.get("node_label_text")
                if label_type and label_text and node_w > 6 * ui_scale and node_h > 6 * ui_scale:
                    text_color = info["node_label_color"]
                    if label_type == "full":
                        font_size = max(6, min(int(11 * ui_scale), int(min(node_w, node_h) * 0.35)))
                        lines = _get_node_label_lines(label_text, font_id, font_size, node_w - 4 * ui_scale, 3)
                        if lines:
                            blf.size(font_id, font_size)
                            line_h = blf.dimensions(font_id, "Ay")[1] + 1
                            ascent_h = blf.dimensions(font_id, "A")[1]
                            text_block_h = (len(lines) - 1) * line_h + ascent_h
                            start_y = by + (node_h - text_block_h) / 2
                            for i, line in enumerate(lines):
                                line_w, _ = blf.dimensions(font_id, line)
                                label_x = bx + (node_w - line_w) / 2
                                label_y = start_y + (len(lines) - 1 - i) * line_h
                                node_labels.append((font_id, line, label_x, label_y, text_color, font_size))
                    else:
                        font_size = max(6, min(int(11 * ui_scale), int(min(node_w, node_h) * 0.45)))
                        blf.size(font_id, font_size)
                        text_w, text_h = blf.dimensions(font_id, label_text)
                        text_x = bx + (node_w - text_w) / 2
                        text_y = by + (node_h - text_h) / 2
                        node_labels.append((font_id, label_text, text_x, text_y, text_color, font_size))

    # Frame labels: resolve collisions in baked space (outer frames win,
    # pushed-down labels keep a clearance gap), then emit backdrop quads into
    # the frames fill batch and text entries for the manual BLF pass.
    for entry in _resolve_frame_label_layout(frame_label_entries, ui_scale):
        rect_x, rect_y, label_w, label_h = entry["rect"]
        label_pad = entry["pad"]
        frame_pos_fill.extend(
            [
                (rect_x, rect_y, 0.0),
                (rect_x + label_w, rect_y, 0.0),
                (rect_x + label_w, rect_y + label_h, 0.0),
                (rect_x, rect_y + label_h, 0.0),
            ]
        )
        label_half_w = label_w / 2
        label_half_h = label_h / 2
        frame_uv_fill.extend(
            [
                (-label_half_w, -label_half_h),
                (label_half_w, -label_half_h),
                (label_half_w, label_half_h),
                (-label_half_w, label_half_h),
            ]
        )
        frame_half_size_fill.extend([(label_half_w, label_half_h)] * 4)
        frame_radius_fill.extend([entry["node_r"]] * 4)
        frame_color_fill.extend([entry["bg_label_color"]] * 4)
        node_labels.append(
            (font_id, entry["text"], rect_x + label_pad, rect_y + label_pad, entry["text_color"], entry["font_size"])
        )

    num_fills = len(all_pos_fill) // 4
    if num_fills > 0:
        shader = _get_batch_rect_shader()
        minimap_state.cache.backdrops_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": all_pos_fill,
                "uv": all_uv_fill,
                "halfSize": all_half_size_fill,
                "radius": all_radius_fill,
                "color": all_color_fill,
            },
            indices=_create_quad_indices(num_fills),
        )
    else:
        minimap_state.cache.backdrops_batch = None

    num_borders = len(all_pos_border) // 4
    if num_borders > 0:
        shader = _get_batch_rect_border_shader()
        minimap_state.cache.borders_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": all_pos_border,
                "uv": all_uv_border,
                "halfSize": all_half_size_border,
                "radius": all_radius_border,
                "color": all_color_border,
                "lineWidth": all_line_width_border,
            },
            indices=_create_quad_indices(num_borders),
        )
    else:
        minimap_state.cache.borders_batch = None

    num_highlight_borders = len(highlight_pos_border) // 4
    if num_highlight_borders > 0:
        shader = _get_batch_rect_border_shader()
        minimap_state.cache.highlight_borders_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": highlight_pos_border,
                "uv": highlight_uv_border,
                "halfSize": highlight_half_size_border,
                "radius": highlight_radius_border,
                "color": highlight_color_border,
                "lineWidth": highlight_line_width_border,
            },
            indices=_create_quad_indices(num_highlight_borders),
        )
    else:
        minimap_state.cache.highlight_borders_batch = None

    num_frame_fills = len(frame_pos_fill) // 4
    if num_frame_fills > 0:
        shader = _get_batch_rect_shader()
        minimap_state.cache.frames_fill_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": frame_pos_fill,
                "uv": frame_uv_fill,
                "halfSize": frame_half_size_fill,
                "radius": frame_radius_fill,
                "color": frame_color_fill,
            },
            indices=_create_quad_indices(num_frame_fills),
        )
    else:
        minimap_state.cache.frames_fill_batch = None

    num_frame_borders = len(frame_pos_border) // 4
    if num_frame_borders > 0:
        shader = _get_batch_rect_border_shader()
        minimap_state.cache.frames_border_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": frame_pos_border,
                "uv": frame_uv_border,
                "halfSize": frame_half_size_border,
                "radius": frame_radius_border,
                "color": frame_color_border,
                "lineWidth": frame_line_width_border,
            },
            indices=_create_quad_indices(num_frame_borders),
        )
    else:
        minimap_state.cache.frames_border_batch = None

    minimap_state.cache.node_labels = node_labels

    # Sockets — auto-hidden below the minimum scale
    pill_h = max(1, tree_data["socket_ph_base"] * bake_scale * ui_scale)
    pill_w = pill_h
    minimap_state.cache.socket_ph = pill_h
    if tree_data["socket_items"] and scale >= _MIN_SOCKET_SCALE:
        half_w = pill_w / 2
        half_h = pill_h / 2
        pill_radius = pill_h / 2
        socket_pos = []
        socket_uv = []
        socket_half_size = []
        socket_radius = []
        socket_color = []
        for color, positions in tree_data["socket_items"].items():
            linear_color = _srgb_to_linear(color)
            for sx_tree, sy_tree in positions:
                socket_baked_x = (sx_tree - origin_x) * bake_scale
                socket_baked_y = (sy_tree - origin_y) * bake_scale
                socket_pad = 1.5
                socket_pos.extend(
                    [
                        (socket_baked_x - half_w - socket_pad, socket_baked_y - half_h - socket_pad, 0.0),
                        (socket_baked_x + half_w + socket_pad, socket_baked_y - half_h - socket_pad, 0.0),
                        (socket_baked_x + half_w + socket_pad, socket_baked_y + half_h + socket_pad, 0.0),
                        (socket_baked_x - half_w - socket_pad, socket_baked_y + half_h + socket_pad, 0.0),
                    ]
                )
                socket_uv.extend(
                    [
                        (-half_w - socket_pad, -half_h - socket_pad),
                        (half_w + socket_pad, -half_h - socket_pad),
                        (half_w + socket_pad, half_h + socket_pad),
                        (-half_w - socket_pad, half_h + socket_pad),
                    ]
                )
                socket_half_size.extend([(half_w, half_h)] * 4)
                socket_radius.extend([pill_radius] * 4)
                socket_color.extend([linear_color] * 4)
        num_sockets = len(socket_pos) // 4
        if num_sockets > 0:
            shader = _get_batch_rect_shader()
            minimap_state.cache.socket_batch = batch_for_shader(
                shader,
                "TRIS",
                {
                    "pos": socket_pos,
                    "uv": socket_uv,
                    "halfSize": socket_half_size,
                    "radius": socket_radius,
                    "color": socket_color,
                },
                indices=_create_quad_indices(num_sockets),
            )
        else:
            minimap_state.cache.socket_batch = None
    else:
        minimap_state.cache.socket_batch = None

    # Wires and markers get their own cache generation so position-only
    # refreshes (drags) skip the O(links) pill rebake entirely. Wires share
    # the same bake_scale as rects so the single content matrix scales both
    # in lockstep; no separate wire tolerance (a larger tolerance would
    # desync wire/node scale between rebuilds).
    if wire_key != minimap_state.cache.wire_key or minimap_state.cache.wire_scale != bake_scale:
        _rebuild_wire_marker_batches(
            minimap_state,
            tree_data,
            origin_x,
            origin_y,
            bake_scale,
            ui_scale,
            min_dim,
            int(wire_curvature),
            wire_thickness,
        )
        minimap_state.cache.wire_key = wire_key
        minimap_state.cache.wire_scale = bake_scale

    minimap_state.cache.batch_key = key
    minimap_state.cache.batch_scale = bake_scale
    minimap_state.cache.batch_anchor = (map_anchor_x, map_anchor_y)


def _rebuild_wire_marker_batches(
    minimap_state: MinimapState,
    tree_data: dict,
    origin_x: float,
    origin_y: float,
    bake_scale: float,
    ui_scale: float,
    min_dim: float,
    wire_curvature: int = 5,
    wire_thickness: float = 1.0,
) -> None:
    """Bake wire batches (noodle strips when curved, pills otherwise) and group markers.

    Called only when tree structure, UI scale, or the scale bucket changed —
    never on position-only drag refreshes.
    """

    # Wires — cubic noodle strips when curvature is on, straight pills at 0.
    # The user wire_thickness multiplier scales the baked thickness.
    thickness = max(1.6, 2.0 * bake_scale) * wire_thickness
    curvature = float(wire_curvature)
    use_curve = curvature > 1e-6
    wire_batches: list[tuple[Any, Any, float]] = []
    per_color_wires: dict[Any, list[tuple[float, float, float, float, float, float, float, float]]] = {}

    for color, items in tree_data["wire_items"].items():
        group_controls: list[tuple[float, float, float, float, float, float, float, float]] = []
        group_pills: list[tuple[float, float, float, float]] = []
        for out_x, out_y, in_x, in_y in items:
            x1 = (out_x - origin_x) * bake_scale
            y1 = (out_y - origin_y) * bake_scale
            x2 = (in_x - origin_x) * bake_scale
            y2 = (in_y - origin_y) * bake_scale
            dx = x2 - x1
            dy = y2 - y1
            if math.hypot(dx, dy) < 0.5:
                continue
            if use_curve:
                # Cubic Bezier matching Blender's node link type (ease-out /
                # ease-in): horizontal handles whose length scales with the
                # horizontal span only, exactly like
                # ``dist = curving * 0.10 * |x3 - x0|`` in
                # ``calculate_inner_link_bezier_points``.
                dist_h = curvature * 0.10 * abs(dx)
                p0x, p0y = x1, y1
                p1x, p1y = x1 + dist_h, y1
                p2x, p2y = x2 - dist_h, y2
                p3x, p3y = x2, y2
                group_controls.append((p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y))
            else:
                length = math.hypot(dx, dy)
                angle = math.atan2(dy, dx)
                mid_x = (x1 + x2) * 0.5
                mid_y = (y1 + y2) * 0.5
                group_pills.append((mid_x, mid_y, length, angle))

        if use_curve and group_controls:
            per_color_wires[color] = group_controls
        elif group_pills:
            _shader, batch = _build_pill_batch(group_pills, thickness)
            if batch is not None:
                wire_batches.append((color, batch, thickness * 0.5))

    if use_curve:
        for color, controls in per_color_wires.items():
            half_thickness = thickness * 0.5
            _shader, batch = _build_noodle_batch(controls, half_thickness)
            if batch is not None:
                wire_batches.append((color, batch, half_thickness))
        minimap_state.cache.wire_batches = wire_batches
        minimap_state.cache.wire_shadow_batch = None
    else:
        minimap_state.cache.wire_batches = wire_batches
        minimap_state.cache.wire_shadow_batch = None

    # Group node underline markers — baked like wires
    marker_batches = []
    group_markers = tree_data.get("group_markers")
    if group_markers:
        marker_offset = 10 * bake_scale
        marker_thickness = max(2.0, 2.0 * ui_scale)
        for marker_color, items in group_markers.items():
            group = []
            for x_mid, y_bot, length in items:
                marker_len = length * bake_scale
                if marker_len < min_dim:
                    continue
                marker_baked_x = (x_mid - origin_x) * bake_scale
                marker_baked_y = (y_bot - origin_y) * bake_scale - marker_offset
                group.append((marker_baked_x, marker_baked_y, marker_len, 0.0))
            if group:
                _marker_shader, marker_batch = _build_pill_batch(group, marker_thickness)
                if marker_batch is not None:
                    marker_batches.append((marker_color, marker_batch))
    minimap_state.cache.marker_batches = marker_batches
