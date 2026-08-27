"""GPU batch building for minimap content."""

import logging
import math

import blf
from gpu_extras.batch import batch_for_shader

from .gpu_draw import (
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
_SCALE_REBUILD_REL = 0.002
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


def _ensure_minimap_batches(
    st: MinimapState,
    mx,
    my,
    mw,
    mh,
    cx,
    cy,
    scale,
    tree_cx,
    tree_cy,
    ui_scale,
    master_alpha,
    show_borders,
    highlight_border=None,
):
    """Bake content batches in map-local space, rebuilding only when stale.

    Vertex data is stored relative to ``tree_data["origin"]`` at the bake-time
    scale, so pan/drag frames only need the matrix transform applied by the
    caller (see draw_minimap). Rebuilds happen when tree positions change,
    the scale drifts past the bucket width (radius/thickness/font buckets),
    styling keys change, or the anchor drifts too far for culling to stay
    conservative. When *highlight_border* is an RGBA color, nodes whose type
    matches ``st.list.hovered_type_label`` (or whose name matches
    ``st.interaction.hovered_node``) get a separate outside outline instead
    of recolouring their own border.
    """
    tree_data = st.cache.tree_data
    if tree_data is None:
        return
    origin = tree_data.get("origin")
    if not origin:
        return

    key = (
        st.cache.position_version,
        round(ui_scale, 3),
        show_borders,
        st.list.hovered_type_label,
        st.interaction.hovered_node,
        bool(highlight_border),
    )
    sb = st.cache.batch_scale
    anchor_x, anchor_y = st.cache.batch_anchor
    # A settle bump changes only tree_data_version, so wire/marker freshness
    # must gate the early return too — otherwise wires stay frozen at their
    # pre-drag positions until an unrelated rebuild trigger fires.
    wire_key = (st.cache.tree_version, round(ui_scale, 3))
    wires_fresh = wire_key == st.cache.wire_key and st.cache.wire_scale == sb
    if (
        key == st.cache.batch_key
        and wires_fresh
        and sb > 0.0
        and abs(scale - sb) <= _SCALE_REBUILD_REL * max(sb, 1e-6)
        and abs(cx - anchor_x) <= _BATCH_DRIFT_PX
        and abs(cy - anchor_y) <= _BATCH_DRIFT_PX
    ):
        return

    ocx, ocy = origin
    # Sticky bake scale: adopt the live scale only when the drift budget is
    # exceeded, so fill and wire generations always share one bake scale
    # (and thus one content-matrix factor) between bucket crossings.
    prev_sb = st.cache.batch_scale
    if prev_sb > 0.0 and abs(scale - prev_sb) <= _SCALE_REBUILD_REL * max(prev_sb, 1e-6):
        sb = prev_sb
    else:
        sb = scale

    font_id = 0
    min_dim = 3.0 * ui_scale
    node_infos = tree_data["node_infos"]
    hovered_type = st.list.hovered_type_label
    hovered_node_name = st.interaction.hovered_node
    hl_outline = None
    if (hovered_type or hovered_node_name) and highlight_border is not None:
        hl_outline = _srgb_to_linear(_alpha_mul(highlight_border, master_alpha))
    hl_margin = 2.0 * ui_scale
    hl_line_w = 2.0

    # Cull window in baked space: the map interior plus slack for anchor
    # drift between rebuilds (nodes outside never reach the GPU batches).
    piv_bx = (tree_cx - ocx) * sb
    piv_by = (tree_cy - ocy) * sb
    cul_l = mx - _CULL_MARGIN_PX - cx + piv_bx
    cul_r = mx + mw + _CULL_MARGIN_PX - cx + piv_bx
    cul_b = my - _CULL_MARGIN_PX - cy + piv_by
    cul_t = my + mh + _CULL_MARGIN_PX - cy + piv_by

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

    hl_pos_border = []
    hl_uv_border = []
    hl_half_size_border = []
    hl_radius_border = []
    hl_color_border = []
    hl_line_width_border = []

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

    cached_text = []

    for info in node_infos:
        bw_raw = info["tree_w"] * sb
        bh_raw = info["tree_h"] * sb
        bx = (info["tree_x"] - ocx) * sb
        by = (info["tree_y"] - ocy) * sb
        bw = max(bw_raw, 1.0)
        bh = max(bh_raw, 1.0)
        is_frame = info["is_frame"]

        # Cull nodes whose quads cannot intersect the minimap interior
        if bx >= cul_r or bx + bw <= cul_l or by >= cul_t or by + bh <= cul_b:
            continue

        if is_frame:
            node_r = info["node_r_base"] * ui_scale * 1.6
        else:
            node_r = info["node_r_base"] * ui_scale * (sb * 2)

        is_tiny = (bw < min_dim or bh < min_dim) and not is_frame

        border_color = info["border_color"]
        border_w = info["border_w"]
        # The normal node border is left as-is (active/selection styling);
        # list hover instead gets a separate outside outline (see below).
        if hl_outline:
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
            bw_final = max(bw, min_dim)
            bh_final = max(bh, min_dim)
            hw = bw_final / 2
            hh = bh_final / 2
            all_pos_fill.extend(
                [
                    (bx, by, 0.0),
                    (bx + bw_final, by, 0.0),
                    (bx + bw_final, by + bh_final, 0.0),
                    (bx, by + bh_final, 0.0),
                ]
            )
            all_uv_fill.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
            all_half_size_fill.extend([(hw, hh)] * 4)
            all_radius_fill.extend([node_r] * 4)
            all_color_fill.extend([info["fill_color"]] * 4)

            if draw_border:
                all_pos_border.extend(
                    [
                        (bx, by, 0.0),
                        (bx + bw_final, by, 0.0),
                        (bx + bw_final, by + bh_final, 0.0),
                        (bx, by + bh_final, 0.0),
                    ]
                )
                all_uv_border.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
                all_half_size_border.extend([(hw, hh)] * 4)
                all_radius_border.extend([node_r] * 4)
                all_color_border.extend([border_color] * 4)
                all_line_width_border.extend([border_w] * 4)

            if is_hovered:
                hbw = bw_final + hl_margin * 2
                hbh = bh_final + hl_margin * 2
                hhw = hbw / 2
                hhh = hbh / 2
                hl_ox = bx - hl_margin
                hl_oy = by - hl_margin
                hl_pos_border.extend(
                    [
                        (hl_ox, hl_oy, 0.0),
                        (hl_ox + hbw, hl_oy, 0.0),
                        (hl_ox + hbw, hl_oy + hbh, 0.0),
                        (hl_ox, hl_oy + hbh, 0.0),
                    ]
                )
                hl_uv_border.extend([(-hhw, -hhh), (hhw, -hhh), (hhw, hhh), (-hhw, hhh)])
                hl_half_size_border.extend([(hhw, hhh)] * 4)
                hl_radius_border.extend([node_r] * 4)
                hl_color_border.extend([hl_outline] * 4)
                hl_line_width_border.extend([hl_line_w] * 4)
        else:
            hw = bw / 2
            hh = bh / 2

            pos_fill = frame_pos_fill if is_frame else all_pos_fill
            uv_fill = frame_uv_fill if is_frame else all_uv_fill
            hs_fill = frame_half_size_fill if is_frame else all_half_size_fill
            rad_fill = frame_radius_fill if is_frame else all_radius_fill
            col_fill = frame_color_fill if is_frame else all_color_fill
            pos_fill.extend(
                [
                    (bx, by, 0.0),
                    (bx + bw, by, 0.0),
                    (bx + bw, by + bh, 0.0),
                    (bx, by + bh, 0.0),
                ]
            )
            uv_fill.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
            hs_fill.extend([(hw, hh)] * 4)
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
                        (bx + bw, by, 0.0),
                        (bx + bw, by + bh, 0.0),
                        (bx, by + bh, 0.0),
                    ]
                )
                ub.extend([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
                hsb.extend([(hw, hh)] * 4)
                rb.extend([node_r] * 4)
                cb.extend([border_color] * 4)
                lwb.extend([border_w] * 4)

            if is_hovered:
                hbw = bw + hl_margin * 2
                hbh = bh + hl_margin * 2
                hhw = hbw / 2
                hhh = hbh / 2
                hl_ox = bx - hl_margin
                hl_oy = by - hl_margin
                hl_pos_border.extend(
                    [
                        (hl_ox, hl_oy, 0.0),
                        (hl_ox + hbw, hl_oy, 0.0),
                        (hl_ox + hbw, hl_oy + hbh, 0.0),
                        (hl_ox, hl_oy + hbh, 0.0),
                    ]
                )
                hl_uv_border.extend([(-hhw, -hhh), (hhw, -hhh), (hhw, hhh), (-hhw, hhh)])
                hl_half_size_border.extend([(hhw, hhh)] * 4)
                hl_radius_border.extend([node_r] * 4)
                hl_color_border.extend([hl_outline] * 4)
                hl_line_width_border.extend([hl_line_w] * 4)

            # Labels
            if is_frame:
                frame_lbl = info.get("frame_label")
                if frame_lbl:
                    text, text_color, bg_color_lbl = frame_lbl
                    label_font_size = max(6, min(11, int(11 * ui_scale * sb * 8)))
                    blf.size(font_id, label_font_size)
                    tw, th = blf.dimensions(font_id, text)
                    lx = bx + (bw - tw) / 2
                    ly = by + bh + 3 * ui_scale
                    label_pad = 2 * ui_scale

                    frame_pos_fill.extend(
                        [
                            (lx - label_pad, ly - label_pad, 0.0),
                            (lx + tw + label_pad, ly - label_pad, 0.0),
                            (lx + tw + label_pad, ly + th + label_pad, 0.0),
                            (lx - label_pad, ly + th + label_pad, 0.0),
                        ]
                    )
                    hw_lp = (tw + 2 * label_pad) / 2
                    hh_lp = (th + 2 * label_pad) / 2
                    frame_uv_fill.extend([(-hw_lp, -hh_lp), (hw_lp, -hh_lp), (hw_lp, hh_lp), (-hw_lp, hh_lp)])
                    frame_half_size_fill.extend([(hw_lp, hh_lp)] * 4)
                    frame_radius_fill.extend([node_r] * 4)
                    frame_color_fill.extend([bg_color_lbl] * 4)
                    cached_text.append((font_id, text, lx, ly, text_color, label_font_size))
            else:
                lbl_type = info.get("node_label_type")
                lbl_text = info.get("node_label_text")
                if lbl_type and lbl_text and bw > 6 * ui_scale and bh > 6 * ui_scale:
                    text_color = info["node_label_color"]
                    if lbl_type == "full":
                        font_size = max(6, min(int(11 * ui_scale), int(min(bw, bh) * 0.35)))
                        lines = _get_node_label_lines(lbl_text, font_id, font_size, bw - 4 * ui_scale, 3)
                        if lines:
                            blf.size(font_id, font_size)
                            line_h = blf.dimensions(font_id, "Ay")[1] + 1
                            asc_h = blf.dimensions(font_id, "A")[1]
                            vis_h = (len(lines) - 1) * line_h + asc_h
                            start_y = by + (bh - vis_h) / 2
                            for i, line in enumerate(lines):
                                lw, _ = blf.dimensions(font_id, line)
                                lx = bx + (bw - lw) / 2
                                ly = start_y + (len(lines) - 1 - i) * line_h
                                cached_text.append((font_id, line, lx, ly, text_color, font_size))
                    else:
                        font_size = max(6, min(int(11 * ui_scale), int(min(bw, bh) * 0.45)))
                        blf.size(font_id, font_size)
                        tw, th = blf.dimensions(font_id, lbl_text)
                        tx = bx + (bw - tw) / 2
                        ty = by + (bh - th) / 2
                        cached_text.append((font_id, lbl_text, tx, ty, text_color, font_size))

    # Compile GPU batches
    num_fills = len(all_pos_fill) // 4
    if num_fills > 0:
        shader = _get_batch_rect_shader()
        st.cache.backdrops_batch = batch_for_shader(
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
        st.cache.backdrops_batch = None

    num_borders = len(all_pos_border) // 4
    if num_borders > 0:
        shader = _get_batch_rect_border_shader()
        st.cache.borders_batch = batch_for_shader(
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
        st.cache.borders_batch = None

    num_hl_borders = len(hl_pos_border) // 4
    if num_hl_borders > 0:
        shader = _get_batch_rect_border_shader()
        st.cache.highlight_borders_batch = batch_for_shader(
            shader,
            "TRIS",
            {
                "pos": hl_pos_border,
                "uv": hl_uv_border,
                "halfSize": hl_half_size_border,
                "radius": hl_radius_border,
                "color": hl_color_border,
                "lineWidth": hl_line_width_border,
            },
            indices=_create_quad_indices(num_hl_borders),
        )
    else:
        st.cache.highlight_borders_batch = None

    num_frame_fills = len(frame_pos_fill) // 4
    if num_frame_fills > 0:
        shader = _get_batch_rect_shader()
        st.cache.frames_fill_batch = batch_for_shader(
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
        st.cache.frames_fill_batch = None

    num_frame_borders = len(frame_pos_border) // 4
    if num_frame_borders > 0:
        shader = _get_batch_rect_border_shader()
        st.cache.frames_border_batch = batch_for_shader(
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
        st.cache.frames_border_batch = None

    st.cache.text = cached_text

    # Sockets — unified batch with per-vertex color + auto-hide by zoom
    ph = max(1, tree_data["socket_ph_base"] * sb * ui_scale)
    pw = ph
    st.cache.socket_ph = ph
    if tree_data["socket_items"] and scale >= _MIN_SOCKET_SCALE:
        half_w = pw / 2
        half_h = ph / 2
        r = ph / 2
        socket_all_pos = []
        socket_all_uv = []
        socket_all_hs = []
        socket_all_r = []
        socket_all_c = []
        for color, positions in tree_data["socket_items"].items():
            linear_color = _srgb_to_linear(color)
            for sx_tree, sy_tree in positions:
                sxb = (sx_tree - ocx) * sb
                syb = (sy_tree - ocy) * sb
                _pad = 1.5
                socket_all_pos.extend(
                    [
                        (sxb - half_w - _pad, syb - half_h - _pad, 0.0),
                        (sxb + half_w + _pad, syb - half_h - _pad, 0.0),
                        (sxb + half_w + _pad, syb + half_h + _pad, 0.0),
                        (sxb - half_w - _pad, syb + half_h + _pad, 0.0),
                    ]
                )
                socket_all_uv.extend(
                    [
                        (-half_w - _pad, -half_h - _pad),
                        (half_w + _pad, -half_h - _pad),
                        (half_w + _pad, half_h + _pad),
                        (-half_w - _pad, half_h + _pad),
                    ]
                )
                socket_all_hs.extend([(half_w, half_h)] * 4)
                socket_all_r.extend([r] * 4)
                socket_all_c.extend([linear_color] * 4)
        num_s = len(socket_all_pos) // 4
        if num_s > 0:
            shader = _get_batch_rect_shader()
            st.cache.socket_batch = batch_for_shader(
                shader,
                "TRIS",
                {
                    "pos": socket_all_pos,
                    "uv": socket_all_uv,
                    "halfSize": socket_all_hs,
                    "radius": socket_all_r,
                    "color": socket_all_c,
                },
                indices=_create_quad_indices(num_s),
            )
        else:
            st.cache.socket_batch = None
    else:
        st.cache.socket_batch = None

    # Wires and markers get their own cache generation so position-only
    # refreshes (drags) skip the O(links) pill rebake entirely. Rebuilds
    # track the sticky bake scale exactly, keeping the shared matrix factor
    # consistent.
    if wire_key != st.cache.wire_key or st.cache.wire_scale != sb:
        _rebuild_wire_marker_batches(st, tree_data, ocx, ocy, sb, ui_scale, min_dim)
        st.cache.wire_key = wire_key
        st.cache.wire_scale = sb

    st.cache.batch_key = key
    st.cache.batch_scale = sb
    st.cache.batch_anchor = (cx, cy)


def _rebuild_wire_marker_batches(
    st: MinimapState,
    tree_data: dict,
    ocx: float,
    ocy: float,
    sb: float,
    ui_scale: float,
    min_dim: float,
) -> None:
    """Bake wire pill batches (per color + shadow underlay) and group markers.

    Called only when tree structure, UI scale, or the scale bucket changed —
    never on position-only drag refreshes.
    """
    # Wires — baked pill batches per color plus a merged thicker shadow underlay
    thickness = max(1.0, 2.0 * sb)
    wire_batches = []
    shadow_points = []
    for color, items in tree_data["wire_items"].items():
        group = []
        for out_x, out_y, in_x, in_y in items:
            x1 = (out_x - ocx) * sb
            y1 = (out_y - ocy) * sb
            x2 = (in_x - ocx) * sb
            y2 = (in_y - ocy) * sb
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length < 0.5:
                continue
            angle = math.atan2(dy, dx)
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            group.append((mid_x, mid_y, length, angle))
        if group:
            _shader, batch = _build_pill_batch(group, thickness)
            if batch is not None:
                wire_batches.append((color, batch))
                shadow_points.extend(group)
    shadow_batch = None
    if shadow_points:
        _shadow_shader, shadow_batch = _build_pill_batch(shadow_points, thickness * 2.5)
    st.cache.wire_batches = wire_batches
    st.cache.wire_shadow_batch = shadow_batch

    # Group node underline markers — baked like wires
    marker_batches = []
    group_markers = tree_data.get("group_markers")
    if group_markers:
        marker_offset = 3 * ui_scale
        marker_thick = max(1.0, 1.5 * ui_scale)
        for marker_color, items in group_markers.items():
            group = []
            for x_mid, y_bot, length in items:
                ln = length * sb
                if ln < min_dim:
                    continue
                mxb = (x_mid - ocx) * sb
                myb = (y_bot - ocy) * sb - marker_offset
                group.append((mxb, myb, ln, 0.0))
            if group:
                _mshader, mbatch = _build_pill_batch(group, marker_thick)
                if mbatch is not None:
                    marker_batches.append((marker_color, mbatch))
    st.cache.marker_batches = marker_batches
