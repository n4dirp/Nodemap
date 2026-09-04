"""Provide GPU batch building for minimap content."""

import logging
import math
from dataclasses import dataclass
from typing import Any

import blf
from gpu_extras.batch import batch_for_shader

from .. import __package__ as base_package
from ..core.constants import (
    BATCH_DRIFT_PX,
    CULL_MARGIN_PX,
    MIN_SOCKET_SCALE,
    SCALE_REBUILD_REL,
)
from ..core.helpers import _get_node_label_lines
from ..core.list_filter import filter_matching_nodes
from ..core.state import MinimapState
from ..core.theme import _alpha_mul, _srgb_to_linear
from .gpu_draw import (
    _build_noodle_batch,
    _build_pill_batch,
    _get_batch_rect_border_shader,
    _get_batch_rect_shader,
)
from .tree_compile import _resolve_wire_items

logger = logging.getLogger(base_package)


def _create_quad_indices(n: int) -> list[tuple[int, int, int]]:
    """Return triangular indices for quad batches."""
    indices = []
    for i in range(n):
        base = i * 4
        indices.append((base, base + 1, base + 2))
        indices.append((base + 2, base + 3, base))
    return indices


def _emit_quad(
    dst: dict,
    left: float,
    top: float,
    right: float,
    bottom: float,
    uv_w: float,
    uv_h: float,
    radius: float,
    color,
    half_size: tuple[float, float] | None = None,
    line_width: float | None = None,
) -> None:
    """Append one rounded-rect quad to a layer's scratch attributes.

    ``left``/``right`` are the exact pos corner values so float results stay
    identical to the inline vertex code. Sockets and reroutes pass a smaller
    ``half_size`` than their panned uv span; everything else derives it from the
    uv span.
    """
    dst["pos"].extend([(left, top, 0.0), (right, top, 0.0), (right, bottom, 0.0), (left, bottom, 0.0)])
    dst["uv"].extend([(-uv_w, -uv_h), (uv_w, -uv_h), (uv_w, uv_h), (-uv_w, uv_h)])
    hs = half_size if half_size is not None else (uv_w, uv_h)
    dst["half_size"].extend([hs] * 4)
    dst["radius"].extend([radius] * 4)
    dst["color"].extend([color] * 4)
    line_widths = dst.get("line_width")
    if line_widths is not None:
        line_widths.extend([line_width] * 4)


def _compute_frame_depths(node_infos: list[dict]) -> dict[int, int]:
    """Return ``{frame ptr: nesting depth}`` for every frame node."""
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
    """Return frame labels with collisions resolved and unplaceable labels dropped."""
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


@dataclass
class _BakeContext:
    """Bake-time shared parameters and per-layer scratch attribute buffers."""

    state: MinimapState
    tree_data: dict
    node_infos: list
    frame_depths: dict
    scale: float
    bake_scale: float
    origin_x: float
    origin_y: float
    ui_scale: float
    font_id: int
    min_dim: float
    cull_left: float
    cull_right: float
    cull_bottom: float
    cull_top: float
    show_borders: bool
    hovered_type: str | None
    hovered_node_name: str | None
    highlight_outline: tuple | None
    highlight_margin: float
    highlight_line_width: float
    filter_names: frozenset[str] | None
    frame_label_entries: list
    node_labels: list
    fill_attr: dict
    frame_fill_attr: dict
    border_attr: dict
    frame_border_attr: dict
    highlight_attr: dict


def _emit_node(ctx: "_BakeContext", info: dict) -> None:
    """Emit one node's fill/border/highlight quads plus its label scratch entries."""
    node_w_raw = info["tree_w"] * ctx.bake_scale
    node_h_raw = info["tree_h"] * ctx.bake_scale
    bx = (info["tree_x"] - ctx.origin_x) * ctx.bake_scale
    by = (info["tree_y"] - ctx.origin_y) * ctx.bake_scale
    node_w = max(node_w_raw, 1.0)
    node_h = max(node_h_raw, 1.0)
    is_frame = info["is_frame"]

    # Cull nodes whose quads cannot intersect the minimap interior
    if bx >= ctx.cull_right or bx + node_w <= ctx.cull_left or by >= ctx.cull_top or by + node_h <= ctx.cull_bottom:
        return

    # Under an active search filter, skip non-matching nodes entirely (body,
    # border, and label together) so the map mirrors the filtered list.
    if ctx.filter_names is not None and info.get("name") not in ctx.filter_names:
        return

    if is_frame:
        node_r = info["node_r_base"] * ctx.ui_scale * 1.6
    else:
        node_r = info["node_r_base"] * ctx.ui_scale * (ctx.bake_scale * 2)

    is_tiny = (node_w < ctx.min_dim or node_h < ctx.min_dim) and not is_frame

    border_color = info["border_color"]
    border_w = info["border_w"]
    # The normal node border is left as-is (active/selection styling);
    # list hover instead gets a separate outside outline (see below).
    if ctx.highlight_outline:
        if ctx.hovered_type is not None and info.get("type_label") == ctx.hovered_type:
            is_hovered = True
        elif ctx.hovered_node_name is not None and info.get("name") == ctx.hovered_node_name:
            is_hovered = True
        else:
            is_hovered = False
    else:
        is_hovered = False

    # Borders always emit vertices regardless of on-screen size so they
    # stay visible at any zoom (hover and normal alike); the SDF shader
    # clamps the line width for tiny nodes.
    draw_border = ctx.show_borders

    if is_tiny:
        node_w_final = max(node_w, ctx.min_dim)
        node_h_final = max(node_h, ctx.min_dim)
        half_w = node_w_final / 2
        half_h = node_h_final / 2
        right = bx + node_w_final
        bottom = by + node_h_final
        _emit_quad(ctx.fill_attr, bx, by, right, bottom, half_w, half_h, node_r, info["fill_color"])
        if draw_border:
            _emit_quad(
                ctx.border_attr, bx, by, right, bottom, half_w, half_h, node_r, border_color, line_width=border_w
            )
        if is_hovered:
            outline_x = bx - ctx.highlight_margin
            outline_y = by - ctx.highlight_margin
            outline_w = node_w_final + ctx.highlight_margin * 2
            outline_h = node_h_final + ctx.highlight_margin * 2
            _emit_quad(
                ctx.highlight_attr,
                outline_x,
                outline_y,
                outline_x + outline_w,
                outline_y + outline_h,
                outline_w / 2,
                outline_h / 2,
                node_r,
                ctx.highlight_outline,
                line_width=ctx.highlight_line_width,
            )
    else:
        half_w = node_w / 2
        half_h = node_h / 2
        right = bx + node_w
        bottom = by + node_h
        _emit_quad(
            ctx.frame_fill_attr if is_frame else ctx.fill_attr,
            bx,
            by,
            right,
            bottom,
            half_w,
            half_h,
            node_r,
            info["fill_color"],
        )
        if draw_border:
            _emit_quad(
                ctx.frame_border_attr if is_frame else ctx.border_attr,
                bx,
                by,
                right,
                bottom,
                half_w,
                half_h,
                node_r,
                border_color,
                line_width=border_w,
            )
        if is_hovered:
            outline_x = bx - ctx.highlight_margin
            outline_y = by - ctx.highlight_margin
            outline_w = node_w + ctx.highlight_margin * 2
            outline_h = node_h + ctx.highlight_margin * 2
            _emit_quad(
                ctx.highlight_attr,
                outline_x,
                outline_y,
                outline_x + outline_w,
                outline_y + outline_h,
                outline_w / 2,
                outline_h / 2,
                node_r,
                ctx.highlight_outline,
                line_width=ctx.highlight_line_width,
            )

        if is_frame:
            # Zoom gate evaluated live per bake (user_zoom drives scale),
            # so labels appear/disappear at the threshold without waiting
            # for a tree recompile.
            if not ctx.state.view.user_zoom >= 0.8:
                return
            frame_label = info.get("frame_label")
            if frame_label:
                text, text_color, bg_label_color = frame_label
                label_font_size = max(6, min(11, int(11 * ctx.ui_scale * ctx.bake_scale * 8)))
                blf.size(ctx.font_id, label_font_size)
                text_w, text_h = blf.dimensions(ctx.font_id, text)
                label_pad = 2 * ctx.ui_scale
                label_w = text_w + 2 * label_pad
                label_h = text_h + 2 * label_pad
                ctx.frame_label_entries.append(
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
                        "depth": ctx.frame_depths.get(info["ptr"], 0),
                        "rect": (
                            bx + (node_w - label_w) / 2,
                            by + node_h + 3 * ctx.ui_scale - label_pad,
                            label_w,
                            label_h,
                        ),
                    }
                )
        else:
            label_type = info.get("node_label_type")
            label_text = info.get("node_label_text")
            if label_type and label_text and node_w > 6 * ctx.ui_scale and node_h > 6 * ctx.ui_scale:
                text_color = info["node_label_color"]
                if label_type == "full":
                    font_size = max(6, min(int(11 * ctx.ui_scale), int(min(node_w, node_h) * 0.35)))
                    lines = _get_node_label_lines(label_text, ctx.font_id, font_size, node_w - 4 * ctx.ui_scale, 3)
                    if lines:
                        blf.size(ctx.font_id, font_size)
                        line_h = blf.dimensions(ctx.font_id, "Ay")[1] + 1
                        ascent_h = blf.dimensions(ctx.font_id, "A")[1]
                        text_block_h = (len(lines) - 1) * line_h + ascent_h
                        start_y = by + (node_h - text_block_h) / 2
                        for i, line in enumerate(lines):
                            line_w, _ = blf.dimensions(ctx.font_id, line)
                            label_x = bx + (node_w - line_w) / 2
                            label_y = start_y + (len(lines) - 1 - i) * line_h
                            ctx.node_labels.append((ctx.font_id, line, label_x, label_y, text_color, font_size))
                else:
                    font_size = max(6, min(int(11 * ctx.ui_scale), int(min(node_w, node_h) * 0.45)))
                    blf.size(ctx.font_id, font_size)
                    text_w, text_h = blf.dimensions(ctx.font_id, label_text)
                    text_x = bx + (node_w - text_w) / 2
                    text_y = by + (node_h - text_h) / 2
                    ctx.node_labels.append((ctx.font_id, label_text, text_x, text_y, text_color, font_size))


def _emit_frame_labels(ctx: "_BakeContext") -> None:
    """Resolve frame-label collisions and emit their backdrops plus text."""
    for entry in _resolve_frame_label_layout(ctx.frame_label_entries, ctx.ui_scale):
        rect_x, rect_y, label_w, label_h = entry["rect"]
        label_pad = entry["pad"]
        _emit_quad(
            ctx.frame_fill_attr,
            rect_x,
            rect_y,
            rect_x + label_w,
            rect_y + label_h,
            label_w / 2,
            label_h / 2,
            entry["node_r"],
            entry["bg_label_color"],
        )
        ctx.node_labels.append(
            (
                ctx.font_id,
                entry["text"],
                rect_x + label_pad,
                rect_y + label_pad,
                entry["text_color"],
                entry["font_size"],
            )
        )


def _bake_rect(ctx: "_BakeContext", src: dict, get_shader, cache_field: str, has_line_width: bool) -> None:
    """Bake one rounded-rect layer's scratch attributes into a GPU batch."""
    cache = ctx.state.cache
    num = len(src["pos"]) // 4
    if num <= 0:
        setattr(cache, cache_field, None)
        return
    data = {
        "pos": src["pos"],
        "uv": src["uv"],
        "halfSize": src["half_size"],
        "radius": src["radius"],
        "color": src["color"],
    }
    if has_line_width:
        data["lineWidth"] = src["line_width"]
    setattr(cache, cache_field, batch_for_shader(get_shader(), "TRIS", data, indices=_create_quad_indices(num)))


def _bake_sockets(ctx: "_BakeContext") -> None:
    """Bake socket-ph, reroute label entries, and the socket pill batch."""
    pill_h = max(1, ctx.tree_data["socket_ph_base"] * ctx.bake_scale * ctx.ui_scale)
    pill_w = pill_h
    ctx.state.cache.socket_ph = pill_h
    # Extend node labels with reroute labels before socket batch cull check,
    # keeping a single BLF measurement pass for all text.
    reroute_labels_raw = ctx.tree_data.get("reroute_labels_raw") if ctx.tree_data.get("reroute_on") else None
    if reroute_labels_raw and ctx.scale >= MIN_SOCKET_SCALE:
        # Quick visibility cull for labels: skip far off-screen reroutes
        # to avoid BLF measurement overhead.
        for entry in reroute_labels_raw:
            tree_x = entry.get("tree_x")
            tree_y = entry.get("tree_y")
            if tree_x is None or tree_y is None:
                continue
            if ctx.filter_names is not None and entry.get("node_name") not in ctx.filter_names:
                continue
            baked_x = (tree_x - ctx.origin_x) * ctx.bake_scale
            baked_y = (tree_y - ctx.origin_y) * ctx.bake_scale
            if (
                baked_x < ctx.cull_left - 60
                or baked_x > ctx.cull_right + 60
                or baked_y < ctx.cull_bottom - 20
                or baked_y > ctx.cull_top + 40
            ):
                continue
            text = entry.get("text", "")
            if not text:
                continue
            text_color = entry.get("color") or _alpha_mul((1.0, 1.0, 1.0, 1.0), 1.0)
            font_size = max(6, min(11, int(10 * ctx.ui_scale)))
            blf.size(ctx.font_id, font_size)
            text_w, text_h = blf.dimensions(ctx.font_id, text)
            label_x = baked_x - text_w * 0.5
            label_y = baked_y + pill_h * 0.5 + 3.0 * ctx.ui_scale
            ctx.state.cache.node_labels.append((ctx.font_id, text, label_x, label_y, text_color, font_size))

    socket_items_by_node = ctx.tree_data.get("socket_items_by_node") or {}
    if socket_items_by_node and ctx.scale >= MIN_SOCKET_SCALE:
        half_w = pill_w / 2
        half_h = pill_h / 2
        pill_radius = pill_h / 2
        socket_pos = []
        socket_uv = []
        socket_half_size = []
        socket_radius = []
        socket_color = []
        socket_attr = {
            "pos": socket_pos,
            "uv": socket_uv,
            "half_size": socket_half_size,
            "radius": socket_radius,
            "color": socket_color,
        }
        socket_pad = 1.5
        if ctx.filter_names is not None:
            name_by_ptr = {info.get("ptr"): info.get("name") for info in ctx.node_infos}
        for node_ptr, dots in socket_items_by_node.items():
            if ctx.filter_names is not None and name_by_ptr.get(node_ptr) not in ctx.filter_names:
                continue
            for color, sx_tree, sy_tree in dots:
                linear_color = _srgb_to_linear(color)
                socket_baked_x = (sx_tree - ctx.origin_x) * ctx.bake_scale
                socket_baked_y = (sy_tree - ctx.origin_y) * ctx.bake_scale
                _emit_quad(
                    socket_attr,
                    socket_baked_x - half_w - socket_pad,
                    socket_baked_y - half_h - socket_pad,
                    socket_baked_x + half_w + socket_pad,
                    socket_baked_y + half_h + socket_pad,
                    half_w + socket_pad,
                    half_h + socket_pad,
                    pill_radius,
                    linear_color,
                    half_size=(half_w, half_h),
                )
        num_sockets = len(socket_pos) // 4
        if num_sockets > 0:
            shader = _get_batch_rect_shader()
            ctx.state.cache.socket_batch = batch_for_shader(
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
            ctx.state.cache.socket_batch = None
    else:
        ctx.state.cache.socket_batch = None


def _bake_reroutes(ctx: "_BakeContext") -> None:
    """Bake the reroute pill batch, scale-gated like sockets."""
    pill_h = max(1, ctx.tree_data["socket_ph_base"] * ctx.bake_scale * ctx.ui_scale)
    pill_w = pill_h
    reroute_items = ctx.tree_data.get("reroute_items") if ctx.tree_data.get("reroute_on") else None
    if reroute_items and ctx.scale >= MIN_SOCKET_SCALE:
        half_w = pill_w / 2
        half_h = pill_h / 2
        pill_radius = pill_h / 2
        reroute_pos: list[tuple[float, float, float]] = []
        reroute_uv: list[tuple[float, float]] = []
        reroute_half_size: list[tuple[float, float]] = []
        reroute_radius: list[float] = []
        reroute_color_list: list[tuple[float, float, float, float]] = []
        reroute_attr = {
            "pos": reroute_pos,
            "uv": reroute_uv,
            "half_size": reroute_half_size,
            "radius": reroute_radius,
            "color": reroute_color_list,
        }
        pad = 1.5
        reroute_by_name = ctx.tree_data.get("reroute_by_name") or {}
        for name, (pill_color, rx_tree, ry_tree) in reroute_by_name.items():
            if ctx.filter_names is not None and name not in ctx.filter_names:
                continue
            baked_x = (rx_tree - ctx.origin_x) * ctx.bake_scale
            baked_y = (ry_tree - ctx.origin_y) * ctx.bake_scale
            # Cull far off-screen reroutes before emitting vertices.
            if (
                baked_x < ctx.cull_left - half_w - 2
                or baked_x > ctx.cull_right + half_w + 2
                or baked_y < ctx.cull_bottom - half_h - 2
                or baked_y > ctx.cull_top + half_h + 2
            ):
                continue
            linear_color = _srgb_to_linear(pill_color)
            _emit_quad(
                reroute_attr,
                baked_x - half_w - pad,
                baked_y - half_h - pad,
                baked_x + half_w + pad,
                baked_y + half_h + pad,
                half_w + pad,
                half_h + pad,
                pill_radius,
                linear_color,
                half_size=(half_w, half_h),
            )
        num_reroutes = len(reroute_pos) // 4
        if num_reroutes > 0:
            shader = _get_batch_rect_shader()
            ctx.state.cache.reroute_batch = batch_for_shader(
                shader,
                "TRIS",
                {
                    "pos": reroute_pos,
                    "uv": reroute_uv,
                    "halfSize": reroute_half_size,
                    "radius": reroute_radius,
                    "color": reroute_color_list,
                },
                indices=_create_quad_indices(num_reroutes),
            )
        else:
            ctx.state.cache.reroute_batch = None
    else:
        ctx.state.cache.reroute_batch = None


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
    list_visible=False,
    highlight_border=None,
    wire_curvature: int = 5,
    wire_thickness: float = 1.0,
):
    """Bake content batches in map-local space, rebuilding only when stale."""
    tree_data = minimap_state.cache.tree_data
    if tree_data is None:
        return
    origin = tree_data.get("origin")
    if not origin:
        return

    query = minimap_state.list.search_query.strip()
    key = (
        minimap_state.cache.position_version,
        round(ui_scale, 3),
        show_borders,
        bool(list_visible),
        minimap_state.list.hovered_type_label,
        minimap_state.interaction.hovered_node_id,
        bool(highlight_border),
        query,
    )
    bake_scale = minimap_state.cache.batch_scale
    anchor_x, anchor_y = minimap_state.cache.batch_anchor
    # A settle bump changes only tree_data_version, so wire/marker freshness
    # must gate the early return too — otherwise wires stay frozen at their
    # pre-drag positions until an unrelated rebuild trigger fires.
    wire_key = (
        minimap_state.cache.tree_version,
        round(ui_scale, 3),
        int(wire_curvature),
        round(wire_thickness, 3),
        query,
    )
    wires_fresh = wire_key == minimap_state.cache.wire_key and minimap_state.cache.wire_scale == bake_scale
    if (
        key == minimap_state.cache.batch_key
        and wires_fresh
        and bake_scale > 0.0
        and abs(scale - bake_scale) <= SCALE_REBUILD_REL * max(bake_scale, 1e-6)
        and abs(map_anchor_x - anchor_x) <= BATCH_DRIFT_PX
        and abs(map_anchor_y - anchor_y) <= BATCH_DRIFT_PX
    ):
        return

    origin_x, origin_y = origin
    # Sticky bake scale: adopt the live scale only when the drift budget is
    # exceeded, so fill and wire generations always share one bake scale
    # (and thus one content-matrix factor) between bucket crossings.
    prev_bake_scale = minimap_state.cache.batch_scale
    if prev_bake_scale > 0.0 and abs(scale - prev_bake_scale) <= SCALE_REBUILD_REL * max(prev_bake_scale, 1e-6):
        bake_scale = prev_bake_scale
    else:
        bake_scale = scale

    font_id = 0
    min_dim = 3.0 * ui_scale
    node_infos = tree_data["node_infos"]
    frame_depths = _compute_frame_depths(node_infos)
    frame_label_entries: list[dict] = []
    # Active search filter: node names to keep visible, or None when the
    # query is empty or the type list is hidden (draw everything). Mirrors
    # the type list's own matching.
    filter_names = (
        filter_matching_nodes(
            tree_data.get("type_stats") or {},
            tree_data.get("type_nodes") or {},
            query,
            tree_data.get("type_search") or None,
        )
        if query and list_visible
        else None
    )
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
    cull_left = map_x - CULL_MARGIN_PX - map_anchor_x + pivot_baked_x
    cull_right = map_x + map_w + CULL_MARGIN_PX - map_anchor_x + pivot_baked_x
    cull_bottom = map_y - CULL_MARGIN_PX - map_anchor_y + pivot_baked_y
    cull_top = map_y + map_h + CULL_MARGIN_PX - map_anchor_y + pivot_baked_y

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

    # Scratch attribute aliases: each layer's dicts feed _emit_quad and the
    # matching batch_for_shader call below shares the same underlying lists.
    fill_attr = {
        "pos": all_pos_fill,
        "uv": all_uv_fill,
        "half_size": all_half_size_fill,
        "radius": all_radius_fill,
        "color": all_color_fill,
    }
    frame_fill_attr = {
        "pos": frame_pos_fill,
        "uv": frame_uv_fill,
        "half_size": frame_half_size_fill,
        "radius": frame_radius_fill,
        "color": frame_color_fill,
    }
    border_attr = {
        "pos": all_pos_border,
        "uv": all_uv_border,
        "half_size": all_half_size_border,
        "radius": all_radius_border,
        "color": all_color_border,
        "line_width": all_line_width_border,
    }
    frame_border_attr = {
        "pos": frame_pos_border,
        "uv": frame_uv_border,
        "half_size": frame_half_size_border,
        "radius": frame_radius_border,
        "color": frame_color_border,
        "line_width": frame_line_width_border,
    }
    highlight_attr = {
        "pos": highlight_pos_border,
        "uv": highlight_uv_border,
        "half_size": highlight_half_size_border,
        "radius": highlight_radius_border,
        "color": highlight_color_border,
        "line_width": highlight_line_width_border,
    }

    ctx = _BakeContext(
        state=minimap_state,
        tree_data=tree_data,
        node_infos=node_infos,
        frame_depths=frame_depths,
        scale=scale,
        bake_scale=bake_scale,
        origin_x=origin_x,
        origin_y=origin_y,
        ui_scale=ui_scale,
        font_id=font_id,
        min_dim=min_dim,
        cull_left=cull_left,
        cull_right=cull_right,
        cull_bottom=cull_bottom,
        cull_top=cull_top,
        show_borders=show_borders,
        hovered_type=hovered_type,
        hovered_node_name=hovered_node_name,
        highlight_outline=highlight_outline,
        highlight_margin=highlight_margin,
        highlight_line_width=highlight_line_width,
        filter_names=filter_names,
        frame_label_entries=frame_label_entries,
        node_labels=node_labels,
        fill_attr=fill_attr,
        frame_fill_attr=frame_fill_attr,
        border_attr=border_attr,
        frame_border_attr=frame_border_attr,
        highlight_attr=highlight_attr,
    )
    for info in node_infos:
        _emit_node(ctx, info)

    # Frame labels: resolve collisions, then emit backdrop quads into the
    # frames fill batch and text entries for the manual BLF pass.
    _emit_frame_labels(ctx)

    _bake_rect(ctx, ctx.fill_attr, _get_batch_rect_shader, "backdrops_batch", False)
    _bake_rect(ctx, ctx.border_attr, _get_batch_rect_border_shader, "borders_batch", True)
    _bake_rect(ctx, ctx.highlight_attr, _get_batch_rect_border_shader, "highlight_borders_batch", True)
    _bake_rect(ctx, ctx.frame_fill_attr, _get_batch_rect_shader, "frames_fill_batch", False)
    _bake_rect(ctx, ctx.frame_border_attr, _get_batch_rect_border_shader, "frames_border_batch", True)
    minimap_state.cache.node_labels = node_labels

    # Sockets and reroutes — pills auto-hidden below the minimum scale;
    # socket-ph is shared and measured in a single BLF pass with the labels.
    _bake_sockets(ctx)
    _bake_reroutes(ctx)

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
            filter_names,
        )
        minimap_state.cache.wire_key = wire_key
        minimap_state.cache.wire_scale = bake_scale

    minimap_state.cache.batch_key = key
    minimap_state.cache.batch_scale = bake_scale
    minimap_state.cache.batch_anchor = (map_anchor_x, map_anchor_y)


def _convert_wire_endpoints(
    items: list[tuple[float, float, float, float]],
    origin_x: float,
    origin_y: float,
    bake_scale: float,
    curvature: float,
) -> tuple[list[tuple[float, ...]], list[tuple[float, float, float, float]]]:
    """Convert tree-space wire endpoints to baked noodle controls or pill quads."""
    use_curve = curvature > 1e-6
    controls: list[tuple[float, ...]] = []
    pills: list[tuple[float, float, float, float]] = []
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
            controls.append((x1, y1, x1 + dist_h, y1, x2 - dist_h, y2, x2, y2))
        else:
            length = math.hypot(dx, dy)
            angle = math.atan2(dy, dx)
            mid_x = (x1 + x2) * 0.5
            mid_y = (y1 + y2) * 0.5
            pills.append((mid_x, mid_y, length, angle))
    return controls, pills


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
    filter_names: frozenset[str] | None = None,
) -> None:
    """Bake wire batches and group markers."""

    # Wires — cubic noodle strips when curvature is on, straight pills at 0.
    # The user wire_thickness multiplier scales the baked thickness.
    thickness = max(1.6, 2.0 * bake_scale) * wire_thickness
    curvature = float(wire_curvature)
    use_curve = curvature > 1e-6
    wire_batches: list[tuple[Any, Any, float]] = []
    per_color_wires: dict[Any, list[tuple[float, float, float, float, float, float, float, float]]] = {}

    # Under an active search filter, resolve wires from raw links so a link
    # stays visible while either endpoint node matches; with no filter the
    # pre-grouped per-color wire_items are used verbatim.
    if filter_names is not None:
        raw_links = tree_data.get("raw_links") or []
        out_pos = tree_data.get("out_pos") or {}
        in_pos = tree_data.get("in_pos") or {}
        highlight_names = tree_data.get("highlight_link_names")
        filtered_links = [link for link in raw_links if link[0] in filter_names or link[2] in filter_names]
        wire_items, wire_highlight_items = _resolve_wire_items(filtered_links, out_pos, in_pos, highlight_names)
    else:
        wire_items = tree_data["wire_items"]
        wire_highlight_items = tree_data.get("wire_highlight_items") or []

    for color, items in wire_items.items():
        group_controls, group_pills = _convert_wire_endpoints(items, origin_x, origin_y, bake_scale, curvature)
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

    # Wires connected to selected nodes — one thicker batch drawn over the
    # regular wires in the theme selection color (see tree_compile).
    highlight_items = wire_highlight_items
    highlight_cache = None
    if highlight_items:
        highlight_thickness = thickness * 1.5
        h_controls, h_pills = _convert_wire_endpoints(highlight_items, origin_x, origin_y, bake_scale, curvature)
        if use_curve and h_controls:
            _shader, batch = _build_noodle_batch(h_controls, highlight_thickness * 0.5)
            if batch is not None:
                highlight_cache = (batch, highlight_thickness * 0.5)
        elif h_pills:
            _shader, batch = _build_pill_batch(h_pills, highlight_thickness)
            if batch is not None:
                highlight_cache = (batch, highlight_thickness * 0.5)
    minimap_state.cache.wire_highlight_batch = highlight_cache

    # Group node underline markers — baked like wires
    marker_batches = []
    if filter_names is not None:
        # Rebuild markers from node infos so group nodes hidden by the
        # search filter drop their underline too (markers carry no names).
        grouped_markers: dict = {}
        for info in tree_data.get("node_infos") or []:
            if info.get("name") not in filter_names:
                continue
            marker_col = info.get("group_marker_col")
            if marker_col:
                grouped_markers.setdefault(marker_col, []).append(
                    (info["tree_x"] + info["tree_w"] / 2, info["tree_y"], info["tree_w"])
                )
        group_markers = grouped_markers
    else:
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
