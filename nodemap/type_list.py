"""Interactive node-type list zone and scrollbar drawing."""

import logging
import math
import time

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

from .batch_build import _create_quad_indices
from .gpu_draw import (
    _draw_filled_rounded_rect,
    _draw_pill,
    _draw_rounded_rect_border,
    _get_batch_rect_shader,
)
from .helpers import (
    _HANDLE_THICKNESS,
    _LIST_COUNT_GAP,
    _LIST_PAD_X,
    _LIST_SWATCH,
    _LIST_SWATCH_GAP,
    STATS_FONT_ID,
    _get_type_list_width,
    _schedule_list_anim_redraw,
)
from .state import MinimapState
from .theme import _COLOR_TAG_TO_THEME_ATTR, _alpha_mul, _srgb_to_linear, _theme_rgba
from .tree_compile import _Timer

logger = logging.getLogger(__package__)

_SCROLLBAR_THICKNESS = 3.0
_SCROLLBAR_THICKNESS_HOVER = 6.0
_SCROLLBAR_INSET = 2.0
_SCROLLBAR_MIN_THUMB = 6.0
_SCROLLBAR_ALPHA = 0.65

_TYPE_LIST_ANIM_AWAIT_TIMEOUT = 1.0
_TYPE_LIST_MIN_LABEL_W = 32.0


def _get_scrollbar_style(ui_scale: float) -> tuple[int, int]:
    """Return the shared scrollbar (thickness, inset) scaled for the UI."""
    return max(2, int(_SCROLLBAR_THICKNESS * ui_scale)), int(_SCROLLBAR_INSET * ui_scale)


def _scrollbar_thickness(ui_scale: float, active: bool = False) -> int:
    """Return the scrollbar thumb thickness; expanded while hovered or dragged."""
    thick, _ = _get_scrollbar_style(ui_scale)
    if not active:
        return thick
    return max(thick + 1, int(_SCROLLBAR_THICKNESS_HOVER * ui_scale))


def _draw_scrollbar_thumb(
    x: float,
    y: float,
    track_len: float,
    visible_frac: float,
    pos_frac: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    horizontal: bool = False,
    active: bool = False,
    pressed: bool = False,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Draw a scrollbar thumb pill; the origin is the track's start end.

    *visible_frac* is the visible/content ratio sizing the thumb;
    *pos_frac* (0..1) slides it along the track from its start end
    (left when horizontal, bottom otherwise). Hovering (*active*) fades
    the thumb to full opacity; dragging (*pressed*) additionally lifts
    each channel by 5/255 like SCROLL_PRESSED in Blender's widget code.
    Returns the drawn ``(thumb_rect, track_rect)`` as ``(x, y, w, h)``
    for hit-testing.
    """
    thick = _scrollbar_thickness(ui_scale, active)
    if not active:
        color = _alpha_mul(colors["scroll_item"], master_alpha * _SCROLLBAR_ALPHA)
    else:
        rgba = colors["scroll_item"]
        if pressed:
            lift = 5.0 / 255.0
            rgba = (
                min(rgba[0] + lift, 1.0),
                min(rgba[1] + lift, 1.0),
                min(rgba[2] + lift, 1.0),
                rgba[3],
            )
        color = _alpha_mul(rgba, master_alpha)
    min_thumb = int(_SCROLLBAR_MIN_THUMB * ui_scale)
    thumb_len = max(min_thumb, int(track_len * visible_frac))
    offset = int((track_len - thumb_len) * min(max(pos_frac, 0.0), 1.0))
    if horizontal:
        _draw_pill(x + offset, y, thumb_len, thick, color)
        return (x + offset, y, thumb_len, thick), (x, y, track_len, thick)
    _draw_pill(x, y + offset, thick, thumb_len, color)
    return (x, y + offset, thick, thumb_len), (x, y, thick, track_len)


def _draw_minimap_scrollbars(
    mx, my, mw, mh, padding, cx, cy, scale, tree_cx, tree_cy, bounds, colors, ui_scale, master_alpha
):
    """Draw horizontal/vertical scrollbar thumbs when zoomed in."""
    inner_l = mx + padding
    inner_r = mx + mw - padding
    inner_b = my + padding
    inner_t = my + mh - padding
    inner_w = mw - 2 * padding
    inner_h = mh - 2 * padding

    bbox_l, bbox_b, bbox_r, bbox_t = bounds
    bbox_w = bbox_r - bbox_l
    bbox_h = bbox_t - bbox_b
    if bbox_w <= 0 or bbox_h <= 0:
        return

    # Convert minimap inner rect corners back to tree coords to find visible extent
    tree_l = tree_cx + (inner_l - cx) / scale
    tree_r = tree_cx + (inner_r - cx) / scale
    tree_b = tree_cy + (inner_b - cy) / scale
    tree_t = tree_cy + (inner_t - cy) / scale

    # Clamp visible area to bbox (viewport cannot extend past tree bounds)
    v_left = max(bbox_l, min(bbox_r, tree_l))
    v_right = max(bbox_l, min(bbox_r, tree_r))
    v_bottom = max(bbox_b, min(bbox_t, tree_b))
    v_top = max(bbox_b, min(bbox_t, tree_t))

    visible_w = v_right - v_left
    visible_h = v_top - v_bottom
    if visible_w >= bbox_w and visible_h >= bbox_h:
        return

    bar_thick, bar_off = _get_scrollbar_style(ui_scale)

    # Horizontal scrollbar (bottom edge)
    if visible_w < bbox_w:
        pos = (v_left - bbox_l) / (bbox_w - visible_w)
        _draw_scrollbar_thumb(
            inner_l,
            my + bar_off,
            inner_w,
            visible_w / bbox_w,
            pos,
            colors,
            master_alpha,
            ui_scale,
            horizontal=True,
        )

    # Vertical scrollbar (right edge)
    if visible_h < bbox_h:
        pos = (v_bottom - bbox_b) / (bbox_h - visible_h)
        _draw_scrollbar_thumb(
            mx + mw - bar_off - bar_thick,
            inner_b,
            inner_h,
            visible_h / bbox_h,
            pos,
            colors,
            master_alpha,
            ui_scale,
        )


def _step_list_width(st: MinimapState, settings, mw: float, ui_scale: float) -> None:
    """Advance the animated type-list zone width for this frame.

    Snaps directly when no animation is running; otherwise eases from the
    recorded start width toward the locked target. An expansion waits (up to
    a timeout) for the pending compile to expose measurable type stats
    before starting its clock.
    """
    list_font_size = settings.type_list_font_size
    target_now = _get_type_list_width(settings, st, mw, ui_scale, list_font_size)
    # During an interactive width drag the live pixel width wins so the zone
    # tracks the cursor per-pixel; percent derivation and animation resume
    # after the drag releases (drag_width cleared in the operator).
    if st.list.drag_width is not None:
        old = st.list.width
        new = st.list.drag_width
        if abs(new - old) >= 0.5:
            from .transforms import _preserve_view_for_list_width

            _preserve_view_for_list_width(st, old, new, ui_scale)
        st.list.width = st.list.drag_width
        return
    if not st.list.anim_active:
        old = st.list.width
        if abs(target_now - old) >= 0.5:
            from .transforms import _preserve_view_for_list_width

            _preserve_view_for_list_width(st, old, target_now, ui_scale)
        st.list.width = target_now
        return

    if st.list.anim_target < 0:
        if target_now > 0:
            st.list.anim_target = target_now
            st.list.anim_start = time.perf_counter()
        elif time.perf_counter() - st.list.anim_start > _TYPE_LIST_ANIM_AWAIT_TIMEOUT:
            st.list.anim_active = False
            st.list.width = target_now
            return
        else:
            st.list.width = st.list.anim_from
            _schedule_list_anim_redraw(st)
            return

    progress = min((time.perf_counter() - st.list.anim_start) / max(st.list.anim_duration, 1e-4), 1.0)
    eased = 1.0 - (1.0 - progress) ** 3
    new_width = st.list.anim_from + (st.list.anim_target - st.list.anim_from) * eased
    if progress >= 1.0:
        new_width = st.list.anim_target
    old = st.list.width
    if abs(new_width - old) >= 0.5:
        from .transforms import _preserve_view_for_list_width

        _preserve_view_for_list_width(st, old, new_width, ui_scale)
    st.list.width = new_width
    if progress >= 1.0:
        st.list.anim_active = False
    else:
        _schedule_list_anim_redraw(st)


def _type_list_cache_key(st: MinimapState, settings, colors: dict, master_alpha: float, ui_scale: float) -> tuple:
    """Return the invalidation key for the cached type-list layout and swatch batch."""
    # The color-tag palette feeds type swatch colors at compile time, so track
    # it here to catch theme edits that do not change the tree fingerprint.
    palette = tuple(
        _theme_rgba(f"node_editor.{attr}", colors["node"])[:3] for attr in _COLOR_TAG_TO_THEME_ATTR.values()
    )
    return (
        st.cache.tree_version,
        settings.type_list_sort,
        settings.colored_nodes,
        settings.show_type_colors,
        frozenset(st.list.expanded),
        ui_scale,
        master_alpha,
        tuple(colors["node"]),
        tuple(colors["text"]),
        palette,
    )


def _build_type_list_cache(
    st: MinimapState, settings, node_tree, key: tuple, colors: dict, master_alpha: float, ui_scale: float
) -> None:
    """Sort type entries once and bake layout metrics plus all swatch pills.

    Swatch vertices are stored in list-local space — x relative to the zone
    origin, y downward from the top row at scroll=0 — so scrolling and width
    animation only move the matrix translate, never rebuild the batch. The
    client (``_draw_type_list``) draws the baked batch under a matrix
    translate ``(zone_x, view_top + scroll)``.
    """
    with _Timer("type_list.cache.build"):
        tree_data = st.cache.tree_data or {}
        type_stats = tree_data.get("type_stats") or {}

        # Live node objects for per-frame selection/active reads; the cache
        # key's tree_version rebuilds this map whenever the node set changes.
        st.cache.list_nodes_by_name = {n.name: n for n in node_tree.nodes} if node_tree else {}

        font_id = STATS_FONT_ID
        font_size = int(settings.type_list_font_size * ui_scale)
        blf.size(font_id, font_size)

        if settings.type_list_sort == "NAME":
            items = sorted(type_stats.items(), key=lambda kv: kv[0].lower())
        else:
            items = sorted(type_stats.items(), key=lambda kv: (-kv[1], kv[0]))

        entries: list[tuple[str, str, float, int]] = []
        widest_count = 0.0
        for label, count in items:
            count_text = str(count)
            count_w = blf.dimensions(font_id, count_text)[0]
            widest_count = max(widest_count, count_w)
            entries.append((label, count_text, count_w, count))

        _, line_h = blf.dimensions(font_id, "Ay")
        row_h = line_h + 4 * ui_scale

        st.cache.list_key = key
        st.cache.list_entries = entries
        st.cache.list_children = tree_data.get("type_nodes") or {}
        st.cache.list_layout = {
            "font_size": font_size,
            "line_h": line_h,
            "row_h": row_h,
            "widest_count": widest_count,
        }
        st.cache.list_swatches_batch = None
        _bake_list_glyph_batch(st, settings, colors, master_alpha, ui_scale, entries, row_h)


def _bake_list_glyph_batch(
    st: MinimapState,
    settings,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    entries: list[tuple[str, str, float, int]],
    row_h: float,
) -> None:
    """Bake all static swatches and expand chevrons into one batched rect pass.

    Header swatches and per-node child swatches keep their type color, while
    expand chevrons use the row text color halved like the live drawing did;
    none of them depend on hover or selection state, so a single baked batch
    drawn under a matrix translate reproduces the per-frame immediate draws.
    The fragment SDF runs in uv space, so chevron bars are baked with rotated
    corner positions and matching rotated uvs (no per-frame matrix work).
    """
    type_colors = (st.cache.tree_data or {}).get("type_colors") or {}
    type_node_colors = (st.cache.tree_data or {}).get("type_node_colors") or {}
    children = st.cache.list_children or {}

    show_type_colors = settings.show_type_colors and settings.colored_nodes
    expanded = st.list.expanded
    count_by_label = {label: count for label, _ct, _cw, count in entries}

    pad_x = _LIST_PAD_X * ui_scale
    swatch = _LIST_SWATCH * ui_scale
    swatch_gap = _LIST_SWATCH_GAP * ui_scale
    icon_col = swatch + swatch_gap
    swatch_col = icon_col if show_type_colors else 0.0

    # Chevron bars use the row text color at half alpha, matching the old live draw.
    text_col = _alpha_mul(colors["text"], 0.65 * master_alpha)
    chevron_col = _srgb_to_linear(_alpha_mul(text_col, 0.5))

    pos: list = []
    uv: list = []
    half_size: list = []
    radius: list = []
    vertex_col: list = []
    quads = 0

    def _push_quad(x: float, y: float, w: float, h: float, r: float, color) -> None:
        nonlocal quads
        hw, hh = w / 2, h / 2
        corners = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
        uvs = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        for (px, py), (ux, uy) in zip(corners, uvs):
            pos.append((px, py, 0.0))
            uv.append((ux, uy))
            half_size.append((hw, hh))
            radius.append(r)
            vertex_col.append(_srgb_to_linear(color))
        quads += 1

    def _push_rotated_bar(m, arm: float, t: float) -> None:
        nonlocal quads
        # Local rect corners span x in [-arm, t/2] (matches the live draw);
        # uvs are centered on the rect so the fragment SDF fills the quad.
        hw = (arm + t / 2.0) / 2.0
        hh = t / 2.0
        local = ((-arm, -hh), (t / 2.0, -hh), (t / 2.0, hh), (-arm, hh))
        uvs = ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
        for (lx, ly), (ux, uy) in zip(local, uvs):
            pos.append((m[0][0] * lx + m[0][1] * ly + m[0][3], m[1][0] * lx + m[1][1] * ly + m[1][3], 0.0))
            uv.append((ux, uy))
            half_size.append((hw, hh))
            radius.append(hh)
            vertex_col.append(chevron_col)
        quads += 1

    for kind, label, node_name, local_y in _iter_type_list_layout(entries, children, expanded, row_h):
        x = pad_x
        if kind == "header":
            if count_by_label.get(label, 0) > 1:
                # Mirror the live chevron's transform tree: translate to the
                # chevron center, base rotation for the expansion direction,
                # offset, then each bar rotated ±45deg (gpu.matrix.multiply_matrix
                # post-multiplies, so order is T1 @ R(base) @ T2 @ R(sign*45)).
                t = max(1.0, 1.2 * ui_scale)
                arm = swatch * 0.6
                cx = pad_x + swatch / 2.0
                cy = local_y - row_h / 2.0
                base_angle = -90.0 if label in expanded else 0.0
                offset_x = (arm * math.cos(math.radians(45.0))) / 2.0
                base_rot = Matrix.Rotation(math.radians(base_angle), 4, "Z")
                center = Matrix.Translation((cx, cy, 0.0))
                offset = Matrix.Translation((offset_x, 0.0, 0.0))
                for sign in (-1, 1):
                    arm_rot = Matrix.Rotation(math.radians(sign * 45.0), 4, "Z")
                    _push_rotated_bar(center @ base_rot @ offset @ arm_rot, arm, t)
            if show_type_colors:
                header_color = _header_type_color(label, children, type_node_colors, type_colors, colors)
                _push_quad(
                    x + icon_col,
                    local_y - (row_h + swatch) / 2.0,
                    swatch,
                    swatch,
                    swatch / 2.0,
                    _alpha_mul(header_color, master_alpha),
                )
        else:
            if show_type_colors:
                node_color = type_node_colors.get(label, {}).get(node_name, type_colors.get(label, colors["node"]))
                _push_quad(
                    x + icon_col + swatch_col,
                    local_y - (row_h + swatch) / 2.0,
                    swatch,
                    swatch,
                    swatch / 2.0,
                    _alpha_mul(node_color, master_alpha),
                )

    if quads == 0:
        st.cache.list_swatches_batch = None
        return
    shader = _get_batch_rect_shader()
    st.cache.list_swatches_batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": pos, "uv": uv, "halfSize": half_size, "radius": radius, "color": vertex_col},
        indices=_create_quad_indices(quads),
    )


def _draw_list_glyph_batch(st: MinimapState, zone_x: float, view_y: float) -> None:
    """Draw the baked swatch/chevron batch under a ``(zone_x, view_y)`` translate."""
    batch = st.cache.list_swatches_batch
    if batch is None:
        return
    shader = _get_batch_rect_shader()
    gpu.matrix.push()
    try:
        gpu.matrix.translate((zone_x, view_y))
        shader.bind()
        shader.uniform_float(
            "ModelViewProjectionMatrix",
            gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix(),
        )
        batch.draw(shader)
    finally:
        gpu.matrix.pop()


def _header_type_color(label: str, children: dict, node_colors: dict, type_colors: dict, colors: dict) -> tuple:
    """Return the header swatch color: the first child's own node color.

    *children* are the alphabetically sorted sub-lists for the type, so the
    first entry is the first row shown under the expanded header. Falls back
    to the type color and then the default node color.
    """
    first_names = children.get(label) or ()
    if first_names:
        first_color = node_colors.get(label, {}).get(first_names[0])
        if first_color is not None:
            return first_color
    return type_colors.get(label, colors["node"])


def _child_label_text(node_name: str, node) -> str:
    """Return the child row text: ``name (label)`` when the node has a label.

    For a group node without a label, falls back to the linked node-tree's name
    (matching what Blender and the minimap display); otherwise falls back to
    the bare node name when *node* is unavailable or has no label.
    """
    try:
        label = getattr(node, "label", "")
    except Exception:
        label = ""
    if label:
        return f"{node_name} ({label})"
    try:
        tree = getattr(node, "node_tree", None)
    except Exception:
        tree = None
    if tree is not None and getattr(tree, "name", ""):
        return tree.name
    return node_name


def _iter_type_list_layout(
    entries: list[tuple[str, str, float, int]],
    children: dict[str, list[str]],
    expanded: set,
    row_h: float,
):
    """Yield ``(kind, label, node_name, local_y_top)`` for each visible list row.

    ``kind`` is ``"header"`` for a type row or ``"child"`` for an individual
    node row. ``local_y_top`` is the list-local top coordinate (0 at the top,
    negative downward) so the same model drives the baked swatch batch and the
    per-frame text/hit-test layout. Only type groups with count > 1 that are
    present in *expanded* emit child rows.
    """
    y = 0.0
    for label, _count_text, _count_w, count in entries:
        yield ("header", label, None, y)
        y -= row_h
        if count > 1 and label in expanded:
            for node_name in children.get(label, ()):
                yield ("child", label, node_name, y)
                y -= row_h


def _draw_type_list(
    settings,
    st: MinimapState,
    mx: float,
    my: float,
    mh: float,
    padding: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> None:
    """Draw the interactive node-type list zone along the minimap's left edge."""
    st.list.row_rects = []
    st.list.node_rects = []
    st.list.toggle_rects = {}
    st.list.scroll_max = 0.0
    st.list.visible_row_keys = []
    # Drawn whenever the zone has width, including while it animates shut,
    # so the content slides out with the panel instead of vanishing.
    if st.list.width <= 0:
        st.list.hovered_type_label = None
        st.list.hovered_node = None
        st.interaction.hovered_node = None
        st.list.hovered_scrollbar = False
        st.list.scrollbar_thumb = None
        st.list.scrollbar_track = None
        st.list.zone_rect = None
        return
    tree_data = st.cache.tree_data
    type_stats = tree_data.get("type_stats") if tree_data else None
    if not type_stats:
        st.list.hovered_type_label = None
        st.list.hovered_node = None
        st.interaction.hovered_node = None
        st.list.hovered_scrollbar = False
        st.list.scrollbar_thumb = None
        st.list.scrollbar_track = None
        st.list.zone_rect = None
        return

    node_tree = bpy.context.space_data.edit_tree if bpy.context.space_data else None

    with _Timer("type_list.cache"):
        key = _type_list_cache_key(st, settings, colors, master_alpha, ui_scale)
        if key != st.cache.list_key or not st.cache.list_layout:
            _build_type_list_cache(st, settings, node_tree, key, colors, master_alpha, ui_scale)
    entries = st.cache.list_entries or []
    nodes_map = st.cache.list_nodes_by_name or {}
    layout = st.cache.list_layout or {}
    font_size = layout.get("font_size", int(settings.type_list_font_size * ui_scale))
    row_h = layout.get("row_h", 16.0)
    line_h = layout.get("line_h", 12.0)
    widest_count = layout.get("widest_count", 0.0)
    st.list.row_height = row_h

    # 1px visual separation between rows: row fills/borders are shrunk by the
    # gap and centered in the slot, leaving hit-testing and layout pitch intact.
    row_gap = 1.0
    row_gap_half = row_gap / 2.0
    row_draw_h = row_h - row_gap

    def _row_y(s_bottom: float) -> float:
        return round(s_bottom + row_gap_half)

    # Text baseline sits centered in the drawn (rounded) row so it stays aligned
    # with the row fill instead of drifting up to 0.5px off from the slot.
    text_y_off = (row_draw_h - line_h) / 2

    pad_x = _LIST_PAD_X * ui_scale
    swatch = _LIST_SWATCH * ui_scale
    swatch_gap = _LIST_SWATCH_GAP * ui_scale
    count_gap = _LIST_COUNT_GAP * ui_scale
    icon_col = swatch + swatch_gap

    # Compute total rows height so the background can shrink-wrap when the
    # list is shorter than the minimap.
    with _Timer("type_list.layout"):
        row_pad_v = 3 * ui_scale
        _children = st.cache.list_children or {}
        total_h = 0.0
        for _ in _iter_type_list_layout(entries, _children, st.list.expanded, row_h):
            total_h += row_h

        # Zone geometry: inset by the resize-handle thickness so edge resize
        # borders stay reachable around the list
        handle_pad = _HANDLE_THICKNESS * ui_scale
        zone_x = mx + handle_pad
        zone_w = mx + padding + st.list.width - 2 * ui_scale - zone_x
        zone_h = min(mh - 2 * handle_pad, total_h + 2 * row_pad_v)
        zone_y = my + mh - zone_h - handle_pad
        st.list.zone_rect = (zone_x, zone_y, zone_w, zone_h)

        zone_r = colors.get("panel_roundness", 4.0) * 0.6
        _draw_filled_rounded_rect(zone_x, zone_y, zone_w, zone_h, zone_r, _alpha_mul(colors["bg"], master_alpha))
        _draw_rounded_rect_border(
            zone_x, zone_y, zone_w, zone_h, zone_r, _alpha_mul(colors["bg_border"], master_alpha), 0.5
        )

        # Scrollable rows viewport inside the zone
        view_t = zone_y + zone_h - row_pad_v
        view_b = zone_y + row_pad_v
        view_h = max(view_t - view_b, row_h)
        scroll_max = max(0.0, total_h - view_h)
        st.list.scroll = min(max(st.list.scroll, 0.0), scroll_max)
        st.list.scroll_max = scroll_max

        show_type_colors = settings.show_type_colors and settings.colored_nodes

        swatch_col = icon_col if show_type_colors else 0.0

        content_x = zone_x + pad_x
        # Static extra margin so counts stay clear of the expanded scrollbar.
        count_right = zone_x + zone_w - pad_x - 4 * ui_scale
        # Labels start past icon columns (expand toggle, plus color swatch if enabled).
        label_x = content_x + icon_col + swatch_col
        # Hide counts when reserving them would squeeze the label below the
        # minimum; names then reclaim the full row width.
        show_counts = count_right - widest_count - count_gap - label_x >= _TYPE_LIST_MIN_LABEL_W * ui_scale
        label_max_w = max(0.0, (count_right - widest_count - count_gap if show_counts else count_right) - label_x)
        text_y_off = (row_h - line_h) / 2

        text_col = _alpha_mul(colors["text"], 0.9 * master_alpha)
        count_col = _alpha_mul(colors["text"], 0.3 * master_alpha)
        sel_col = _alpha_mul(colors["node_selected"], 0.95 * master_alpha)
        active_col = _alpha_mul(colors["indicator"], master_alpha)
        # Light fill under selected/active rows; selection color at low alpha (drawn
        # before text so BLF's alpha-disable cannot clobber the SDF fill blend).
        sel_fill_col = _alpha_mul(colors["node_selected"], 0.05 * master_alpha)
        active_fill_col = _alpha_mul(colors["indicator"], 0.05 * master_alpha)

        # Per-type selection state (compiled) drives font (not icon) recoloring.
        type_colors = tree_data.get("type_colors") or {}
        type_node_colors = tree_data.get("type_node_colors") or {}
        type_selected = tree_data.get("type_selected_counts") or {}
        type_active = tree_data.get("type_active_label")
        hover_col = _alpha_mul(colors["text"], 0.025 * master_alpha)

        pill_x = zone_x + 2 * ui_scale
        pill_w = zone_w - 4 * ui_scale

    # Clip rows to the zone interior so partial rows never bleed onto the map
    saved_scissor = None
    try:
        was_active = gpu.state.scissor_test_get()
        saved_scissor = (was_active, gpu.state.scissor_get() if was_active else None)
    except Exception:
        saved_scissor = None
    try:
        gpu.state.scissor_set(int(zone_x + 1), int(zone_y + 1), max(0, int(zone_w - 2)), max(0, int(zone_h - 2)))
        gpu.state.scissor_test_set(True)
        gpu.state.blend_set("ALPHA")

        with _Timer("type_list.rows"):
            entry_map = {lbl: (ct, cw, cnt) for (lbl, ct, cw, cnt) in entries}
            expanded = st.list.expanded
            hovered = st.list.hovered_type_label
            hovered_child = st.list.hovered_node

            # Walk the shared layout model; cull rows outside the viewport.
            visible_rows = []
            for row_idx, (kind, label, node_name, local_y) in enumerate(
                _iter_type_list_layout(entries, _children, expanded, row_h)
            ):
                s_top = view_t + st.list.scroll + local_y
                s_bottom = s_top - row_h
                if s_top <= view_b or s_bottom >= view_t:
                    continue
                visible_rows.append((kind, label, node_name, s_top, s_bottom, row_idx))

            st.list.visible_row_keys = [
                ("header", label) if kind == "header" else ("child", label, node_name)
                for kind, label, node_name, *_rest in visible_rows
            ]

        with _Timer("type_list.pills"):
            # Zebra bands keyed on the absolute layout index so they stay attached
            # to rows while scrolling; drawn beneath pills, text, and icons.
            band_col = (0.0, 0.0, 0.0, 0.15 * master_alpha)
            for _kind, _label, _node_name, s_top, s_bottom, row_idx in visible_rows:
                if row_idx % 2 == 1:
                    _draw_filled_rounded_rect(pill_x, _row_y(s_bottom), pill_w, row_draw_h, 0.0, band_col)

            # Header hover pills + hit rects (rows + expand toggle slots)
            header_rects = []
            toggle_rects = {}
            for kind, label, _node_name, _s_top, s_bottom, _row_idx in visible_rows:
                if kind != "header":
                    continue

                if label == type_active:
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(s_bottom), pill_w, row_draw_h, 4.0 * ui_scale, active_fill_col
                    )
                elif type_selected.get(label, 0) > 0:
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(s_bottom), pill_w, row_draw_h, 4.0 * ui_scale, sel_fill_col
                    )

                if hovered == label:
                    _draw_filled_rounded_rect(pill_x, _row_y(s_bottom), pill_w, row_draw_h, 4.0 * ui_scale, hover_col)

                if label == type_active:
                    # Active outline drawn here (pre-text) so BLF cannot clobber the
                    # alpha blend state the SDF fill relies on.
                    _draw_rounded_rect_border(
                        pill_x, _row_y(s_bottom), pill_w, row_draw_h, 4.0 * ui_scale, sel_col, 0.5 * ui_scale
                    )
                header_rects.append((pill_x, s_bottom, pill_w, row_h, label))
                if entry_map.get(label, ("", 0.0, 1))[2] > 1:
                    toggle_rects[label] = (content_x, s_bottom, swatch + swatch_gap, row_h)
                    if label in expanded:
                        child_count = len(_children.get(label, ()))
                        if child_count > 0:
                            header_color = _header_type_color(label, _children, type_node_colors, type_colors, colors)
                            line_color = (
                                _alpha_mul(header_color, master_alpha)
                                if show_type_colors
                                else _alpha_mul(colors["text"], 0.1 * master_alpha)
                            )
                            _draw_expand_guide_line(
                                content_x + swatch / 2, s_bottom - 1, child_count * row_h - 2, ui_scale, line_color
                            )
            st.list.row_rects = header_rects
            st.list.toggle_rects = toggle_rects

            # Active child hit-test lookups run up here so the outline can be drawn
            # in this pre-text pass (BLF disables alpha blending after glyph draws).
            active_node = node_tree.nodes.active if node_tree else None

            # Child hover pills + hit rects
            child_rects = []
            for kind, label, node_name, _s_top, s_bottom, _row_idx in visible_rows:
                if kind != "child":
                    continue
                child_selected = False
                child_active = False
                try:
                    node = nodes_map.get(node_name) if nodes_map else None
                    child_active = bool(active_node and node == active_node)
                    child_selected = bool(node and node.select)
                except Exception:
                    node = None

                if child_active:
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(s_bottom), pill_w, row_draw_h, 4.0 * ui_scale, active_fill_col
                    )
                elif child_selected:
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(s_bottom), pill_w, row_draw_h, 4.0 * ui_scale, sel_fill_col
                    )
                if child_active:
                    _draw_rounded_rect_border(
                        pill_x, _row_y(s_bottom), pill_w, row_draw_h, 4.0 * ui_scale, sel_col, 0.5 * ui_scale
                    )

                if hovered_child == (label, node_name):
                    _draw_filled_rounded_rect(pill_x, _row_y(s_bottom), pill_w, row_draw_h, 4.0 * ui_scale, hover_col)

                child_rects.append((pill_x, s_bottom, pill_w, row_h, label, node_name))
            st.list.node_rects = child_rects

        # Baked swatches/chevrons draw first so text renders above them; the
        # batch only moves via a matrix translate, so this is a single draw.
        with _Timer("type_list.text"):
            gpu.state.blend_set("ALPHA")
            _draw_list_glyph_batch(st, zone_x, view_t + st.list.scroll)

            font_id = STATS_FONT_ID
            blf.size(font_id, font_size)
            with_shadow = settings.show_text_shadow
            if with_shadow:
                blf.enable(font_id, blf.SHADOW)
                blf.shadow(font_id, 3, 0, 0, 0, 255)
                blf.shadow_offset(font_id, 0, -1)

            # Child rows: indented by one column/tab relative to headers.
            # When swatches are enabled, the child swatch sits indented under the
            # header label start (baked), and the child name follows it.
            child_indent = icon_col
            child_label_x = label_x + child_indent
            # Child rows never draw a count, so their names always run to the
            # right edge instead of reserving the header count column.
            child_label_max_w = max(0.0, count_right - child_label_x)
            child_clip_l = int(child_label_x)
            child_clip_r = int(child_label_x + child_label_max_w)
            clip_t = int(zone_y - row_h)
            clip_b = int(zone_y + zone_h + row_h)
            header_clip_l = int(label_x)
            header_clip_r = int(label_x + label_max_w)
            if show_counts:
                count_clip_r = int(count_right + widest_count)

            # CLIPPING stays enabled for the whole pass; per-row only the box
            # changes (labels, counts, and child names use distinct boxes).
            blf.enable(font_id, blf.CLIPPING)

            for kind, label, node_name, _s_top, s_bottom, _row_idx in visible_rows:
                text_y = _row_y(s_bottom) + text_y_off
                if kind == "header":
                    # Icons keep the type color; selection/active state shows in the
                    # row text color instead (active brightest, then selected).
                    is_active = label == type_active
                    is_sel = type_selected.get(label, 0) > 0
                    if is_active:
                        label_col = active_col
                    elif is_sel:
                        label_col = sel_col
                    else:
                        label_col = text_col

                    blf.clipping(font_id, header_clip_l, clip_t, header_clip_r, clip_b)
                    blf.position(font_id, label_x, text_y, 0)
                    blf.color(font_id, *label_col)
                    blf.draw(font_id, label)
                    # Counts sit right of the label clip box; BLF discards glyphs
                    # past the box instead of clipping them.
                    if show_counts:
                        count_text, count_w, _cnt = entry_map.get(label, ("", 0.0, 1))
                        blf.clipping(font_id, header_clip_r, clip_t, count_clip_r, clip_b)
                        blf.position(font_id, count_right - count_w, text_y, 0)
                        blf.color(font_id, *count_col)
                        blf.draw(font_id, count_text)
                else:
                    # Child row: the type-colored swatch is baked; text shows the
                    # selection state.
                    is_active = False
                    is_sel = False
                    try:
                        node = nodes_map.get(node_name) if nodes_map else None
                        is_active = bool(active_node and node == active_node)
                        is_sel = bool(node and node.select)
                    except Exception:
                        node = None
                    if is_active:
                        label_col = active_col
                    elif is_sel:
                        label_col = sel_col
                    else:
                        label_col = text_col

                    blf.clipping(font_id, child_clip_l, clip_t, child_clip_r, clip_b)
                    label_text = _child_label_text(node_name, node)
                    blf.position(font_id, child_label_x, text_y, 0)
                    blf.color(font_id, *label_col)
                    blf.draw(font_id, label_text)
            blf.disable(font_id, blf.CLIPPING)
            if with_shadow:
                blf.disable(font_id, blf.SHADOW)
            gpu.state.blend_set("ALPHA")
    finally:
        try:
            was_active, old_rect = saved_scissor or (False, None)
            if was_active and old_rect:
                gpu.state.scissor_set(*old_rect)
                gpu.state.scissor_test_set(True)
            else:
                gpu.state.scissor_set(0, 0, 65535, 65535)
                gpu.state.scissor_test_set(False)
        except Exception:
            pass

    # Scrollbar thumb when the list overflows (same style as the map scrollbars)
    with _Timer("type_list.scrollbar"):
        st.list.scrollbar_thumb = None
        st.list.scrollbar_track = None
        if scroll_max > 0:
            gpu.state.blend_set("ALPHA")
            _bar_thick, bar_off = _get_scrollbar_style(ui_scale)
            frac = st.list.scroll / scroll_max
            # Hover and drag share one expanded look (Blender overlay style).
            active = st.list.hovered_scrollbar or st.list.scrollbar_dragging
            thick = _scrollbar_thickness(ui_scale, active)
            thumb_rect, track_rect = _draw_scrollbar_thumb(
                zone_x + zone_w - thick - bar_off,
                zone_y + bar_off,
                zone_h - 2 * bar_off,
                view_h / total_h,
                1.0 - frac,
                colors,
                master_alpha,
                ui_scale,
                active=active,
            )
            st.list.scrollbar_thumb = thumb_rect
            st.list.scrollbar_track = track_rect


def _draw_expand_guide_line(x: float, top: float, height: float, ui_scale: float, color) -> None:
    """Draw a vertical guide under the expand icon, spanning the child rows."""
    t = max(1.0, 1.0 * ui_scale)
    _draw_filled_rounded_rect(x - t / 2, top - height, t, height, t / 2, color)
