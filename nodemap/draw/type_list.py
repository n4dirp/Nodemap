"""Provide an interactive node-type list zone and scrollbar drawing."""

import logging
import math
import time

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

from .. import __package__ as base_package
from ..core.constants import (
    BUTTON_SIZE,
    HANDLE_THICKNESS,
    LIST_COUNT_GAP,
    LIST_PAD_X,
    LIST_SWATCH,
    LIST_SWATCH_GAP,
    SCROLLBAR_ALPHA,
    SCROLLBAR_INSET,
    SCROLLBAR_MIN_THUMB,
    SCROLLBAR_THICKNESS,
    SCROLLBAR_THICKNESS_HOVER,
    TYPE_LIST_ANIM_AWAIT_TIMEOUT,
    TYPE_LIST_FONT_ID,
    TYPE_LIST_MIN_LABEL_W,
)
from ..core.helpers import (
    _get_type_list_width,
    _schedule_list_anim_redraw,
)
from ..core.list_filter import filter_type_list, match_span, normalize_query
from ..core.state import MinimapState
from ..core.theme import _COLOR_TAG_TO_THEME_ATTR, _alpha_mul, _srgb_to_linear, _theme_rgba
from .batch_build import _create_quad_indices
from .gpu_draw import (
    _draw_filled_rounded_rect,
    _draw_pill,
    _draw_pill_border,
    _draw_rounded_rect_border,
    _get_batch_rect_shader,
)
from .tree_compile import _Timer

logger = logging.getLogger(base_package)


def _get_scrollbar_style(ui_scale: float) -> tuple[int, int]:
    """Return the shared scrollbar (thickness, inset) scaled for the UI."""
    return max(2, int(SCROLLBAR_THICKNESS * ui_scale)), int(SCROLLBAR_INSET * ui_scale)


def _scrollbar_thickness(ui_scale: float, active: bool = False) -> int:
    """Return the scrollbar thumb thickness; expand while hovered or dragged."""
    thick, _ = _get_scrollbar_style(ui_scale)
    if not active:
        return thick
    return max(thick + 1, int(SCROLLBAR_THICKNESS_HOVER * ui_scale))


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
        color = _alpha_mul(colors["scroll_item"], master_alpha * SCROLLBAR_ALPHA)
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
    min_thumb_len = int(SCROLLBAR_MIN_THUMB * ui_scale)
    thumb_len = max(min_thumb_len, int(track_len * visible_frac))
    offset = int((track_len - thumb_len) * min(max(pos_frac, 0.0), 1.0))
    if horizontal:
        _draw_pill(x + offset, y, thumb_len, thick, color)
        return (x + offset, y, thumb_len, thick), (x, y, track_len, thick)
    _draw_pill(x, y + offset, thick, thumb_len, color)
    return (x, y + offset, thick, thumb_len), (x, y, thick, track_len)


def _draw_minimap_scrollbars(
    map_x,
    map_y,
    map_w,
    map_h,
    padding,
    map_anchor_x,
    map_anchor_y,
    scale,
    tree_center_x,
    tree_center_y,
    content_bounds,
    colors,
    ui_scale,
    master_alpha,
):
    """Draw horizontal/vertical scrollbar thumbs when zoomed in.

    *content_bounds* should be the raw node bounds (not the inflated
    ``tree_bounds``) so that scrollbars appear only when actual node
    content leaves the visible minimap interior.
    """
    inner_l = map_x + padding
    inner_r = map_x + map_w - padding
    inner_b = map_y + padding
    inner_t = map_y + map_h - padding
    inner_w = map_w - 2 * padding
    inner_h = map_h - 2 * padding

    bbox_l, bbox_b, bbox_r, bbox_t = content_bounds
    bbox_w = bbox_r - bbox_l
    bbox_h = bbox_t - bbox_b
    if bbox_w <= 0 or bbox_h <= 0:
        return

    # Convert minimap inner rect corners back to tree coords to find visible extent
    tree_l = tree_center_x + (inner_l - map_anchor_x) / scale
    tree_r = tree_center_x + (inner_r - map_anchor_x) / scale
    tree_b = tree_center_y + (inner_b - map_anchor_y) / scale
    tree_t = tree_center_y + (inner_t - map_anchor_y) / scale

    # Clamp visible area to content bounds
    v_left = max(bbox_l, min(bbox_r, tree_l))
    v_right = max(bbox_l, min(bbox_r, tree_r))
    v_bottom = max(bbox_b, min(bbox_t, tree_b))
    v_top = max(bbox_b, min(bbox_t, tree_t))

    visible_w = v_right - v_left
    visible_h = v_top - v_bottom
    if visible_w >= bbox_w and visible_h >= bbox_h:
        return

    bar_thickness, bar_offset = _get_scrollbar_style(ui_scale)

    # Horizontal scrollbar (bottom edge)
    if visible_w < bbox_w:
        frac_h = (v_left - bbox_l) / (bbox_w - visible_w)
        _draw_scrollbar_thumb(
            inner_l,
            map_y + bar_offset,
            inner_w,
            visible_w / bbox_w,
            frac_h,
            colors,
            master_alpha,
            ui_scale,
            horizontal=True,
        )

    # Vertical scrollbar (right edge)
    if visible_h < bbox_h:
        frac_v = (v_bottom - bbox_b) / (bbox_h - visible_h)
        _draw_scrollbar_thumb(
            map_x + map_w - bar_offset - bar_thickness,
            inner_b,
            inner_h,
            visible_h / bbox_h,
            frac_v,
            colors,
            master_alpha,
            ui_scale,
        )


def _step_list_width(state: MinimapState, settings, map_w: float, ui_scale: float) -> None:
    """Advance the animated type-list zone width for this frame.

    Snap directly when no animation is running; otherwise eases from the
    recorded start width toward the locked target. An expansion waits (up to
    a timeout) for the pending compile to expose measurable type stats
    before starting its clock.
    """
    list_font_size = settings.type_list_font_size
    target_width = _get_type_list_width(settings, state, map_w, ui_scale, list_font_size)
    # During an interactive width drag the live pixel width wins so the zone
    # tracks the cursor per-pixel; percent derivation and animation resume
    # after the drag releases (drag_width cleared in the operator).
    if state.list.dragging_width is not None:
        current_width = state.list.list_width
        new_width = state.list.dragging_width
        if abs(new_width - current_width) >= 0.5:
            from ..geo.transforms import _preserve_view_for_list_width

            _preserve_view_for_list_width(state, current_width, new_width, ui_scale)
        state.list.list_width = state.list.dragging_width
        return
    if not state.list.anim_active:
        current_width = state.list.list_width
        if abs(target_width - current_width) >= 0.5:
            from ..geo.transforms import _preserve_view_for_list_width

            _preserve_view_for_list_width(state, current_width, target_width, ui_scale)
        state.list.list_width = target_width
        return

    if state.list.anim_target < 0:
        if target_width > 0:
            state.list.anim_target = target_width
            state.list.anim_start = time.perf_counter()
        elif time.perf_counter() - state.list.anim_start > TYPE_LIST_ANIM_AWAIT_TIMEOUT:
            state.list.anim_active = False
            state.list.list_width = target_width
            return
        else:
            state.list.list_width = state.list.anim_from
            _schedule_list_anim_redraw(state)
            return

    progress = min((time.perf_counter() - state.list.anim_start) / max(state.list.anim_duration, 1e-4), 1.0)
    eased = 1.0 - (1.0 - progress) ** 3
    new_width = state.list.anim_from + (state.list.anim_target - state.list.anim_from) * eased
    if progress >= 1.0:
        new_width = state.list.anim_target
    current_width = state.list.list_width
    if abs(new_width - current_width) >= 0.5:
        from ..geo.transforms import _preserve_view_for_list_width

        _preserve_view_for_list_width(state, current_width, new_width, ui_scale)
    state.list.list_width = new_width
    if progress >= 1.0:
        state.list.anim_active = False
    else:
        _schedule_list_anim_redraw(state)


def _type_list_cache_key(state: MinimapState, settings, colors: dict, master_alpha: float, ui_scale: float) -> tuple:
    """Return the invalidation key for the cached type-list layout and swatch batch."""
    # The color-tag palette feeds type swatch colors at compile time, so track
    # it here to catch theme edits that do not change the tree fingerprint.
    palette = tuple(
        _theme_rgba(f"node_editor.{attr}", colors["node"])[:3] for attr in _COLOR_TAG_TO_THEME_ATTR.values()
    )
    return (
        state.cache.tree_version,
        settings.type_list_sort,
        settings.show_node_colors,
        settings.show_type_colors,
        frozenset(state.list.expanded),
        state.list.search_query,
        ui_scale,
        master_alpha,
        tuple(colors["node"]),
        tuple(colors["text"]),
        palette,
    )


def _build_type_list_cache(
    state: MinimapState, settings, node_tree, key: tuple, colors: dict, master_alpha: float, ui_scale: float
) -> None:
    """Sort type entries once and bake layout metrics plus all swatch pills.

    Swatch vertices are stored in list-local space — x relative to the zone
    origin, y downward from the top row at scroll=0 — so scrolling and width
    animation only move the matrix translate, never rebuild the batch. The
    client (``_draw_type_list``) draws the baked batch under a matrix
    translate ``(zone_x, view_top + scroll)``.
    """
    with _Timer("type_list.cache.build"):
        tree_data = state.cache.tree_data or {}
        type_stats = tree_data.get("type_stats") or {}

        # Live node objects for per-frame selection/active reads; the cache
        # key's tree_version rebuilds this map whenever the node set changes.
        state.cache.list_nodes_by_name = {n.name: n for n in node_tree.nodes} if node_tree else {}

        font_id = TYPE_LIST_FONT_ID
        font_size = int(settings.type_list_font_size * ui_scale)
        blf.size(font_id, font_size)

        children = tree_data.get("type_nodes") or {}
        search_texts = tree_data.get("type_search") or None
        # Filter before sorting so the COUNT order can use the (possibly
        # reduced) match counts; the full per-type count is restored from
        # type_stats for display logic (chevrons, lone-group labels).
        visible, effective_expanded, filtered_children = filter_type_list(
            type_stats,
            children,
            state.list.expanded,
            state.list.search_query,
            search_texts=search_texts,
        )
        if state.list.search_query.strip():
            pass
        elif settings.type_list_sort == "NAME":
            visible.sort(key=lambda label_count: label_count[0].lower())
        else:
            visible.sort(key=lambda label_count: (-label_count[1], label_count[0]))

        entries: list[tuple[str, str, float, int]] = []
        widest_count = 0.0
        for label, display_count in visible:
            count_text = str(display_count)
            count_width = blf.dimensions(font_id, count_text)[0]
            widest_count = max(widest_count, count_width)
            entries.append((label, count_text, count_width, type_stats.get(label, display_count)))

        _, line_h = blf.dimensions(font_id, "Ay")
        row_h = line_h + 4 * ui_scale

        state.cache.list_key = key
        state.cache.list_entries = entries
        state.cache.list_effective_expanded = effective_expanded
        state.cache.list_children = tree_data.get("type_nodes") or {}
        state.cache.list_filtered_children = filtered_children
        state.cache.list_layout = {
            "font_size": font_size,
            "line_h": line_h,
            "row_h": row_h,
            "widest_count": widest_count,
        }
        state.cache.list_swatches_batch = None
        _bake_list_glyph_batch(state, settings, colors, master_alpha, ui_scale, entries, row_h)


def _bake_list_glyph_batch(
    state: MinimapState,
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
    type_colors = (state.cache.tree_data or {}).get("type_colors") or {}
    type_node_colors = (state.cache.tree_data or {}).get("type_node_colors") or {}
    children = state.cache.list_filtered_children or state.cache.list_children or {}

    show_type_colors = settings.show_type_colors and settings.show_node_colors
    expanded = state.cache.list_effective_expanded or state.list.expanded
    count_by_label = {label: count for label, _ct, _cw, count in entries}

    pad_x = LIST_PAD_X * ui_scale
    swatch = LIST_SWATCH * ui_scale
    swatch_gap = LIST_SWATCH_GAP * ui_scale
    icon_col_x = swatch + swatch_gap
    swatch_col_x = icon_col_x if show_type_colors else 0.0

    # Chevron bars use the row text color at half alpha.
    text_color = _alpha_mul(colors["text"], 0.65 * master_alpha)
    chevron_color = _srgb_to_linear(_alpha_mul(text_color, 0.5))

    pos: list = []
    uv: list = []
    half_size: list = []
    radius: list = []
    vertex_color: list = []
    quads = 0

    def _push_quad(x: float, y: float, w: float, h: float, r: float, color) -> None:
        nonlocal quads
        half_w, half_h = w / 2, h / 2
        corners = ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
        uvs = ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
        for (px, py), (ux, uy) in zip(corners, uvs):
            pos.append((px, py, 0.0))
            uv.append((ux, uy))
            half_size.append((half_w, half_h))
            radius.append(r)
            vertex_color.append(_srgb_to_linear(color))
        quads += 1

    def _push_rotated_bar(matrix, arm: float, bar_thickness: float) -> None:
        nonlocal quads
        # Local rect corners span x in [-arm, t/2] (matches the live draw);
        # uvs are centered on the rect so the fragment SDF fills the quad.
        half_w = (arm + bar_thickness / 2.0) / 2.0
        half_h = bar_thickness / 2.0
        local = ((-arm, -half_h), (bar_thickness / 2.0, -half_h), (bar_thickness / 2.0, half_h), (-arm, half_h))
        uvs = ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
        for (lx, ly), (ux, uy) in zip(local, uvs):
            pos.append(
                (
                    matrix[0][0] * lx + matrix[0][1] * ly + matrix[0][3],
                    matrix[1][0] * lx + matrix[1][1] * ly + matrix[1][3],
                    0.0,
                )
            )
            uv.append((ux, uy))
            half_size.append((half_w, half_h))
            radius.append(half_h)
            vertex_color.append(chevron_color)
        quads += 1

    for kind, label, node_name, local_y in _iter_type_list_layout(entries, children, expanded, row_h):
        x = pad_x
        if kind == "header":
            if count_by_label.get(label, 0) > 1:
                # Mirror the live chevron's transform tree: translate to the
                # chevron center, base rotation for the expansion direction,
                # offset, then each bar rotated ±45deg (gpu.matrix.multiply_matrix
                # post-multiplies, so order is T1 @ R(base) @ T2 @ R(sign*45)).
                bar_thickness = max(1.0, 1.2 * ui_scale)
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
                    _push_rotated_bar(center @ base_rot @ offset @ arm_rot, arm, bar_thickness)
            if show_type_colors:
                header_color = _header_type_color(label, children, type_node_colors, type_colors, colors)
                _push_quad(
                    x + icon_col_x,
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
                    x + icon_col_x + swatch_col_x,
                    local_y - (row_h + swatch) / 2.0,
                    swatch,
                    swatch,
                    swatch / 2.0,
                    _alpha_mul(node_color, master_alpha),
                )

    if quads == 0:
        state.cache.list_swatches_batch = None
        return
    shader = _get_batch_rect_shader()
    state.cache.list_swatches_batch = batch_for_shader(
        shader,
        "TRIS",
        {"pos": pos, "uv": uv, "halfSize": half_size, "radius": radius, "color": vertex_color},
        indices=_create_quad_indices(quads),
    )


def _draw_list_glyph_batch(state: MinimapState, zone_x: float, view_y: float) -> None:
    """Draw the baked swatch/chevron batch under a ``(zone_x, view_y)`` translate."""
    batch = state.cache.list_swatches_batch
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
    """Return the header swatch color: the first child own node color.

    *children* are the alphabetically sorted sub-lists for the type, so the
    first entry is the first row shown under the expanded header. Falls back
    to the type color and then the default node color.
    """
    child_names = children.get(label) or ()
    if child_names:
        first_color = node_colors.get(label, {}).get(child_names[0])
        if first_color is not None:
            return first_color
    return type_colors.get(label, colors["node"])


def _group_header_text(label: str, count: int, children: dict, nodes_by_name: dict) -> str:
    """Return the header text for a group type, appending ``(label)`` for a lone node.

    A group type with exactly one node shows ``Group (label)`` where *label*
    is that node's custom label or, failing that, its linked node-tree's name.
    All other headers keep the bare type name.
    """
    if count != 1:
        return label
    names = children.get(label, ())
    if not names:
        return label
    node = (nodes_by_name or {}).get(names[0])
    if node is None or getattr(node, "type", "") != "GROUP":
        return label
    try:
        node_label = getattr(node, "label", "")
    except Exception:
        node_label = ""
    try:
        tree = getattr(node, "node_tree", None)
        tree_name = getattr(tree, "name", "") if tree is not None else ""
    except Exception:
        tree_name = ""
    sub = node_label or tree_name
    if not sub:
        return label
    return f"{label} ({sub})"


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


def _draw_text_with_match(font_id, x: float, y: float, text: str, base_color, match_color, norm_query: str) -> None:
    """Draw *text* at ``(x, y)``, tinting the first *norm_query* match.

    The matched substring is drawn in *match_color* to highlight it inside a
    search result row; the rest uses *base_color*. With no match the whole
    string is drawn in *base_color*. Each segment is positioned explicitly by
    its cumulative width so the highlight never overlaps the surrounding text
    (BLF does not advance the cursor across repeated ``blf.draw`` calls while
    clipping is enabled).
    """
    start = match_span(norm_query, text) if norm_query else -1
    if start < 0:
        blf.position(font_id, x, y, 0)
        blf.color(font_id, *base_color)
        blf.draw(font_id, text)
        return
    end = start + len(norm_query)
    if start > 0:
        blf.position(font_id, x, y, 0)
        blf.color(font_id, *base_color)
        blf.draw(font_id, text[:start])
    match_x = x + blf.dimensions(font_id, text[:start])[0]
    blf.position(font_id, match_x, y, 0)
    blf.color(font_id, *match_color)
    blf.draw(font_id, text[start:end])
    if end < len(text):
        after_x = x + blf.dimensions(font_id, text[:end])[0]
        blf.position(font_id, after_x, y, 0)
        blf.color(font_id, *base_color)
        blf.draw(font_id, text[end:])


def _draw_search_clear_button(rect, hovered: bool, colors: dict, master_alpha: float, ui_scale: float) -> None:
    """Draw the clear-query (X) button inside the search pill.

    Two thin bars rotated ±45deg form the cross; the GPU matrix transform keeps
    it crisp and independent of the active BLF font/clip state.
    """
    clear_x, clear_y, clear_size, _ = rect
    icon_color = _alpha_mul(colors["text"], 0.35 * master_alpha if not hovered else master_alpha)
    cx = round(clear_x + clear_size / 2)
    cy = round(clear_y + clear_size / 2)
    stroke = max(1.0, 0.8 * ui_scale)
    length = clear_size * 1.0
    gpu.matrix.push()
    try:
        gpu.matrix.translate((cx, cy, 0.0))
        for angle in (45.0, -45.0):
            gpu.matrix.push()
            try:
                gpu.matrix.multiply_matrix(Matrix.Rotation(math.radians(angle), 4, "Z"))
                _draw_filled_rounded_rect(-length / 2, -stroke / 2, length, stroke, 0.0, icon_color)
            finally:
                gpu.matrix.pop()
    finally:
        gpu.matrix.pop()


def _draw_search_icon(cx: float, cy: float, size: float, color, ui_scale: float) -> None:
    """Draw a magnifying-glass search icon centered at ``(cx, cy)``.

    A thin circular ring forms the lens and a single diagonal bar the handle,
    matching the crisp SDF cross drawn for the clear button.
    """
    ring_r = size * 0.28
    stroke = max(1.0, 1.2 * ui_scale)
    # Lens ring (a circular pill border).
    _draw_pill_border(
        cx - ring_r,
        cy - ring_r + stroke,
        ring_r * 2,
        ring_r * 2,
        color,
        stroke,
    )
    # Handle: one diagonal bar emerging from the lens ring toward the lower-right.
    handle_length = size * 0.32
    gpu.matrix.push()
    try:
        gpu.matrix.translate((cx, cy, 0.0))
        gpu.matrix.multiply_matrix(Matrix.Rotation(math.radians(-45.0), 4, "Z"))
        _draw_pill(ring_r, -stroke / 2, handle_length, stroke, color)
    finally:
        gpu.matrix.pop()


def _draw_type_list(
    settings,
    state: MinimapState,
    map_x: float,
    map_y: float,
    map_h: float,
    padding: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> None:
    """Draw the interactive node-type list zone along the minimap's left edge."""
    state.list.row_rects = []
    state.list.node_rects = []
    state.list.toggle_rects = {}
    state.list.scroll_max = 0.0
    state.list.visible_row_keys = []
    # Drawn whenever the zone has width, including while it animates shut,
    # so the content slides out with the panel instead of vanishing.
    if state.list.list_width <= 0:
        state.list.hovered_type_label = None
        state.list.hovered_list_row = None
        state.interaction.hovered_node_id = None
        state.list.hovered_scrollbar = False
        state.list.scrollbar_thumb = None
        state.list.scrollbar_track = None
        state.list.list_zone_rect = None
        state.list.search_rect = None
        state.list.search_clear_rect = None
        state.list.search_clear_hovered = False
        return
    tree_data = state.cache.tree_data
    type_stats = tree_data.get("type_stats") if tree_data else None
    if not type_stats:
        state.list.hovered_type_label = None
        state.list.hovered_list_row = None
        state.interaction.hovered_node_id = None
        state.list.hovered_scrollbar = False
        state.list.scrollbar_thumb = None
        state.list.scrollbar_track = None
        state.list.list_zone_rect = None
        state.list.search_rect = None
        state.list.search_clear_rect = None
        state.list.search_clear_hovered = False
        return

    node_tree = bpy.context.space_data.edit_tree if bpy.context.space_data else None

    with _Timer("type_list.cache"):
        key = _type_list_cache_key(state, settings, colors, master_alpha, ui_scale)
        if key != state.cache.list_key or not state.cache.list_layout:
            _build_type_list_cache(state, settings, node_tree, key, colors, master_alpha, ui_scale)
    entries = state.cache.list_entries or []
    expanded = state.cache.list_effective_expanded or state.list.expanded
    nodes_by_name = state.cache.list_nodes_by_name or {}
    layout = state.cache.list_layout or {}
    font_size = layout.get("font_size", int(settings.type_list_font_size * ui_scale))
    row_h = layout.get("row_h", 16.0)
    line_h = layout.get("line_h", 12.0)
    widest_count = layout.get("widest_count", 0.0)
    state.list.row_height = row_h

    # 1px visual separation between rows: row fills/borders are shrunk by the
    # gap and centered in the slot, leaving hit-testing and layout pitch intact.
    row_gap = 1.0
    row_gap_half = round(row_gap / 2.0)
    row_draw_h = row_h - row_gap

    def _row_y(slot_bottom: float) -> float:
        return round(slot_bottom + row_gap_half)

    # Text baseline sits centered in the drawn (rounded) row so it stays aligned
    # with the row fill instead of drifting up to 0.5px off from the slot.

    pad_x = LIST_PAD_X * ui_scale
    swatch = LIST_SWATCH * ui_scale
    swatch_gap = LIST_SWATCH_GAP * ui_scale
    count_gap = LIST_COUNT_GAP * ui_scale
    icon_col_x = swatch + swatch_gap

    with _Timer("type_list.layout"):
        row_pad_v = 1 * ui_scale
        children = state.cache.list_filtered_children or state.cache.list_children or {}
        total_h = 0.0
        for _ in _iter_type_list_layout(entries, children, expanded, row_h):
            total_h += row_h

        # Zone geometry: inset by the resize-handle thickness so edge resize
        # borders stay reachable around the list
        handle_pad = HANDLE_THICKNESS * ui_scale
        zone_x = map_x + handle_pad
        zone_w = map_x + padding + state.list.list_width - 2 * ui_scale - zone_x
        # The fixed search row sits above the scrolled content, with one row
        # padding above and below it. Its height matches the minimap chrome
        # buttons so the search bar reads as the same control size.
        if settings.show_search_bar:
            search_h = (BUTTON_SIZE - 1) * ui_scale
        else:
            search_h = 0.0
            state.list.search_focused = False
            state.list.search_rect = None
            state.list.search_clear_rect = None
            state.list.search_clear_hovered = False
        # Reserve at least one content row so an empty result still has room
        # to show the "No matches" message instead of collapsing the view.
        zone_h = min(map_h - 2 * handle_pad, max(total_h, row_h) + search_h + 3 * row_pad_v)
        zone_y = round(map_y + map_h - zone_h - handle_pad)
        state.list.list_zone_rect = (zone_x, zone_y, zone_w, zone_h)
        search_top = zone_y + zone_h - 1
        search_bottom = search_top - search_h
        search_draw_h = search_h - row_gap
        search_pad_v = round((search_draw_h - line_h) / 2.0) + 1

        zone_radius = colors.get("panel_roundness", 4.0) * 0.6
        _draw_filled_rounded_rect(zone_x, zone_y, zone_w, zone_h, zone_radius, _alpha_mul(colors["bg"], master_alpha))
        _draw_rounded_rect_border(
            zone_x, zone_y, zone_w, zone_h, zone_radius, _alpha_mul(colors["bg_border"], master_alpha), 0.5
        )

        view_top = search_bottom - row_pad_v
        view_bottom = zone_y + row_pad_v
        view_h = max(view_top - view_bottom, row_h)
        scroll_max = max(0.0, total_h - view_h)
        state.list.scroll = min(max(state.list.scroll, 0.0), scroll_max)
        state.list.scroll_max = scroll_max

        # Per-header slot bottoms in viewport coords, recorded for every header
        # regardless of culling so expand guide lines keep drawing even when the
        # header itself is scrolled out of view (only its children remain).
        header_slot_bottom: dict[str, float] = {}
        for kind, label, _node_name, local_y in _iter_type_list_layout(entries, children, expanded, row_h):
            if kind == "header":
                header_slot_bottom[label] = view_top + state.list.scroll + local_y - row_h

        show_type_colors = settings.show_type_colors and settings.show_node_colors

        swatch_col_x = icon_col_x if show_type_colors else 0.0

        content_x = zone_x + pad_x
        # Static extra margin so counts stay clear of the expanded scrollbar.
        count_right = zone_x + zone_w - pad_x - 4 * ui_scale
        label_x = content_x + icon_col_x + swatch_col_x
        # Hide counts when reserving them would squeeze the label below the
        # minimum; names then reclaim the full row width.
        show_counts = count_right - widest_count - count_gap - label_x >= TYPE_LIST_MIN_LABEL_W * ui_scale
        label_max_width = max(0.0, (count_right - widest_count - count_gap if show_counts else count_right) - label_x)
        text_y_off = round((row_h - line_h) / 2)

        text_color = _alpha_mul(colors["text"], 0.9 * master_alpha)
        count_color = _alpha_mul(colors["text"], 0.3 * master_alpha)
        selection_color = _alpha_mul(colors["node_selected"], 0.95 * master_alpha)
        active_color = _alpha_mul(colors["indicator"], master_alpha)
        # Matched substring in search results; distinct from selection/active so
        # it still reads against those row states.
        match_color = _alpha_mul(colors["indicator"], 0.85 * master_alpha)
        # Light fill under selected/active rows; selection color at low alpha (drawn
        # before text so BLF's alpha-disable cannot clobber the SDF fill blend).
        selection_fill_color = _alpha_mul(colors["node_selected"], 0.05 * master_alpha)
        active_fill_color = _alpha_mul(colors["indicator"], 0.05 * master_alpha)

        # Per-type selection state (compiled) drives font (not icon) recoloring.
        type_colors = tree_data.get("type_colors") or {}
        type_node_colors = tree_data.get("type_node_colors") or {}
        type_selected_counts = tree_data.get("type_selected_counts") or {}
        type_active = tree_data.get("type_active_label")
        hover_color = _alpha_mul(colors["text"], 0.025 * master_alpha)

        pill_x = zone_x + 2 * ui_scale
        pill_w = zone_w - 4 * ui_scale
        state.list.search_rect = (pill_x, search_bottom, pill_w, search_h)

        search_text_x = zone_x + pad_x

        # Clear (X) button at the pill's right edge; shown only while a filter
        # query is present so the query text can reclaim the full row width.
        if state.list.search_query:
            clear_size = max(int(12 * ui_scale), int(search_h * 0.6))
            clear_x = pill_x + pill_w - clear_size - 4 * ui_scale
            clear_y = round(search_bottom - 0.5 + (search_h - clear_size) / 2)
            state.list.search_clear_rect = (clear_x, clear_y, clear_size, clear_size)
        else:
            state.list.search_clear_rect = None
            state.list.search_clear_hovered = False

    # Clip rows to the view interior (below the fixed search row) so scrolled
    # content can never draw over/behind the search box.
    saved_scissor = None
    try:
        was_active = gpu.state.scissor_test_get()
        saved_scissor = (was_active, gpu.state.scissor_get() if was_active else None)
    except Exception:
        saved_scissor = None
    zone_scissor = (
        int(zone_x + 1),
        int(zone_y + 1),
        max(0, int(zone_w - 2)),
        max(0, int(zone_h - 2)),
    )
    view_scissor = (
        int(zone_x + 1),
        int(view_bottom),
        max(0, int(zone_w - 2)),
        max(0, int(view_top - view_bottom)),
    )
    try:
        gpu.state.scissor_set(*view_scissor)
        gpu.state.scissor_test_set(True)
        gpu.state.blend_set("ALPHA")

        with _Timer("type_list.rows"):
            entry_map = {lbl: (ct, cw, cnt) for (lbl, ct, cw, cnt) in entries}
            hovered = state.list.hovered_type_label
            hovered_child = state.list.hovered_list_row

            visible_rows = []
            for row_idx, (kind, label, node_name, local_y) in enumerate(
                _iter_type_list_layout(entries, children, expanded, row_h)
            ):
                slot_top = view_top + state.list.scroll + local_y
                slot_bottom = slot_top - row_h
                if slot_top <= view_bottom or slot_bottom >= view_top:
                    continue
                visible_rows.append((kind, label, node_name, slot_top, slot_bottom, row_idx))

            state.list.visible_row_keys = [
                ("header", label) if kind == "header" else ("child", label, node_name)
                for kind, label, node_name, *_rest in visible_rows
            ]
            state.list.visible_row_index_map = {key: idx for idx, key in enumerate(state.list.visible_row_keys)}

            # Labels with any visible row (a child row carries its type label, so a
            # header scrolled out of view while its children remain still registers).
            header_has_visible = {label for _kind, label, *_rest in visible_rows}

        with _Timer("type_list.pills"):
            # Zebra bands keyed on the absolute layout index so they stay attached
            # to rows while scrolling; drawn beneath pills, text, and icons.
            band_color = (1.0, 1.0, 1.0, 0.002 * master_alpha)

            # Fixed search row above the scrolled content; focused state reuses
            # the active-row styling (light fill + selection border).
            if settings.show_search_bar:
                search_pill_y = search_bottom + row_gap_half
                gpu.state.scissor_set(*zone_scissor)
                if state.list.search_focused:
                    _draw_filled_rounded_rect(
                        pill_x,
                        search_pill_y,
                        pill_w,
                        search_draw_h,
                        4.0 * ui_scale,
                        (0, 0, 0, 0.6 * master_alpha),
                    )
                    # _draw_rounded_rect_border(
                    #     pill_x,
                    #     search_pill_y,
                    #     pill_w,
                    #     search_draw_h,
                    #     4.0 * ui_scale,
                    #     _alpha_mul(colors["node_active"], master_alpha),
                    #     0.5 * ui_scale,
                    # )
                else:
                    _draw_filled_rounded_rect(
                        pill_x,
                        search_pill_y,
                        pill_w,
                        search_draw_h,
                        4.0 * ui_scale,
                        (0.0, 0.0, 0.0, 0.3 * master_alpha),
                    )
                _draw_rounded_rect_border(
                    pill_x,
                    search_pill_y,
                    pill_w,
                    search_draw_h,
                    4.0 * ui_scale,
                    _alpha_mul(colors["bg_border"], master_alpha),
                    0.5,
                )
                gpu.state.scissor_set(*view_scissor)
            for _kind, _label, _node_name, slot_top, slot_bottom, row_idx in visible_rows:
                if row_idx % 2 == 1:
                    _draw_filled_rounded_rect(pill_x, _row_y(slot_bottom), pill_w, row_draw_h, 0.0, band_color)

            header_rects = []
            toggle_rects = {}
            for kind, label, _node_name, _slot_top, slot_bottom, _row_idx in visible_rows:
                if kind != "header":
                    continue

                if label == type_active:
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(slot_bottom), pill_w, row_draw_h, 4.0 * ui_scale, active_fill_color
                    )
                elif type_selected_counts.get(label, 0) > 0:
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(slot_bottom), pill_w, row_draw_h, 4.0 * ui_scale, selection_fill_color
                    )

                if hovered == label:
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(slot_bottom), pill_w, row_draw_h, 4.0 * ui_scale, hover_color
                    )

                if label == type_active:
                    # Active outline drawn here (pre-text) so BLF cannot clobber the
                    # alpha blend state the SDF fill relies on.
                    _draw_rounded_rect_border(
                        pill_x, _row_y(slot_bottom), pill_w, row_draw_h, 4.0 * ui_scale, selection_color, 0.5 * ui_scale
                    )
                header_rects.append((pill_x, slot_bottom, pill_w, row_h, label))
                if entry_map.get(label, ("", 0.0, 1))[2] > 1:
                    toggle_rects[label] = (content_x, slot_bottom, swatch + swatch_gap, row_h)
            state.list.row_rects = header_rects
            state.list.toggle_rects = toggle_rects

            # Expand guide lines. Drawn for every expanded header whose content has
            # any visible row, independent of the header itself being on screen: a
            # header scrolled out of view still connects its visible children, so the
            # vertical line stays under them instead of vanishing. The zone scissor
            # clips the full-height guide to the visible interior.
            for label in header_has_visible:
                if label not in expanded:
                    continue
                child_count = len(children.get(label, ()))
                if child_count <= 0:
                    continue
                header_color = _header_type_color(label, children, type_node_colors, type_colors, colors)
                line_color = (
                    _alpha_mul(header_color, master_alpha)
                    if show_type_colors
                    else _alpha_mul(colors["text"], 0.1 * master_alpha)
                )
                _draw_expand_guide_line(
                    round(content_x + swatch / 2),
                    header_slot_bottom[label] - 1,
                    child_count * row_h - 2,
                    ui_scale,
                    line_color,
                )

            active_node = node_tree.nodes.active if node_tree else None

            child_rects = []
            for kind, label, node_name, _slot_top, slot_bottom, _row_idx in visible_rows:
                if kind != "child":
                    continue
                child_selected = False
                child_active = False
                try:
                    node = nodes_by_name.get(node_name) if nodes_by_name else None
                    child_active = bool(active_node and node == active_node)
                    child_selected = bool(node and node.select)
                except Exception:
                    node = None

                if child_active:
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(slot_bottom), pill_w, row_draw_h, 4.0 * ui_scale, active_fill_color
                    )
                elif child_selected:
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(slot_bottom), pill_w, row_draw_h, 4.0 * ui_scale, selection_fill_color
                    )
                if child_active:
                    _draw_rounded_rect_border(
                        pill_x, _row_y(slot_bottom), pill_w, row_draw_h, 4.0 * ui_scale, selection_color, 0.5 * ui_scale
                    )

                if hovered_child == (label, node_name):
                    _draw_filled_rounded_rect(
                        pill_x, _row_y(slot_bottom), pill_w, row_draw_h, 4.0 * ui_scale, hover_color
                    )

                child_rects.append((pill_x, slot_bottom, pill_w, row_h, label, node_name))
            state.list.node_rects = child_rects

        # Baked swatches/chevrons draw first so text renders above them; the
        # batch only moves via a matrix translate, so this is a single draw.
        with _Timer("type_list.text"):
            gpu.state.blend_set("ALPHA")
            _draw_list_glyph_batch(state, zone_x, view_top + state.list.scroll)

            font_id = TYPE_LIST_FONT_ID
            blf.size(font_id, font_size)
            with_shadow = settings.show_text_shadow
            if with_shadow:
                blf.enable(font_id, blf.SHADOW)
                blf.shadow(font_id, 3, 0, 0, 0, 255)
                blf.shadow_offset(font_id, 0, -1)

            # Child rows indent by one column relative to headers.
            child_indent = icon_col_x
            child_label_x = label_x + child_indent
            # Child rows never draw a count, so their names always run to the
            # right edge instead of reserving the header count column.
            child_label_max_width = max(0.0, count_right - child_label_x)
            child_clip_left = int(child_label_x)
            child_clip_right = int(child_label_x + child_label_max_width)
            clip_top = int(zone_y - row_h)
            clip_bottom = int(zone_y + zone_h + row_h)
            header_clip_left = int(label_x)
            header_clip_right = int(label_x + label_max_width)
            if show_counts:
                count_clip_right = int(count_right + widest_count)

            # CLIPPING stays enabled for the whole pass; per-row only the box
            # changes (labels, counts, and child names use distinct boxes).
            blf.enable(font_id, blf.CLIPPING)

            # Fixed search row: placeholder or query text plus a static caret
            # while focused (no blink timer, to avoid a permanent redraw loop).
            search_query = state.list.search_query
            norm_query = normalize_query(search_query)
            if settings.show_search_bar:
                search_cursor = min(max(state.list.search_cursor, 0), len(search_query))
                search_text_y = search_pill_y + search_pad_v
                # The query text/caret stop short of the clear button so they never
                # run underneath it.
                clear_rect = state.list.search_clear_rect
                search_text_right = clear_rect[0] - 2 * ui_scale if clear_rect else count_right

                # Search-box glyphs (caret, placeholder/query text, clear button) use
                # the full zone scissor since they sit above the scrolled content.
                gpu.state.scissor_set(*zone_scissor)

                if state.list.search_focused:
                    caret_x = round(
                        search_text_x
                        + (blf.dimensions(font_id, search_query[:search_cursor])[0] if search_query else 0.0)
                    )
                    _draw_filled_rounded_rect(
                        caret_x,
                        round(search_pill_y),
                        max(2.0, 2.2 * ui_scale),
                        search_draw_h,
                        0.0,
                        _alpha_mul(colors["indicator"], master_alpha),
                    )
                search_text = search_query if search_query else "Filter"
                blf.clipping(font_id, int(search_text_x), clip_top, int(search_text_right), clip_bottom)
                blf.position(font_id, search_text_x, search_text_y, 0)
                blf.color(font_id, *(text_color if search_query else count_color))
                blf.draw(font_id, search_text)
                if clear_rect:
                    _draw_search_clear_button(
                        clear_rect, state.list.search_clear_hovered, colors, master_alpha, ui_scale
                    )
                gpu.state.scissor_set(*view_scissor)
                if not entries and search_query:
                    no_match_y = (view_bottom + view_top - line_h) / 2 + 1
                    blf.clipping(
                        font_id,
                        int(search_text_x),
                        int(view_bottom - row_h),
                        int(count_right),
                        int(view_top + row_h),
                    )
                    blf.position(font_id, search_text_x, no_match_y, 0)
                    blf.color(font_id, *count_color)
                    blf.draw(font_id, "No matches")

            for kind, label, node_name, _slot_top, slot_bottom, _row_idx in visible_rows:
                text_y = _row_y(slot_bottom) + text_y_off
                if kind == "header":
                    # Icons keep the type color; selection/active state shows in the
                    # row text color instead (active brightest, then selected).
                    is_active = label == type_active
                    is_sel = type_selected_counts.get(label, 0) > 0
                    if is_active:
                        label_color = active_color
                    elif is_sel:
                        label_color = selection_color
                    else:
                        label_color = text_color

                    header_text = _group_header_text(
                        label, entry_map.get(label, ("", 0.0, 1))[2], children, nodes_by_name
                    )
                    blf.clipping(font_id, header_clip_left, clip_top, header_clip_right, clip_bottom)
                    _draw_text_with_match(font_id, label_x, text_y, header_text, label_color, match_color, norm_query)
                    # Counts sit right of the label clip box; BLF discards glyphs
                    # past the box instead of clipping them.
                    if show_counts:
                        count_text, count_width, _cnt = entry_map.get(label, ("", 0.0, 1))
                        blf.clipping(font_id, header_clip_right, clip_top, count_clip_right, clip_bottom)
                        blf.position(font_id, count_right - count_width, text_y, 0)
                        blf.color(font_id, *count_color)
                        blf.draw(font_id, count_text)
                else:
                    # Child row: the type-colored swatch is baked; text shows the
                    # selection state.
                    is_active = False
                    is_sel = False
                    try:
                        node = nodes_by_name.get(node_name) if nodes_by_name else None
                        is_active = bool(active_node and node == active_node)
                        is_sel = bool(node and node.select)
                    except Exception:
                        node = None
                    if is_active:
                        label_color = active_color
                    elif is_sel:
                        label_color = selection_color
                    else:
                        label_color = text_color

                    blf.clipping(font_id, child_clip_left, clip_top, child_clip_right, clip_bottom)
                    label_text = _child_label_text(node_name, node)
                    _draw_text_with_match(
                        font_id, child_label_x, text_y, label_text, label_color, match_color, norm_query
                    )
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

    with _Timer("type_list.scrollbar"):
        state.list.scrollbar_thumb = None
        state.list.scrollbar_track = None
        if scroll_max > 0:
            gpu.state.blend_set("ALPHA")
            bar_thickness, bar_offset = _get_scrollbar_style(ui_scale)
            frac = state.list.scroll / scroll_max
            # Hover and drag share one expanded look (Blender overlay style).
            active = state.list.hovered_scrollbar or state.list.scrollbar_dragging
            thick = _scrollbar_thickness(ui_scale, active)
            thumb_rect, track_rect = _draw_scrollbar_thumb(
                round(zone_x + zone_w - thick - bar_offset),
                zone_y + bar_offset,
                max(view_top - zone_y - 2 * bar_offset, 0.0),
                view_h / total_h,
                1.0 - frac,
                colors,
                master_alpha,
                ui_scale,
                active=active,
            )
            state.list.scrollbar_thumb = thumb_rect
            state.list.scrollbar_track = track_rect


def _draw_expand_guide_line(x: float, top: float, height: float, ui_scale: float, color) -> None:
    """Draw a vertical guide under the expand icon, spanning the child rows."""
    t = max(1.0, 1.0 * ui_scale)
    _draw_filled_rounded_rect(x - t / 2, top - height, t, height, t / 2, color)
