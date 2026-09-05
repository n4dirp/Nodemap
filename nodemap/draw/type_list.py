"""Provide an interactive node-type list zone and scrollbar drawing."""

import math
import time

import blf
import bpy
import gpu
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix

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
from ..core.list_filter import (
    filter_type_list,
    match_span,
    normalize_query,
)
from ..core.state import MinimapState
from ..core.theme import (
    _COLOR_TAG_TO_THEME_ATTR,
    _alpha_mul,
    _srgb_to_linear,
    _theme_rgba,
)
from .batch_build import _create_quad_indices
from .gpu_draw import (
    _draw_filled_quad,
    _draw_filled_rounded_rect,
    _draw_pill,
    _draw_rounded_rect_border,
    _get_batch_rect_shader,
)

_ROW_HEADER = "header"
_ROW_CHILD = "child"
_DEFAULT_ENTRY = ("", 0.0, 1)


# ---------------------------------------------------------------------------
# Scrollbars
# ---------------------------------------------------------------------------


def _get_scrollbar_style(ui_scale: float) -> tuple[int, int]:
    """Return the shared scrollbar `(thickness, inset)` scaled for the UI."""
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
    """Draw a scrollbar thumb pill and return `(thumb_rect, track_rect)`.

    ``visible_frac`` sizes the thumb, ``pos_frac`` slides it along the track.
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
    """Draw horizontal/vertical minimap scrollbar thumbs when zoomed in."""
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

    tree_l = tree_center_x + (inner_l - map_anchor_x) / scale
    tree_r = tree_center_x + (inner_r - map_anchor_x) / scale
    tree_b = tree_center_y + (inner_b - map_anchor_y) / scale
    tree_t = tree_center_y + (inner_t - map_anchor_y) / scale

    v_left = max(bbox_l, min(bbox_r, tree_l))
    v_right = max(bbox_l, min(bbox_r, tree_r))
    v_bottom = max(bbox_b, min(bbox_t, tree_b))
    v_top = max(bbox_b, min(bbox_t, tree_t))

    visible_w = v_right - v_left
    visible_h = v_top - v_bottom

    if visible_w >= bbox_w and visible_h >= bbox_h:
        return

    bar_thickness, bar_offset = _get_scrollbar_style(ui_scale)

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


# ---------------------------------------------------------------------------
# Type-list width animation
# ---------------------------------------------------------------------------


def _maybe_preserve_view_for_list_width(
    state: MinimapState,
    current_width: float,
    new_width: float,
    ui_scale: float,
) -> None:
    """Preserve view alignment when the list width changes meaningfully."""
    if abs(new_width - current_width) >= 0.5:
        from ..geo.transforms import _preserve_view_for_list_width

        _preserve_view_for_list_width(state, current_width, new_width, ui_scale)


def _step_list_width(state: MinimapState, settings, map_w: float, ui_scale: float) -> None:
    """Advance the animated type-list zone width for this frame."""
    list_font_size = settings.type_list_font_size
    target_width = _get_type_list_width(settings, state, map_w, ui_scale, list_font_size)

    # During an interactive width drag, the live pixel width wins.
    if state.list.dragging_width is not None:
        _maybe_preserve_view_for_list_width(
            state,
            state.list.list_width,
            state.list.dragging_width,
            ui_scale,
        )
        state.list.list_width = state.list.dragging_width
        return

    if not state.list.anim_active:
        _maybe_preserve_view_for_list_width(
            state,
            state.list.list_width,
            target_width,
            ui_scale,
        )
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

    progress = min(
        (time.perf_counter() - state.list.anim_start) / max(state.list.anim_duration, 1e-4),
        1.0,
    )
    eased = 1.0 - (1.0 - progress) ** 3
    new_width = state.list.anim_from + (state.list.anim_target - state.list.anim_from) * eased

    if progress >= 1.0:
        new_width = state.list.anim_target

    _maybe_preserve_view_for_list_width(
        state,
        state.list.list_width,
        new_width,
        ui_scale,
    )
    state.list.list_width = new_width

    if progress >= 1.0:
        state.list.anim_active = False
        # The list animation is over; drop the batch key so the next draw
        # rebuilds at the final scale immediately instead of waiting the
        # zoom settle window.
        # state.cache.batch_key = None
    else:
        _schedule_list_anim_redraw(state)


# ---------------------------------------------------------------------------
# Cache build / bake
# ---------------------------------------------------------------------------


def _type_list_cache_key(
    state: MinimapState,
    settings,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> tuple:
    """Return the invalidation key for cached layout and swatch batch."""
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
    state: MinimapState,
    settings,
    node_tree,
    key: tuple,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> None:
    """Build cached entries, row layout, entry map, and baked glyph batch."""
    tree_data = state.cache.tree_data or {}
    type_stats = tree_data.get("type_stats") or {}

    state.cache.list_nodes_by_name = {n.name: n for n in node_tree.nodes} if node_tree else {}

    font_id = TYPE_LIST_FONT_ID
    font_size = int(settings.type_list_font_size * ui_scale)
    blf.size(font_id, font_size)

    children = tree_data.get("type_nodes") or {}
    search_texts = tree_data.get("type_search") or None

    visible, effective_expanded, filtered_children = filter_type_list(
        type_stats,
        children,
        state.list.expanded,
        state.list.search_query,
        search_texts=search_texts,
    )

    effective_expanded = effective_expanded or set()
    filtered_children = filtered_children or {}

    # While searching, preserve filter/match order.
    if not state.list.search_query.strip():
        if settings.type_list_sort == "NAME":
            visible.sort(key=lambda label_count: label_count[0].lower())
        else:
            visible.sort(key=lambda label_count: (-label_count[1], label_count[0]))

    entries: list[tuple[str, str, float, int]] = []
    entry_map: dict[str, tuple[str, float, int]] = {}
    widest_count = 0.0

    for label, display_count in visible:
        count_text = str(display_count)
        count_width = blf.dimensions(font_id, count_text)[0]
        widest_count = max(widest_count, count_width)

        full_count = type_stats.get(label, display_count)
        entries.append((label, count_text, count_width, full_count))
        entry_map[label] = (count_text, count_width, full_count)

    _, line_h = blf.dimensions(font_id, "Ay")
    row_h = line_h + 4 * ui_scale

    # Build the canonical row layout once.
    rows = tuple(
        _iter_type_list_layout(
            entries,
            filtered_children,
            effective_expanded,
            row_h,
        )
    )

    header_local_bottom = {label: local_y - row_h for kind, label, _node_name, local_y in rows if kind == _ROW_HEADER}

    state.cache.list_key = key
    state.cache.list_entries = entries
    state.cache.list_effective_expanded = effective_expanded
    state.cache.list_children = children
    state.cache.list_filtered_children = filtered_children

    state.cache.list_layout = {
        "font_size": font_size,
        "line_h": line_h,
        "row_h": row_h,
        "widest_count": widest_count,
        "rows": rows,
        "total_h": len(rows) * row_h,
        "header_local_bottom": header_local_bottom,
        "entry_map": entry_map,
    }

    state.cache.list_swatches_batch = None

    _bake_list_glyph_batch(
        state,
        settings,
        colors,
        master_alpha,
        ui_scale,
        row_h,
        rows,
        entry_map,
    )


def _bake_list_glyph_batch(
    state: MinimapState,
    settings,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    row_h: float,
    rows: tuple,
    entry_map: dict,
) -> None:
    """Bake static swatches and expand chevrons into one batched rect pass."""
    type_colors = (state.cache.tree_data or {}).get("type_colors") or {}
    type_node_colors = (state.cache.tree_data or {}).get("type_node_colors") or {}
    children = state.cache.list_filtered_children or state.cache.list_children or {}

    show_type_colors = settings.show_type_colors and settings.show_node_colors

    expanded = getattr(state.cache, "list_effective_expanded", None)
    if expanded is None:
        expanded = state.list.expanded

    pad_x = LIST_PAD_X * ui_scale
    swatch = LIST_SWATCH * ui_scale
    swatch_gap = LIST_SWATCH_GAP * ui_scale
    icon_col_x = swatch + swatch_gap
    swatch_col_x = icon_col_x if show_type_colors else 0.0

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

        half_w = (arm + bar_thickness / 2.0) / 2.0
        half_h = bar_thickness / 2.0

        local = (
            (-arm, -half_h),
            (bar_thickness / 2.0, -half_h),
            (bar_thickness / 2.0, half_h),
            (-arm, half_h),
        )

        uvs = (
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        )

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

    for kind, label, node_name, local_y in rows:
        x = pad_x

        if kind == _ROW_HEADER:
            if entry_map.get(label, _DEFAULT_ENTRY)[2] > 1:
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
                header_color = _header_type_color(
                    label,
                    children,
                    type_node_colors,
                    type_colors,
                    colors,
                )

                _push_quad(
                    x + icon_col_x,
                    round(local_y - (row_h + swatch) / 2.0),
                    swatch,
                    swatch,
                    swatch / 2.0,
                    _alpha_mul(header_color, master_alpha),
                )

        else:
            if show_type_colors:
                node_color = type_node_colors.get(label, {}).get(
                    node_name,
                    type_colors.get(label, colors["node"]),
                )

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
        {
            "pos": pos,
            "uv": uv,
            "halfSize": half_size,
            "radius": radius,
            "color": vertex_color,
        },
        indices=_create_quad_indices(quads),
    )


def _draw_list_glyph_batch(state: MinimapState, zone_x: float, view_y: float) -> None:
    """Draw the baked swatch/chevron batch under a `(zone_x, view_y)` translate."""
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


# ---------------------------------------------------------------------------
# Pure label / layout / text helpers
# ---------------------------------------------------------------------------


def _header_type_color(
    label: str,
    children: dict,
    node_colors: dict,
    type_colors: dict,
    colors: dict,
) -> tuple:
    """Return the header swatch color: first child node color, then fallbacks."""
    child_names = children.get(label) or ()
    if child_names:
        first_color = node_colors.get(label, {}).get(child_names[0])
        if first_color is not None:
            return first_color

    return type_colors.get(label, colors["node"])


def _group_header_text(label: str, count: int, children: dict, nodes_by_name: dict) -> str:
    """Return header text, appending `(label)` for a lone group node."""
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
    """Return child row text: `name (label)` or group tree-name fallback."""
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
    """Yield `(kind, label, node_name, local_y_top)` for each list row."""
    y = 0.0

    for label, _count_text, _count_w, count in entries:
        yield (_ROW_HEADER, label, None, y)
        y -= row_h

        if count > 1 and label in expanded:
            for node_name in children.get(label, ()):
                yield (_ROW_CHILD, label, node_name, y)
                y -= row_h


def _draw_expand_guide_line(x: float, top: float, height: float, ui_scale: float, color) -> None:
    """Draw a vertical guide under the expand icon, spanning child rows."""
    t = max(1.0, 1.0 * ui_scale)
    _draw_filled_rounded_rect(x - t / 2, top - height, t, height, t / 2, color)


def _draw_text_with_match(
    font_id,
    x: float,
    y: float,
    text: str,
    base_color,
    match_color,
    norm_query: str,
) -> None:
    """Draw text with the first normalized-query match highlighted."""
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


def _draw_search_clear_button(
    rect,
    hovered: bool,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> None:
    """Draw the clear-query (X) button inside the search pill."""
    clear_x, clear_y, clear_size, _ = rect

    icon_color = _alpha_mul(
        colors["text"],
        0.35 * master_alpha if not hovered else master_alpha,
    )

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
                _draw_filled_rounded_rect(
                    -length / 2,
                    -stroke / 2,
                    length,
                    stroke,
                    0.0,
                    icon_color,
                )
            finally:
                gpu.matrix.pop()
    finally:
        gpu.matrix.pop()


def _draw_search_filter_icon(
    x: float,
    y: float,
    size: float,
    color,
    ui_scale: float,
) -> None:
    """Draw a filled filter (funnel) icon for the search pill."""
    cx = round(x + size / 2)
    cy = round(y + size / 2)

    # Overall icon box
    w = size * 0.8  # width of the rim / funnel mouth
    h = size * 0.85  # overall height

    half_w = w / 2
    top_y = -h / 2
    bottom_y = h / 2

    # Vertical proportions (fractions of total height), mirrored vertically
    rim_h = h * 0.13  # thickness of the bottom rim
    rim_top_y = bottom_y - rim_h
    point_y = bottom_y - h * 0.62  # where the body meets the stem
    stem_w = max(2.0 * ui_scale, w * 0.15)  # width of the stem
    stem_radius = min(stem_w / 2, h * 0.03)

    gpu.matrix.push()
    try:
        gpu.matrix.translate((cx, cy, 0.0))

        _draw_filled_rounded_rect(-half_w, rim_top_y, w, rim_h, rim_h / 2, color)

        _draw_filled_quad(
            (-stem_w / 2, point_y), (stem_w / 2, point_y), (half_w, rim_top_y), (-half_w, rim_top_y), color
        )

        _draw_filled_rounded_rect(-stem_w / 2, top_y, stem_w, point_y - top_y, stem_radius, color)
    finally:
        gpu.matrix.pop()


# ---------------------------------------------------------------------------
# Type-list draw pipeline: geometry, fills, text, scrollbar
# ---------------------------------------------------------------------------


def _clear_list_interaction(state: MinimapState) -> None:
    """Reset hover, scroll, search, and row visibility state."""
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
    state.list.visible_row_keys = []
    state.list.visible_row_index_map = {}


def _resolve_child_state(nodes_by_name, node_name, active_node):
    """Return `(node, is_active, is_selected)` for a child row."""
    try:
        node = nodes_by_name.get(node_name) if nodes_by_name else None
        is_active = bool(active_node and node == active_node)
        is_selected = bool(node and node.select)
    except Exception:
        node = None
        is_active = False
        is_selected = False

    return node, is_active, is_selected


def _compute_zone_geometry(
    state: MinimapState,
    settings,
    layout: dict,
    expanded: set,
    row_gap: float,
    row_gap_half: float,
    row_draw_h: float,
    map_x: float,
    map_y: float,
    map_h: float,
    padding: float,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
) -> dict:
    """Compute list zone metrics and draw the zone background."""
    font_size = layout["font_size"]
    row_h = layout["row_h"]
    line_h = layout["line_h"]
    widest_count = layout["widest_count"]
    total_h = layout["total_h"]
    header_local_bottom = layout["header_local_bottom"]

    pad_x = LIST_PAD_X * ui_scale
    swatch = LIST_SWATCH * ui_scale
    swatch_gap = LIST_SWATCH_GAP * ui_scale
    count_gap = LIST_COUNT_GAP * ui_scale
    icon_col_x = swatch + swatch_gap
    row_pad_v = ui_scale

    children = state.cache.list_filtered_children or state.cache.list_children or {}

    handle_pad = HANDLE_THICKNESS * ui_scale
    zone_x = map_x + handle_pad
    zone_w = map_x + padding + state.list.list_width - 2 * ui_scale - zone_x

    if settings.show_search_bar:
        search_h = (BUTTON_SIZE - 1) * ui_scale
    else:
        search_h = 0.0
        state.list.search_focused = False
        state.list.search_clear_rect = None
        state.list.search_clear_hovered = False

    zone_h = min(
        map_h - 2 * handle_pad,
        max(total_h, row_h) + search_h + 3 * row_pad_v,
    )

    zone_y = round(map_y + map_h - zone_h - handle_pad)
    state.list.list_zone_rect = (zone_x, zone_y, zone_w, zone_h)

    search_top = zone_y + zone_h - 1
    search_bottom = search_top - search_h
    search_draw_h = max(0.0, search_h - row_gap)
    search_pad_v = round((search_draw_h - line_h) / 2.0) + 1 if search_h > 0 else 0

    zone_radius = colors.get("panel_roundness", 4.0) * 0.6

    _draw_filled_rounded_rect(
        zone_x,
        zone_y,
        zone_w,
        zone_h,
        zone_radius,
        _alpha_mul(colors["bg"], master_alpha),
    )

    _draw_rounded_rect_border(
        zone_x,
        zone_y,
        zone_w,
        zone_h,
        zone_radius,
        _alpha_mul(colors["bg_border"], master_alpha),
        0.5,
    )

    view_top = search_bottom - row_pad_v + 1
    view_bottom = zone_y + row_pad_v + 1
    view_h = max(view_top - view_bottom, row_h)

    scroll_max = max(0.0, total_h - view_h)
    state.list.scroll = min(max(state.list.scroll, 0.0), scroll_max)
    state.list.scroll_max = scroll_max

    header_slot_bottom = {
        label: view_top + state.list.scroll + local_bottom for label, local_bottom in header_local_bottom.items()
    }

    show_type_colors = settings.show_type_colors and settings.show_node_colors
    swatch_col_x = icon_col_x if show_type_colors else 0.0
    content_x = zone_x + pad_x

    count_right = zone_x + zone_w - pad_x - 4 * ui_scale
    label_x = content_x + icon_col_x + swatch_col_x

    show_counts = count_right - widest_count - count_gap - label_x >= TYPE_LIST_MIN_LABEL_W * ui_scale

    label_max_width = max(
        0.0,
        (count_right - widest_count - count_gap if show_counts else count_right) - label_x,
    )

    text_y_off = round((row_h - line_h) / 2)

    text_color = _alpha_mul(colors["text"], 0.9 * master_alpha)
    count_color = _alpha_mul(colors["text"], 0.3 * master_alpha)
    selection_color = _alpha_mul(colors["node_selected"], 0.95 * master_alpha)
    active_color = _alpha_mul(colors["indicator"], master_alpha)
    match_color = _alpha_mul(colors["indicator"], 0.85 * master_alpha)

    selection_fill_color = _alpha_mul(colors["viewport_fill"], 0.2 * master_alpha)
    active_fill_color = _alpha_mul(colors["viewport_fill"], 0.4 * master_alpha)
    active_border_color = colors["viewport_fill"]

    tree_data = state.cache.tree_data or {}
    type_colors = tree_data.get("type_colors") or {}
    type_node_colors = tree_data.get("type_node_colors") or {}
    type_selected_counts = tree_data.get("type_selected_counts") or {}
    type_active = tree_data.get("type_active_label")

    hover_color = _alpha_mul(colors["text"], 0.025 * master_alpha)

    pill_x = zone_x + 2 * ui_scale
    pill_w = zone_w - 4 * ui_scale

    if settings.show_search_bar:
        state.list.search_rect = (pill_x, search_bottom, pill_w, search_h)
    else:
        state.list.search_rect = None

    search_text_x = zone_x + pad_x

    if settings.show_search_bar and state.list.search_query:
        clear_size = max(int(12 * ui_scale), int(search_h * 0.6))
        clear_x = pill_x + pill_w - clear_size - 4 * ui_scale
        clear_y = round(search_bottom - 0.5 + (search_h - clear_size) / 2)
        state.list.search_clear_rect = (clear_x, clear_y, clear_size, clear_size)
    else:
        state.list.search_clear_rect = None
        state.list.search_clear_hovered = False

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

    return {
        "font_size": font_size,
        "row_h": row_h,
        "line_h": line_h,
        "widest_count": widest_count,
        "expanded": expanded,
        "row_draw_h": row_draw_h,
        "swatch": swatch,
        "swatch_gap": swatch_gap,
        "icon_col_x": icon_col_x,
        "children": children,
        "total_h": total_h,
        "zone_x": zone_x,
        "zone_w": zone_w,
        "zone_y": zone_y,
        "zone_h": zone_h,
        "search_draw_h": search_draw_h,
        "search_pad_v": search_pad_v,
        "search_pill_y": search_bottom + row_gap_half,
        "view_top": view_top,
        "view_bottom": view_bottom,
        "view_h": view_h,
        "scroll_max": scroll_max,
        "header_slot_bottom": header_slot_bottom,
        "show_type_colors": show_type_colors,
        "content_x": content_x,
        "count_right": count_right,
        "label_x": label_x,
        "label_max_width": label_max_width,
        "show_counts": show_counts,
        "text_y_off": text_y_off,
        "text_color": text_color,
        "count_color": count_color,
        "selection_color": selection_color,
        "active_color": active_color,
        "match_color": match_color,
        "selection_fill_color": selection_fill_color,
        "active_fill_color": active_fill_color,
        "active_border_color": active_border_color,
        "type_colors": type_colors,
        "type_node_colors": type_node_colors,
        "type_selected_counts": type_selected_counts,
        "type_active": type_active,
        "hover_color": hover_color,
        "pill_x": pill_x,
        "pill_w": pill_w,
        "search_text_x": search_text_x,
        "zone_scissor": zone_scissor,
        "view_scissor": view_scissor,
    }


def _draw_list_fills(
    state: MinimapState,
    settings,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    geo: dict,
    visible_rows: list,
    entry_map: dict,
    header_has_visible: set,
) -> None:
    """Draw search pill, zebra bands, row fills, guides, and hit rects."""
    pill_x = geo["pill_x"]
    pill_w = geo["pill_w"]
    row_draw_h = geo["row_draw_h"]
    row_h = geo["row_h"]

    active_fill_color = geo["active_fill_color"]
    active_border_color = geo["active_border_color"]

    selection_fill_color = geo["selection_fill_color"]
    hover_color = geo["hover_color"]

    type_active = geo["type_active"]
    type_selected_counts = geo["type_selected_counts"]

    content_x = geo["content_x"]
    swatch = geo["swatch"]
    swatch_gap = geo["swatch_gap"]

    children = geo["children"]
    expanded = geo["expanded"]

    type_colors = geo["type_colors"]
    type_node_colors = geo["type_node_colors"]
    show_type_colors = geo["show_type_colors"]

    zone_scissor = geo["zone_scissor"]
    view_scissor = geo["view_scissor"]

    search_pill_y = geo["search_pill_y"]
    search_draw_h = geo["search_draw_h"]
    header_slot_bottom = geo["header_slot_bottom"]

    fill = _draw_filled_rounded_rect
    border = _draw_rounded_rect_border
    guide = _draw_expand_guide_line
    header_color = _header_type_color

    radius = 4.0 * ui_scale
    active_border_w = 0.5 * ui_scale

    band_color = (1.0, 1.0, 1.0, 0.002 * master_alpha)

    hovered = state.list.hovered_type_label
    hovered_child = state.list.hovered_list_row

    if settings.show_search_bar:
        gpu.state.scissor_set(*zone_scissor)

        if state.list.search_focused:
            fill(
                pill_x,
                search_pill_y,
                pill_w,
                search_draw_h,
                radius,
                (0.0, 0.0, 0.0, 0.6 * master_alpha),
            )
        else:
            fill(
                pill_x,
                search_pill_y,
                pill_w,
                search_draw_h,
                radius,
                (0.0, 0.0, 0.0, 0.3 * master_alpha),
            )

        border(
            pill_x,
            search_pill_y,
            pill_w,
            search_draw_h,
            radius,
            _alpha_mul(colors["bg_border"], master_alpha),
            0.5,
        )

        gpu.state.scissor_set(*view_scissor)

    # Zebra bands.
    for (
        _kind,
        _label,
        _node_name,
        _slot_bottom,
        row_idx,
        draw_y,
        _node,
        _child_active,
        _child_selected,
    ) in visible_rows:
        if row_idx & 1:
            fill(pill_x, draw_y, pill_w, row_draw_h, 0.0, band_color)

    # Header fills, outlines, and hit rects.
    header_rects = []
    toggle_rects = {}

    for (
        kind,
        label,
        _node_name,
        slot_bottom,
        _row_idx,
        draw_y,
        _node,
        _child_active,
        _child_selected,
    ) in visible_rows:
        if kind != _ROW_HEADER:
            continue

        is_active = label == type_active

        if is_active:
            fill(pill_x, draw_y, pill_w, row_draw_h, radius, active_fill_color)
        elif type_selected_counts.get(label, 0) > 0:
            fill(pill_x, draw_y, pill_w, row_draw_h, radius, selection_fill_color)

        if hovered == label:
            fill(pill_x, draw_y, pill_w, row_draw_h, radius, hover_color)

        if is_active:
            border(pill_x, draw_y, pill_w, row_draw_h, radius, active_border_color, active_border_w)

        header_rects.append((pill_x, slot_bottom, pill_w, row_h, label))

        if entry_map.get(label, _DEFAULT_ENTRY)[2] > 1:
            toggle_rects[label] = (content_x, slot_bottom, swatch + swatch_gap, row_h)

    state.list.row_rects = header_rects
    state.list.toggle_rects = toggle_rects

    # Child fills.
    child_rects = []

    for (
        kind,
        label,
        node_name,
        slot_bottom,
        _row_idx,
        draw_y,
        _node,
        child_active,
        child_selected,
    ) in visible_rows:
        if kind != _ROW_CHILD:
            continue

        if child_active:
            fill(pill_x, draw_y, pill_w, row_draw_h, radius, active_fill_color)
        elif child_selected:
            fill(pill_x, draw_y, pill_w, row_draw_h, radius, selection_fill_color)

        if child_active:
            border(pill_x, draw_y, pill_w, row_draw_h, radius, active_border_color, active_border_w)

        if hovered_child == (label, node_name):
            fill(pill_x, draw_y, pill_w, row_draw_h, radius, hover_color)

        child_rects.append((pill_x, slot_bottom, pill_w, row_h, label, node_name))

    state.list.node_rects = child_rects

    # Expand guide lines (on top of the row fills).
    children_get = children.get

    for label in header_has_visible:
        if label not in expanded:
            continue

        child_count = len(children_get(label, ()))
        if child_count <= 0:
            continue

        guide_top = header_slot_bottom.get(label)
        if guide_top is None:
            continue

        color = header_color(label, children, type_node_colors, type_colors, colors)

        line_color = (
            _alpha_mul(color, master_alpha) if show_type_colors else _alpha_mul(colors["text"], 0.1 * master_alpha)
        )

        guide(
            round(content_x + swatch / 2),
            guide_top - 1,
            child_count * row_h - 2,
            ui_scale,
            line_color,
        )


def _draw_list_text(
    state: MinimapState,
    settings,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    entries,
    nodes_by_name: dict,
    geo: dict,
    visible_rows: list,
    entry_map: dict,
) -> None:
    """Draw glyph batch, row labels, counts, and search UI text."""
    _draw_list_glyph_batch(
        state,
        geo["zone_x"],
        round(geo["view_top"] + state.list.scroll),
    )

    font_id = TYPE_LIST_FONT_ID
    blf.size(font_id, geo["font_size"])

    with_shadow = settings.show_text_shadow
    if with_shadow:
        blf.enable(font_id, blf.SHADOW)
        blf.shadow(font_id, 3, 0, 0, 0, 255)
        blf.shadow_offset(font_id, 0, -1)

    label_x = geo["label_x"]
    label_max_width = geo["label_max_width"]
    icon_col_x = geo["icon_col_x"]
    count_right = geo["count_right"]
    row_h = geo["row_h"]
    line_h = geo["line_h"]
    zone_y = geo["zone_y"]
    zone_h = geo["zone_h"]
    show_counts = geo["show_counts"]
    text_y_off = geo["text_y_off"]

    text_color = geo["text_color"]
    count_color = geo["count_color"]
    selection_color = geo["selection_color"]
    active_color = geo["active_color"]
    match_color = geo["match_color"]

    type_active = geo["type_active"]
    type_selected_counts = geo["type_selected_counts"]
    children = geo["children"]

    search_pill_y = geo["search_pill_y"]
    search_pad_v = geo["search_pad_v"]
    search_draw_h = geo["search_draw_h"]
    search_text_x = geo["search_text_x"]

    view_top = geo["view_top"]
    view_bottom = geo["view_bottom"]
    zone_scissor = geo["zone_scissor"]
    view_scissor = geo["view_scissor"]

    child_label_x = label_x + icon_col_x
    child_label_max_width = max(0.0, count_right - child_label_x)

    child_clip_left = int(child_label_x)
    child_clip_right = int(child_label_x + child_label_max_width)

    clip_top = int(zone_y - row_h)
    clip_bottom = int(zone_y + zone_h + row_h)

    header_clip_left = int(label_x)
    header_clip_right = int(label_x + label_max_width)

    count_clip_right = int(count_right + geo["widest_count"]) if show_counts else 0

    blf.enable(font_id, blf.CLIPPING)

    search_query = state.list.search_query
    norm_query = normalize_query(search_query)

    if settings.show_search_bar:
        search_cursor = min(max(state.list.search_cursor, 0), len(search_query))
        search_text_y = search_pill_y + search_pad_v

        clear_rect = state.list.search_clear_rect
        search_text_right = clear_rect[0] - 2 * ui_scale if clear_rect else count_right

        gpu.state.scissor_set(*zone_scissor)

        icon_color = text_color if search_query else count_color
        icon_size = search_draw_h * 0.55
        _draw_search_filter_icon(
            search_text_x,
            search_pill_y + (search_draw_h - icon_size) / 2,
            icon_size,
            icon_color,
            ui_scale,
        )

        search_text_start_x = search_text_x + icon_size + 3 * ui_scale
        state.list.search_text_start_x = search_text_start_x

        if state.list.search_focused:
            caret_x = round(
                search_text_start_x
                + (blf.dimensions(font_id, search_query[:search_cursor])[0] if search_query else 0.0)
            )

            _draw_filled_rounded_rect(
                caret_x,
                round(search_pill_y),
                max(2.0, 2.2 * ui_scale),
                search_draw_h,
                0.0,
                _alpha_mul(geo["active_border_color"], master_alpha),
            )

        search_text = search_query if search_query else "Filter"

        blf.clipping(font_id, int(search_text_start_x), clip_top, int(search_text_right), clip_bottom)
        blf.position(font_id, search_text_start_x, search_text_y, 0)
        blf.color(font_id, *(text_color if search_query else count_color))
        blf.draw(font_id, search_text)

        if clear_rect:
            _draw_search_clear_button(
                clear_rect,
                state.list.search_clear_hovered,
                colors,
                master_alpha,
                ui_scale,
            )

        gpu.state.scissor_set(*view_scissor)

        if not entries and search_query:
            no_match_y = (view_bottom + view_top - line_h) / 2 + 1

            blf.clipping(
                font_id,
                int(search_text_start_x),
                int(view_bottom - row_h),
                int(count_right),
                int(view_top + row_h),
            )

            blf.position(font_id, search_text_start_x, no_match_y, 0)
            blf.color(font_id, *count_color)
            blf.draw(font_id, "No matches")

    for (
        kind,
        label,
        node_name,
        _slot_bottom,
        _row_idx,
        draw_y,
        node,
        child_active,
        child_selected,
    ) in visible_rows:
        text_y = draw_y + text_y_off

        if kind == _ROW_HEADER:
            count_text, count_width, full_count = entry_map.get(label, _DEFAULT_ENTRY)

            is_active = label == type_active
            is_sel = type_selected_counts.get(label, 0) > 0

            if is_active:
                label_color = active_color
            elif is_sel:
                label_color = selection_color
            else:
                label_color = text_color

            header_text = _group_header_text(label, full_count, children, nodes_by_name)

            blf.clipping(font_id, header_clip_left, clip_top, header_clip_right, clip_bottom)
            _draw_text_with_match(
                font_id,
                label_x,
                text_y,
                header_text,
                label_color,
                match_color,
                norm_query,
            )

            if show_counts:
                blf.clipping(font_id, header_clip_right, clip_top, count_clip_right, clip_bottom)
                blf.position(font_id, count_right - count_width, text_y, 0)
                blf.color(font_id, *count_color)
                blf.draw(font_id, count_text)

        else:
            if child_active:
                label_color = active_color
            elif child_selected:
                label_color = selection_color
            else:
                label_color = text_color

            blf.clipping(font_id, child_clip_left, clip_top, child_clip_right, clip_bottom)

            label_text = _child_label_text(node_name, node)

            _draw_text_with_match(
                font_id,
                child_label_x,
                text_y,
                label_text,
                label_color,
                match_color,
                norm_query,
            )

    blf.disable(font_id, blf.CLIPPING)

    if with_shadow:
        blf.disable(font_id, blf.SHADOW)

    gpu.state.blend_set("ALPHA")


def _draw_list_scrollbar(
    state: MinimapState,
    colors: dict,
    master_alpha: float,
    ui_scale: float,
    geo: dict,
) -> None:
    """Draw the vertical list scrollbar thumb when content overflows."""
    state.list.scrollbar_thumb = None
    state.list.scrollbar_track = None

    scroll_max = geo["scroll_max"]
    total_h = geo["total_h"]

    if scroll_max <= 0 or total_h <= 0:
        return

    gpu.state.blend_set("ALPHA")

    _bar_thickness, bar_offset = _get_scrollbar_style(ui_scale)

    frac = state.list.scroll / scroll_max
    active = state.list.hovered_scrollbar or state.list.scrollbar_dragging
    thick = _scrollbar_thickness(ui_scale, active)

    thumb_rect, track_rect = _draw_scrollbar_thumb(
        round(geo["zone_x"] + geo["zone_w"] - thick - bar_offset),
        geo["zone_y"] + bar_offset,
        max(geo["view_top"] - geo["zone_y"] - 2 * bar_offset, 0.0),
        geo["view_h"] / total_h,
        1.0 - frac,
        colors,
        master_alpha,
        ui_scale,
        active=active,
    )

    state.list.scrollbar_thumb = thumb_rect
    state.list.scrollbar_track = track_rect


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


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
    state.list.visible_row_index_map = {}

    if state.list.list_width <= 0:
        _clear_list_interaction(state)
        return

    tree_data = state.cache.tree_data
    type_stats = tree_data.get("type_stats") if tree_data else None

    if not type_stats:
        _clear_list_interaction(state)
        return

    node_tree = bpy.context.space_data.edit_tree if bpy.context.space_data else None

    key = _type_list_cache_key(state, settings, colors, master_alpha, ui_scale)

    if key != state.cache.list_key or not state.cache.list_layout:
        _build_type_list_cache(
            state,
            settings,
            node_tree,
            key,
            colors,
            master_alpha,
            ui_scale,
        )

    entries = state.cache.list_entries or []

    expanded = getattr(state.cache, "list_effective_expanded", None)
    if expanded is None:
        expanded = state.list.expanded

    nodes_by_name = state.cache.list_nodes_by_name or {}
    layout = state.cache.list_layout or {}

    font_size = layout.get("font_size", int(settings.type_list_font_size * ui_scale))
    row_h = layout.get("row_h", 16.0)
    line_h = layout.get("line_h", 12.0)
    widest_count = layout.get("widest_count", 0.0)

    state.list.row_height = row_h

    row_gap = 1.0
    row_gap_half = round(row_gap / 2.0)
    row_draw_h = row_h - row_gap

    rows = layout.get("rows")

    # Safety fallback for older/incomplete caches.
    if rows is None:
        children = state.cache.list_filtered_children or state.cache.list_children or {}
        rows = tuple(_iter_type_list_layout(entries, children, expanded, row_h))
        total_h = len(rows) * row_h
        header_local_bottom = {
            label: local_y - row_h for kind, label, _node_name, local_y in rows if kind == _ROW_HEADER
        }
        entry_map = {lbl: (ct, cw, cnt) for lbl, ct, cw, cnt in entries}
    else:
        total_h = layout.get("total_h", len(rows) * row_h)
        header_local_bottom = layout.get("header_local_bottom") or {}
        entry_map = layout.get("entry_map") or {lbl: (ct, cw, cnt) for lbl, ct, cw, cnt in entries}

    geometry_layout = {
        "font_size": font_size,
        "line_h": line_h,
        "row_h": row_h,
        "widest_count": widest_count,
        "total_h": total_h,
        "header_local_bottom": header_local_bottom,
    }

    geo = _compute_zone_geometry(
        state,
        settings,
        geometry_layout,
        expanded,
        row_gap,
        row_gap_half,
        row_draw_h,
        map_x,
        map_y,
        map_h,
        padding,
        colors,
        master_alpha,
        ui_scale,
    )

    saved_scissor = None

    try:
        was_active = gpu.state.scissor_test_get()
        saved_scissor = (was_active, gpu.state.scissor_get() if was_active else None)
    except Exception:
        saved_scissor = None

    try:
        gpu.state.scissor_set(*geo["view_scissor"])
        gpu.state.scissor_test_set(True)
        gpu.state.blend_set("ALPHA")

        active_node = node_tree.nodes.active if node_tree else None

        view_top = geo["view_top"]
        view_bottom = geo["view_bottom"]
        scroll = state.list.scroll
        row_h = geo["row_h"]

        visible_rows = []
        visible_keys = []
        visible_index = {}
        header_has_visible = set()

        resolve_child = _resolve_child_state

        for row_idx, (kind, label, node_name, local_y) in enumerate(rows):
            slot_top = view_top + scroll + local_y
            slot_bottom = slot_top - row_h

            if slot_top <= view_bottom or slot_bottom >= view_top:
                continue

            draw_y = round(slot_bottom + row_gap_half)

            if kind == _ROW_CHILD:
                node, child_active, child_selected = resolve_child(
                    nodes_by_name,
                    node_name,
                    active_node,
                )
            else:
                node = None
                child_active = False
                child_selected = False

            visible_rows.append(
                (
                    kind,
                    label,
                    node_name,
                    slot_bottom,
                    row_idx,
                    draw_y,
                    node,
                    child_active,
                    child_selected,
                )
            )

            if kind == _ROW_HEADER:
                row_key = (_ROW_HEADER, label)
            else:
                row_key = (_ROW_CHILD, label, node_name)

            visible_index[row_key] = len(visible_keys)
            visible_keys.append(row_key)
            header_has_visible.add(label)

        state.list.visible_row_keys = visible_keys
        state.list.visible_row_index_map = visible_index

        _draw_list_fills(
            state,
            settings,
            colors,
            master_alpha,
            ui_scale,
            geo,
            visible_rows,
            entry_map,
            header_has_visible,
        )

        _draw_list_text(
            state,
            settings,
            colors,
            master_alpha,
            ui_scale,
            entries,
            nodes_by_name,
            geo,
            visible_rows,
            entry_map,
        )

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

    _draw_list_scrollbar(state, colors, master_alpha, ui_scale, geo)
