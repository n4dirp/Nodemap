"""Draw baked minimap content batches in the manifest-defined layer order."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import gpu
from mathutils import Matrix

from ..core.theme import _srgb_to_linear
from .content_layers import CONTENT_LAYERS
from .gpu_draw import (
    _draw_text_with_shadow,
    _get_batch_noodle_shader,
    _get_batch_pill_shader,
    _get_batch_rect_border_shader,
    _get_batch_rect_shader,
)

if TYPE_CHECKING:
    from ..core.state import MinimapState
    from ..ui.preferences import NODEMAP_PG_settings


def _content_pivot(
    state: MinimapState, scale: float, tree_center_x: float, tree_center_y: float
) -> tuple[tuple[float, float] | None, float, float, float]:
    """Return the origin, content scale factor, and pivot for the content matrix."""
    origin = state.cache.tree_data.get("origin") if state.cache.tree_data else None
    if not origin:
        return None, 1.0, 0.0, 0.0
    batch_scale = state.cache.batch_scale if state.cache.batch_scale > 0.0 else scale
    content_scale_factor = scale / batch_scale
    pivot_x = (tree_center_x - origin[0]) * batch_scale
    pivot_y = (tree_center_y - origin[1]) * batch_scale
    return origin, content_scale_factor, pivot_x, pivot_y


def _draw_layer_frames(state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]) -> None:
    """Draw frame node fill and border batches."""
    frames_fill_batch = state.cache.frames_fill_batch
    frames_border_batch = state.cache.frames_border_batch
    if not frames_fill_batch and not frames_border_batch:
        return
    if frames_fill_batch:
        fill_shader = _get_batch_rect_shader()
        fill_shader.bind()
        fill_shader.uniform_float("ModelViewProjectionMatrix", mvp)
        frames_fill_batch.draw(fill_shader)
    if frames_border_batch:
        border_shader = _get_batch_rect_border_shader()
        border_shader.bind()
        border_shader.uniform_float("ModelViewProjectionMatrix", mvp)
        frames_border_batch.draw(border_shader)


def _draw_layer_wires(state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]) -> None:
    """Draw link wires (baked batches; shadow underlay first, then colors)."""
    wire_batches = state.cache.wire_batches or []
    wire_shadow_batch = state.cache.wire_shadow_batch
    if not (settings.show_wires and (wire_shadow_batch or wire_batches)):
        return
    wire_curved = int(params["wire_curvature"]) > 0
    shadow_alpha = 0.35 * params["master_alpha"]
    if wire_curved:
        noodle_shader = _get_batch_noodle_shader()
        noodle_shader.bind()
        noodle_shader.uniform_float("ModelViewProjectionMatrix", mvp)
        if wire_shadow_batch is not None and shadow_alpha > 0:
            if isinstance(wire_shadow_batch, tuple):
                shadow_batch, shadow_half = wire_shadow_batch
            else:
                shadow_batch, shadow_half = wire_shadow_batch, 1.0
            noodle_shader.uniform_float("color", (0.0, 0.0, 0.0, shadow_alpha))
            noodle_shader.uniform_float("halfThick", float(shadow_half))
            shadow_batch.draw(noodle_shader)
        for entry in wire_batches:
            if len(entry) == 3:
                wire_color, batch, half = entry
            else:
                wire_color, batch = entry
                half = 1.0
            noodle_shader.uniform_float("color", _srgb_to_linear(wire_color))
            noodle_shader.uniform_float("halfThick", float(half))
            batch.draw(noodle_shader)
    else:
        pill_shader = _get_batch_pill_shader()
        pill_shader.bind()
        pill_shader.uniform_float("ModelViewProjectionMatrix", mvp)
        if wire_shadow_batch is not None and shadow_alpha > 0:
            # Straight-wire shadow is a plain batch.
            shadow_batch = wire_shadow_batch[0] if isinstance(wire_shadow_batch, tuple) else wire_shadow_batch
            pill_shader.uniform_float("color", (0.0, 0.0, 0.0, shadow_alpha))
            shadow_batch.draw(pill_shader)
        for entry in wire_batches:
            if len(entry) == 3:
                wire_color, batch = entry[0], entry[1]
            else:
                wire_color, batch = entry
            pill_shader.uniform_float("color", _srgb_to_linear(wire_color))
            batch.draw(pill_shader)


def _draw_layer_wire_highlight(
    state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]
) -> None:
    """Draw wires connected to selected nodes (thicker stroke over regular wires)."""
    highlight_batch = state.cache.wire_highlight_batch
    if not (settings.show_wires and highlight_batch):
        return
    tree_data = state.cache.tree_data
    wire_color = tree_data.get("wire_highlight_color") if tree_data else None
    if wire_color is None:
        return
    batch, half = highlight_batch
    if int(params["wire_curvature"]) > 0:
        noodle_shader = _get_batch_noodle_shader()
        noodle_shader.bind()
        noodle_shader.uniform_float("ModelViewProjectionMatrix", mvp)
        noodle_shader.uniform_float("color", _srgb_to_linear(wire_color))
        noodle_shader.uniform_float("halfThick", float(half))
        batch.draw(noodle_shader)
    else:
        pill_shader = _get_batch_pill_shader()
        pill_shader.bind()
        pill_shader.uniform_float("ModelViewProjectionMatrix", mvp)
        pill_shader.uniform_float("color", _srgb_to_linear(wire_color))
        batch.draw(pill_shader)


def _draw_layer_backdrops(state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]) -> None:
    """Draw node fill background batches."""
    backdrops_batch = state.cache.backdrops_batch
    if not backdrops_batch:
        return
    fill_shader = _get_batch_rect_shader()
    fill_shader.bind()
    fill_shader.uniform_float("ModelViewProjectionMatrix", mvp)
    backdrops_batch.draw(fill_shader)


def _draw_layer_borders(state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]) -> None:
    """Draw node border batches."""
    borders_batch = state.cache.borders_batch
    if not borders_batch:
        return
    border_shader = _get_batch_rect_border_shader()
    border_shader.bind()
    border_shader.uniform_float("ModelViewProjectionMatrix", mvp)
    borders_batch.draw(border_shader)


def _draw_layer_highlight(state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]) -> None:
    """Draw list-hover outside outlines (rendered above regular borders)."""
    highlight_borders_batch = state.cache.highlight_borders_batch
    if not highlight_borders_batch:
        return
    border_shader = _get_batch_rect_border_shader()
    border_shader.bind()
    border_shader.uniform_float("ModelViewProjectionMatrix", mvp)
    highlight_borders_batch.draw(border_shader)


def _draw_layer_markers(state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]) -> None:
    """Draw group node underline marker batches."""
    marker_batches = state.cache.marker_batches or []
    if not marker_batches:
        return
    pill_shader = _get_batch_pill_shader()
    pill_shader.bind()
    pill_shader.uniform_float("ModelViewProjectionMatrix", mvp)
    for marker_color, batch in marker_batches:
        pill_shader.uniform_float("color", _srgb_to_linear(marker_color))
        batch.draw(pill_shader)


def _draw_layer_socket(state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]) -> None:
    """Draw socket indicator pills (a single per-vertex colored batch)."""
    socket_batch = state.cache.socket_batch
    if not (settings.show_socket_indicators and socket_batch):
        return
    shader = _get_batch_rect_shader()
    shader.bind()
    shader.uniform_float("ModelViewProjectionMatrix", mvp)
    socket_batch.draw(shader)


def _draw_layer_reroute(state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]) -> None:
    """Draw reroute pills (same SDF as sockets, per-vertex color, batched by color)."""
    reroute_batch = state.cache.reroute_batch
    if not (getattr(settings, "show_reroutes", True) and reroute_batch):
        return
    shader = _get_batch_rect_shader()
    shader.bind()
    shader.uniform_float("ModelViewProjectionMatrix", mvp)
    reroute_batch.draw(shader)


def _draw_layer_labels(state: MinimapState, settings: NODEMAP_PG_settings, mvp: Any, params: dict[str, Any]) -> None:
    """Draw node text labels in screen space (BLF never sees the content matrix)."""
    label_entries = state.cache.node_labels or []
    if not (label_entries and params["origin"]):
        return
    gpu.state.blend_set("ALPHA")
    offset_x = params["map_anchor_x"] - params["content_scale_factor"] * params["pivot_x"]
    offset_y = params["map_anchor_y"] - params["content_scale_factor"] * params["pivot_y"]
    for font_id, text, label_x, label_y, text_color, font_size in label_entries:
        _draw_text_with_shadow(
            font_id,
            text,
            round(params["content_scale_factor"] * label_x + offset_x),
            round(params["content_scale_factor"] * label_y + offset_y),
            text_color,
            font_size,
            settings.show_text_shadow,
        )
    gpu.state.blend_set("ALPHA")


_DRAW: dict[str, Callable] = {
    "frames": _draw_layer_frames,
    "wires": _draw_layer_wires,
    "wire_highlight": _draw_layer_wire_highlight,
    "backdrops": _draw_layer_backdrops,
    "borders": _draw_layer_borders,
    "highlight": _draw_layer_highlight,
    "markers": _draw_layer_markers,
    "socket": _draw_layer_socket,
    "reroute": _draw_layer_reroute,
    "labels": _draw_layer_labels,
}


def draw_content_batches(
    state: MinimapState,
    settings: NODEMAP_PG_settings,
    scale: float,
    tree_center_x: float,
    tree_center_y: float,
    map_anchor_x: float,
    map_anchor_y: float,
    master_alpha: float,
    wire_curvature: Any,
) -> None:
    """Draw all map-content batches in the order given by CONTENT_LAYERS."""
    origin, content_scale_factor, pivot_x, pivot_y = _content_pivot(state, scale, tree_center_x, tree_center_y)
    if origin is None:
        return

    # Content batches are baked in map-local space; place them with one
    # matrix transform (translate -> scale about the view pivot) instead of
    # rebuilding vertex data on pan/drag frames.
    content_matrix = (
        Matrix.Translation((map_anchor_x, map_anchor_y, 0.0))
        @ Matrix.Scale(content_scale_factor, 4)
        @ Matrix.Translation((-pivot_x, -pivot_y, 0.0))
    )
    params: dict[str, Any] = {
        "master_alpha": master_alpha,
        "wire_curvature": wire_curvature,
        "map_anchor_x": map_anchor_x,
        "map_anchor_y": map_anchor_y,
        "content_scale_factor": content_scale_factor,
        "pivot_x": pivot_x,
        "pivot_y": pivot_y,
        "origin": origin,
    }
    mvp: Any = None
    gpu.matrix.push()
    try:
        gpu.matrix.multiply_matrix(content_matrix)
        mvp = gpu.matrix.get_projection_matrix() @ gpu.matrix.get_model_view_matrix()
        for layer in CONTENT_LAYERS:
            if layer.in_content_matrix:
                _DRAW[layer.name](state, settings, mvp, params)
    finally:
        gpu.matrix.pop()
    for layer in CONTENT_LAYERS:
        if not layer.in_content_matrix:
            _DRAW[layer.name](state, settings, mvp, params)
