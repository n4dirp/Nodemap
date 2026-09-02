"""Define the canonical minimap content layers and their draw order."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    """Describe a map-content visual layer and the RenderCache fields it owns."""

    name: str
    cache_fields: tuple[str, ...]
    in_content_matrix: bool = True


CONTENT_LAYERS: tuple[Layer, ...] = (
    Layer("frames", ("frames_fill_batch", "frames_border_batch")),
    Layer("wires", ("wire_shadow_batch", "wire_batches")),
    Layer("wire_highlight", ("wire_highlight_batch",)),
    Layer("backdrops", ("backdrops_batch",)),
    Layer("borders", ("borders_batch",)),
    Layer("highlight", ("highlight_borders_batch",)),
    Layer("markers", ("marker_batches",)),
    Layer("socket", ("socket_batch",)),
    Layer("reroute", ("reroute_batch",)),
    Layer("labels", ("node_labels",), in_content_matrix=False),
)
