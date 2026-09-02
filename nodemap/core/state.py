"""Provide per-area minimap state."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import bpy

from .. import __package__ as base_package
from .helpers import get_addon_preferences

logger = logging.getLogger(base_package)

Rect = tuple[float, float, float, float]
Vec2 = tuple[float, float]

# Guard flag: set True during handle drags to suppress property update
# callbacks, preventing tree_data invalidation and the resulting one-frame
# content flash.
_suppress_update = False


@contextmanager
def suppress_update_callbacks():
    """Suppress property update callbacks while the context is active."""
    global _suppress_update
    _suppress_update = True
    try:
        yield
    finally:
        _suppress_update = False


class ResizeHandle(StrEnum):
    """Identify which minimap edge or corner a resize grab is attached to."""

    LIST = "LIST"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    TOP = "TOP"
    BOTTOM = "BOTTOM"
    TOP_LEFT = "TOP_LEFT"
    TOP_RIGHT = "TOP_RIGHT"
    BOTTOM_LEFT = "BOTTOM_LEFT"
    BOTTOM_RIGHT = "BOTTOM_RIGHT"


@dataclass
class ViewState:
    """Store viewport geometry and zoom for the minimap content rect."""

    rect: Rect = (0.0, 0.0, 0.0, 0.0)
    tree_bounds: Rect = (0.0, 0.0, 0.0, 0.0)
    outer_margin: float = 10.0
    inner_padding: float = 6.0
    map_scale: float = 1.0
    user_zoom: float = 1.0
    anchor_zoom: float = 1.0
    pan: Vec2 = (0.0, 0.0)
    width_clamped: bool = False
    height_clamped: bool = False
    moving: bool = False
    snapped: bool = False


@dataclass
class ButtonState:
    """Store hit rects and hover for the minimap frame buttons."""

    rects: dict[str, Rect] = field(default_factory=dict)
    hovered_button_id: str | None = None
    node_count_anchor: tuple[float, float] | None = None


@dataclass
class InteractionState:
    """Store transient hover and press state for minimap interaction."""

    hovered_node_id: str | None = None
    hovered_handle: ResizeHandle | None = None
    resize_active: ResizeHandle | None = None
    pressed: bool = False


@dataclass
class ListState:
    """Store geometry and animation state for the node-type list zone."""

    list_width: float = 0.0
    dragging_width: float | None = None
    width_clamped: bool = False
    scroll: float = 0.0
    scroll_max: float = 0.0
    row_height: float = 16.0
    hovered_type_label: str | None = None
    hovered_list_row: tuple | None = None
    hovered_scrollbar: bool = False
    scrollbar_dragging: bool = False
    expanded: set[str] = field(default_factory=set)
    row_rects: list[Rect] = field(default_factory=list)
    node_rects: list[Rect] = field(default_factory=list)
    toggle_rects: dict[str, Rect] = field(default_factory=dict)
    scrollbar_thumb: Rect | None = None
    scrollbar_track: Rect | None = None
    list_zone_rect: Rect | None = None
    visible_row_keys: list[tuple] = field(default_factory=list)
    visible_row_index_map: dict[tuple, int] = field(default_factory=dict)
    anim_active: bool = False
    anim_from: float = 0.0
    anim_target: float = -1.0
    anim_start: float = 0.0
    anim_duration: float = 0.33
    anim_timer: Any = None


@dataclass
class RenderCache:
    """Cache compiled tree data and GPU batches."""

    fingerprint: Any = None
    pending_timer: Any = None
    pending_timer_deadline: float = 0.0
    pending_fingerprint: Any = None
    force_immediate: bool = False
    tree_data: dict | None = None
    backdrops_batch: Any = None
    borders_batch: Any = None
    highlight_borders_batch: Any = None
    frames_fill_batch: Any = None
    frames_border_batch: Any = None
    node_labels: list[tuple[int, str, float, float, tuple[float, ...], float]] | None = None
    wire_batches: list | None = None
    wire_shadow_batch: Any = None
    wire_highlight_batch: Any = None
    marker_batches: list | None = None
    socket_batch: Any = None
    socket_shadow: list | None = None
    reroute_batch: Any = None
    list_key: Any = None
    list_entries: list | None = None
    list_layout: dict | None = None
    list_children: dict = field(default_factory=dict)
    list_nodes_by_name: dict = field(default_factory=dict)
    list_swatches_batch: Any = None
    tree_version: int = 0
    position_version: int = 0
    batch_key: Any = None
    batch_scale: float = 1.0
    batch_anchor: Vec2 = (0.0, 0.0)
    wire_key: Any = None
    wire_scale: float = 1.0
    pending_settle_flush: bool = False
    last_move_refresh: float = 0.0
    _batches_dirty: bool = False

    # Field categories for declarative invalidation.
    _BATCH_FIELDS: tuple[str, ...] = (
        "backdrops_batch",
        "borders_batch",
        "highlight_borders_batch",
        "frames_fill_batch",
        "frames_border_batch",
        "node_labels",
        "wire_batches",
        "wire_shadow_batch",
        "wire_highlight_batch",
        "marker_batches",
        "socket_batch",
        "socket_shadow",
        "reroute_batch",
        "batch_key",
        "wire_key",
        "list_key",
        "list_entries",
        "list_layout",
        "list_children",
        "list_nodes_by_name",
        "list_swatches_batch",
    )

    def _reset_fields(self, field_names: tuple[str, ...]) -> None:
        """Reset the given fields to their default values."""
        defaults = {
            f.name: f.default if f.default is not f.default_factory else f.default_factory()
            for f in self.__dataclass_fields__.values()
        }
        for name in field_names:
            setattr(self, name, defaults[name])

    def invalidate_all(self) -> None:
        """Clear all compiled batch data, fingerprints, and tree data."""
        self._reset_fields(self._BATCH_FIELDS)
        self.fingerprint = None
        self.tree_data = None

    def invalidate_batches_only(self) -> None:
        """Clear GPU batch data while preserving tree data and fingerprints.

        Use for display-only preference changes that affect rendering, not
        tree data.
        """
        self._reset_fields(self._BATCH_FIELDS)


@dataclass
class MinimapState:
    """Store per-area minimap state combining view, interaction, and cache."""

    enabled: bool = True
    view: ViewState = field(default_factory=ViewState)
    interaction: InteractionState = field(default_factory=InteractionState)
    list: ListState = field(default_factory=ListState)
    cache: RenderCache = field(default_factory=RenderCache)
    buttons: ButtonState = field(default_factory=ButtonState)
    last_tree_ptr: int | None = None
    tree_views: dict[int, tuple[float, float, float]] = field(default_factory=dict)


_minimap_state: dict[int, MinimapState] = {}
_minimap_window_operators: dict[int, Any] = {}
_registration_state: dict[str, bool] = {"done": False}

# Define interactive minimap buttons as (id, show-preference attribute).
# Order defines the top-edge horizontal capsule (right-aligned).
# Frame order left to right: SELECTED, VIEW, ALL.
_MINIMAP_BUTTONS: tuple[tuple[str, str], ...] = (
    ("SELECTED", "show_frame_selected_btn"),
    ("VIEW", "show_frame_view_btn"),
    ("ALL", "show_frame_all_btn"),
    ("LIST", "show_list_toggle_btn"),
)


def _state(area_ptr: int | None = None) -> MinimapState:
    """Return the minimap state for the given area, initializing defaults if needed."""
    if area_ptr is None:
        try:
            area_ptr = bpy.context.area.as_pointer()
        except (AttributeError, ReferenceError):
            return MinimapState()
    if area_ptr not in _minimap_state:
        state = MinimapState()
        try:
            prefs = get_addon_preferences()
            if prefs:
                state.enabled = prefs.settings.show_by_default
        except (AttributeError, ReferenceError):
            pass
        _minimap_state[area_ptr] = state
    return _minimap_state[area_ptr]


def _cleanup_area_states() -> None:
    """Remove stale entries from `_minimap_state` for closed NODE_EDITOR areas."""
    window_manager = bpy.context.window_manager
    if not window_manager:
        return
    active_area_pointers: set[int] = set()
    for window in window_manager.windows:
        if not window or not window.screen:
            continue
        for area in window.screen.areas:
            if area.type == "NODE_EDITOR":
                active_area_pointers.add(area.as_pointer())
    stale_pointers = [area_ptr for area_ptr in _minimap_state if area_ptr not in active_area_pointers]
    for area_ptr in stale_pointers:
        del _minimap_state[area_ptr]
    if stale_pointers:
        logger.debug("_cleanup_area_states: removed %d stale entries", len(stale_pointers))


def _ensure_area_states() -> None:
    """Pre-populate state for all existing NODE_EDITOR areas (called at registration)."""
    _cleanup_area_states()
    window_manager = bpy.context.window_manager
    if not window_manager:
        logger.debug("_ensure_area_states: no window_manager")
        return
    count = 0
    for window in window_manager.windows:
        if not window or not window.screen:
            continue
        for area in window.screen.areas:
            if area.type == "NODE_EDITOR":
                area_ptr = area.as_pointer()
                _state(area_ptr)
                count += 1
                window_name = window.screen.name if window.screen else "?"
                logger.debug("_ensure_area_states: created state for area %d (window %s)", area_ptr, window_name)
    logger.debug("_ensure_area_states: %d NODE_EDITOR areas processed", count)
