"""Minimap state management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import bpy

logger = logging.getLogger(__package__)

Rect = tuple[float, float, float, float]
Vec2 = tuple[float, float]


class ResizeHandle(StrEnum):
    """Resize handle identifiers for the minimap edge/corner interaction."""

    W = "W"
    H = "H"
    C = "C"
    LIST = "LIST"


@dataclass
class ViewState:
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


@dataclass
class ButtonState:
    rects: dict[str, Rect] = field(default_factory=dict)
    hovered_button_id: str | None = None


@dataclass
class InteractionState:
    hovered_node_id: str | None = None
    hovered_handle: ResizeHandle | None = None
    resize_active: ResizeHandle | None = None
    pressed: bool = False


@dataclass
class ListState:
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
    visible_row_keys: list[str] = field(default_factory=list)
    anim_active: bool = False
    anim_from: float = 0.0
    anim_target: float = -1.0
    anim_start: float = 0.0
    anim_duration: float = 0.33
    anim_timer: Any = None


@dataclass
class RenderCache:
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
    marker_batches: list | None = None
    socket_batch: Any = None
    socket_shadow: list | None = None
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

    def invalidate_all(self) -> None:
        """Clear all compiled batch data, fingerprints, and tree data."""
        self.fingerprint = None
        self.tree_data = None
        self.backdrops_batch = None
        self.borders_batch = None
        self.highlight_borders_batch = None
        self.frames_fill_batch = None
        self.frames_border_batch = None
        self.node_labels = None
        self.wire_batches = None
        self.wire_shadow_batch = None
        self.marker_batches = None
        self.socket_batch = None
        self.socket_shadow = None
        self.batch_key = None
        self.wire_key = None
        self.list_key = None
        self.list_entries = None
        self.list_layout = None
        self.list_children = {}
        self.list_nodes_by_name = {}
        self.list_swatches_batch = None

    def invalidate_batches_only(self) -> None:
        """Clear GPU batch data while preserving tree_data and fingerprints.

        Used by display-only preference changes (size, position, opacity, etc.)
        that affect how content is rendered but not what tree data is needed.
        """
        self.backdrops_batch = None
        self.borders_batch = None
        self.highlight_borders_batch = None
        self.frames_fill_batch = None
        self.frames_border_batch = None
        self.node_labels = None
        self.wire_batches = None
        self.wire_shadow_batch = None
        self.marker_batches = None
        self.socket_batch = None
        self.socket_shadow = None
        self.batch_key = None
        self.wire_key = None
        self.list_key = None
        self.list_entries = None
        self.list_layout = None
        self.list_children = {}
        self.list_swatches_batch = None


@dataclass
class MinimapState:
    enabled: bool = True
    view: ViewState = field(default_factory=ViewState)
    interaction: InteractionState = field(default_factory=InteractionState)
    list: ListState = field(default_factory=ListState)
    cache: RenderCache = field(default_factory=RenderCache)
    buttons: ButtonState = field(default_factory=ButtonState)
    last_tree_ptr: int | None = None
    tree_views: dict[int, tuple[float, float, float]] = field(default_factory=dict)


# Socket indicator pill size multiplier (in tree units).
SOCKET_PILL_SIZE_MULTIPLIER = 2.0

_minimap_state: dict[int, MinimapState] = {}
_minimap_window_operators: dict[int, Any] = {}
_registration_state: dict[str, bool] = {"done": False}

# Interactive minimap buttons as (id, show-preference attr).
# Order defines the top-edge horizontal capsule (right-aligned); "LIST" renders standalone.
# Frame order left-to-right: SELECTED, VIEW, ALL.
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
            prefs = bpy.context.preferences.addons.get(__package__)
            if prefs:
                state.enabled = prefs.preferences.settings.show_by_default
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
