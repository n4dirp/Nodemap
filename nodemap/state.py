"""Minimap state management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import bpy

logger = logging.getLogger(__package__)


@dataclass
class MinimapState:
    rect: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    tree_bounds: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    margin: float = 10.0
    padding: float = 6.0
    scale: float = 1.0
    hovered_node: str | None = None
    zoom: float = 1.0
    base_zoom: float = 1.0
    pan: list[float] = field(default_factory=lambda: [0.0, 0.0])
    enabled: bool = True
    frame_all_btn: tuple[float, float, float, float] | None = None
    frame_view_btn: tuple[float, float, float, float] | None = None
    frame_selected_btn: tuple[float, float, float, float] | None = None
    list_toggle_btn: tuple[float, float, float, float] | None = None
    hovered_frame_btn: str | None = None
    width_clamped: bool = False
    height_clamped: bool = False
    hovered_handle: str | None = None
    resize_active: str | None = None
    pressed: bool = False
    list_width: float = 0.0
    list_scroll: float = 0.0
    list_scroll_max: float = 0.0
    list_row_h: float = 16.0
    hovered_type_label: str | None = None
    list_row_rects: list = field(default_factory=list)
    list_expanded: set = field(default_factory=set)
    list_node_rects: list = field(default_factory=list)
    list_toggle_rects: dict = field(default_factory=dict)
    hovered_list_node: tuple | None = None
    hovered_list_scrollbar: bool = False
    list_scroll_dragging: bool = False
    list_scrollbar_thumb: tuple[float, float, float, float] | None = None
    list_scrollbar_track: tuple[float, float, float, float] | None = None
    list_zone_rect: tuple[float, float, float, float] | None = None
    list_anim_active: bool = False
    list_anim_from: float = 0.0
    list_anim_target: float = -1.0
    list_anim_start: float = 0.0
    list_anim_duration: float = 0.33
    list_anim_timer: Any = None
    cached_fingerprint: Any = None
    pending_timer: Any = None
    pending_timer_deadline: float = 0.0
    pending_fingerprint: Any = None
    force_immediate_compile: bool = False
    tree_data: dict | None = None
    cached_backdrops_batch: Any = None
    cached_borders_batch: Any = None
    cached_frames_fill_batch: Any = None
    cached_frames_border_batch: Any = None
    cached_text: list | None = None
    cached_wire_batches: list | None = None
    cached_wire_shadow_batch: Any = None
    cached_marker_batches: list | None = None
    cached_socket_batch: Any = None
    cached_socket_ph: float = 2.0
    cached_socket_shadow: list | None = None
    list_cache_key: Any = None
    cached_list_entries: list | None = None
    cached_list_layout: dict | None = None
    cached_list_children: dict = field(default_factory=dict)
    cached_list_swatches_batch: Any = None
    tree_data_version: int = 0
    pos_data_version: int = 0
    batch_cache_key: Any = None
    batch_scale: float = 1.0
    batch_anchor: tuple[float, float] = (0.0, 0.0)
    wire_cache_key: Any = None
    wire_scale: float = 1.0
    pending_settle_flush: bool = False
    last_move_refresh: float = 0.0
    _profiler: Any = field(default=None, repr=False)
    _profiling_active: bool = field(default=False, repr=False)
    _profiling_frame_count: int = field(default=0, repr=False)


_minimap_state: dict[int, MinimapState] = {}
_minimap_window_operators: dict[int, Any] = {}
_registration_state: dict[str, bool] = {"done": False}

# Interactive minimap buttons as (id, show-preference attr, MinimapState attr).
# Order defines the right-edge capsule stack; "LIST" renders standalone.
_MINIMAP_BUTTONS: tuple[tuple[str, str, str], ...] = (
    ("ALL", "show_frame_all_btn", "frame_all_btn"),
    ("VIEW", "show_frame_view_btn", "frame_view_btn"),
    ("SELECTED", "show_frame_selected_btn", "frame_selected_btn"),
    ("LIST", "show_list_toggle_btn", "list_toggle_btn"),
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
                state.enabled = getattr(prefs.preferences.settings, "show_by_default", True)
        except (AttributeError, ReferenceError):
            pass
        _minimap_state[area_ptr] = state
    return _minimap_state[area_ptr]


def _ensure_area_states() -> None:
    """Pre-populate state for all existing NODE_EDITOR areas (called at registration)."""
    wm = bpy.context.window_manager
    if not wm:
        logger.debug("_ensure_area_states: no window_manager")
        return
    count = 0
    for window in wm.windows:
        if not window or not window.screen:
            continue
        for area in window.screen.areas:
            if area.type == "NODE_EDITOR":
                ptr = area.as_pointer()
                _state(ptr)
                count += 1
                win_name = window.screen.name if window.screen else "?"
                logger.debug("_ensure_area_states: created state for area %d (window %s)", ptr, win_name)
    logger.debug("_ensure_area_states: %d NODE_EDITOR areas processed", count)
