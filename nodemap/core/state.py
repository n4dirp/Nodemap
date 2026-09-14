"""Provide per-area minimap state."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import bpy

from .. import __package__ as base_package
from .constants import CONTENT_PADDING
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
    raw_tree_bounds: Rect | None = None
    snapshot_selected_bounds: Rect | None = None
    outer_margin: float = 10.0
    inner_padding: float = CONTENT_PADDING
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
    pressed_button_id: str | None = None


@dataclass
class InteractionState:
    """Store transient hover and press state for minimap interaction."""

    hovered_minimap: bool = False
    hovered_node_id: str | None = None
    hovered_handle: ResizeHandle | None = None
    resize_active: ResizeHandle | None = None
    pressed: bool = False
    marquee_active: bool = False
    marquee_start: tuple[int, int] | None = None
    marquee_end: tuple[int, int] | None = None


@dataclass
class ListState:
    """Store geometry and animation state for the node-type list zone."""

    list_width: float = 0.0
    # Zone placement for the current frame: "LEFT" (default) or "TOP" when the
    # position preference (or Auto on a tall minimap) moves the list on top.
    list_placement: str = "LEFT"
    search_query: str = ""
    search_cursor: int = 0
    search_focused: bool = False
    search_esc_armed: bool = False
    search_rect: Rect | None = None
    search_text_start_x: float = 0.0
    search_clear_rect: Rect | None = None
    search_clear_hovered: bool = False
    dragging_width: float | None = None
    width_clamped: bool = False
    scroll: float = 0.0
    scroll_max: float = 0.0
    h_scroll: float = 0.0
    h_scroll_max: float = 0.0
    # Follow Active tracking: last revealed active node and one with a pending reveal.
    followed_active: str | None = None
    follow_pending: str | None = None
    row_height: float = 16.0
    hovered_type_label: str | None = None
    hovered_list_row: tuple | None = None
    hovered_scrollbar: bool = False
    scrollbar_dragging: bool = False
    hovered_h_scrollbar: bool = False
    h_scrollbar_dragging: bool = False
    expanded: set[str] = field(default_factory=set)
    row_rects: list[Rect] = field(default_factory=list)
    node_rects: list[Rect] = field(default_factory=list)
    toggle_rects: dict[str, Rect] = field(default_factory=dict)
    scrollbar_thumb: Rect | None = None
    scrollbar_track: Rect | None = None
    h_scrollbar_thumb: Rect | None = None
    h_scrollbar_track: Rect | None = None
    list_zone_rect: Rect | None = None
    visible_row_keys: list[tuple] = field(default_factory=list)
    visible_row_index_map: dict[tuple, int] = field(default_factory=dict)
    arrow_key: tuple | None = None
    anim_active: bool = False
    anim_from: float = 0.0
    anim_target: float = -1.0
    anim_start: float = 0.0
    anim_duration: float = 0.33
    anim_timer: Any = None


@dataclass
class SharedTreeCache:
    """Cache compiled tree data and its compile schedule, shared per node tree.

    All areas showing the same node tree share one ``tree_data`` compile,
    one fingerprint, and one pending settle timer instead of compiling per
    area. GPU batches stay per-area because culling, scale, and hover state
    differ between minimaps.
    """

    fingerprint: Any = None
    tree_data: dict | None = None
    tree_version: int = 0
    position_version: int = 0
    tree_ptr: int | None = None
    pending_timer: Any = None
    pending_timer_deadline: float = 0.0
    pending_fingerprint: Any = None
    pending_immediate: bool = False
    pending_settle_flush: bool = False
    force_immediate: bool = False


@dataclass
class RenderCache:
    """Cache per-area GPU batches and type-list layout state."""

    backdrops_batch: Any = None
    borders_batch: Any = None
    highlight_borders_batch: Any = None
    frames_fill_batch: Any = None
    frames_border_batch: Any = None
    node_labels: list[tuple[int, str, float, float, tuple[float, ...], float]] | None = None
    wire_batches: list | None = None
    wire_highlight_batch: Any = None
    marker_batch: Any = None
    socket_batch: Any = None
    socket_shadow: list | None = None
    reroute_batch: Any = None
    list_key: Any = None
    list_entries: list | None = None
    list_layout: dict | None = None
    list_children: dict = field(default_factory=dict)
    list_filtered_children: dict = field(default_factory=dict)
    list_effective_expanded: set = field(default_factory=set)
    list_swatches_batch: Any = None
    list_swatches_border_batch: Any = None
    batch_key: Any = None
    batch_scale: float = 1.0
    batch_anchor: Vec2 = (0.0, 0.0)
    wire_key: Any = None
    wire_scale: float = 1.0
    last_seen_scale: float = 0.0
    scale_last_change_ts: float = 0.0
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
        "wire_highlight_batch",
        "marker_batch",
        "socket_batch",
        "socket_shadow",
        "reroute_batch",
        "batch_key",
        "wire_key",
        "list_key",
        "list_entries",
        "list_layout",
        "list_children",
        "list_filtered_children",
        "list_swatches_batch",
        "list_swatches_border_batch",
    )

    def _reset_fields(self, field_names: tuple[str, ...]) -> None:
        """Reset the given fields to their default values."""
        defaults = {
            f.name: f.default if f.default is not f.default_factory else f.default_factory()
            for f in self.__dataclass_fields__.values()
        }
        for name in field_names:
            setattr(self, name, defaults[name])

    def invalidate_batches_only(self) -> None:
        """Clear GPU batch data for display-only preference changes.

        Use when a setting affects rendering but not what tree data must be
        compiled; the shared tree data and fingerprints are left untouched.
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
    shared: SharedTreeCache | None = None

    def tree_data(self) -> dict | None:
        """Return the compiled tree data shared by this area's node tree."""
        shared = self.shared
        return shared.tree_data if shared is not None else None

    def request_immediate_compile(self) -> None:
        """Flag the shared tree cache so the next draw compiles immediately."""
        shared = self.shared
        if shared is not None:
            shared.force_immediate = True


_minimap_state: dict[int, MinimapState] = {}
_minimap_window_operators: dict[int, Any] = {}
_shared_tree_caches: dict[int, SharedTreeCache] = {}
_registration_state: dict[str, bool] = {"done": False}


def _shared_tree_cache(tree_ptr: int) -> SharedTreeCache:
    """Return the shared compile cache for a node tree pointer, creating it if needed."""
    shared = _shared_tree_caches.get(tree_ptr)
    if shared is None:
        shared = SharedTreeCache(tree_ptr=tree_ptr)
        _shared_tree_caches[tree_ptr] = shared
    return shared


def _unregister_pending_timer(shared: SharedTreeCache) -> None:
    """Unregister the shared cache's pending settle timer, if any."""
    if shared.pending_timer is None:
        return
    try:
        bpy.app.timers.unregister(shared.pending_timer)
    except (ValueError, RuntimeError):
        pass
    shared.pending_timer = None


def _cleanup_shared_tree_caches() -> None:
    """Unregister pending settle timers and drop all shared tree caches (unload)."""
    for shared in _shared_tree_caches.values():
        _unregister_pending_timer(shared)
    _shared_tree_caches.clear()


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
    """Remove stale entries from `_minimap_state` and `_shared_tree_caches`."""
    window_manager = bpy.context.window_manager
    if not window_manager:
        return
    active_area_pointers: set[int] = set()
    live_tree_pointers: set[int] = set()
    for window in window_manager.windows:
        if not window or not window.screen:
            continue
        for area in window.screen.areas:
            if area.type != "NODE_EDITOR":
                continue
            active_area_pointers.add(area.as_pointer())
            space = area.spaces.active if area.spaces else None
            tree = getattr(space, "edit_tree", None) if space is not None else None
            if tree is not None:
                try:
                    live_tree_pointers.add(tree.as_pointer())
                except ReferenceError:
                    pass
    stale_pointers = [area_ptr for area_ptr in _minimap_state if area_ptr not in active_area_pointers]
    for area_ptr in stale_pointers:
        del _minimap_state[area_ptr]
    if stale_pointers:
        logger.debug("_cleanup_area_states: removed %d stale entries", len(stale_pointers))
    stale_tree_pointers = [tree_ptr for tree_ptr in _shared_tree_caches if tree_ptr not in live_tree_pointers]
    for tree_ptr in stale_tree_pointers:
        _unregister_pending_timer(_shared_tree_caches[tree_ptr])
        del _shared_tree_caches[tree_ptr]
    if stale_tree_pointers:
        logger.debug("_cleanup_area_states: removed %d stale tree caches", len(stale_tree_pointers))


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
