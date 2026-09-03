"""Animation controller for smooth pan, zoom, and inertia effects."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import bpy

from .. import __package__ as base_package
from ..core.helpers import get_addon_preferences

if TYPE_CHECKING:
    from bpy.types import Context, Region, SpaceNodeEditor

    from ..core.state import MinimapState
    from .navigate import NODEMAP_OT_navigate

logger = logging.getLogger(base_package)


class AnimationController:
    """Manages smooth animations for the minimap operator.

    Encapsulates inertia, smooth-drag, center-animation, frame-zoom,
    and editor-viewport animations. Holds a weak reference to the parent
    operator for accessing shared state (space, region, override_ctx).
    """

    __slots__ = (
        "_op",
        "smooth_timer",
        "inertia_active",
        "inertia_mode",
        "smooth_velocity",
        "anim_active",
        "anim_target",
        "anim_applied",
        "anim_progress",
        "anim_acc",
        "drag_target",
        "drag_active",
        "frame_anim_active",
        "frame_anim_start_zoom",
        "frame_anim_start_pan",
        "frame_anim_target_zoom",
        "frame_anim_target_pan",
        "frame_anim_progress",
        "editor_anim_active",
        "editor_anim_progress",
        "editor_anim_start_rect",
        "editor_anim_target_rect",
    )

    def __init__(self, op: NODEMAP_OT_navigate) -> None:
        self._op = op
        self.smooth_timer: str | None = None
        self.inertia_active: bool = False
        self.inertia_mode: str | None = None
        self.smooth_velocity: list[float] = [0.0, 0.0]
        self.anim_active: bool = False
        self.anim_target: list[float] = [0.0, 0.0]
        self.anim_applied: list[float] = [0.0, 0.0]
        self.anim_progress: float = 0.0
        self.anim_acc: list[float] = [0.0, 0.0]
        self.drag_target: list[float] = [0.0, 0.0]
        self.drag_active: bool = False
        self.frame_anim_active: bool = False
        self.frame_anim_start_zoom: float = 1.0
        self.frame_anim_start_pan: list[float] = [0.0, 0.0]
        self.frame_anim_target_zoom: float = 1.0
        self.frame_anim_target_pan: list[float] = [0.0, 0.0]
        self.frame_anim_progress: float = 0.0
        self.editor_anim_active: bool = False
        self.editor_anim_progress: float = 0.0
        self.editor_anim_start_rect: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.editor_anim_target_rect: list[float] = [0.0, 0.0, 0.0, 0.0]

    def reset(self) -> None:
        """Reset all animation state to defaults."""
        self.smooth_timer = None
        self.inertia_active = False
        self.inertia_mode = None
        self.smooth_velocity = [0.0, 0.0]
        self.anim_active = False
        self.anim_target = [0.0, 0.0]
        self.anim_applied = [0.0, 0.0]
        self.anim_progress = 0.0
        self.anim_acc = [0.0, 0.0]
        self.drag_target = [0.0, 0.0]
        self.drag_active = False
        self.frame_anim_active = False
        self.frame_anim_start_zoom = 1.0
        self.frame_anim_start_pan = [0.0, 0.0]
        self.frame_anim_target_zoom = 1.0
        self.frame_anim_target_pan = [0.0, 0.0]
        self.frame_anim_progress = 0.0
        self.editor_anim_active = False
        self.editor_anim_progress = 0.0
        self.editor_anim_start_rect = [0.0, 0.0, 0.0, 0.0]
        self.editor_anim_target_rect = [0.0, 0.0, 0.0, 0.0]

    def any_active(self) -> bool:
        """Return True when any animation is currently running."""
        return (
            self.anim_active
            or self.inertia_active
            or self.drag_active
            or self.frame_anim_active
            or self.editor_anim_active
        )

    def create_timer(self, context: Context) -> None:
        if self.smooth_timer:
            return
        self.smooth_timer = context.window_manager.event_timer_add(1 / 60, window=context.window)

    def destroy_timer(self, context: Context) -> None:
        if self.smooth_timer:
            try:
                context.window_manager.event_timer_remove(self.smooth_timer)
            except (RuntimeError, ValueError):
                pass
            self.smooth_timer = None

    def cancel_smooth(self, context: Context) -> None:
        """Snap all active animations to their targets and stop."""
        from ..geo.transforms import _clamp_pan_to_viewport

        op = self._op
        if self.inertia_active:
            self.inertia_active = False
            self.inertia_mode = None
            self.smooth_velocity = [0.0, 0.0]
            self.destroy_timer(context)
        if self.anim_active:
            if self.anim_applied[0] != self.anim_target[0] or self.anim_applied[1] != self.anim_target[1]:
                remaining_x = self.anim_target[0] - self.anim_applied[0]
                remaining_y = self.anim_target[1] - self.anim_applied[1]
                if abs(remaining_x) >= 0.5 or abs(remaining_y) >= 0.5:
                    try:
                        with op._override_ctx(context):
                            bpy.ops.view2d.pan(deltax=int(remaining_x), deltay=int(remaining_y))
                    except RuntimeError:
                        pass
            self.anim_active = False
            self.destroy_timer(context)
        if self.frame_anim_active:
            state: MinimapState | None = op._state
            if state:
                state.view.anchor_zoom = self.frame_anim_target_zoom
                state.view.user_zoom = self.frame_anim_target_zoom
                state.view.pan = (self.frame_anim_target_pan[0], self.frame_anim_target_pan[1])
                _clamp_pan_to_viewport(op._space, op._region, state)
            self.frame_anim_active = False
            self.frame_anim_progress = 0.0
            self.destroy_timer(context)
        if self.editor_anim_active:
            self._cancel_editor_animation(context)

    def apply_inertia(self, context: Context) -> None:
        """Decay inertia and apply pan deltas."""
        from ..geo.transforms import _clamp_pan_to_viewport

        op = self._op
        decay = 0.92
        self.smooth_velocity[0] *= decay
        self.smooth_velocity[1] *= decay
        speed = max(abs(self.smooth_velocity[0]), abs(self.smooth_velocity[1]))
        if speed < 0.5:
            self.inertia_active = False
            self.inertia_mode = None
            self.destroy_timer(context)
            return
        if self.inertia_mode == "PAN":
            state: MinimapState | None = op._state
            if state:
                op._pan_acc[0] += self.smooth_velocity[0]
                op._pan_acc[1] += self.smooth_velocity[1]
                dx = int(op._pan_acc[0])
                dy = int(op._pan_acc[1])
                op._pan_acc[0] -= dx
                op._pan_acc[1] -= dy
                if dx != 0 or dy != 0:
                    state.view.pan = (state.view.pan[0] + dx, state.view.pan[1] + dy)
                    _clamp_pan_to_viewport(op._space, op._region, state)
        elif self.inertia_mode == "VIEW":
            op._pan_acc[0] += self.smooth_velocity[0]
            op._pan_acc[1] += self.smooth_velocity[1]
            dx = int(op._pan_acc[0])
            dy = int(op._pan_acc[1])
            op._pan_acc[0] -= dx
            op._pan_acc[1] -= dy
            if dx != 0 or dy != 0:
                try:
                    with op._override_ctx(context):
                        bpy.ops.view2d.pan(deltax=dx, deltay=dy)
                except RuntimeError:
                    pass
                _clamp_pan_to_viewport(op._space, op._region, op._state)
        op._redraw_ui()

    def apply_smooth_drag(self, context: Context) -> None:
        """Chase the drag target with a spring-like follow."""
        from ..geo.transforms import _clamp_pan_to_viewport

        op = self._op
        if not self.drag_active:
            return
        magnitude = (self.drag_target[0] ** 2 + self.drag_target[1] ** 2) ** 0.5
        raw = magnitude / 200.0
        follow = 0.25 + raw * 0.55
        follow = min(follow, 0.8)
        max_move = 120.0 + magnitude * 0.15
        max_move = min(max_move, 800.0)
        dx = self.drag_target[0] * follow
        dy = self.drag_target[1] * follow
        dx = max(min(dx, max_move), -max_move)
        dy = max(min(dy, max_move), -max_move)
        op._pan_acc[0] += dx
        op._pan_acc[1] += dy
        self.drag_target[0] -= dx
        self.drag_target[1] -= dy
        pan_x = int(op._pan_acc[0])
        pan_y = int(op._pan_acc[1])
        op._pan_acc[0] -= pan_x
        op._pan_acc[1] -= pan_y
        if pan_x != 0 or pan_y != 0:
            try:
                with op._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
            except RuntimeError:
                pass
            _clamp_pan_to_viewport(op._space, op._region, op._state)
        if not op._dragging:
            self.drag_active = False
        op._redraw_ui()

    def apply_center_animation(self, context: Context) -> None:
        """Ease the view toward the center-animation target."""
        op = self._op
        if not self.anim_active:
            return
        addon = get_addon_preferences(context)
        settings = addon.settings if addon else None
        speed = settings.pan_speed if settings else "MEDIUM"
        frames = {"FAST": 10, "MEDIUM": 20}.get(speed, 24)
        self.anim_progress += 1 / frames
        if self.anim_progress >= 1.0:
            remaining_x = self.anim_target[0] - self.anim_applied[0]
            remaining_y = self.anim_target[1] - self.anim_applied[1]
            if abs(remaining_x) >= 0.5 or abs(remaining_y) >= 0.5:
                try:
                    with op._override_ctx(context):
                        bpy.ops.view2d.pan(deltax=int(remaining_x), deltay=int(remaining_y))
                except RuntimeError:
                    pass
            self.anim_active = False
            self.destroy_timer(context)
            return
        eased = 1.0 - (1.0 - self.anim_progress) ** 3
        desired_x = self.anim_target[0] * eased
        desired_y = self.anim_target[1] * eased
        delta_x = desired_x - self.anim_applied[0]
        delta_y = desired_y - self.anim_applied[1]
        self.anim_applied[0] += delta_x
        self.anim_applied[1] += delta_y
        self.anim_acc[0] += delta_x
        self.anim_acc[1] += delta_y
        dx = int(self.anim_acc[0])
        dy = int(self.anim_acc[1])
        self.anim_acc[0] -= dx
        self.anim_acc[1] -= dy
        if dx != 0 or dy != 0:
            try:
                with op._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=dx, deltay=dy)
            except RuntimeError:
                pass
        op._redraw_ui()

    def start_frame_animation(self, context: Context, target_zoom: float, target_pan: list[float]) -> None:
        """Begin a zoom+pan animation toward the given target."""
        op = self._op
        state: MinimapState | None = op._state
        if not state:
            return
        if self.frame_anim_active:
            self.frame_anim_active = False
        self.frame_anim_progress = 0.0
        self.frame_anim_start_zoom = state.view.user_zoom
        self.frame_anim_start_pan = [state.view.pan[0], state.view.pan[1]]
        self.frame_anim_target_zoom = target_zoom
        self.frame_anim_target_pan = [target_pan[0], target_pan[1]]
        self.frame_anim_active = True
        self.create_timer(context)

    def apply_frame_animation(self, context: Context) -> None:
        """Step the frame zoom+pan animation forward one frame."""
        from ..geo.transforms import _clamp_pan_to_viewport

        op = self._op
        if not self.frame_anim_active:
            return
        state: MinimapState | None = op._state
        if not state:
            self.frame_anim_active = False
            self.destroy_timer(context)
            return
        addon = get_addon_preferences(context)
        settings = addon.settings if addon else None
        speed = settings.pan_speed if settings else "MEDIUM"
        frames = {"FAST": 10, "MEDIUM": 20}.get(speed, 24)
        progress = self.frame_anim_progress + 1 / frames
        self.frame_anim_progress = progress
        if progress >= 1.0:
            state.view.anchor_zoom = self.frame_anim_target_zoom
            state.view.user_zoom = self.frame_anim_target_zoom
            state.view.pan = (self.frame_anim_target_pan[0], self.frame_anim_target_pan[1])
            _clamp_pan_to_viewport(op._space, op._region, state)
            self.frame_anim_active = False
            self.frame_anim_progress = 0.0
            self.destroy_timer(context)
            op._redraw_ui()
            return
        eased = 1.0 - (1.0 - progress) ** 3
        state.view.user_zoom = (
            self.frame_anim_start_zoom + (self.frame_anim_target_zoom - self.frame_anim_start_zoom) * eased
        )
        state.view.anchor_zoom = state.view.user_zoom
        state.view.pan = (
            self.frame_anim_start_pan[0] + (self.frame_anim_target_pan[0] - self.frame_anim_start_pan[0]) * eased,
            self.frame_anim_start_pan[1] + (self.frame_anim_target_pan[1] - self.frame_anim_start_pan[1]) * eased,
        )
        _clamp_pan_to_viewport(op._space, op._region, state)
        op._redraw_ui()

    def view_selected_animated(self, context: Context, settings) -> bool:
        """Ease the editor viewport onto the selected nodes; True when started."""
        from ..geo.framing import _compute_editor_frame_selected_targets

        op = self._op
        if not self._animations_enabled(settings, context):
            return False
        targets = _compute_editor_frame_selected_targets(op._space, op._region)
        if targets is None:
            return False
        self.start_editor_animation(context, list(targets))
        return True

    def start_editor_animation(self, context: Context, target_rect: list[float]) -> None:
        """Begin animating the editor viewport toward the target tree-space rect."""
        from ..geo.transforms import _get_visible_rect

        op = self._op
        visible = _get_visible_rect(op._space, op._region)
        if not visible:
            return
        if self._editor_view_close(visible, target_rect):
            return
        self.editor_anim_progress = 0.0
        self.editor_anim_start_rect = [visible[0], visible[1], visible[2], visible[3]]
        self.editor_anim_target_rect = target_rect
        self.editor_anim_active = True
        self.create_timer(context)

    def apply_editor_animation(self, context: Context) -> None:
        """Step the editor viewport animation forward one frame."""
        op = self._op
        if not self.editor_anim_active:
            return
        if not op._space or not op._region or not op._state:
            self.editor_anim_active = False
            self.destroy_timer(context)
            return
        addon = get_addon_preferences(context)
        settings = addon.settings if addon else None
        speed = settings.pan_speed if settings else "MEDIUM"
        frames = {"FAST": 10, "MEDIUM": 20}.get(speed, 24)
        progress = self.editor_anim_progress + 1 / frames
        if progress >= 1.0:
            self._correct_editor_view(context, self.editor_anim_target_rect)
            self.editor_anim_active = False
            self.editor_anim_progress = 0.0
            self.destroy_timer(context)
            op._redraw_ui()
            return
        self.editor_anim_progress = progress
        eased = 1.0 - (1.0 - progress) ** 3
        desired = [
            start + (target - start) * eased
            for start, target in zip(self.editor_anim_start_rect, self.editor_anim_target_rect)
        ]
        self._correct_editor_view(context, desired)
        op._redraw_ui()

    def _animations_enabled(self, settings, context: Context, default: bool = True) -> bool:
        if context.preferences.view.use_reduce_motion:
            return False
        if settings is None:
            return default
        return settings.use_animations

    def _editor_view_close(self, visible: tuple[float, float, float, float], target: list[float]) -> bool:
        """Return True when the editor viewport already frames *target*."""
        region = self._op._region
        if not region:
            return False
        cur_w = max(visible[2] - visible[0], 1e-6)
        cur_h = max(visible[3] - visible[1], 1e-6)
        des_w = max(target[2] - target[0], 1e-6)
        des_h = max(target[3] - target[1], 1e-6)
        ratio = min(max(des_w / cur_w, des_h / cur_h), 1e6)
        if ratio < 1.0 - 0.005 or ratio > 1.0 + 0.005:
            return False
        vzx = region.width / cur_w
        vzy = region.height / cur_h
        dcx = (target[0] + target[2] - visible[0] - visible[2]) / 2
        dcy = (target[1] + target[3] - visible[1] - visible[3]) / 2
        return abs(dcx * vzx) <= 0.5 and abs(dcy * vzy) <= 0.5

    def _correct_editor_view(self, context: Context, desired: list[float]) -> None:
        """Nudge the editor view2d one monotonic step toward the desired rect."""
        from ..geo.transforms import _get_visible_rect

        op = self._op
        space: SpaceNodeEditor | None = op._space
        region: Region | None = op._region
        if not space or not region:
            return

        current = _get_visible_rect(space, region)
        if not current:
            return

        cur_w = max(current[2] - current[0], 1e-6)
        cur_h = max(current[3] - current[1], 1e-6)
        des_w = max(desired[2] - desired[0], 1e-6)
        des_h = max(desired[3] - desired[1], 1e-6)
        ratio = min(max(des_w / cur_w, des_h / cur_h), 1e6)
        if ratio < 1.0 - 0.005:
            fac = min((1.0 - ratio) / 2.0, 0.4)
            try:
                with op._override_ctx(context):
                    bpy.ops.view2d.zoom_in(zoomfacx=fac, zoomfacy=fac)
            except RuntimeError:
                pass
        elif ratio > 1.0 + 0.005:
            fac = max((1.0 / ratio - 1.0) / 2.0, -0.4)
            try:
                with op._override_ctx(context):
                    bpy.ops.view2d.zoom_out(zoomfacx=fac, zoomfacy=fac)
            except RuntimeError:
                pass

        current = _get_visible_rect(space, region)
        if not current:
            return

        from .navigate import _view_zoom_factors

        view_zoom_x, view_zoom_y = _view_zoom_factors(space, region, current)
        dcx = (desired[0] + desired[2] - current[0] - current[2]) / 2
        dcy = (desired[1] + desired[3] - current[1] - current[3]) / 2
        pan_x = int(round(dcx * view_zoom_x))
        pan_y = int(round(dcy * view_zoom_y))
        if pan_x != 0 or pan_y != 0:
            try:
                with op._override_ctx(context):
                    bpy.ops.view2d.pan(deltax=pan_x, deltay=pan_y)
            except RuntimeError:
                pass

    def _cancel_editor_animation(self, context: Context) -> None:
        """Snap the editor viewport to the animation target and stop stepping."""
        if not self.editor_anim_active:
            return
        self.editor_anim_active = False
        self.editor_anim_progress = 0.0
        if self._op._space and self._op._region:
            self._correct_editor_view(context, self.editor_anim_target_rect)
        self.destroy_timer(context)
