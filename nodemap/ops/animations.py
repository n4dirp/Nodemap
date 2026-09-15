"""Animation controller for smooth pan, zoom, and inertia effects."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING

import bpy

from .. import __package__ as base_package
from ..core.constants import INERTIA_PAN_DECAY, PAN_ANIM_INTERVAL, PAN_FRAMES, PAN_MIN_FRAMES
from ..core.helpers import get_addon_preferences
from ..geo.framing import _compute_editor_frame_selected_targets
from ..geo.transforms import (
    _clamp_pan_to_viewport,
    _get_visible_rect,
    _interp_rect,
    _minimap_view_from_world_rect,
    _minimap_world_rect,
    _smooth_view_fac,
)

if TYPE_CHECKING:
    from bpy.types import Context, Region, SpaceNodeEditor

    from ..core.state import MinimapState
    from .navigate import NODEMAP_OT_navigate

logger = logging.getLogger(base_package)

# Smooth-drag follow uses frame-rate independent exponential damping. Inertia
# decays the released view velocity each tick until it falls below the stop
# speed. Each drag tick applies the fraction ``1 - exp(-rate * dt)`` of the
# remaining target, so the per-second catch-up is identical at any timer
# rate and heavy redraws (big trees at low fps) do not stretch the lag. The
# rate grows with the remaining distance plus recent speed, so slow drags
# stay smooth while rapid or far drags catch up faster.
_INERTIA_DECAY: float = 0.92
_INERTIA_STOP_SPEED: float = 0.5
_DRAG_MAX_FRAME_DT: float = 0.25
_DRAG_DT_MIN: float = 0.001
_DRAG_DT_MAX: float = 0.10
_DRAG_LAMBDA: float = 14.0
_DRAG_BOOST_GAIN: float = 0.4
_DRAG_LAMBDA_MAX: float = 32.0
_ANIM_FINISH_EPS: float = 0.5


def _drag_rate(dist: float) -> float:
    """Return the drag catch-up rate for remaining distance *dist* in pixels.

    Pure: base rate plus a square-root boost, clamped to the maximum, so
    small corrections keep the smooth base feel while rapid or far drags
    chase harder without snapping.
    """
    if dist <= 0.0:
        return _DRAG_LAMBDA
    return min(_DRAG_LAMBDA + _DRAG_BOOST_GAIN * math.sqrt(dist), _DRAG_LAMBDA_MAX)


def _drag_alpha(dt: float, rate: float = _DRAG_LAMBDA) -> float:
    """Return the drag catch-up fraction for a frame delta of *dt* seconds.

    Pure: exponential damping ``1 - exp(-rate * dt)`` clamped to [0, 1], so
    the view covers the same share of the remaining distance per second at
    any tick rate. Non-positive deltas apply nothing.
    """
    if dt <= 0.0:
        return 0.0
    return min(1.0 - math.exp(-rate * dt), 1.0)


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
        "anim_total",
        "drag_target",
        "drag_active",
        "_last_drag_tick",
        "frame_anim_active",
        "frame_anim_start_zoom",
        "frame_anim_start_pan",
        "frame_anim_target_zoom",
        "frame_anim_target_pan",
        "frame_anim_progress",
        "frame_anim_total",
        "editor_anim_active",
        "editor_anim_progress",
        "editor_anim_start_rect",
        "editor_anim_target_rect",
        "editor_anim_total",
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
        self.anim_total: float = 1.0
        self.drag_target: list[float] = [0.0, 0.0]
        self.drag_active: bool = False
        self._last_drag_tick: float = 0.0
        self.frame_anim_active: bool = False
        self.frame_anim_start_zoom: float = 1.0
        self.frame_anim_start_pan: list[float] = [0.0, 0.0]
        self.frame_anim_target_zoom: float = 1.0
        self.frame_anim_target_pan: list[float] = [0.0, 0.0]
        self.frame_anim_progress: float = 0.0
        self.frame_anim_total: float = 1.0
        self.editor_anim_active: bool = False
        self.editor_anim_progress: float = 0.0
        self.editor_anim_start_rect: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.editor_anim_target_rect: list[float] = [0.0, 0.0, 0.0, 0.0]
        self.editor_anim_total: float = 1.0

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
        self.anim_total = 1.0
        self.drag_target = [0.0, 0.0]
        self.drag_active = False
        self._last_drag_tick = 0.0
        self.frame_anim_active = False
        self.frame_anim_start_zoom = 1.0
        self.frame_anim_start_pan = [0.0, 0.0]
        self.frame_anim_target_zoom = 1.0
        self.frame_anim_target_pan = [0.0, 0.0]
        self.frame_anim_progress = 0.0
        self.frame_anim_total = 1.0
        self.editor_anim_active = False
        self.editor_anim_progress = 0.0
        self.editor_anim_start_rect = [0.0, 0.0, 0.0, 0.0]
        self.editor_anim_target_rect = [0.0, 0.0, 0.0, 0.0]
        self.editor_anim_total = 1.0

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
        self.smooth_timer = context.window_manager.event_timer_add(PAN_ANIM_INTERVAL, window=context.window)

    def destroy_timer(self, context: Context) -> None:
        if self.smooth_timer:
            try:
                context.window_manager.event_timer_remove(self.smooth_timer)
            except (RuntimeError, ValueError):
                pass
            self.smooth_timer = None

    def _settings(self, context: Context):
        """Return the add-on settings for *context*."""
        return get_addon_preferences(context).settings

    def _total_frames(self, context: Context, fac: float) -> float:
        """Return the animation duration in frames for magnitude factor *fac*.

        Match Blender's ``view2d_smooth_view`` duration scaling
        (``smooth_viewtx * fac``): far view changes take the full pan-speed
        budget while near changes take fewer frames, never below the minimum
        visible floor.
        """
        base = PAN_FRAMES
        return max(min(base * fac, base), PAN_MIN_FRAMES)

    @staticmethod
    def _ease(progress: float) -> float:
        """Ease-out cubic interpolation for *progress* in [0, 1].

        Match the 1.5.0 FAST feel (``1 - (1 - t)^3``): fast start with a
        gentle landing, used by the click-to-pan center animation.
        """
        return 1.0 - (1.0 - progress) ** 3

    @staticmethod
    def _ease_smooth(progress: float) -> float:
        """Ease-in-out (smoothstep) interpolation for *progress* in [0, 1].

        Match the timer step of Blender's ``view2d_smooth_view``
        (``3t^2 - 2t^3``); frame and editor viewport animations share it.
        """
        return progress * progress * (3.0 - 2.0 * progress)

    def _take_pan(self, vx: float, vy: float) -> tuple[int, int]:
        """Accumulate *vx/vy* in the shared pan buffer and return the integer part."""
        op = self._op
        op._pan_acc[0] += vx
        op._pan_acc[1] += vy
        dx = int(op._pan_acc[0])
        dy = int(op._pan_acc[1])
        op._pan_acc[0] -= dx
        op._pan_acc[1] -= dy
        return dx, dy

    def _pan_editor(self, context: Context, dx: int, dy: int) -> None:
        """Pan the editor viewport by *dx/dy* pixels, ignoring no-ops."""
        if dx == 0 and dy == 0:
            return
        try:
            with self._op._override_ctx(context):
                bpy.ops.view2d.pan(deltax=dx, deltay=dy)
        except RuntimeError:
            pass

    def _finish_center_animation(self, context: Context) -> None:
        """Snap the view to the center-animation target and stop."""
        remaining_x = self.anim_target[0] - self.anim_applied[0]
        remaining_y = self.anim_target[1] - self.anim_applied[1]
        if abs(remaining_x) >= _ANIM_FINISH_EPS or abs(remaining_y) >= _ANIM_FINISH_EPS:
            self._pan_editor(context, int(remaining_x), int(remaining_y))
        self.anim_active = False
        self.destroy_timer(context)

    def _finish_frame_animation(self, context: Context) -> None:
        """Snap the minimap view to the frame-animation target and stop."""
        op = self._op
        state: MinimapState | None = op._state
        if state:
            state.view.anchor_zoom = self.frame_anim_target_zoom
            state.view.user_zoom = self.frame_anim_target_zoom
            state.view.pan = (self.frame_anim_target_pan[0], self.frame_anim_target_pan[1])
            _clamp_pan_to_viewport(op._space, op._region, state)
            # The final frame lands inside the zoom-settle window, which would
            # defer the batch rebuild; force it so the bake matches the target
            # scale even if no later redraw follows.
            state.cache._batches_dirty = True
        self.frame_anim_active = False
        self.frame_anim_progress = 0.0
        self.destroy_timer(context)

    def cancel_smooth(self, context: Context) -> None:
        """Snap all active animations to their targets and stop."""
        if self.inertia_active:
            self.inertia_active = False
            self.inertia_mode = None
            self.smooth_velocity = [0.0, 0.0]
            self.destroy_timer(context)
        if self.anim_active:
            self._finish_center_animation(context)
        if self.frame_anim_active:
            self._finish_frame_animation(context)
        if self.editor_anim_active:
            self._cancel_editor_animation(context)

    def apply_inertia(self, context: Context) -> None:
        """Decay inertia and apply pan deltas."""
        op = self._op
        if self.inertia_mode == "PAN":
            decay = INERTIA_PAN_DECAY
        else:
            decay = _INERTIA_DECAY
        self.smooth_velocity[0] *= decay
        self.smooth_velocity[1] *= decay
        speed = max(abs(self.smooth_velocity[0]), abs(self.smooth_velocity[1]))
        if speed < _INERTIA_STOP_SPEED:
            self.inertia_active = False
            self.inertia_mode = None
            self.destroy_timer(context)
            return
        dx, dy = self._take_pan(self.smooth_velocity[0], self.smooth_velocity[1])
        if self.inertia_mode == "PAN":
            state: MinimapState | None = op._state
            if state:
                if dx != 0 or dy != 0:
                    state.view.pan = (state.view.pan[0] + dx, state.view.pan[1] + dy)
                    _clamp_pan_to_viewport(op._space, op._region, state)
                    # State changed directly (no view2d operator): redraw now.
                    op._redraw_ui()
        elif self.inertia_mode == "VIEW":
            if dx != 0 or dy != 0:
                # view2d.pan tags the region redraw itself; no explicit call needed.
                self._pan_editor(context, dx, dy)
                _clamp_pan_to_viewport(op._space, op._region, op._state)

    def apply_smooth_drag(self, context: Context) -> None:
        """Chase the drag target with distance-adaptive exponential damping.

        Each tick applies the fraction ``1 - exp(-rate * dt)`` of the
        remaining target, so the per-second catch-up is identical at any
        tick rate and there is a single int() quantization point in
        ``_take_pan`` instead of magnitude-split micro-steps. The rate grows
        with the remaining distance plus recent speed, so far or rapid drags
        accelerate while small corrections keep the smooth base feel.
        """
        op = self._op
        if not self.drag_active:
            return
        now = time.monotonic()
        dt = now - self._last_drag_tick if self._last_drag_tick > 0.0 else PAN_ANIM_INTERVAL
        if dt <= 0.0 or dt > _DRAG_MAX_FRAME_DT:
            dt = PAN_ANIM_INTERVAL
        self._last_drag_tick = now
        dt = min(max(dt, _DRAG_DT_MIN), _DRAG_DT_MAX)
        dist = math.hypot(self.drag_target[0], self.drag_target[1])
        speed = math.hypot(self.smooth_velocity[0], self.smooth_velocity[1])
        follow = _drag_alpha(dt, _drag_rate(dist + speed))
        dx = self.drag_target[0] * follow
        dy = self.drag_target[1] * follow
        self.drag_target[0] -= dx
        self.drag_target[1] -= dy
        pan_x, pan_y = self._take_pan(dx, dy)
        if pan_x != 0 or pan_y != 0:
            # view2d.pan tags the region redraw itself; no explicit call needed.
            self._pan_editor(context, pan_x, pan_y)
            _clamp_pan_to_viewport(op._space, op._region, op._state)
        if not op._dragging:
            self.drag_active = False

    def start_center_animation(
        self, context: Context, pan_x: float, pan_y: float, visible: tuple[float, float, float, float]
    ) -> None:
        """Begin a pan-only center animation toward the given editor pixel delta."""
        from .navigate import _view_zoom_factors

        op = self._op
        view_zoom_x, view_zoom_y = _view_zoom_factors(op._space, op._region, visible)
        target_rect = (
            visible[0] + pan_x / view_zoom_x,
            visible[1] + pan_y / view_zoom_y,
            visible[2] + pan_x / view_zoom_x,
            visible[3] + pan_y / view_zoom_y,
        )
        self.anim_target = [pan_x, pan_y]
        self.anim_applied = [0.0, 0.0]
        self.anim_progress = 0.0
        self.anim_acc = [0.0, 0.0]
        self.anim_total = self._total_frames(context, _smooth_view_fac(visible, target_rect))
        self.anim_active = True
        self.create_timer(context)

    def apply_center_animation(self, context: Context) -> None:
        """Ease the view toward the center-animation target."""
        if not self.anim_active:
            return
        self.anim_progress += 1 / self.anim_total
        if self.anim_progress >= 1.0:
            self._finish_center_animation(context)
            return
        eased = self._ease(self.anim_progress)
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
            # view2d.pan tags the region redraw itself; no explicit call needed.
            self._pan_editor(context, dx, dy)

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
        start_world = _minimap_world_rect(state, self.frame_anim_start_zoom, self.frame_anim_start_pan)
        target_world = _minimap_world_rect(state, target_zoom, self.frame_anim_target_pan)
        self.frame_anim_total = self._total_frames(context, _smooth_view_fac(start_world, target_world))
        self.frame_anim_active = True
        self.create_timer(context)

    def apply_frame_animation(self, context: Context) -> None:
        """Step the frame zoom+pan animation forward one frame."""
        op = self._op
        if not self.frame_anim_active:
            return
        state: MinimapState | None = op._state
        if not state:
            self.frame_anim_active = False
            self.destroy_timer(context)
            return
        progress = self.frame_anim_progress + 1 / self.frame_anim_total
        self.frame_anim_progress = progress
        if progress >= 1.0:
            self._finish_frame_animation(context)
            op._redraw_ui()
            return
        eased = self._ease_smooth(progress)
        start_world = _minimap_world_rect(state, self.frame_anim_start_zoom, self.frame_anim_start_pan)
        target_world = _minimap_world_rect(state, self.frame_anim_target_zoom, self.frame_anim_target_pan)
        zoom, pan = _minimap_view_from_world_rect(state, _interp_rect(start_world, target_world, eased))
        state.view.user_zoom = zoom
        state.view.anchor_zoom = zoom
        state.view.pan = pan
        _clamp_pan_to_viewport(op._space, op._region, state)
        op._redraw_ui()

    def view_selected_animated(self, context: Context) -> bool:
        """Ease the editor viewport onto the selected nodes; True when started."""
        if not self._animations_enabled(context):
            return False
        targets = _compute_editor_frame_selected_targets(self._op._space, self._op._region)
        if targets is None:
            return False
        self.start_editor_animation(context, list(targets))
        return True

    def start_editor_animation(self, context: Context, target_rect: list[float]) -> None:
        """Begin animating the editor viewport toward the target tree-space rect."""
        op = self._op
        visible = _get_visible_rect(op._space, op._region)
        if not visible:
            return
        if self._editor_view_close(visible, target_rect):
            return
        self.editor_anim_progress = 0.0
        self.editor_anim_start_rect = [visible[0], visible[1], visible[2], visible[3]]
        self.editor_anim_target_rect = target_rect
        self.editor_anim_total = self._total_frames(context, _smooth_view_fac(visible, tuple(target_rect)))
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
        progress = self.editor_anim_progress + 1 / self.editor_anim_total
        if progress >= 1.0:
            self._correct_editor_view(context, self.editor_anim_target_rect)
            self.editor_anim_active = False
            self.editor_anim_progress = 0.0
            self.destroy_timer(context)
            return
        self.editor_anim_progress = progress
        eased = self._ease_smooth(progress)
        desired = [
            start + (target - start) * eased
            for start, target in zip(self.editor_anim_start_rect, self.editor_anim_target_rect)
        ]
        # _correct_editor_view drives view2d operators that tag the region
        # redraw themselves; no explicit redraw needed here.
        self._correct_editor_view(context, desired)

    def _animations_enabled(self, context: Context) -> bool:
        """Return True when animations are allowed by preferences and accessibility."""
        if context.preferences.view.use_reduce_motion:
            return False
        return self._settings(context).use_animations

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
        self._pan_editor(context, pan_x, pan_y)

    def _cancel_editor_animation(self, context: Context) -> None:
        """Snap the editor viewport to the animation target and stop stepping."""
        if not self.editor_anim_active:
            return
        self.editor_anim_active = False
        self.editor_anim_progress = 0.0
        if self._op._space and self._op._region:
            self._correct_editor_view(context, self.editor_anim_target_rect)
        self.destroy_timer(context)
