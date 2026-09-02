"""Shared numeric and string constants for the nodemap extension."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Minimap geometry
# ---------------------------------------------------------------------------
MIN_MAP_WIDTH: int = 120
MIN_MAP_HEIGHT: int = 80

# ---------------------------------------------------------------------------
# Interaction handles
# ---------------------------------------------------------------------------
HANDLE_THICKNESS: int = 6

# ---------------------------------------------------------------------------
# Framing / zoom
# ---------------------------------------------------------------------------
MAX_FRAME_ZOOM: float = 20.0
EDITOR_FIT_MARGIN: float = 0.15

# ---------------------------------------------------------------------------
# Node labels
# ---------------------------------------------------------------------------
LABEL_MARGIN_PX: float = 12.0

# ---------------------------------------------------------------------------
# Type-list zone
# ---------------------------------------------------------------------------
TYPE_LIST_FONT_ID: int = 0
TYPE_LIST_FONT_SIZE: int = 10
TYPE_LIST_MIN_WIDTH: float = 70.0
TYPE_LIST_MAX_WIDTH_PCT: float = 0.5
LIST_PAD_X: float = 6.0
LIST_SWATCH: float = 8.0
LIST_SWATCH_GAP: float = 5.0
LIST_COUNT_GAP: float = 8.0
SCROLLBAR_HIT_PAD: float = 6.0
LIST_ANIM_FRAMES: dict[str, int] = {"FAST": 10, "MEDIUM": 20}
EMPTY_FINGERPRINT: tuple = (0, 0.0, "", 0, 0, 0, 0.0, 0.0, 0)

# Scrollbar appearance
SCROLLBAR_THICKNESS: float = 3.0
SCROLLBAR_THICKNESS_HOVER: float = 6.0
SCROLLBAR_INSET: float = 2.0
SCROLLBAR_MIN_THUMB: float = 6.0
SCROLLBAR_ALPHA: float = 0.65
TYPE_LIST_ANIM_AWAIT_TIMEOUT: float = 1.0
TYPE_LIST_MIN_LABEL_W: float = 32.0

# ---------------------------------------------------------------------------
# Minimap chrome (header buttons, font)
# ---------------------------------------------------------------------------
FONT_SIZE: int = 11
BUTTON_SIZE: int = 20
BUTTON_MARGIN: int = 0
BUTTON_HOVER_ALPHA: float = 0.015

# ---------------------------------------------------------------------------
# Batch building
# ---------------------------------------------------------------------------
MIN_SOCKET_SCALE: float = 0.15
SCALE_REBUILD_REL: float = 0.015
BATCH_DRIFT_PX: float = 256.0
CULL_MARGIN_PX: float = BATCH_DRIFT_PX + 32.0

# ---------------------------------------------------------------------------
# Socket indicator
# ---------------------------------------------------------------------------
SOCKET_PILL_SIZE_MULTIPLIER: float = 2.0

# ---------------------------------------------------------------------------
# Node rendering
# ---------------------------------------------------------------------------
NODE_ROUNDNESS_DEFAULT: float = 2.0
