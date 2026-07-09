"""Nodemap add-on preferences and logging infrastructure."""

import logging
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty
from bpy.types import AddonPreferences, PropertyGroup

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def _trace_logger(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, msg, args, **kwargs)


logging.Logger.trace = _trace_logger


def _update_logger_from_prefs():
    """Configures the logger based on user preferences (Opt-in logging)."""
    logger = logging.getLogger(__package__)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    enabled = False
    level = "INFO"
    try:
        prefs = bpy.context.preferences.addons.get(__package__).preferences
        enabled = getattr(prefs, "logging_enabled", False)
        level = getattr(prefs, "logging_level", "INFO")
    except (KeyError, AttributeError, ReferenceError):
        pass

    if not enabled:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return

    level_map = {"INFO": logging.INFO, "DEBUG": logging.DEBUG, "TRACE": TRACE_LEVEL}
    handler = logging.StreamHandler()
    handler.setFormatter(AddonLogFormatter(with_level=True))

    logger.addHandler(handler)
    logger.setLevel(level_map[level])


def _update_minimap_cache(self, context):
    """Invalidate compiled batches in all active states and trigger UI redraw."""
    try:
        # Avoid module-level circular imports
        from .helpers import _minimap_state, redraw_ui

        for state in _minimap_state.values():
            if "cached_key" in state:
                state["cached_key"] = None
            state.pop("cached_fingerprint", None)
        redraw_ui("NODE_EDITOR")
    except (ImportError, AttributeError):
        pass


class AddonLogFormatter(logging.Formatter):
    """Custom formatter to provide timestamped and addon-prefixed logs."""

    def __init__(self, with_level=False):
        super().__init__()
        self.start_time = time.time()
        self.with_level = with_level

    def format(self, record):
        """Formats the log record with relative timestamps."""
        rel_time = record.created - self.start_time
        minutes, seconds = divmod(rel_time, 60)
        timestamp = f"{int(minutes):02d}:{seconds:06.3f}"
        short_name = __package__.rsplit(".", 1)[-1]

        if self.with_level:
            return f"{timestamp}  {short_name:<16} | {record.levelname.title()}: {record.getMessage()}"

        return f"{timestamp}  {short_name:<16} | {record.getMessage()}"


class NODEMAP_PG_settings(PropertyGroup):
    """Preferences for the Nodes Minimap."""

    show_by_default: BoolProperty(
        name="Show by Default",
        description="Show minimap on newly opened Node Editor areas",
        default=True,
    )

    position: EnumProperty(
        name="Position",
        description="Minimap corner position",
        items=[
            ("TOP_LEFT", "Top Left", "Display in the top-left corner"),
            ("TOP_RIGHT", "Top Right", "Display in the top-right corner"),
            ("BOTTOM_LEFT", "Bottom Left", "Display in the bottom-left corner"),
            ("BOTTOM_RIGHT", "Bottom Right", "Display in the bottom-right corner"),
        ],
        default="TOP_LEFT",
        update=_update_minimap_cache,
    )

    minimap_width: IntProperty(
        name="Size X",
        description="Minimap width in pixels",
        default=256,
        min=64,
        subtype="PIXEL",
        update=_update_minimap_cache,
    )

    minimap_height: IntProperty(
        name="Size Y",
        description="Minimap height in pixels",
        default=128,
        min=64,
        subtype="PIXEL",
        update=_update_minimap_cache,
    )

    max_width_pct: IntProperty(
        name="Max X %",
        description="Maximum minimap width as percentage of the safe region width",
        default=75,
        min=10,
        max=100,
        subtype="PERCENTAGE",
        update=_update_minimap_cache,
    )

    max_height_pct: IntProperty(
        name="Max Y %",
        description="Maximum minimap height as percentage of the safe region height",
        default=75,
        min=10,
        max=100,
        subtype="PERCENTAGE",
        update=_update_minimap_cache,
    )

    opacity: FloatProperty(
        name="Opacity",
        description="Adjusts the overall opacity of the minimap",
        default=1.0,
        min=0.1,
        max=1.0,
        precision=3,
        subtype="FACTOR",
        update=_update_minimap_cache,
    )

    custom_bg_color: BoolProperty(
        name="Custom Background",
        description="Use a custom background color instead of the Blender theme color",
        default=False,
        update=_update_minimap_cache,
    )

    bg_color: FloatVectorProperty(
        name="Background Color",
        description="Custom background color for the minimap overlay",
        default=(0.45, 0.45, 0.45, 0.95),
        size=4,
        min=0.0,
        max=1.0,
        subtype="COLOR_GAMMA",
        update=_update_minimap_cache,
    )

    show_viewport_overlay: BoolProperty(
        name="Viewport Overlay",
        description="Show darkened overlay with viewport cutout in the minimap",
        default=True,
    )

    viewport_overlay_color: FloatVectorProperty(
        name="Viewport Overlay Color",
        description="Color of the viewport overlay",
        default=(0.0, 0.0, 0.0, 0.5),
        size=4,
        min=0.0,
        max=1.0,
        subtype="COLOR_GAMMA",
    )

    show_node_count: BoolProperty(
        name="Show Node Count",
        description="Display node count at the bottom of the minimap",
        default=True,
        update=_update_minimap_cache,
    )

    show_frame_all_btn: BoolProperty(
        name="Frame All Button",
        description="Show a frame-all button inside the minimap",
        default=False,
        update=_update_minimap_cache,
    )

    show_names: BoolProperty(
        name="Show Node Labels",
        description="Display labels inside minimap nodes",
        default=True,
        update=_update_minimap_cache,
    )

    show_frames: BoolProperty(
        name="Show Frames",
        description="Display frame node backgrounds in the minimap",
        default=True,
        update=_update_minimap_cache,
    )

    show_frame_labels: BoolProperty(
        name="Show Frame Labels",
        description="Display labels above frame nodes in the minimap",
        default=True,
        update=_update_minimap_cache,
    )

    node_label_mode: EnumProperty(
        name="Node Labels",
        description="How labels appear in the minimap",
        items=[
            ("COMPACT", "Initials", "Display abbreviated initials"),
            ("FULL", "Name", "Display full name split across lines"),
        ],
        default="COMPACT",
        update=_update_minimap_cache,
    )

    colored_nodes: BoolProperty(
        name="Colored Nodes",
        description="Use custom node colors and color tags",
        default=True,
        update=_update_minimap_cache,
    )

    show_wires: BoolProperty(
        name="Show Wires",
        description="Display node connections in the minimap",
        default=True,
        update=_update_minimap_cache,
    )

    show_wire_color: BoolProperty(
        name="Socket Wire Colors",
        description="Color wires by the output socket type",
        default=True,
        update=_update_minimap_cache,
    )

    show_socket_indicators: BoolProperty(
        name="Socket Indicators",
        description="Display colored indicator pills on node sockets",
        default=False,
        update=_update_minimap_cache,
    )

    debounce_interval: FloatProperty(
        name="Update Delay",
        description="Delay in seconds before the minimap updates after a change (0 = instant)",
        default=0.08,
        min=0.0,
        max=0.5,
        step=0.01,
        unit="TIME_ABSOLUTE",
    )

    auto_frame_selected: BoolProperty(
        name="Auto Frame Selected",
        description="Automatically frame the selected node",
        default=True,
    )

    interactive: BoolProperty(
        name="Interactive",
        description="Enable mouse and keyboard interaction with the minimap",
        default=True,
        update=_update_minimap_cache,
    )

    scroll_wheel_mode: EnumProperty(
        name="Scroll Wheel",
        description="Choose what the scroll wheel zooms (Hold Alt to switch)",
        items=[
            ("NODE_EDITOR", "Editor Zoom", "Zoom the node editor view"),
            ("MINIMAP", "Minimap Zoom", "Zoom the minimap view"),
        ],
        default="NODE_EDITOR",
    )

    follow_view: BoolProperty(
        name="Follow View",
        description="Keep the editor viewport inside the minimap by adjusting the minimap pan automatically",
        default=False,
        update=_update_minimap_cache,
    )

    frame_view_fill: BoolProperty(
        name="Frame View Fill",
        description="Zoom in to the viewport while keeping it fully visible, instead of capping zoom at 1x",
        default=True,
    )

    left_click_action: EnumProperty(
        name="Left Click",
        description="Left click behavior in the minimap",
        items=[
            ("PAN", "Pan View", "Center the view on the clicked location"),
            ("SELECT", "Select Node", "Select the node under the cursor"),
            ("PAN_SELECT", "Pan + Select", "Center the view and select the node"),
        ],
        default="PAN",
    )

    right_click_action: EnumProperty(
        name="Right Click",
        description="Right click behavior in the minimap",
        items=[
            ("PAN", "Pan View", "Center the view on the clicked location"),
            ("SELECT", "Select Node", "Select the node under the cursor"),
            ("PAN_SELECT", "Pan + Select", "Center the view and select the node"),
        ],
        default="SELECT",
    )

    smooth_pan: BoolProperty(
        name="Smooth Pan",
        description="Apply inertia and smooth animations when panning the view with the minimap",
        default=True,
    )

    pan_speed: EnumProperty(
        name="Pan Speed",
        description="Animation speed for click-to-pan",
        items=[
            ("FAST", "Fast", "Quick snap (0.2s)"),
            ("MEDIUM", "Medium", "Balanced (0.4s)"),
            ("SLOW", "Slow", "Leisurely (0.67s)"),
        ],
        default="MEDIUM",
    )


class NODEMAP_AddonPreferences(AddonPreferences):
    """Add-on preferences for Nodes Minimap."""

    bl_idname = __package__

    settings: PointerProperty(type=NODEMAP_PG_settings)

    logging_enabled: BoolProperty(
        name="Enable Console Logging",
        description="Output add-on log messages to the console",
        default=False,
        update=lambda self, context: _update_logger_from_prefs(),
    )

    logging_level: EnumProperty(
        name="Log Level",
        items=[
            ("INFO", "Info", "Major events and state changes"),
            ("DEBUG", "Debug", "Detailed operational information"),
            ("TRACE", "Verbose", "Performance timing and cache operations"),
        ],
        default="INFO",
        update=lambda self, context: _update_logger_from_prefs(),
    )

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        layout.label(text="Performance")
        layout.prop(self.settings, "debounce_interval", text="Update Delay")

        layout.label(text="Development")
        row = layout.row(align=True, heading="Console Logging")
        row.prop(self, "logging_enabled", text="")
        sub = row.row(align=True)
        sub.active = self.logging_enabled
        sub.prop(self, "logging_level", text="")


classes = (
    NODEMAP_PG_settings,
    NODEMAP_AddonPreferences,
)
