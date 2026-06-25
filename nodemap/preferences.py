"""Nodes Minimap add-on preferences and logging infrastructure."""

import logging
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, PointerProperty
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


class NODES_MINIMAP_PG_settings(PropertyGroup):
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
    )

    minimap_width: IntProperty(
        name="Size X",
        description="Minimap width in pixels",
        default=256,
        min=64,
        subtype="PIXEL",
    )

    minimap_height: IntProperty(
        name="Size Y",
        description="Minimap height in pixels",
        default=128,
        min=64,
        subtype="PIXEL",
    )

    max_width_pct: IntProperty(
        name="Max X %",
        description="Maximum minimap width as percentage of the safe region width",
        default=75,
        min=10,
        max=100,
        subtype="PERCENTAGE",
    )

    max_height_pct: IntProperty(
        name="Max Y %",
        description="Maximum minimap height as percentage of the safe region height",
        default=75,
        min=10,
        max=100,
        subtype="PERCENTAGE",
    )

    opacity: FloatProperty(
        name="Opacity",
        description="Background opacity of the minimap",
        default=1.0,
        min=0.1,
        max=1.0,
        precision=3,
        subtype="FACTOR",
    )

    show_node_count: BoolProperty(
        name="Show Node Count",
        description="Display node count at the bottom of the minimap",
        default=True,
    )

    show_names: BoolProperty(
        name="Show Node Labels",
        description="Display labels inside minimap nodes",
        default=True,
    )

    node_label_mode: EnumProperty(
        name="",
        description="How to display node labels in the minimap",
        items=[
            ("COMPACT", "Name Initials", "Display abbreviated initials"),
            ("FULL", "Full Name", "Display full name split across lines"),
        ],
        default="COMPACT",
    )

    colored_nodes: BoolProperty(
        name="Colored Nodes",
        description="Use node custom colors and color tags; when disabled all nodes use the theme default",
        default=True,
    )

    show_wires: BoolProperty(
        name="Show Wires",
        description="Display node connections in the minimap",
        default=True,
    )

    show_wire_color: BoolProperty(
        name="Socket Wire Colors",
        description="Color wires by the output socket type",
        default=True,
    )

    auto_frame_selected: BoolProperty(
        name="Auto Frame Selected",
        description="Automatically frame the selected node when clicking in the minimap",
        default=True,
    )

    interactive: BoolProperty(
        name="Interactive",
        description="Enable mouse and keyboard interaction with the minimap",
        default=True,
    )

    scroll_wheel_mode: EnumProperty(
        name="Scroll Wheel",
        description="What the scroll wheel controls when hovering the minimap",
        items=[
            ("MINIMAP", "Minimap Zoom", "Zoom the minimap's internal view"),
            ("NODE_EDITOR", "Node Editor Zoom", "Zoom the actual node editor viewport"),
        ],
        default="MINIMAP",
    )

    left_click_action: EnumProperty(
        name="Left Click",
        description="Left click behavior in the minimap",
        items=[
            ("PAN", "Pan View", "Center the view on the clicked location"),
            ("SELECT", "Select Node", "Select the node under the cursor"),
            ("PAN_SELECT", "Pan + Select", "Center the view and select the node"),
        ],
        default="SELECT",
    )

    right_click_action: EnumProperty(
        name="Right Click",
        description="Right click behavior in the minimap",
        items=[
            ("PAN", "Pan View", "Center the view on the clicked location"),
            ("SELECT", "Select Node", "Select the node under the cursor"),
            ("PAN_SELECT", "Pan + Select", "Center the view and select the node"),
        ],
        default="PAN",
    )


class NODES_MINIMAP_AddonPreferences(AddonPreferences):
    """Add-on preferences for Nodes Minimap."""

    bl_idname = __package__

    settings: PointerProperty(type=NODES_MINIMAP_PG_settings)

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

        layout.label(text="Development")
        row = layout.row(align=True, heading="Console Logging")
        row.prop(self, "logging_enabled", text="")
        sub = row.row(align=True)
        sub.active = self.logging_enabled
        sub.prop(self, "logging_level", text="")


classes = (
    NODES_MINIMAP_PG_settings,
    NODES_MINIMAP_AddonPreferences,
)
