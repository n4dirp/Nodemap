"""Nodemap add-on preferences and logging infrastructure."""

import logging
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty
from bpy.types import AddonPreferences, PropertyGroup

from .helpers import MIN_MAP_HEIGHT, MIN_MAP_WIDTH

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")

# Guard flag: set True during handle drags to suppress property update
# callbacks, preventing tree_data invalidation and the resulting one-frame
# content flash.
_suppress_update = False


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
        enabled = prefs.logging_enabled
        level = prefs.logging_level
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


def _update_invalidate_all(self, context):
    """Invalidate batches and schedule recompile for structural preference changes.

    Used when a setting affects what tree data must be compiled (e.g. wire
    visibility, node labels). Keeps tree_data as a fallback so the next draw
    has content to show (no blank frame), clears the fingerprint so the draw
    detects the change and schedules a recompile, and sets force_immediate so
    the recompile fires on the next event loop iteration (~1 frame).
    """
    if _suppress_update:
        return
    try:
        from .helpers import redraw_ui
        from .state import _minimap_state

        for state in _minimap_state.values():
            state.cache.fingerprint = None
            state.cache._batches_dirty = True
            state.cache.force_immediate = True
        redraw_ui("NODE_EDITOR")
    except (ImportError, AttributeError):
        pass


def _update_invalidate_batches(self, context):
    """Invalidate GPU batches only for display preference changes.

    Used when a setting affects how content is rendered (size, position,
    opacity, colors) but not what tree data is needed. Preserves tree_data
    to avoid an expensive one-frame flash from a full tree recompile.
    """
    if _suppress_update:
        return
    try:
        from .helpers import redraw_ui
        from .state import _minimap_state

        for state in _minimap_state.values():
            state.cache.invalidate_batches_only()
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


_CLICK_ACTION_ITEMS = [
    ("PAN", "Pan View", "Center the view on the clicked location"),
    ("SELECT", "Select Node", "Select the node under the cursor"),
    ("SELECT_FRAME", "Select Node + Frame View", "Select the node and frame it in the editor"),
    ("SELECT_PAN", "Select Node + Pan View", "Select the node and pan the view"),
]


class NODEMAP_PG_settings(PropertyGroup):
    """Preferences for the Nodes Minimap."""

    show_by_default: BoolProperty(
        name="Show by Default",
        description="Show minimap on newly opened Node Editor areas",
        default=False,
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
        default="BOTTOM_RIGHT",
        update=_update_invalidate_batches,
    )

    minimap_width: IntProperty(
        name="Size X",
        description="Minimap width in pixels",
        default=300,
        min=MIN_MAP_WIDTH,
        subtype="PIXEL",
        update=_update_invalidate_batches,
    )

    minimap_height: IntProperty(
        name="Size Y",
        description="Minimap height in pixels",
        default=128,
        min=MIN_MAP_HEIGHT,
        subtype="PIXEL",
        update=_update_invalidate_batches,
    )

    max_width_pct: IntProperty(
        name="Max X %",
        description="Largest the minimap can be across, as a share of the available space",
        default=100,
        min=10,
        max=100,
        subtype="PERCENTAGE",
        update=_update_invalidate_batches,
    )

    max_height_pct: IntProperty(
        name="Max Y %",
        description="Largest the minimap can be up and down, as a share of the available space",
        default=100,
        min=10,
        max=100,
        subtype="PERCENTAGE",
        update=_update_invalidate_batches,
    )

    opacity: FloatProperty(
        name="Opacity",
        description="Adjusts the overall opacity of the minimap",
        default=0.95,
        min=0.15,
        max=1.0,
        precision=3,
        subtype="FACTOR",
        update=_update_invalidate_batches,
    )

    custom_bg_color: BoolProperty(
        name="Custom Background",
        description="Use a custom background color instead of the Blender theme color",
        default=False,
        update=_update_invalidate_batches,
    )

    bg_color: FloatVectorProperty(
        name="Background Color",
        description="Custom background color for the minimap overlay",
        default=(0.45, 0.45, 0.45, 0.95),
        size=4,
        min=0.0,
        max=1.0,
        subtype="COLOR_GAMMA",
        update=_update_invalidate_batches,
    )

    show_viewport_overlay: BoolProperty(
        name="Viewport Overlay",
        description="Show darkened overlay with viewport cutout in the minimap",
        default=True,
    )

    viewport_overlay_color: FloatVectorProperty(
        name="Viewport Overlay Color",
        description="Color of the viewport overlay",
        default=(0.1, 0.1, 0.1, 0.5),
        size=4,
        min=0.0,
        max=1.0,
        subtype="COLOR_GAMMA",
    )

    viewport_fill_rect: BoolProperty(
        name="Active View Fill",
        description="Fill the active view rect with a color",
        default=True,
    )

    viewport_fill_color: FloatVectorProperty(
        name="Active View Fill Color",
        description="Color of the active view fill rect",
        default=(0.278, 0.447, 0.702, 0.4),
        size=4,
        min=0.0,
        max=1.0,
        subtype="COLOR_GAMMA",
    )

    show_node_count: BoolProperty(
        name="Show Node Count",
        description="Display node count at the bottom of the minimap",
        default=True,
        update=_update_invalidate_batches,
    )

    show_frame_all_btn: BoolProperty(
        name="Frame All Button",
        description="Show a Frame-all button inside the minimap",
        default=True,
        update=_update_invalidate_batches,
    )

    show_frame_view_btn: BoolProperty(
        name="Frame View Button",
        description="Show a Frame-view button inside the minimap",
        default=True,
        update=_update_invalidate_batches,
    )

    show_frame_selected_btn: BoolProperty(
        name="Frame Selected Button",
        description="Show a Frame-selected button inside the minimap",
        default=True,
        update=_update_invalidate_batches,
    )

    show_list_toggle_btn: BoolProperty(
        name="List Toggle Button",
        description="Show a button in the minimap to toggle the node-type list",
        default=True,
        update=_update_invalidate_batches,
    )

    show_names: BoolProperty(
        name="Show Node Labels",
        description="Display labels inside minimap nodes",
        default=True,
        update=_update_invalidate_all,
    )

    show_frames: BoolProperty(
        name="Show Frames",
        description="Display frame node backgrounds in the minimap",
        default=True,
        update=_update_invalidate_all,
    )

    show_frame_labels: BoolProperty(
        name="Show Frame Labels",
        description="Display labels above frame nodes in the minimap",
        default=True,
        update=_update_invalidate_all,
    )

    node_label_mode: EnumProperty(
        name="Node Labels",
        description="How labels appear in the minimap",
        items=[
            ("COMPACT", "Initials", "Display abbreviated initials"),
            ("FULL", "Name", "Display full name split across lines"),
        ],
        default="COMPACT",
        update=_update_invalidate_all,
    )

    colored_nodes: BoolProperty(
        name="Colored Nodes",
        description="Use custom node colors and color tags",
        default=True,
        update=_update_invalidate_all,
    )

    show_wires: BoolProperty(
        name="Show Wires",
        description="Display node connections in the minimap",
        default=True,
        update=_update_invalidate_all,
    )

    show_wire_color: BoolProperty(
        name="Socket Wire Colors",
        description="Color wires by the output socket type",
        default=True,
        update=_update_invalidate_all,
    )

    show_socket_indicators: BoolProperty(
        name="Socket Indicators",
        description="Display colored indicator pills on node sockets",
        default=True,
        update=_update_invalidate_all,
    )

    show_node_borders: BoolProperty(
        name="Node Borders",
        description="Display borders around nodes, highlighting selection and active state",
        default=True,
        update=_update_invalidate_batches,
    )

    show_type_list: BoolProperty(
        name="Type List",
        description=(
            "Show an interactive node-type list beside the map; "
            "hovering a row highlights those nodes, clicking selects them"
        ),
        default=False,
        update=_update_invalidate_all,
    )

    show_type_colors: BoolProperty(
        name="Type Colors",
        description="Draw a colored swatch icon next to each entry in the type list",
        default=True,
        update=_update_invalidate_all,
    )

    type_list_sort: EnumProperty(
        name="Type List Sort",
        description="How entries are ordered in the node-type list",
        items=[
            ("NAME", "Alphabetical", "Order alphabetically by type name"),
            ("COUNT", "Count", "Order by node count, highest first"),
        ],
        default="NAME",
    )

    type_list_font_size: IntProperty(
        name="Type List Font Size",
        description="Font size for the node-type list entries (pixels)",
        default=10,
        min=8,
        max=20,
        update=_update_invalidate_all,
    )

    type_list_width_pct: IntProperty(
        name="Type List Width %",
        description="Width of the node-type list as a percentage of the minimap width",
        default=35,
        min=15,
        max=50,
        subtype="PERCENTAGE",
        update=_update_invalidate_batches,
    )

    debounce_interval: FloatProperty(
        name="Update Delay",
        description="Delay in seconds before the minimap updates after a change (0 = instant)",
        default=0.1,
        min=0.0,
        max=0.5,
        step=0.01,
        unit="TIME_ABSOLUTE",
    )

    interactive: BoolProperty(
        name="Interactive",
        description="Enable mouse and keyboard interaction with the minimap",
        default=True,
        update=_update_invalidate_all,
    )

    scroll_wheel_mode: EnumProperty(
        name="Scroll Wheel",
        description="Choose what the scroll wheel zooms (Hold Alt to switch)",
        items=[
            ("MINIMAP", "Minimap Zoom", "Zoom the minimap view"),
            ("NODE_EDITOR", "Node Editor Zoom", "Zoom the node editor view"),
        ],
        default="NODE_EDITOR",
    )

    follow_view: BoolProperty(
        name="Follow View",
        description="Keep the editor viewport inside the minimap by adjusting the minimap pan automatically",
        default=False,
        update=_update_invalidate_batches,
    )

    frame_view_fill: BoolProperty(
        name="Frame View Fill",
        description="Zoom in to the viewport while keeping it fully visible, instead of capping zoom at 1x",
        default=True,
    )

    left_click_action: EnumProperty(
        name="Left Click",
        description="Left click behavior in the minimap",
        items=_CLICK_ACTION_ITEMS,
        default="PAN",
    )

    right_click_action: EnumProperty(
        name="Right Click",
        description="Right click behavior in the minimap",
        items=_CLICK_ACTION_ITEMS,
        default="SELECT_FRAME",
    )

    animations: BoolProperty(
        name="Animations",
        description=(
            "Enable smooth animations for panning, framing, and the type list\n"
            "* Overridden by the Reduce Motion accessibility option"
        ),
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
        default="FAST",
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
        settings = self.settings

        layout.label(text="Nodemap")
        layout.prop(settings, "show_by_default", text="Show in New Editors")

        layout.separator()
        sub_body = layout.column()
        sub_body.active = settings.interactive
        col = sub_body.column()
        col.label(text="Navigation")

        col = sub_body.column(heading="Animations")
        _reduce_motion = context.preferences.view.use_reduce_motion
        col.active = not _reduce_motion
        row = col.row(align=True, heading="")
        row.prop(settings, "animations", text="")
        sub = row.row(align=True)
        sub.active = settings.animations
        sub.row().prop(settings, "pan_speed", expand=True)

        col.separator()
        col.prop(settings, "left_click_action", text="Left Click")
        col.prop(settings, "right_click_action", text="Right Click")
        col.row().prop(settings, "scroll_wheel_mode", expand=True)

        col.separator()
        split = col.split(factor=0.4)
        sub = split.column()
        sub.alignment = "RIGHT"
        sub.label(text="Keymap")
        col = split.column()
        wm = context.window_manager
        kc = wm.keyconfigs.user
        from . import addon_keymaps

        row = col.row()
        row.use_property_split = False
        for km_addon, kmi_addon in addon_keymaps:
            km = kc.keymaps.get(km_addon.name)
            if not km:
                continue
            kmi = km.keymap_items.get(kmi_addon.idname)
            if kmi:
                from rna_keymap_ui import draw_kmi

                draw_kmi([], kc, km, kmi, row, 0)
            else:
                layout.operator("nodemap.restore_keymap", text="Restore")

        layout.separator()
        layout.label(text="Layout")
        col = layout.column(align=True)
        col.prop(settings, "minimap_width", text="Size X")
        col.prop(settings, "minimap_height", text="Y")

        col = layout.column(align=True)
        col.prop(settings, "max_width_pct", text="Max Region X")
        col.prop(settings, "max_height_pct", text="Y")

        layout.separator()
        layout.label(text="Type List")
        col = layout.column()
        col.active = settings.show_type_list
        row = col.row()
        row.prop(settings, "type_list_sort", text="Sort", expand=True)
        col.prop(settings, "type_list_font_size", text="Font Size")
        sub = col.row()
        sub.active = settings.colored_nodes
        sub.prop(settings, "show_type_colors", text="Type Colors")

        layout.separator()
        layout.label(text="Performance")
        layout.prop(self.settings, "debounce_interval", text="Update Delay")

        layout.separator()
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
