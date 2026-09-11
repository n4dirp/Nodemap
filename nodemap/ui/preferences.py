"""Nodemap add-on preferences and logging infrastructure."""

import logging
import time

from bpy.props import BoolProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty, PointerProperty
from bpy.types import AddonPreferences, PropertyGroup

from .. import __package__ as base_package
from ..core.constants import MIN_MAP_HEIGHT, MIN_MAP_WIDTH, TYPE_LIST_MIN_WIDTH
from ..core.helpers import get_addon_preferences
from ..core.state import _suppress_update
from .panels import NODEMAP_PT_presets

TRACE_LEVEL = 5
logging.addLevelName(TRACE_LEVEL, "TRACE")


def _trace_logger(self, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL):
        self._log(TRACE_LEVEL, msg, args, **kwargs)


logging.Logger.trace = _trace_logger


def _update_logger_from_prefs():
    """Configure the logger based on user preferences (Opt-in logging)."""
    logger = logging.getLogger(base_package)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    enabled = False
    level = "INFO"
    try:
        prefs = get_addon_preferences()
        enabled = prefs.use_logging
        level = prefs.logging_level
    except (KeyError, AttributeError, ReferenceError):
        pass

    if not enabled:
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return

    level_map = {"INFO": logging.INFO, "DEBUG": logging.DEBUG, "TRACE": TRACE_LEVEL}
    handler = logging.StreamHandler()
    handler.setFormatter(AddonLogFormatter(with_level=False))

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
        from ..core.helpers import redraw_ui
        from ..core.state import _minimap_state, _shared_tree_caches

        for state in _minimap_state.values():
            state.cache._batches_dirty = True
        for shared in _shared_tree_caches.values():
            shared.fingerprint = None
            shared.force_immediate = True
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
        from ..core.helpers import redraw_ui
        from ..core.state import _minimap_state

        for state in _minimap_state.values():
            state.cache.invalidate_batches_only()
        redraw_ui("NODE_EDITOR")
    except (ImportError, AttributeError):
        pass


class AddonLogFormatter(logging.Formatter):
    """Provide timestamped and addon-prefixed logs."""

    def __init__(self, with_level=False):
        super().__init__()
        self.start_time = time.time()
        self.with_level = with_level

    def format(self, record):
        """Format the log record with relative timestamps."""
        rel_time = record.created - self.start_time
        minutes, seconds = divmod(rel_time, 60)
        timestamp = f"{int(minutes):02d}:{seconds:06.3f}"
        package_short_name = base_package.rsplit(".", 1)[-1]

        if self.with_level:
            return f"{timestamp}  {package_short_name:<16} | {record.levelname.title()}: {record.getMessage()}"

        return f"{timestamp}  {package_short_name:<16} | {record.getMessage()}"


_CLICK_ACTION_ITEMS = [
    ("PAN", "Pan View", "Center the view on the clicked location"),
    ("SELECT", "Select Node", "Select the node under the cursor"),
    ("SELECT_FRAME", "Select Node + Frame View", "Select the node and frame it in the editor"),
    ("SELECT_PAN", "Select Node + Pan View", "Select the node and pan the view"),
]

_DRAG_ACTION_ITEMS = [
    ("CENTER_PAN", "Center Pan", "Center the view on the cursor and drag to pan it around that point"),
    ("PAN", "Pan View", "Drag to pan the node editor view"),
    ("FRAME_RECT", "Frame Region", "Drag to draw a rectangle and frame that area"),
    ("FRAME_RECT_EDITOR", "Frame Region (Editor)", "Drag to draw a rectangle and frame that area in the editor"),
]


class NODEMAP_PG_settings(PropertyGroup):
    """Store preferences for the Nodes Minimap."""

    show_by_default: BoolProperty(
        name="Show by Default",
        description="Show minimap on newly opened Node Editor areas",
        default=False,
    )

    dock_mode: EnumProperty(
        name="Dock Mode",
        description="How to dock the minimap in the editor",
        items=[
            ("CORNER", "Corner", "Dock to one of the four corners"),
            ("EDGE", "Edge", "Dock along one of the four edges"),
            ("FREE", "Free", "Place the minimap anywhere by dragging"),
        ],
        default="CORNER",
        update=_update_invalidate_batches,
    )

    corner_position: EnumProperty(
        name="Corner",
        description="Which corner to dock the minimap in",
        items=[
            ("TOP_LEFT", "Top-Left", "Dock in the top-left corner"),
            ("TOP_RIGHT", "Top-Right", "Dock in the top-right corner"),
            ("BOTTOM_LEFT", "Bottom-Left", "Dock in the bottom-left corner"),
            ("BOTTOM_RIGHT", "Bottom-Right", "Dock in the bottom-right corner"),
        ],
        default="TOP_LEFT",
        update=_update_invalidate_batches,
    )

    edge_position: EnumProperty(
        name="Edge",
        description="Which edge to dock the minimap along",
        items=[
            ("TOP_BORDER", "Top", "Dock along the top edge"),
            ("BOTTOM_BORDER", "Bottom", "Dock along the bottom edge"),
            ("LEFT_BORDER", "Left", "Dock along the left edge"),
            ("RIGHT_BORDER", "Right", "Dock along the right edge"),
        ],
        default="TOP_BORDER",
        update=_update_invalidate_batches,
    )

    @property
    def current_position(self) -> str:
        """Resolve the effective dock position from dock_mode + sub-position."""
        if self.dock_mode == "FREE":
            return "FREE"
        if self.dock_mode == "EDGE":
            return self.edge_position
        return self.corner_position

    offset_x: IntProperty(
        name="Offset X",
        description="Horizontal offset from the default dock position, in pixels",
        default=0,
        subtype="PIXEL",
        update=_update_invalidate_batches,
    )

    offset_y: IntProperty(
        name="Offset Y",
        description="Vertical offset from the default dock position, in pixels",
        default=0,
        subtype="PIXEL",
        update=_update_invalidate_batches,
    )

    use_snap_to_borders: BoolProperty(
        name="Snap to Borders",
        description="Snap the minimap to editor borders and corners when dragging it",
        default=True,
        update=_update_invalidate_batches,
    )

    minimap_width: IntProperty(
        name="Size X",
        description="Minimap width in pixels",
        default=400,
        min=MIN_MAP_WIDTH,
        subtype="PIXEL",
        update=_update_invalidate_batches,
    )

    minimap_height: IntProperty(
        name="Size Y",
        description="Minimap height in pixels",
        default=124,
        min=MIN_MAP_HEIGHT,
        subtype="PIXEL",
        update=_update_invalidate_batches,
    )

    max_width_percent: IntProperty(
        name="Max Width",
        description="Maximum share of the available width the minimap can occupy",
        default=100,
        min=10,
        max=100,
        subtype="PERCENTAGE",
        update=_update_invalidate_batches,
    )

    max_height_percent: IntProperty(
        name="Max Height",
        description="Maximum share of the available height the minimap can occupy",
        default=100,
        min=10,
        max=100,
        subtype="PERCENTAGE",
        update=_update_invalidate_batches,
    )

    opacity: FloatProperty(
        name="Opacity",
        description="Adjusts the overall opacity of the minimap",
        default=1.0,
        min=0.15,
        max=1.0,
        precision=3,
        subtype="FACTOR",
        update=_update_invalidate_all,
    )

    use_custom_background: BoolProperty(
        name="Custom Background",
        description="Use a custom background color instead of the Blender theme color",
        default=True,
        update=_update_invalidate_batches,
    )

    background_color: FloatVectorProperty(
        name="Background Color",
        description="Custom background color for the minimap overlay",
        default=(0.157, 0.157, 0.157, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype="COLOR_GAMMA",
        update=_update_invalidate_batches,
    )

    use_custom_text: BoolProperty(
        name="Custom Text Color",
        description="Override the Blender theme text color with a custom color",
        default=False,
        update=_update_invalidate_all,
    )

    text_color: FloatVectorProperty(
        name="Text Color",
        description="Custom text color for minimap labels and the type list",
        default=(1.0, 1.0, 1.0, 1.0),
        size=4,
        min=0.0,
        max=1.0,
        subtype="COLOR_GAMMA",
        update=_update_invalidate_all,
    )

    show_text_shadow: BoolProperty(
        name="Text Shadows",
        description="Draw a shadow behind minimap and type-list text",
        default=True,
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
        default=(0.0, 0.0, 0.0, 0.4),
        size=4,
        min=0.0,
        max=1.0,
        subtype="COLOR_GAMMA",
    )

    use_custom_viewport_fill: BoolProperty(
        name="Active View Fill",
        description="Fill the active view rect with a color",
        default=True,
    )

    viewport_fill_color: FloatVectorProperty(
        name="Active View Fill Color",
        description="Color of the active view fill rect",
        default=(0.278, 0.447, 0.702, 1.0),
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

    show_frame_all_button: BoolProperty(
        name="Frame All Button",
        description="Show a Frame-all button inside the minimap",
        default=True,
        update=_update_invalidate_batches,
    )

    show_frame_view_button: BoolProperty(
        name="Frame View Button",
        description="Show a Frame-view button inside the minimap",
        default=True,
        update=_update_invalidate_batches,
    )

    show_frame_selected_button: BoolProperty(
        name="Frame Selected Button",
        description="Show a Frame-selected button inside the minimap",
        default=True,
        update=_update_invalidate_batches,
    )

    show_list_toggle_button: BoolProperty(
        name="List Toggle Button",
        description="Show a button in the minimap to toggle the node-type list",
        default=True,
        update=_update_invalidate_batches,
    )

    show_move_button: BoolProperty(
        name="Show Move Handle",
        description="Show a button in the minimap to drag and reposition the minimap",
        default=True,
        update=_update_invalidate_batches,
    )

    show_node_labels: BoolProperty(
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

    compact_node_labels: BoolProperty(
        name="Compact Node Labels",
        description="Display abbreviated initials instead of full node names",
        default=True,
        update=_update_invalidate_all,
    )

    show_node_colors: BoolProperty(
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
    highlight_selected_wires: BoolProperty(
        name="Highlight Selected Wires",
        description="Draw wires connected to selected nodes in the theme selection color",
        default=True,
        update=_update_invalidate_all,
    )
    use_custom_noodle_curving: BoolProperty(
        name="Custom Noodle Curving",
        description="Use a custom noodle curving value instead of the Blender theme value",
        default=False,
        update=_update_invalidate_batches,
    )
    noodle_curving: IntProperty(
        name="Noodle Curving",
        description="Curving of the noodle",
        default=0,
        min=0,
        max=10,
        update=_update_invalidate_batches,
    )
    wire_thickness: FloatProperty(
        name="Wire Thickness",
        description="Multiplier applied to the minimap wire thickness",
        default=0.5,
        min=0.25,
        soft_max=2.0,
        max=3.0,
        precision=2,
        step=0.1,
        update=_update_invalidate_batches,
    )
    wire_opacity: FloatProperty(
        name="Wire Color Alpha",
        description="Multiplier applied to the alpha of minimap wire colors",
        default=0.5,
        min=0.1,
        max=1.0,
        precision=3,
        step=0.1,
        update=_update_invalidate_all,
    )
    show_wire_color: BoolProperty(
        name="Socket Wire Colors",
        description="Color wires by the output socket type",
        default=True,
        update=_update_invalidate_all,
    )
    show_dashed_wires: BoolProperty(
        name="Dashed Field Wires",
        description="Draw field and modifier wires dashed in Geometry Node trees",
        default=True,
        update=_update_invalidate_all,
    )

    show_socket_indicators: BoolProperty(
        name="Socket Indicators",
        description="Display colored indicator pills on node sockets",
        default=True,
        update=_update_invalidate_all,
    )

    show_reroutes: BoolProperty(
        name="Reroute Nodes",
        description="Display reroute nodes as colored pills",
        default=True,
        update=_update_invalidate_all,
    )

    show_node_outline: BoolProperty(
        name="Node Outline",
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

    use_group_by_type: BoolProperty(
        name="Group by Type",
        description="Group nodes by type in the type list. Disable to show one row per node",
        default=True,
        update=_update_invalidate_batches,
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

    type_list_width: IntProperty(
        name="Type List Width",
        description="Size of the node-type list in pixels (height when the list is on top)",
        default=160,
        min=int(TYPE_LIST_MIN_WIDTH),
        max=10000,
        subtype="PIXEL",
        update=_update_invalidate_batches,
    )

    type_list_position: EnumProperty(
        name="Type List Position",
        description="Where to place the node-type list (Auto moves it on top when the minimap is taller than wide)",
        items=[
            ("AUTO", "Auto", "Place the list on top when the minimap is taller than wide, else on the left"),
            ("LEFT", "Left", "Always place the list along the left edge"),
            ("TOP", "Top", "Always place the list across the top edge"),
        ],
        default="AUTO",
        update=_update_invalidate_batches,
    )

    show_search_bar: BoolProperty(
        name="Filter Bar",
        description="Show the filter field at the top of the node-type list",
        default=True,
        update=_update_invalidate_all,
    )

    use_follow_active: BoolProperty(
        name="Follow Active",
        description="Scroll the type list to reveal the active node whenever it changes",
        default=True,
        update=_update_invalidate_batches,
    )

    debounce_delay: FloatProperty(
        name="Debounce Delay",
        description="Delay in seconds before the minimap updates after a change (0 = instant)",
        default=0.1,
        min=0.0,
        max=0.5,
        step=0.01,
        unit="TIME_ABSOLUTE",
    )

    use_interactive: BoolProperty(
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

    use_follow_view: BoolProperty(
        name="Follow View",
        description="Keep the editor viewport inside the minimap by adjusting the minimap pan automatically",
        default=False,
        update=_update_invalidate_batches,
    )

    use_auto_zoom: BoolProperty(
        name="Auto Zoom",
        description="Automatically adjust the minimap zoom when node tree bounds change",
        default=True,
        update=_update_invalidate_batches,
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

    left_drag_action: EnumProperty(
        name="Left Drag",
        description="Left drag behavior in the minimap",
        items=_DRAG_ACTION_ITEMS,
        default="CENTER_PAN",
    )

    right_drag_action: EnumProperty(
        name="Right Drag",
        description="Right drag behavior in the minimap",
        items=_DRAG_ACTION_ITEMS,
        default="FRAME_RECT",
    )

    use_animations: BoolProperty(
        name="Animations",
        description=(
            "Enable smooth animations for panning, framing, and the type list\n"
            "* Overridden by the Reduce Motion accessibility option"
        ),
        default=True,
    )


class NODEMAP_AddonPreferences(AddonPreferences):
    """Store add-on preferences for the Nodes Minimap."""

    bl_idname = base_package

    settings: PointerProperty(type=NODEMAP_PG_settings)

    use_logging: BoolProperty(
        name="Console Logging",
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

        row = layout.row()
        row.label(text="General")

        NODEMAP_PT_presets.draw_panel_header(row)

        layout.prop(settings, "show_by_default", text="Show in New Editors")

        layout.separator(type="LINE")
        group = layout.column()
        group.label(text="Layout")

        group.row().prop(settings, "dock_mode", text="Dock Mode", expand=True)

        col = group.column(align=True)
        if settings.dock_mode == "CORNER":
            col.prop(settings, "corner_position", text="Corner")
        elif settings.dock_mode == "EDGE":
            col.prop(settings, "edge_position", text="Edge")
        else:  # FREE
            col.prop(settings, "offset_x", text="Offset X")
            col.prop(settings, "offset_y", text="Y")
            group.prop(settings, "use_snap_to_borders", text="Snap to Borders")
        group.separator()

        col = group.column(align=True)
        col.prop(settings, "minimap_width", text="Size X")
        col.prop(settings, "minimap_height", text="Y")

        col = group.column(align=True)
        col.prop(settings, "max_width_percent", text="Max Width")
        col.prop(settings, "max_height_percent", text="Max Height")

        layout.separator(type="LINE")
        group = layout.column()
        group.label(text="Objects")
        split = group.split(factor=0.4)
        split.label(text="")
        row = split.row()
        row.use_property_split = False
        col = row.column(heading="Show")
        col.prop(settings, "show_frames", text="Frames")
        col.prop(settings, "show_node_colors", text="Node Colors")
        col.prop(settings, "show_node_outline", text="Node Outline")
        col.prop(settings, "show_socket_indicators", text="Node Sockets")
        col.prop(settings, "show_reroutes", text="Reroutes")
        col.prop(settings, "show_node_count", text="Total Count")
        sub = col.row()
        sub.active = settings.use_interactive
        sub.prop(settings, "show_type_list", text="Type List")
        col.prop(settings, "show_wires", text="Wires")

        col = row.column(heading="Labels")
        col.prop(settings, "show_node_labels", text="Node Labels")
        if settings.show_node_labels:
            col.prop(settings, "compact_node_labels", text="Compact Node Labels")
        if settings.show_frames:
            col.prop(settings, "show_frame_labels", text="Frame Labels")

        col = row.column(heading="Buttons")
        col.active = settings.use_interactive
        col.prop(settings, "show_frame_all_button", text="Frame All")
        col.prop(settings, "show_frame_view_button", text="Frame View")
        if not settings.use_follow_view:
            col.prop(settings, "show_frame_selected_button", text="Frame Selected")
        col.prop(settings, "show_list_toggle_button", text="List Toggle")
        col.prop(settings, "show_move_button", text="Move Handle")

        group.separator()
        group = group.column()
        group.label(text="Type List")
        col = group.column()
        col.active = settings.show_type_list
        col.row().prop(settings, "type_list_position", text="Position", expand=True)
        col.prop(settings, "type_list_font_size", text="Font Size")
        row = col.row()
        row.prop(settings, "show_search_bar", text="Filter Bar")
        sub = row.row()
        sub.active = settings.show_node_colors
        sub.prop(settings, "show_type_colors", text="Type Colors")
        row.prop(settings, "use_follow_active")

        col.prop(settings, "use_group_by_type")
        if settings.use_group_by_type:
            col.row().prop(settings, "type_list_sort", text="Sort", expand=True)

        group.separator()
        group = group.column()
        group.label(text="Wires")
        col = group.column()
        col.active = settings.show_wires
        row = col.row(align=True)
        row.prop(settings, "show_wire_color", text="Wire Colors")
        row.prop(settings, "show_dashed_wires", text="Dashed Fields")
        row.prop(settings, "highlight_selected_wires", text="Highlight Selection")

        sub = col.column(heading="Noodle Curving")
        row = sub.row(align=True, heading="")
        row.prop(settings, "use_custom_noodle_curving", text="")
        sub = row.row(align=True)
        sub.active = settings.use_custom_noodle_curving
        sub.row().prop(settings, "noodle_curving", text="", expand=True)

        col.prop(settings, "wire_thickness", text="Thickness")
        col.prop(settings, "wire_opacity", text="Opacity", slider=True)

        layout.separator(type="LINE")
        group = layout.column()
        group.label(text="Theme")
        col = group.column()
        col.prop(settings, "opacity", text="Opacity")

        row = col.row(align=True, heading="Colors")
        row.prop(settings, "use_custom_viewport_fill", text="View Highlight")
        sub = row.row(align=True)
        sub.active = settings.use_custom_viewport_fill
        sub.prop(settings, "viewport_fill_color", text="")

        row = col.row(align=True)
        row.prop(settings, "show_viewport_overlay", text="View Dimming")
        sub = row.row(align=True)
        sub.active = settings.show_viewport_overlay
        sub.prop(settings, "viewport_overlay_color", text="")

        row = col.row(align=True)
        row.prop(settings, "use_custom_background", text="Background")
        sub = row.row(align=True)
        sub.active = settings.use_custom_background
        sub.prop(settings, "background_color", text="")

        row = col.row(align=True)
        row.prop(settings, "use_custom_text", text="Text Color")
        sub = row.row(align=True)
        sub.active = settings.use_custom_text
        sub.prop(settings, "text_color", text="")
        row = col.row(align=True)
        row.prop(settings, "show_text_shadow", text="Text Shadows")

        layout.separator(type="LINE")
        group = layout.column()
        group.label(text="Navigation")
        split = group.split(factor=0.4)
        split.label(text="")
        row = split.row()
        row.use_property_split = False
        col = row.column()
        col.prop(settings, "use_interactive", text="Interactive Map")
        sub = col.column()
        sub.active = settings.use_interactive and not context.preferences.view.use_reduce_motion
        sub.prop(settings, "use_animations", text="Animations")

        col = row.column()
        col.prop(settings, "use_follow_view", text="Follow View")
        col.prop(settings, "use_auto_zoom", text="Auto Zoom")

        group.separator()
        interaction_column = group.column()
        interaction_column.active = settings.use_interactive
        col = interaction_column.column(align=False)
        col.prop(settings, "left_click_action", text="Left Click")
        col.prop(settings, "right_click_action", text="Right Click")
        interaction_column.separator()
        col = interaction_column.column(align=False)
        col.prop(settings, "left_drag_action", text="Left Drag")
        col.prop(settings, "right_drag_action", text="Right Drag")

        interaction_column.separator()
        interaction_column.row().prop(settings, "scroll_wheel_mode", expand=True)

        interaction_column.separator()
        split = interaction_column.split(factor=0.4)
        row = split.row(align=True)
        row.alignment = "RIGHT"
        row.label(text="")
        box = split.box()
        box.label(text="Key Modifiers:", icon="INFO")
        flow = box.column_flow(columns=2, align=True)
        flow.label(text="Shift Drag Frame Region", icon="DOT")
        flow.label(text="Ctrl Drag Pan View", icon="DOT")
        flow.label(text="Alt Drag Frame Region (Editor)", icon="DOT")
        flow.label(text="Shift Click Extend", icon="DOT")
        flow.label(text="Ctrl Click Toggle selection", icon="DOT")
        flow.label(text="Alt Scroll Toggle Zoom", icon="DOT")

        layout.separator()
        split = layout.split(factor=0.4)
        sub = split.column(align=True)
        sub.alignment = "RIGHT"
        sub.label(text="Shortcuts")
        col = split.column(align=True)
        window_manager = context.window_manager
        user_keyconfig = window_manager.keyconfigs.user
        from .. import addon_keymap_bindings

        row = col.row(align=True)
        row.use_property_split = False
        for km_addon, kmi_addon in addon_keymap_bindings:
            user_keymap = user_keyconfig.keymaps.get(km_addon.name)
            if not user_keymap:
                continue
            user_keymap_item = user_keymap.keymap_items.get(kmi_addon.idname)
            if user_keymap_item:
                from rna_keymap_ui import draw_kmi

                draw_kmi([], user_keyconfig, user_keymap, user_keymap_item, row, 0)
            else:
                col.operator("nodemap.restore_keymap", text="Restore")

        layout.separator(type="LINE")
        group = layout.column()
        group.label(text="Performance")
        group.prop(self.settings, "debounce_delay", text="Debounce Delay")

        layout.separator(type="LINE")
        group = layout.column()
        group.label(text="Development")
        row = group.row(align=True, heading="Console Logging")
        row.prop(self, "use_logging", text="")
        sub = row.row(align=True)
        sub.active = self.use_logging
        sub.prop(self, "logging_level", text="")


classes = (
    NODEMAP_PG_settings,
    NODEMAP_AddonPreferences,
)
