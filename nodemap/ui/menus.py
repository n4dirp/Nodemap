"""Provide right-click context menus for the minimap buttons."""

import bpy
from bpy.types import Menu

from ..core.buttons import BUTTONS, GROUPS, MENU_ORDER
from ..core.helpers import get_addon_preferences

_BUTTON_MENU_ID = "NODEMAP_MT_minimap_button"
_BUTTONS_MENU_ID = "NODEMAP_MT_minimap_buttons"

_context_button_id: str | None = None


def _draw_list_toggles(layout, settings) -> None:
    """Draw type-list toggles for the list button menu."""
    options = layout.column()
    options.active = settings.show_type_list
    options.prop(settings, "use_group_by_type", text="Group by Type")
    options.prop(settings, "show_search_bar", text="Filter Bar")
    options.prop(settings, "use_follow_active", text="Follow Active")
    options.prop(settings, "show_type_colors", text="Type Colors")


def _draw_move_toggles(layout, settings) -> None:
    """Draw move-related toggles for the move handle menu."""
    layout.prop(settings, "use_snap_to_borders", text="Snap to Borders")


def _draw_frame_toggles(layout, settings) -> None:
    """Draw frame-related toggles for the frame button menu."""
    layout.prop(settings, "use_auto_zoom", text="Auto Zoom")
    layout.prop(settings, "use_follow_view", text="Follow View")


_MENU_SECTION_BUILDERS = {
    "list": _draw_list_toggles,
    "move": _draw_move_toggles,
    "frame": _draw_frame_toggles,
}


class NODEMAP_MT_minimap_button(Menu):
    """Toggle options for the minimap button that opened the menu."""

    bl_idname = _BUTTON_MENU_ID
    bl_label = "Nodemap"

    def draw(self, context):
        """Draw toggle rows relevant to the right-clicked minimap button."""
        prefs = get_addon_preferences(context)
        if prefs is None:
            return
        settings = prefs.settings
        layout = self.layout
        button_def = BUTTONS.get(_context_button_id) if _context_button_id else None
        menu_group = button_def.menu_group if button_def is not None else "frame"

        # Options
        _MENU_SECTION_BUILDERS[menu_group](layout, settings)

        # Button toggles
        layout.separator()
        layout.menu(_BUTTONS_MENU_ID)


class NODEMAP_MT_minimap_buttons(Menu):
    """Visibility toggles for the minimap buttons."""

    bl_idname = _BUTTONS_MENU_ID
    bl_label = "Buttons"

    def draw(self, context):
        """Draw the minimap button visibility toggles."""
        prefs = get_addon_preferences(context)
        if prefs is None:
            return
        settings = prefs.settings
        layout = self.layout
        for group_index, group in enumerate(GROUPS):
            if group_index:
                layout.separator()
            for button_id in MENU_ORDER:
                button_def = BUTTONS[button_id]
                if button_def.group != group:
                    continue
                if button_def.disable_when_follow_view:
                    row = layout.row()
                    row.active = not settings.use_follow_view
                    row.prop(settings, button_def.pref_attr, text=button_def.label)
                else:
                    layout.prop(settings, button_def.pref_attr, text=button_def.label)


def open_minimap_button_menu(context, button_id: str) -> None:
    """Open the toggle menu for the minimap button with *button_id*."""
    global _context_button_id
    _context_button_id = button_id
    try:
        bpy.ops.wm.call_menu(name=_BUTTON_MENU_ID)
    except RuntimeError:
        _context_button_id = None


classes = (NODEMAP_MT_minimap_button, NODEMAP_MT_minimap_buttons)
