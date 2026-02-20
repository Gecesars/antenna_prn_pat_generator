from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from PySide6.QtCore import QPoint
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMenu


@dataclass
class ContextInfo:
    widget: str
    mouse_pos: Optional[QPoint] = None
    global_pos: Optional[QPoint] = None
    picked_object_id: Optional[str] = None
    selected_ids: List[str] = field(default_factory=list)
    tool_mode: str = "Select"
    picked_point_3d: Optional[tuple] = None
    table_row: Optional[int] = None
    field_name: str = ""


class ContextMenuDispatcher:
    """Central dispatcher for right-click menus in every mechanics surface."""

    def __init__(self, page):
        self.page = page

    def _call(self, method_name: str, *args, **kwargs):
        fn = getattr(self.page, method_name, None)
        if callable(fn):
            return fn(*args, **kwargs)
        fallback = getattr(self.page, "action_not_implemented", None)
        if callable(fallback):
            return fallback(method_name)
        return None

    def _add_action(self, menu: QMenu, text: str, callback=None, enabled: bool = True, tooltip: str = "") -> QAction:
        act = menu.addAction(text)
        act.setEnabled(bool(enabled))
        if tooltip:
            act.setToolTip(str(tooltip))
            act.setStatusTip(str(tooltip))
        if callback is not None and enabled:
            act.triggered.connect(callback)
        return act

    def _add_undo_redo(self, menu: QMenu):
        menu.addSeparator()
        can_undo = bool(self.page.engine.command_stack.can_undo)
        can_redo = bool(self.page.engine.command_stack.can_redo)
        ulabel = self.page.engine.command_stack.undo_label or "Undo"
        rlabel = self.page.engine.command_stack.redo_label or "Redo"
        self._add_action(menu, f"Undo ({ulabel})", lambda: self._call("action_undo"), enabled=can_undo)
        self._add_action(menu, f"Redo ({rlabel})", lambda: self._call("action_redo"), enabled=can_redo)

    def build_menu(self, ctx: ContextInfo) -> QMenu:
        menu = QMenu(self.page)
        target = ctx.picked_object_id or (ctx.selected_ids[0] if ctx.selected_ids else "-")
        self._add_action(menu, f"Target: {target}", enabled=False)
        menu.addSeparator()

        widget = str(ctx.widget or "").lower()
        if widget == "viewport":
            if ctx.picked_object_id:
                self._build_viewport_object_menu(menu, ctx)
            elif len(ctx.selected_ids) >= 2:
                self._build_viewport_multi_menu(menu, ctx)
            else:
                self._build_viewport_empty_menu(menu, ctx)
        elif widget == "scene_tree":
            self._build_tree_menu(menu, ctx)
        elif widget == "properties":
            self._build_properties_menu(menu, ctx)
        elif widget == "measurements":
            self._build_measurements_menu(menu, ctx)
        else:
            self._build_global_empty_menu(menu, ctx)
        return menu

    def _build_viewport_empty_menu(self, menu: QMenu, ctx: ContextInfo):
        m_view = menu.addMenu("View")
        self._add_action(m_view, "Reset View", lambda: self._call("action_reset_view"))
        self._add_action(m_view, "Fit All", lambda: self._call("action_fit_all"))
        self._add_action(m_view, "Toggle Grid", lambda: self._call("action_toggle_grid"))
        self._add_action(m_view, "Toggle Axes", lambda: self._call("action_toggle_axes"))
        m_rm = m_view.addMenu("Render Mode")
        self._add_action(m_rm, "Solid", lambda: self._call("action_set_render_mode", "solid"))
        self._add_action(m_rm, "Wireframe", lambda: self._call("action_set_render_mode", "wireframe"))
        self._add_action(m_rm, "Solid + Edges", lambda: self._call("action_set_render_mode", "solid_edges"))
        m_bg = m_view.addMenu("Background")
        self._add_action(m_bg, "Dark", lambda: self._call("action_set_background", "dark"))
        self._add_action(m_bg, "Light", lambda: self._call("action_set_background", "light"))
        self._add_action(m_view, "Screenshot...", lambda: self._call("action_screenshot"))

        m_create = menu.addMenu("Create")
        p = m_create.addMenu("Add Primitive")
        self._add_action(p, "Box...", lambda: self._call("action_create_primitive_dialog", "box"))
        self._add_action(p, "Cylinder...", lambda: self._call("action_create_primitive_dialog", "cylinder"))
        self._add_action(p, "Sphere...", lambda: self._call("action_create_primitive_dialog", "sphere"))
        self._add_action(p, "Cone...", lambda: self._call("action_create_primitive_dialog", "cone"))
        self._add_action(p, "Plane...", lambda: self._call("action_create_primitive_dialog", "plane"))
        self._add_action(m_create, "Create Marker...", lambda: self._call("action_add_marker_at_cursor", ctx))

        m_clip = menu.addMenu("Clipping / Sections")
        self._add_action(m_clip, "Add Clipping Plane XY", lambda: self._call("action_add_clipping_plane", "xy"))
        self._add_action(m_clip, "Add Clipping Plane XZ", lambda: self._call("action_add_clipping_plane", "xz"))
        self._add_action(m_clip, "Add Clipping Plane YZ", lambda: self._call("action_add_clipping_plane", "yz"))
        self._add_action(m_clip, "Remove All Clipping Planes", lambda: self._call("action_clear_clipping"))
        self._add_action(m_clip, "Toggle Clipping", lambda: self._call("action_toggle_clipping"))

        m_sel = menu.addMenu("Selection")
        self._add_action(m_sel, "Select All", lambda: self._call("action_select_all"))
        self._add_action(m_sel, "Select None", lambda: self._call("action_select_none"))
        self._add_action(m_sel, "Invert Selection", lambda: self._call("action_select_invert"))

        m_tools = menu.addMenu("Tools")
        m_tm = m_tools.addMenu("Enter Tool Mode")
        for mode in ["Select", "Move", "Rotate", "Scale", "Measure"]:
            self._add_action(m_tm, mode, lambda _c=False, m=mode: self._call("action_set_tool_mode", m))
        self._add_action(m_tools, "Preferences...", lambda: self._call("action_open_preferences"))

        self._add_undo_redo(menu)

    def _build_viewport_object_menu(self, menu: QMenu, ctx: ContextInfo):
        target = str(ctx.picked_object_id or "")
        selected = list(ctx.selected_ids)
        multi_ready = len(selected) >= 2

        m_sel = menu.addMenu("Selection")
        self._add_action(m_sel, "Select This", lambda: self._call("action_select_this", target))
        self._add_action(m_sel, "Add to Selection", lambda: self._call("action_add_to_selection", target))
        self._add_action(m_sel, "Remove from Selection", lambda: self._call("action_remove_from_selection", target))
        self._add_action(m_sel, "Isolate", lambda: self._call("action_isolate_selection", [target]))
        self._add_action(m_sel, "Show Only Selected", lambda: self._call("action_show_only_selected"))
        self._add_action(m_sel, "Show All", lambda: self._call("action_show_all"))

        m_tr = menu.addMenu("Transform")
        self._add_action(m_tr, "Move...", lambda: self._call("action_transform_dialog", "move", [target]))
        self._add_action(m_tr, "Rotate...", lambda: self._call("action_transform_dialog", "rotate", [target]))
        self._add_action(m_tr, "Scale...", lambda: self._call("action_transform_dialog", "scale", [target]))
        m_align = m_tr.addMenu("Align")
        self._add_action(m_align, "Align to World Axes", lambda: self._call("action_align_world", [target]))
        self._add_action(m_align, "Align to Plane...", lambda: self._call("action_align_plane_dialog", [target]))
        m_snap = m_tr.addMenu("Snap")
        self._add_action(m_snap, "Snap to Grid", lambda: self._call("action_toggle_snap"))
        self._add_action(m_snap, "Set Snap Step...", lambda: self._call("action_set_snap_step"))

        m_edit = menu.addMenu("Edit")
        self._add_action(m_edit, "Duplicate", lambda: self._call("action_duplicate", [target]))
        self._add_action(m_edit, "Mirror...", lambda: self._call("action_mirror_dialog", [target]))
        self._add_action(m_edit, "Array...", lambda: self._call("action_array_dialog", [target]))
        self._add_action(m_edit, "Rename...", lambda: self._call("action_rename_dialog", target))
        self._add_action(m_edit, "Set Color...", lambda: self._call("action_set_color_dialog", [target]))
        self._add_action(m_edit, "Set Opacity...", lambda: self._call("action_set_opacity_dialog", [target]))
        self._add_action(m_edit, "Lock/Unlock", lambda: self._call("action_toggle_lock", [target]))

        m_bool = menu.addMenu("Boolean")
        self._add_action(
            m_bool,
            "Union (A U B)",
            lambda: self._call("action_boolean_union"),
            enabled=multi_ready,
            tooltip="Select at least 2 objects.",
        )
        self._add_action(
            m_bool,
            "Subtract (A - B)",
            lambda: self._call("action_boolean_subtract"),
            enabled=multi_ready,
            tooltip="Select primary + cutters.",
        )
        self._add_action(
            m_bool,
            "Intersect (A n B)",
            lambda: self._call("action_boolean_intersect"),
            enabled=multi_ready,
            tooltip="Select at least 2 objects.",
        )
        self._add_action(m_bool, "Diagnose", lambda: self._call("action_boolean_diagnose"))
        self._add_action(m_bool, "Boolean Settings...", lambda: self._call("action_boolean_settings"))

        m_mea = menu.addMenu("Measure / Analyze")
        self._add_action(m_mea, "Show Bounding Box", lambda: self._call("action_measure_bbox", [target]))
        self._add_action(m_mea, "Compute Volume", lambda: self._call("action_measure_volume", [target]))
        self._add_action(m_mea, "Compute Area", lambda: self._call("action_measure_area", [target]))
        self._add_action(m_mea, "Compute Centroid", lambda: self._call("action_measure_centroid", [target]))
        self._add_action(m_mea, "Measure Distance...", lambda: self._call("action_enter_measure_mode", "distance"))
        self._add_action(m_mea, "Measure Angle...", lambda: self._call("action_enter_measure_mode", "angle"))

        m_mark = menu.addMenu("Markers")
        self._add_action(m_mark, "Add Marker at Pick Point", lambda: self._call("action_add_marker_at_cursor", ctx))
        self._add_action(m_mark, "Add Label (name/value)", lambda: self._call("action_add_label_at_cursor", ctx))
        self._add_action(m_mark, "Add Custom Math Marker...", lambda: self._call("action_add_custom_math_marker", ctx))

        m_exp = menu.addMenu("Export")
        self._add_action(m_exp, "Export Selected as STL...", lambda: self._call("action_export_selected", "stl"))
        self._add_action(m_exp, "Export Selected as OBJ/PLY...", lambda: self._call("action_export_selected", "obj"))

        m_dz = menu.addMenu("Danger Zone")
        self._add_action(m_dz, "Delete Selected", lambda: self._call("action_delete_selected"))
        self._add_action(m_dz, "Reset Transform", lambda: self._call("action_reset_transform", [target]))

        if str(ctx.tool_mode).lower() != "select":
            menu.addSeparator()
            self._add_action(menu, "Exit Tool Mode", lambda: self._call("action_set_tool_mode", "Select"))

        self._add_undo_redo(menu)

    def _build_viewport_multi_menu(self, menu: QMenu, ctx: ContextInfo):
        self._add_action(menu, "Group...", lambda: self._call("action_group_selected"))
        self._add_action(menu, "Ungroup...", lambda: self._call("action_ungroup_selected"))
        menu.addSeparator()
        m_align = menu.addMenu("Align")
        self._add_action(m_align, "Align centers X", lambda: self._call("action_align_centers", "x"))
        self._add_action(m_align, "Align centers Y", lambda: self._call("action_align_centers", "y"))
        self._add_action(m_align, "Align centers Z", lambda: self._call("action_align_centers", "z"))
        self._add_action(m_align, "Distribute equally X", lambda: self._call("action_distribute", "x"))
        self._add_action(m_align, "Distribute equally Y", lambda: self._call("action_distribute", "y"))
        self._add_action(m_align, "Distribute equally Z", lambda: self._call("action_distribute", "z"))
        m_bool = menu.addMenu("Boolean")
        self._add_action(m_bool, "Union (all)", lambda: self._call("action_boolean_union"))
        self._add_action(m_bool, "Subtract (primary - others)", lambda: self._call("action_boolean_subtract"))
        self._add_action(m_bool, "Intersect (all)", lambda: self._call("action_boolean_intersect"))
        self._add_action(m_bool, "Diagnose", lambda: self._call("action_boolean_diagnose"))
        m_tr = menu.addMenu("Transform")
        self._add_action(m_tr, "Move...", lambda: self._call("action_transform_dialog", "move", ctx.selected_ids))
        self._add_action(m_tr, "Rotate...", lambda: self._call("action_transform_dialog", "rotate", ctx.selected_ids))
        self._add_action(m_tr, "Scale...", lambda: self._call("action_transform_dialog", "scale", ctx.selected_ids))
        m_vis = menu.addMenu("Visibility")
        self._add_action(m_vis, "Hide Selected", lambda: self._call("action_hide_selected"))
        self._add_action(m_vis, "Show Selected", lambda: self._call("action_show_selected"))
        menu.addSeparator()
        self._add_action(menu, "Export Selection...", lambda: self._call("action_export_selected", "obj"))
        self._add_action(menu, "Delete Selected", lambda: self._call("action_delete_selected"))
        self._add_undo_redo(menu)

    def _build_tree_menu(self, menu: QMenu, ctx: ContextInfo):
        has_item = bool(ctx.picked_object_id)
        if has_item:
            oid = str(ctx.picked_object_id)
            self._add_action(menu, "Select / Focus", lambda: self._call("action_focus_object", oid))
            self._add_action(menu, "Rename...", lambda: self._call("action_rename_dialog", oid))
            vis = menu.addMenu("Visibility")
            self._add_action(vis, "Show / Hide", lambda: self._call("action_toggle_visibility", [oid]))
            self._add_action(vis, "Isolate", lambda: self._call("action_isolate_selection", [oid]))
            self._add_action(menu, "Lock/Unlock", lambda: self._call("action_toggle_lock", [oid]))
            self._add_action(menu, "Duplicate", lambda: self._call("action_duplicate", [oid]))
            self._add_action(menu, "Delete", lambda: self._call("action_delete_selected"))
            self._add_action(menu, "Color/Opacity...", lambda: self._call("action_style_dialog", [oid]))
            self._add_action(menu, "Export...", lambda: self._call("action_export_selected", "obj"))
            m_mea = menu.addMenu("Measurements")
            self._add_action(m_mea, "BBox", lambda: self._call("action_measure_bbox", [oid]))
            self._add_action(m_mea, "Area", lambda: self._call("action_measure_area", [oid]))
            self._add_action(m_mea, "Volume", lambda: self._call("action_measure_volume", [oid]))
            self._add_action(m_mea, "Centroid", lambda: self._call("action_measure_centroid", [oid]))
            m_mark = menu.addMenu("Add Marker")
            self._add_action(m_mark, "Marker no centroide", lambda: self._call("action_marker_centroid", oid))
            self._add_action(m_mark, "Label com nome", lambda: self._call("action_label_centroid", oid))
        else:
            self._add_action(menu, "Create primitive", lambda: self._call("action_create_primitive_dialog", "box"))
            self._add_action(menu, "Import mesh", lambda: self._call("action_import_mesh"))
            self._add_action(menu, "Show all", lambda: self._call("action_show_all"))
        self._add_undo_redo(menu)

    def _build_properties_menu(self, menu: QMenu, ctx: ContextInfo):
        self._add_action(menu, "Copy Value", lambda: self._call("action_copy_property_value", ctx.field_name))
        self._add_action(menu, "Copy All Properties (JSON)", lambda: self._call("action_copy_all_properties"))
        self._add_action(menu, "Paste Transform", lambda: self._call("action_paste_transform"))
        self._add_action(menu, "Reset Transform", lambda: self._call("action_reset_transform"))
        self._add_action(menu, "Apply Transform to Mesh", lambda: self._call("action_bake_transform"))
        self._add_action(menu, "Rename...", lambda: self._call("action_rename_selected"))
        self._add_action(menu, "Lock/Unlock", lambda: self._call("action_toggle_lock_selected"))
        self._add_action(menu, "Delete", lambda: self._call("action_delete_selected"))
        self._add_undo_redo(menu)

    def _build_measurements_menu(self, menu: QMenu, ctx: ContextInfo):
        self._add_action(menu, "Copy row", lambda: self._call("action_copy_measurement_row", ctx.table_row))
        self._add_action(menu, "Copy table", lambda: self._call("action_copy_measurement_table"))
        self._add_action(menu, "Export CSV", lambda: self._call("action_export_measurements_csv"))
        self._add_action(menu, "Delete selected measure", lambda: self._call("action_delete_measurement_row", ctx.table_row))
        menu.addSeparator()
        self._add_action(menu, "Exit Measure Mode", lambda: self._call("action_set_tool_mode", "Select"))
        self._add_action(menu, "Clear Current Measure", lambda: self._call("action_clear_current_measure"))
        self._add_action(menu, "Save Measure as Marker", lambda: self._call("action_save_measure_as_marker"))
        self._add_action(menu, "Copy last measure", lambda: self._call("action_copy_last_measure"))
        self._add_undo_redo(menu)

    def _build_global_empty_menu(self, menu: QMenu, _ctx: ContextInfo):
        self._add_action(menu, "Help (context) / Atalhos", lambda: self._call("action_show_shortcuts"))
        self._add_action(menu, "Reset View", lambda: self._call("action_reset_view"))
        menu.addSeparator()
        self._add_action(menu, "No actions available", enabled=False)
