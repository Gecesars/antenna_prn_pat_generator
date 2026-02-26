from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import math
import time

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QToolButton, QFrame

from mech.engine.scene_object import SceneObject, Marker
from .context_menu import ContextInfo


class ViewportPyVista(QWidget):
    selectionRequested = Signal(list, str)  # ids, mode
    contextRequested = Signal(object)  # ContextInfo
    pickedInfo = Signal(dict)  # object_id, picked_point, picked_cell_id, modifiers
    hoverInfo = Signal(dict)  # object_id, picked_point, picked_cell_id
    statusMessage = Signal(str)
    measurePointPicked = Signal(tuple)  # x,y,z
    pieceCommandRequested = Signal(dict)  # direct mouse command on picked piece
    quickActionRequested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = None
        self._pv = None
        self._selection = []
        self._objects_cache: Dict[str, SceneObject] = {}
        self._boundary_map: Dict[str, List[str]] = {}
        self._actor_by_id: Dict[str, object] = {}
        self._marker_actors = []
        self._selection_overlay_actors = []
        self._hover_overlay_actor = None
        self._render_mode = "solid_edges"
        self._refresh_in_progress = False
        self._pending_refresh = None
        self._render_in_progress = False
        self._render_failures = 0
        self._render_disabled = False
        self._zoom_step = 1.10
        self._zoom_step_fine = 1.04
        self._zoom_step_fast = 1.18
        self._tool_mode = "Select"
        self._selection_mode = "object"
        self._snap_enabled = False
        self._show_grid = True
        self._show_axes = True
        self._grid_step = 5.0
        self._grid_size = 2000.0
        self._grid_color = "#888888"
        self._grid_plane = True
        self._xray_mode = False
        self._clipping_enabled = True
        self._clipping_planes = []
        self._sub_face: Optional[tuple] = None
        self._sub_edge: Optional[tuple] = None
        self._sub_face_actor = None
        self._sub_edge_actor = None
        self._hover_object_id: str = ""
        self._active_object_id: str = ""
        self._last_pick_point: Optional[tuple] = None
        self._hover_pick_interval_s = 1.0 / 45.0
        self._last_hover_pick_ts = 0.0
        self._last_hover_pos: Optional[Tuple[int, int]] = None
        self._last_hover_event_pos: Optional[Tuple[int, int]] = None
        self._last_hover_event_ts = 0.0
        self._last_hover_move_ts = 0.0
        self._hover_force_interval_s = 0.055
        self._pending_hover_pos: Optional[Tuple[int, int]] = None
        self._hover_debounce_timer = QTimer(self)
        self._hover_debounce_timer.setSingleShot(True)
        self._hover_debounce_timer.timeout.connect(self._flush_hover_pick)
        self._navigation_active = False
        self._lod_during_navigation = True
        self._lod_active = False
        self._left_button_down = False
        self._left_dragging = False
        self._left_press_pos: Optional[Tuple[int, int]] = None
        self._left_press_ts = 0.0
        self._left_press_modifiers = 0
        self._left_drag_threshold_px = 5
        self._left_click_max_duration_s = 0.42
        self._last_left_click_ts = 0.0
        self._last_left_click_object_id = ""
        self._suspend_hover_until = 0.0
        self._last_wheel_ts = 0.0
        self._wheel_accum_steps = 0
        self._wheel_accum_modifiers = 0
        self._wheel_last_event_ts = 0.0
        self._wheel_debounce_timer = QTimer(self)
        self._wheel_debounce_timer.setSingleShot(True)
        self._wheel_debounce_timer.timeout.connect(self._flush_wheel_zoom)
        self._zoom_far_cap = 2500.0
        self._quick_buttons: Dict[str, QToolButton] = {}
        self._perf = {
            "render_count": 0,
            "hover_pick_count": 0,
            "hover_skipped": 0,
            "wheel_count": 0,
            "wheel_skipped": 0,
            "wheel_clamped": 0,
            "last_hover_ms": 0.0,
            "last_wheel_ms": 0.0,
            "interaction_starts": 0,
            "interaction_ends": 0,
        }
        self._build_ui()

    def _safe_render(self):
        if not self.is_available():
            return
        if self._render_disabled:
            return
        if self._render_in_progress:
            return
        self._render_in_progress = True
        try:
            self.plotter.render()
            self._perf["render_count"] = int(self._perf.get("render_count", 0)) + 1
            self._render_failures = 0
        except Exception:
            self._render_failures += 1
            if self._render_failures >= 3:
                self._render_disabled = True
                self.statusMessage.emit("Viewport rendering disabled after repeated OpenGL/render failures.")
        finally:
            self._render_in_progress = False

    def _clear_subselection_actors(self):
        if not self.is_available():
            self._sub_face_actor = None
            self._sub_edge_actor = None
            return
        for actor in (self._sub_face_actor, self._sub_edge_actor):
            if actor is None:
                continue
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                continue
        self._sub_face_actor = None
        self._sub_edge_actor = None

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.quick_toolbar = QWidget(self)
        self.quick_toolbar.setMinimumHeight(44)
        self.quick_toolbar.setStyleSheet(
            "QWidget{background:rgba(18,24,31,232); border:1px solid rgba(210,230,250,55); border-radius:8px;}"
            "QToolButton{color:#ecf4ff; padding:4px 8px; border-radius:5px; min-height:26px; font-size:12px; font-weight:600;}"
            "QToolButton:hover{background:rgba(255,255,255,36);}"
            "QToolButton:pressed{background:rgba(102,180,255,82);}"
            "QToolButton:checked{background:rgba(64,170,255,112); border:1px solid rgba(164,222,255,140);}"
            "QLabel{color:#d5e7fb; font-size:11px; font-weight:700; padding:0 3px;}"
        )
        row = QHBoxLayout(self.quick_toolbar)
        row.setContentsMargins(6, 4, 6, 4)
        row.setSpacing(4)
        self._quick_buttons = {}

        def _btn(text: str, token: str, tip: str, *, checkable: bool = False):
            b = QToolButton(self.quick_toolbar)
            b.setText(text)
            b.setAutoRaise(True)
            b.setCheckable(bool(checkable))
            b.setToolTip(tip)
            b.clicked.connect(lambda _checked=False, t=token: self._on_quick_action(t))
            row.addWidget(b)
            self._quick_buttons[str(token)] = b
            return b

        def _sep():
            sep = QFrame(self.quick_toolbar)
            sep.setFrameShape(QFrame.VLine)
            sep.setFrameShadow(QFrame.Plain)
            sep.setStyleSheet("color: rgba(255,255,255,35);")
            row.addWidget(sep)

        row.addWidget(QLabel("Camera", self.quick_toolbar))
        _btn("Iso", "view_iso", "Isometric view")
        _btn("Top", "view_top", "Top view")
        _btn("Front", "view_front", "Front view")
        _btn("Right", "view_right", "Right view")
        _btn("Fit", "fit_all", "Fit all geometry")
        _btn("Sel", "fit_selection", "Fit current selection")
        _btn("+", "zoom_in", "Zoom in")
        _btn("-", "zoom_out", "Zoom out")

        _sep()
        row.addWidget(QLabel("View", self.quick_toolbar))
        _btn("Shaded", "render_solid_edges", "Shaded + edges", checkable=True)
        _btn("Wire", "render_wireframe", "Wireframe", checkable=True)
        _btn("X-Ray", "render_xray", "X-Ray mode", checkable=True)
        _btn("Grid", "toggle_grid", "Toggle grid", checkable=True)
        _btn("Axes", "toggle_axes", "Toggle axes", checkable=True)
        _btn("Sect", "section_xy", "Add XY section plane")

        _sep()
        row.addWidget(QLabel("Interact", self.quick_toolbar))
        _btn("Sel:Object", "selection_cycle", "Cycle selection mode: object -> face -> edge -> vertex -> body -> component")
        _btn("Snap", "toggle_snap", "Toggle grid snap", checkable=True)
        _btn("LOD", "toggle_lod", "Toggle adaptive level of detail during navigation", checkable=True)
        _btn("Detail", "detail_focus", "Focus selected geometry details")
        _btn("Shot", "screenshot", "Save screenshot")
        _btn("UI", "toggle_panels", "Show/hide side and bottom controls")
        row.addStretch(1)
        layout.addWidget(self.quick_toolbar, 0)
        self.mouse_hint = QLabel(
            "Mouse: LMB select | Shift+LMB add | Ctrl+LMB toggle | Wheel zoom | Ctrl/Shift wheel fine | Alt wheel fast",
            self,
        )
        self.mouse_hint.setStyleSheet(
            "QLabel{background:rgba(20,28,36,210); color:#dbe9fa; border:1px solid rgba(210,230,250,40); border-radius:5px; padding:4px 8px; font-size:11px;}"
        )
        layout.addWidget(self.mouse_hint, 0)
        self._sync_quick_toolbar_state()
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor

            self._pv = pv
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter.interactor)
            self.plotter.set_background("#1f2430")
            self.plotter.add_axes(line_width=2)
            self._apply_grid()
            iren = self.plotter.interactor
            iren.AddObserver("LeftButtonPressEvent", self._on_left_press_vtk, 1.0)
            iren.AddObserver("LeftButtonReleaseEvent", self._on_left_release_vtk, 1.0)
            iren.AddObserver("RightButtonPressEvent", self._on_right_click_vtk, 1.0)
            iren.AddObserver("MouseMoveEvent", self._on_mouse_move_vtk, 1.0)
            iren.AddObserver("MouseWheelForwardEvent", self._on_mouse_wheel_forward_vtk, 2.0)
            iren.AddObserver("MouseWheelBackwardEvent", self._on_mouse_wheel_backward_vtk, 2.0)
            iren.AddObserver("StartInteractionEvent", self._on_start_interaction_vtk, 1.0)
            iren.AddObserver("EndInteractionEvent", self._on_end_interaction_vtk, 1.0)
        except Exception as e:
            lbl = QLabel(f"Viewport 3D indisponivel.\nInstale: pyside6 pyvista pyvistaqt vtk\n\nDetalhe: {e}", self)
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)

    def is_available(self) -> bool:
        return self.plotter is not None

    @staticmethod
    def _enum_to_int(value) -> int:
        try:
            return int(value)
        except Exception:
            pass
        try:
            return int(getattr(value, "value"))
        except Exception:
            pass
        try:
            return int(value.__index__())
        except Exception:
            return 0

    @staticmethod
    def _qt_modifier_flag(name: str):
        km = getattr(Qt, "KeyboardModifier", None)
        if km is not None and hasattr(km, name):
            return getattr(km, name)
        return getattr(Qt, name, 0)

    def _has_modifier(self, modifiers, flag) -> bool:
        try:
            return bool(modifiers & flag)
        except Exception:
            m = self._enum_to_int(modifiers)
            f = self._enum_to_int(flag)
            return bool(m & f)

    def _set_button_checked(self, token: str, checked: bool):
        btn = self._quick_buttons.get(str(token))
        if btn is None:
            return
        if bool(btn.isCheckable()) and bool(btn.isChecked()) != bool(checked):
            btn.blockSignals(True)
            btn.setChecked(bool(checked))
            btn.blockSignals(False)

    def _sync_quick_toolbar_state(self):
        mode = str(self._selection_mode or "object").strip().lower()
        mode_short = mode[:4].capitalize() if mode else "Obj"
        btn = self._quick_buttons.get("selection_cycle")
        if btn is not None:
            btn.setText(f"Sel:{mode_short}")
            btn.setToolTip(f"Selection mode: {mode}. Click to cycle mode.")

        self._set_button_checked("toggle_grid", bool(self._show_grid))
        self._set_button_checked("toggle_axes", bool(self._show_axes))
        self._set_button_checked("toggle_snap", bool(self._snap_enabled))
        self._set_button_checked("toggle_lod", bool(self._lod_during_navigation))

        is_wire = str(self._render_mode or "").strip().lower() == "wireframe"
        is_shaded = str(self._render_mode or "").strip().lower() in {"solid", "solid_edges"} and not bool(self._xray_mode)
        self._set_button_checked("render_wireframe", is_wire)
        self._set_button_checked("render_solid_edges", is_shaded)
        self._set_button_checked("render_xray", bool(self._xray_mode))

    def set_selection_mode_indicator(self, mode: str):
        token = str(mode or "object").strip().lower()
        if not token:
            token = "object"
        self._selection_mode = token
        self._sync_quick_toolbar_state()

    def set_snap_enabled(self, enabled: bool):
        self._snap_enabled = bool(enabled)
        self._sync_quick_toolbar_state()

    def set_navigation_lod(self, enabled: bool):
        on = bool(enabled)
        if on == self._lod_during_navigation:
            self._sync_quick_toolbar_state()
            return
        self._lod_during_navigation = on
        if not on and self._lod_active:
            self._set_navigation_lod_state(False)
        self._sync_quick_toolbar_state()
        self.statusMessage.emit(f"Navigation LOD {'enabled' if on else 'disabled'}.")

    def toggle_navigation_lod(self) -> bool:
        self.set_navigation_lod(not self._lod_during_navigation)
        return bool(self._lod_during_navigation)

    def diagnostics_snapshot(self) -> dict:
        rows = dict(self._perf)
        rows["hover_target_hz"] = round(1.0 / float(self._hover_pick_interval_s), 2) if self._hover_pick_interval_s > 0 else 0.0
        rows["selection_mode"] = str(self._selection_mode)
        rows["snap_enabled"] = bool(self._snap_enabled)
        rows["lod_navigation_enabled"] = bool(self._lod_during_navigation)
        rows["navigation_active"] = bool(self._navigation_active)
        return rows

    def _apply_grid(self):
        if not self.is_available():
            return
        if not self._show_grid:
            try:
                self.plotter.remove_bounds_axes()
            except Exception:
                pass
            return
        step = max(1e-6, float(self._grid_step))
        size = max(step * 2.0, float(self._grid_size))
        half = 0.5 * size
        try:
            self.plotter.show_grid(
                color=str(self._grid_color or "#888888"),
                bounds=(-half, half, -half, half, -half, half),
                xtitle="X",
                ytitle="Y",
                ztitle="Z",
                location="outer",
                ticks="outside",
            )
        except Exception:
            try:
                self.plotter.show_grid(color=str(self._grid_color or "#888888"))
            except Exception:
                pass

    def _poly_from_object(self, obj: SceneObject):
        faces = np.asarray(obj.mesh.faces, dtype=int)
        verts = np.asarray(obj.mesh.vertices, dtype=float)
        if faces.size == 0 or verts.size == 0:
            return None
        ff = np.empty((faces.shape[0], 4), dtype=np.int64)
        ff[:, 0] = 3
        ff[:, 1:] = faces.astype(np.int64)
        poly = self._pv.PolyData(verts, ff.reshape(-1))
        return poly

    def _capture_camera_state(self):
        if not self.is_available():
            return None
        try:
            cam = self.plotter.renderer.GetActiveCamera()
            if cam is None:
                return None
            return {
                "position": tuple(float(x) for x in cam.GetPosition()),
                "focal": tuple(float(x) for x in cam.GetFocalPoint()),
                "viewup": tuple(float(x) for x in cam.GetViewUp()),
                "parallel": bool(cam.GetParallelProjection()),
                "parallel_scale": float(cam.GetParallelScale()),
                "view_angle": float(cam.GetViewAngle()),
                "clipping": tuple(float(x) for x in cam.GetClippingRange()),
            }
        except Exception:
            return None

    def _restore_camera_state(self, state):
        if state is None or not self.is_available():
            return
        try:
            cam = self.plotter.renderer.GetActiveCamera()
            if cam is None:
                return
            pos = tuple(state.get("position", ()))
            foc = tuple(state.get("focal", ()))
            up = tuple(state.get("viewup", ()))
            if len(pos) == 3:
                cam.SetPosition(*[float(x) for x in pos])
            if len(foc) == 3:
                cam.SetFocalPoint(*[float(x) for x in foc])
            if len(up) == 3:
                cam.SetViewUp(*[float(x) for x in up])
            cam.SetParallelProjection(1 if bool(state.get("parallel", False)) else 0)
            if "parallel_scale" in state:
                cam.SetParallelScale(float(state.get("parallel_scale", 1.0)))
            if "view_angle" in state:
                cam.SetViewAngle(float(state.get("view_angle", 30.0)))
            cl = tuple(state.get("clipping", ()))
            if len(cl) == 2:
                cam.SetClippingRange(float(cl[0]), float(cl[1]))
        except Exception:
            return

    def _bounds_for_ids(self, ids: List[str]) -> Optional[list]:
        if not self.is_available():
            return None
        bounds = None
        for oid in [str(x) for x in ids if str(x) in self._actor_by_id]:
            actor = self._actor_by_id.get(str(oid))
            if actor is None:
                continue
            try:
                b = actor.GetBounds()
            except Exception:
                b = None
            if not b:
                continue
            if bounds is None:
                bounds = [float(v) for v in b]
                continue
            bounds[0] = min(bounds[0], float(b[0]))
            bounds[1] = max(bounds[1], float(b[1]))
            bounds[2] = min(bounds[2], float(b[2]))
            bounds[3] = max(bounds[3], float(b[3]))
            bounds[4] = min(bounds[4], float(b[4]))
            bounds[5] = max(bounds[5], float(b[5]))
        return bounds

    @staticmethod
    def _bounds_center_diag(bounds: Sequence[float]) -> Tuple[np.ndarray, float]:
        b = np.asarray(bounds, dtype=float).reshape(6)
        center = np.asarray([(b[0] + b[1]) * 0.5, (b[2] + b[3]) * 0.5, (b[4] + b[5]) * 0.5], dtype=float)
        ext = np.asarray([max(1e-12, b[1] - b[0]), max(1e-12, b[3] - b[2]), max(1e-12, b[5] - b[4])], dtype=float)
        diag = float(np.linalg.norm(ext))
        return center, diag

    def _focus_camera_to_point(self, point: Sequence[float], blend: float = 0.45):
        if not self.is_available():
            return
        try:
            cam = self.plotter.renderer.GetActiveCamera()
            if cam is None:
                return
            target = np.asarray(point, dtype=float).reshape(3)
            fp = np.asarray(cam.GetFocalPoint(), dtype=float).reshape(3)
            pos = np.asarray(cam.GetPosition(), dtype=float).reshape(3)
            alpha = float(max(0.0, min(1.0, blend)))
            new_fp = (1.0 - alpha) * fp + alpha * target
            delta = new_fp - fp
            cam.SetFocalPoint(float(new_fp[0]), float(new_fp[1]), float(new_fp[2]))
            new_pos = pos + delta
            cam.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
        except Exception:
            return

    def focus_on_ids(self, ids: Sequence[str], *, detail: bool = False) -> bool:
        if not self.is_available():
            return False
        bounds = self._bounds_for_ids([str(x) for x in ids])
        if bounds is None:
            return False
        try:
            if not detail:
                self.plotter.renderer.ResetCamera(*bounds)
                self.plotter.renderer.ResetCameraClippingRange()
                self._safe_render()
                return True

            center, diag = self._bounds_center_diag(bounds)
            cam = self.plotter.renderer.GetActiveCamera()
            if cam is None:
                self.plotter.renderer.ResetCamera(*bounds)
                self._safe_render()
                return True
            fp = np.asarray(cam.GetFocalPoint(), dtype=float).reshape(3)
            pos = np.asarray(cam.GetPosition(), dtype=float).reshape(3)
            view_vec = pos - fp
            norm = float(np.linalg.norm(view_vec))
            if norm <= 1e-12:
                view_vec = np.asarray([1.0, 1.0, 1.0], dtype=float)
                norm = float(np.linalg.norm(view_vec))
            view_dir = view_vec / norm
            # Bring camera closer than regular fit for detail inspection.
            target_dist = max(1e-3, diag * 0.85)
            new_pos = center + view_dir * target_dist
            cam.SetFocalPoint(float(center[0]), float(center[1]), float(center[2]))
            cam.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
            near = max(1e-6, target_dist * 0.02)
            far = max(near * 20.0, target_dist * 20.0)
            cam.SetClippingRange(float(near), float(far))
            self._safe_render()
            return True
        except Exception:
            return False

    def refresh_scene(
        self,
        objects: Dict[str, SceneObject],
        selection,
        markers: Dict[str, Marker],
        boundary_map: Optional[Dict[str, List[str]]] = None,
        *,
        hover_id: str = "",
        active_id: str = "",
    ):
        if self._refresh_in_progress:
            self._pending_refresh = (
                dict(objects),
                list(selection),
                dict(markers),
                dict(boundary_map or {}),
                str(hover_id or ""),
                str(active_id or ""),
            )
            return
        self._refresh_in_progress = True
        self._selection = [str(x) for x in selection]
        self._objects_cache = dict(objects)
        bc_map = dict(boundary_map or {})
        self._boundary_map = dict(bc_map)
        self._hover_object_id = str(hover_id or self._hover_object_id or "")
        self._active_object_id = str(active_id or (self._selection[0] if self._selection else ""))
        try:
            if not self.is_available():
                return
            cam_state = self._capture_camera_state()
            self.plotter.clear()
            self._actor_by_id = {}
            self._marker_actors = []
            self._selection_overlay_actors = []
            self._hover_overlay_actor = None

            for oid, obj in objects.items():
                if not obj.visible:
                    continue
                poly = self._poly_from_object(obj)
                if poly is None:
                    continue
                is_sel = oid in self._selection
                is_active = bool(is_sel and oid == self._active_object_id)
                opacity = float(obj.opacity)
                if bool(getattr(obj, "locked", False)):
                    opacity = min(opacity, 0.45)
                if self._xray_mode:
                    opacity = max(0.05, min(0.35, opacity * 0.35))
                actor = self.plotter.add_mesh(
                    poly,
                    name=str(oid),
                    color=str(obj.color),
                    opacity=opacity,
                    show_edges=bool(is_sel or self._xray_mode or self._render_mode == "solid_edges"),
                    edge_color="#ffe066" if is_active else ("#00d4ff" if is_sel else "#222222"),
                    line_width=2.8 if is_active else (1.6 if is_sel else 0.2),
                    pickable=True,
                )
                self._actor_by_id[str(oid)] = actor

            for mk in markers.values():
                if not mk.visible:
                    continue
                p = np.asarray(mk.position, dtype=float).reshape(3)
                color = str(mk.style.get("color", "#ffd166"))
                size = float(mk.style.get("size", 12.0))
                actor = self.plotter.add_points([p], color=color, point_size=size, render_points_as_spheres=True, pickable=False)
                self._marker_actors.append(actor)
                lbl = mk.name
                if mk.last_values:
                    vals = ", ".join([f"{k}={v:.4g}" for k, v in mk.last_values.items()])
                    lbl = f"{mk.name}: {vals}"
                self.plotter.add_point_labels([p], [lbl], font_size=10, shape=None, text_color=color)

            # Boundary overlay: annotate selected bodies with assigned boundary types.
            for oid in self._selection:
                if oid not in objects:
                    continue
                labels = [str(x) for x in bc_map.get(str(oid), []) if str(x).strip()]
                if not labels:
                    continue
                obj = objects[oid]
                if not bool(obj.visible):
                    continue
                verts = np.asarray(obj.mesh.vertices, dtype=float)
                if verts.ndim != 2 or verts.shape[0] <= 0 or verts.shape[1] != 3:
                    continue
                c = np.mean(verts, axis=0)
                short = ", ".join(labels[:3])
                if len(labels) > 3:
                    short = short + ", ..."
                text = f"BC[{len(labels)}]: {short}"
                try:
                    self.plotter.add_point_labels([c], [text], font_size=10, shape=None, text_color="#ffcc66")
                except Exception:
                    continue

            if self._show_axes:
                self.plotter.add_axes(line_width=2)
            if self._show_grid:
                self._apply_grid()
                if self._grid_plane:
                    try:
                        step = max(1e-6, float(self._grid_step))
                        size = max(step * 2.0, float(self._grid_size))
                        res = min(200, max(2, int(round(size / step))))
                        plane = self._pv.Plane(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0), i_size=size, j_size=size, i_resolution=res, j_resolution=res)
                        self.plotter.add_mesh(
                            plane,
                            style="wireframe",
                            color=str(self._grid_color or "#888888"),
                            opacity=0.22,
                            line_width=0.6,
                            pickable=False,
                            lighting=False,
                        )
                    except Exception:
                        pass
            if self._clipping_enabled:
                for pl in self._clipping_planes:
                    try:
                        self.plotter.add_mesh_clip_plane(None, normal=pl["normal"], invert=False)
                    except Exception:
                        continue
            self._apply_actor_visuals()
            if self._lod_active:
                self._apply_navigation_lod_visuals()
            else:
                self._draw_selection_overlays()
                self._draw_subselection()
            self._sync_quick_toolbar_state()
            self._restore_camera_state(cam_state)
            self._safe_render()
        finally:
            self._refresh_in_progress = False
            pending = self._pending_refresh
            self._pending_refresh = None
            if pending is not None:
                p_objects, p_selection, p_markers, p_bc, p_hover, p_active = pending
                self.refresh_scene(
                    p_objects,
                    p_selection,
                    p_markers,
                    boundary_map=p_bc,
                    hover_id=p_hover,
                    active_id=p_active,
                )

    def _apply_actor_visuals(self):
        if not self.is_available():
            return
        token = str(self._render_mode or "solid_edges").strip().lower()
        selected = {str(x) for x in self._selection}
        for oid, actor in list(self._actor_by_id.items()):
            obj = self._objects_cache.get(str(oid))
            if obj is None:
                continue
            is_sel = str(oid) in selected
            is_active = bool(is_sel and str(oid) == str(self._active_object_id))
            try:
                prop = actor.GetProperty()
            except Exception:
                continue
            try:
                if token == "wireframe":
                    prop.SetRepresentationToWireframe()
                    prop.EdgeVisibilityOn()
                else:
                    prop.SetRepresentationToSurface()
                    if token == "solid_edges" or is_sel or self._xray_mode:
                        prop.EdgeVisibilityOn()
                    else:
                        prop.EdgeVisibilityOff()
                if is_active:
                    prop.SetEdgeColor(tuple(float(x) for x in self._pv.Color("#ffe066").float_rgb))
                    prop.SetLineWidth(2.8)
                elif is_sel:
                    prop.SetEdgeColor(tuple(float(x) for x in self._pv.Color("#00d4ff").float_rgb))
                    prop.SetLineWidth(1.6)
                else:
                    prop.SetEdgeColor(tuple(float(x) for x in self._pv.Color("#222222").float_rgb))
                    prop.SetLineWidth(0.2)
            except Exception:
                pass
            try:
                prop.SetColor(tuple(float(x) for x in self._pv.Color(str(obj.color)).float_rgb))
            except Exception:
                pass
            try:
                opacity = float(getattr(obj, "opacity", 0.85))
                if self._xray_mode:
                    opacity = max(0.05, min(0.35, opacity * 0.35))
                if bool(getattr(obj, "locked", False)):
                    opacity = min(opacity, 0.45)
                prop.SetOpacity(float(opacity))
            except Exception:
                pass

    def update_selection(self, selection, *, hover_id: str = "", active_id: str = ""):
        self._selection = [str(x) for x in selection]
        if hover_id:
            self._hover_object_id = str(hover_id)
        if active_id:
            self._active_object_id = str(active_id)
        elif self._selection:
            self._active_object_id = str(self._selection[0])
        else:
            self._active_object_id = ""
        if not self.is_available():
            return
        self._apply_actor_visuals()
        if self._lod_active:
            self._apply_navigation_lod_visuals()
        else:
            self._draw_selection_overlays()
            self._draw_subselection()
        self._sync_quick_toolbar_state()
        self._safe_render()

    def _draw_selection_overlays(self):
        if not self.is_available():
            return
        for actor in list(self._selection_overlay_actors):
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                continue
        self._selection_overlay_actors = []
        if self._hover_overlay_actor is not None:
            try:
                self.plotter.remove_actor(self._hover_overlay_actor, render=False)
            except Exception:
                pass
            self._hover_overlay_actor = None

        for oid in self._selection:
            obj = self._objects_cache.get(str(oid))
            if obj is None or not bool(getattr(obj, "visible", True)):
                continue
            poly = self._poly_from_object(obj)
            if poly is None:
                continue
            is_active = str(oid) == str(self._active_object_id)
            color = "#ffe066" if is_active else "#00d4ff"
            width = 3.6 if is_active else 2.4
            try:
                wire = self.plotter.add_mesh(
                    poly,
                    style="wireframe",
                    color=color,
                    line_width=float(width),
                    opacity=1.0,
                    pickable=False,
                    lighting=False,
                )
                self._selection_overlay_actors.append(wire)
                fill = self.plotter.add_mesh(
                    poly,
                    color=color,
                    opacity=0.08 if is_active else 0.05,
                    pickable=False,
                    lighting=False,
                    show_edges=False,
                )
                self._selection_overlay_actors.append(fill)
            except Exception:
                continue

        hov = str(self._hover_object_id or "").strip()
        if hov and hov not in {str(x) for x in self._selection}:
            obj = self._objects_cache.get(hov)
            if obj is not None and bool(getattr(obj, "visible", True)):
                poly = self._poly_from_object(obj)
                if poly is not None:
                    try:
                        self._hover_overlay_actor = self.plotter.add_mesh(
                            poly,
                            style="wireframe",
                            color="#34e5ff",
                            line_width=2.8,
                            opacity=1.0,
                            pickable=False,
                            lighting=False,
                        )
                    except Exception:
                        self._hover_overlay_actor = None

    def _draw_subselection(self):
        if not self.is_available():
            return
        self._clear_subselection_actors()
        if self._sub_face is not None:
            oid, face_idx = self._sub_face
            obj = self._objects_cache.get(str(oid))
            if obj is not None:
                faces = np.asarray(obj.mesh.faces, dtype=int)
                verts = np.asarray(obj.mesh.vertices, dtype=float)
                if 0 <= int(face_idx) < int(faces.shape[0]):
                    tri = faces[int(face_idx)]
                    tri_pts = np.asarray([verts[int(tri[0])], verts[int(tri[1])], verts[int(tri[2])]], dtype=float)
                    try:
                        poly = self._pv.PolyData(tri_pts, np.array([3, 0, 1, 2], dtype=np.int64))
                        self._sub_face_actor = self.plotter.add_mesh(
                            poly,
                            color="#ffcc00",
                            opacity=0.55,
                            show_edges=True,
                            edge_color="#ff8800",
                            line_width=3.0,
                            pickable=False,
                        )
                    except Exception:
                        self._sub_face_actor = None
        if self._sub_edge is not None:
            oid, edge = self._sub_edge
            obj = self._objects_cache.get(str(oid))
            if obj is not None:
                verts = np.asarray(obj.mesh.vertices, dtype=float)
                try:
                    a = int(edge[0])
                    b = int(edge[1])
                except Exception:
                    a, b = -1, -1
                if 0 <= a < int(verts.shape[0]) and 0 <= b < int(verts.shape[0]):
                    line = self._pv.Line(verts[a], verts[b], resolution=1)
                    try:
                        self._sub_edge_actor = self.plotter.add_mesh(
                            line,
                            color="#00d4ff",
                            line_width=8.0,
                            opacity=1.0,
                            pickable=False,
                            render_lines_as_tubes=True,
                        )
                    except Exception:
                        self._sub_edge_actor = None

    def _apply_navigation_lod_visuals(self):
        if not self.is_available():
            return
        for actor in list(self._actor_by_id.values()):
            try:
                prop = actor.GetProperty()
                prop.SetRepresentationToSurface()
                prop.EdgeVisibilityOff()
                prop.SetLineWidth(0.1)
            except Exception:
                continue
        for actor in list(self._selection_overlay_actors):
            try:
                self.plotter.remove_actor(actor, render=False)
            except Exception:
                continue
        self._selection_overlay_actors = []
        if self._hover_overlay_actor is not None:
            try:
                self.plotter.remove_actor(self._hover_overlay_actor, render=False)
            except Exception:
                pass
            self._hover_overlay_actor = None

    def _set_navigation_lod_state(self, enabled: bool):
        if not self.is_available():
            self._lod_active = bool(enabled)
            return
        target = bool(enabled)
        if target:
            self._lod_active = True
            self._apply_navigation_lod_visuals()
            self._safe_render()
            return
        if not self._lod_active:
            return
        self._lod_active = False
        self._apply_actor_visuals()
        self._draw_selection_overlays()
        self._draw_subselection()
        self._safe_render()

    def _on_start_interaction_vtk(self, _obj, _ev):
        if self._navigation_active:
            return
        self._navigation_active = True
        self._perf["interaction_starts"] = int(self._perf.get("interaction_starts", 0)) + 1
        self._suspend_hover_until = time.perf_counter() + 0.08
        if self._hover_debounce_timer.isActive():
            self._hover_debounce_timer.stop()
        if self._lod_during_navigation:
            self._set_navigation_lod_state(True)

    def _on_end_interaction_vtk(self, _obj, _ev):
        if not self._navigation_active:
            return
        self._navigation_active = False
        self._perf["interaction_ends"] = int(self._perf.get("interaction_ends", 0)) + 1
        self._suspend_hover_until = time.perf_counter() + 0.04
        if self._wheel_debounce_timer.isActive():
            self._wheel_debounce_timer.stop()
            self._flush_wheel_zoom()
        if self._lod_active:
            self._set_navigation_lod_state(False)
        if self._pending_hover_pos is not None:
            self._hover_debounce_timer.start(18)

    @staticmethod
    def _is_valid_point(point: Optional[Sequence[float]]) -> bool:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            return False
        try:
            vals = np.asarray(point, dtype=float).reshape(3)
            return bool(np.all(np.isfinite(vals)))
        except Exception:
            return False

    def _scene_bounds(self) -> Optional[list]:
        if not self._actor_by_id:
            return None
        return self._bounds_for_ids(list(self._actor_by_id.keys()))

    def _resolve_zoom_focus(self, picked_id: Optional[str], picked_point: Optional[tuple]) -> np.ndarray:
        # Focus priority: picked point -> current selection -> camera focal -> scene center.
        if picked_id and self._is_valid_point(picked_point):
            return np.asarray(picked_point, dtype=float).reshape(3)
        sel_bounds = self._bounds_for_ids([str(x) for x in self._selection])
        if sel_bounds is not None:
            center, _ = self._bounds_center_diag(sel_bounds)
            return center
        if self.is_available():
            try:
                cam = self.plotter.renderer.GetActiveCamera()
                if cam is not None:
                    return np.asarray(cam.GetFocalPoint(), dtype=float).reshape(3)
            except Exception:
                pass
        scene_bounds = self._scene_bounds()
        if scene_bounds is not None:
            center, _ = self._bounds_center_diag(scene_bounds)
            return center
        return np.asarray([0.0, 0.0, 0.0], dtype=float)

    def _update_adaptive_clipping(self, cam, focus: np.ndarray, scene_diag: float):
        try:
            pos = np.asarray(cam.GetPosition(), dtype=float).reshape(3)
            dist = float(np.linalg.norm(pos - focus))
            base = max(1e-9, float(scene_diag))
            near = max(base * 1e-6, dist * 0.0015)
            far = max(near * 120.0, dist + base * 8.0)
            cam.SetClippingRange(float(near), float(far))
        except Exception:
            try:
                self.plotter.renderer.ResetCameraClippingRange()
            except Exception:
                pass

    def _vtk_pick(self, x: Optional[int] = None, y: Optional[int] = None) -> Tuple[Optional[str], Optional[tuple], Optional[int]]:
        if not self.is_available():
            return None, None, None
        try:
            if x is None or y is None:
                iren = self.plotter.interactor
                x, y = iren.GetEventPosition()
            picker = self._pv._vtk.vtkCellPicker()
            mode = str(self._selection_mode or "object").strip().lower()
            tol_map = {
                "object": 0.00055,
                "body": 0.00055,
                "component": 0.00055,
                "face": 0.00085,
                "edge": 0.0014,
                "vertex": 0.0020,
            }
            picker.SetTolerance(float(tol_map.get(mode, 0.0006)))
            picker.Pick(x, y, 0, self.plotter.renderer)
            actor = picker.GetActor()
            point = tuple(float(v) for v in picker.GetPickPosition())
            try:
                cell_id = int(picker.GetCellId())
            except Exception:
                cell_id = -1
            if actor is None:
                return None, point, (cell_id if cell_id >= 0 else None)
            picked_id = None
            for oid, act in self._actor_by_id.items():
                try:
                    if actor == act:
                        picked_id = oid
                        break
                except Exception:
                    continue
            return picked_id, point, (cell_id if cell_id >= 0 else None)
        except Exception:
            return None, None, None

    def _on_left_press_vtk(self, _obj, _ev):
        if not self.is_available():
            return
        self._left_button_down = True
        self._left_dragging = False
        self._left_press_ts = time.perf_counter()
        try:
            x, y = self.plotter.interactor.GetEventPosition()
            self._left_press_pos = (int(x), int(y))
        except Exception:
            self._left_press_pos = None
        mod = 0
        try:
            from PySide6.QtWidgets import QApplication

            mod = QApplication.keyboardModifiers()
        except Exception:
            mod = 0
        self._left_press_modifiers = int(self._enum_to_int(mod))

    def _on_left_release_vtk(self, obj, _ev):
        if not self.is_available():
            return
        was_down = bool(self._left_button_down)
        was_dragging = bool(self._left_dragging)
        press_pos = self._left_press_pos
        press_ts = float(self._left_press_ts)
        mod = int(self._left_press_modifiers)
        self._left_button_down = False
        self._left_dragging = False
        self._left_press_pos = None
        self._left_press_ts = 0.0
        self._left_press_modifiers = 0
        if not was_down:
            return
        try:
            x, y = self.plotter.interactor.GetEventPosition()
            rel_pos = (int(x), int(y))
        except Exception:
            rel_pos = press_pos
        if press_pos is not None and rel_pos is not None:
            dist = float(math.hypot(float(rel_pos[0] - press_pos[0]), float(rel_pos[1] - press_pos[1])))
        else:
            dist = 0.0
        dt = time.perf_counter() - press_ts if press_ts > 0.0 else 0.0
        is_click = (not was_dragging) and dist <= float(self._left_drag_threshold_px) and dt <= float(self._left_click_max_duration_s)
        if not is_click:
            return
        self._on_left_click_vtk(obj, _ev, pick_pos=rel_pos, modifiers=mod, abort_event=True)

    def _on_left_click_vtk(self, obj, _ev, *, pick_pos: Optional[Tuple[int, int]] = None, modifiers=None, abort_event: bool = True):
        picked_id, picked_point, picked_cell = self._vtk_pick(*(pick_pos or (None, None)))
        if picked_point is not None:
            self._last_pick_point = tuple(float(x) for x in picked_point)
        mod = modifiers
        if mod is None:
            mod = 0
            try:
                from PySide6.QtWidgets import QApplication

                mod = QApplication.keyboardModifiers()
            except Exception:
                mod = 0
        ctrl_mod = self._qt_modifier_flag("ControlModifier")
        alt_mod = self._qt_modifier_flag("AltModifier")
        shift_mod = self._qt_modifier_flag("ShiftModifier")
        mod_int = self._enum_to_int(mod)

        if picked_id and self._has_modifier(mod, ctrl_mod) and self._has_modifier(mod, alt_mod):
            self.pieceCommandRequested.emit(
                {
                    "command": "toggle_visibility",
                    "object_id": str(picked_id),
                    "picked_point": tuple(picked_point) if picked_point is not None else None,
                    "picked_cell_id": int(picked_cell) if picked_cell is not None else None,
                    "modifiers": int(mod_int),
                }
            )
            if abort_event:
                try:
                    obj.SetAbortFlag(1)
                except Exception:
                    pass
            return

        if picked_id and self._has_modifier(mod, alt_mod):
            boundary_type = "fixed"
            if self._has_modifier(mod, shift_mod):
                boundary_type = "force"
            self.pieceCommandRequested.emit(
                {
                    "command": "apply_boundary",
                    "boundary_type": str(boundary_type),
                    "object_id": str(picked_id),
                    "picked_point": tuple(picked_point) if picked_point is not None else None,
                    "picked_cell_id": int(picked_cell) if picked_cell is not None else None,
                    "modifiers": int(mod_int),
                }
            )
            if abort_event:
                try:
                    obj.SetAbortFlag(1)
                except Exception:
                    pass
            return

        mode = "replace"
        if self._has_modifier(mod, shift_mod):
            mode = "add"
        elif self._has_modifier(mod, ctrl_mod):
            mode = "toggle"
        if picked_id:
            self.selectionRequested.emit([picked_id], mode)
            self.pickedInfo.emit(
                {
                    "object_id": str(picked_id),
                    "picked_point": tuple(picked_point) if picked_point is not None else None,
                    "picked_cell_id": int(picked_cell) if picked_cell is not None else None,
                    "modifiers": int(mod_int),
                    "selection_mode": str(mode),
                }
            )
            now = time.perf_counter()
            oid = str(picked_id)
            if oid and (now - float(self._last_left_click_ts)) <= 0.34 and oid == str(self._last_left_click_object_id):
                self.focus_on_ids([oid], detail=True)
                self.statusMessage.emit(f"Detail focus: {oid}")
                self._last_left_click_ts = 0.0
                self._last_left_click_object_id = ""
            else:
                self._last_left_click_ts = now
                self._last_left_click_object_id = oid
        else:
            self.selectionRequested.emit([], "replace")
            self.pickedInfo.emit(
                {
                    "object_id": "",
                    "picked_point": tuple(picked_point) if picked_point is not None else None,
                    "picked_cell_id": None,
                    "modifiers": int(mod_int),
                    "selection_mode": "replace",
                }
            )
            self._last_left_click_ts = 0.0
            self._last_left_click_object_id = ""
        if str(self._tool_mode).lower() == "measure" and picked_point is not None:
            self.measurePointPicked.emit(tuple(picked_point))
        if abort_event:
            try:
                obj.SetAbortFlag(1)
            except Exception:
                pass

    def _on_mouse_move_vtk(self, _obj, _ev):
        if not self.is_available():
            return
        now = time.perf_counter()
        try:
            x, y = self.plotter.interactor.GetEventPosition()
            pos = (int(x), int(y))
        except Exception:
            return
        if self._left_button_down:
            if self._left_press_pos is not None:
                dx = float(pos[0] - self._left_press_pos[0])
                dy = float(pos[1] - self._left_press_pos[1])
                if math.hypot(dx, dy) > float(self._left_drag_threshold_px):
                    self._left_dragging = True
            if self._left_dragging:
                self._suspend_hover_until = time.perf_counter() + 0.05
                return
        self._pending_hover_pos = pos
        self._last_hover_move_ts = now
        self._schedule_hover_pick(now, pos)

    def _schedule_hover_pick(self, now: float, pos: Tuple[int, int]):
        if self._refresh_in_progress or self._render_in_progress:
            self._perf["hover_skipped"] = int(self._perf.get("hover_skipped", 0)) + 1
            return
        if self._navigation_active:
            self._perf["hover_skipped"] = int(self._perf.get("hover_skipped", 0)) + 1
            return
        if now < float(self._suspend_hover_until):
            remain_ms = int(max(6.0, min(70.0, (float(self._suspend_hover_until) - now) * 1000.0 + 6.0)))
            self._hover_debounce_timer.start(remain_ms)
            self._perf["hover_skipped"] = int(self._perf.get("hover_skipped", 0)) + 1
            return
        if pos == self._last_hover_pos and (now - float(self._last_hover_pick_ts)) < 0.22:
            return

        prev_pos = self._last_hover_event_pos
        prev_ts = float(self._last_hover_event_ts)
        self._last_hover_event_pos = pos
        self._last_hover_event_ts = now

        speed = 0.0
        if prev_pos is not None and prev_ts > 0.0:
            dt = max(1e-6, now - prev_ts)
            dx = float(pos[0] - prev_pos[0])
            dy = float(pos[1] - prev_pos[1])
            speed = float(math.hypot(dx, dy) / dt)

        target_hz = 60.0
        if speed > 1400.0:
            target_hz = 22.0
        elif speed > 700.0:
            target_hz = 32.0
        elif speed > 280.0:
            target_hz = 44.0
        min_interval = max(1e-3, 1.0 / target_hz)
        self._hover_pick_interval_s = float(min_interval)

        elapsed = now - float(self._last_hover_pick_ts)
        if elapsed >= max(float(self._hover_pick_interval_s), float(self._hover_force_interval_s)):
            self._flush_hover_pick()
            return

        wait_ms = int(max(6.0, min(55.0, (float(self._hover_pick_interval_s) - elapsed) * 1000.0)))
        if not self._hover_debounce_timer.isActive() or wait_ms < self._hover_debounce_timer.remainingTime():
            self._hover_debounce_timer.start(wait_ms)

    def _flush_hover_pick(self):
        if not self.is_available():
            return
        pos = self._pending_hover_pos
        if pos is None:
            return
        now = time.perf_counter()
        if self._refresh_in_progress or self._render_in_progress or self._navigation_active:
            self._perf["hover_skipped"] = int(self._perf.get("hover_skipped", 0)) + 1
            return
        if now < float(self._suspend_hover_until):
            self._hover_debounce_timer.start(int(max(6.0, min(60.0, (float(self._suspend_hover_until) - now) * 1000.0 + 6.0))))
            self._perf["hover_skipped"] = int(self._perf.get("hover_skipped", 0)) + 1
            return
        if pos == self._last_hover_pos and (now - float(self._last_hover_pick_ts)) < 0.2:
            return

        self._pending_hover_pos = None
        self._last_hover_pick_ts = now
        t0 = time.perf_counter()
        picked_id, picked_point, picked_cell = self._vtk_pick(pos[0], pos[1])
        self._perf["hover_pick_count"] = int(self._perf.get("hover_pick_count", 0)) + 1
        self._perf["last_hover_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
        self._last_hover_pos = pos
        if picked_point is not None:
            self._last_pick_point = tuple(float(x) for x in picked_point)
        current = str(self._hover_object_id or "")
        incoming = str(picked_id or "")
        if incoming == current:
            return
        self._hover_object_id = incoming
        self.hoverInfo.emit(
            {
                "object_id": incoming,
                "picked_point": tuple(picked_point) if picked_point is not None else None,
                "picked_cell_id": int(picked_cell) if picked_cell is not None else None,
            }
        )
        try:
            self._draw_selection_overlays()
            self._safe_render()
        except Exception:
            pass

    def _selection_anchor_point(self) -> Optional[tuple]:
        if self._last_pick_point is not None:
            return tuple(self._last_pick_point)
        bounds = self._bounds_for_ids([str(x) for x in self._selection])
        if bounds is None:
            return None
        center, _diag = self._bounds_center_diag(bounds)
        return (float(center[0]), float(center[1]), float(center[2]))

    def _wheel_zoom_factor(self, modifiers, camera_distance: float, scene_diag: float) -> float:
        ctrl_mod = self._qt_modifier_flag("ControlModifier")
        shift_mod = self._qt_modifier_flag("ShiftModifier")
        alt_mod = self._qt_modifier_flag("AltModifier")
        base_step = float(self._zoom_step)
        if self._has_modifier(modifiers, ctrl_mod) or self._has_modifier(modifiers, shift_mod):
            base_step = float(self._zoom_step_fine)
        elif self._has_modifier(modifiers, alt_mod):
            base_step = float(self._zoom_step_fast)
        ratio = max(1e-9, float(camera_distance)) / max(1e-9, float(scene_diag))
        adapt = 0.62 + 0.45 * max(0.35, min(2.8, math.sqrt(ratio)))
        return float(math.exp(math.log(max(1.0001, base_step)) * adapt))

    def _apply_wheel_zoom(self, steps: int, modifiers) -> None:
        if not self.is_available():
            return
        steps_i = int(steps)
        if steps_i == 0:
            return
        direction = 1 if steps_i > 0 else -1
        pulses = abs(steps_i)
        t0 = time.perf_counter()
        try:
            cam = self.plotter.renderer.GetActiveCamera()
            if cam is None:
                return
            picked_id, picked_point, _picked_cell = self._vtk_pick()
            if self._is_valid_point(picked_point):
                self._last_pick_point = tuple(float(x) for x in picked_point)
            focus = self._resolve_zoom_focus(picked_id, picked_point)
            fp = np.asarray(cam.GetFocalPoint(), dtype=float).reshape(3)
            pos = np.asarray(cam.GetPosition(), dtype=float).reshape(3)
            ctrl_mod = self._qt_modifier_flag("ControlModifier")
            shift_mod = self._qt_modifier_flag("ShiftModifier")
            blend = 0.36
            if self._has_modifier(modifiers, ctrl_mod) or self._has_modifier(modifiers, shift_mod):
                blend = 0.64
            new_fp = (1.0 - blend) * fp + blend * focus
            view_vec = pos - new_fp
            dist = float(np.linalg.norm(view_vec))
            if dist <= 1e-12:
                view_vec = np.asarray([1.0, 1.0, 1.0], dtype=float)
                dist = float(np.linalg.norm(view_vec))
            view_dir = view_vec / max(1e-12, dist)

            scene_bounds = self._scene_bounds()
            if scene_bounds is not None:
                _center, scene_diag = self._bounds_center_diag(scene_bounds)
            else:
                scene_diag = max(1e-3, dist)
            factor = self._wheel_zoom_factor(modifiers, dist, scene_diag)
            effective_pulses = min(8.0, 1.0 + 0.82 * float(max(0, pulses - 1)))
            factor = float(max(1.0001, factor) ** effective_pulses)
            now = time.perf_counter()
            dt = now - float(self._last_wheel_ts)
            self._last_wheel_ts = now
            if 0.0 < dt < 0.018:
                # Touchpads emit high-frequency wheel events; smooth each step to avoid bounce.
                factor = 1.0 + (factor - 1.0) * 0.52

            if bool(cam.GetParallelProjection()):
                scale = float(cam.GetParallelScale())
                min_scale = max(1e-9, scene_diag * 1e-6)
                max_scale = max(min_scale * 1200.0, scene_diag * 1200.0)
                new_scale = (scale / factor) if int(direction) > 0 else (scale * factor)
                new_scale = max(min_scale, min(max_scale, float(new_scale)))
                cam.SetParallelScale(float(new_scale))
                shift = new_fp - fp
                new_pos = pos + shift
                cam.SetFocalPoint(float(new_fp[0]), float(new_fp[1]), float(new_fp[2]))
                cam.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
                self._update_adaptive_clipping(cam, new_fp, scene_diag)
            else:
                min_dist = max(1e-6, scene_diag * 1e-5)
                if self._has_modifier(modifiers, ctrl_mod) or self._has_modifier(modifiers, shift_mod):
                    min_dist *= 0.5
                max_dist = max(min_dist * 200.0, scene_diag * float(self._zoom_far_cap))
                target_dist = (dist / factor) if int(direction) > 0 else (dist * factor)
                clamped_dist = max(min_dist, min(max_dist, float(target_dist)))
                if abs(clamped_dist - target_dist) > 1e-12:
                    self._perf["wheel_clamped"] = int(self._perf.get("wheel_clamped", 0)) + 1
                new_pos = new_fp + view_dir * clamped_dist
                cam.SetFocalPoint(float(new_fp[0]), float(new_fp[1]), float(new_fp[2]))
                cam.SetPosition(float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))
                self._update_adaptive_clipping(cam, new_fp, scene_diag)
            self._suspend_hover_until = time.perf_counter() + 0.03
            self._safe_render()
        except Exception:
            return
        finally:
            self._perf["wheel_count"] = int(self._perf.get("wheel_count", 0)) + 1
            self._perf["last_wheel_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

    def _queue_wheel_zoom(self, step: int, modifiers):
        s = 1 if int(step) > 0 else -1
        now = time.perf_counter()
        mod_i = int(self._enum_to_int(modifiers))

        # Flush pending batch when modifiers change to keep behavior deterministic.
        if self._wheel_accum_steps != 0 and mod_i != int(self._wheel_accum_modifiers):
            self._flush_wheel_zoom()

        # Anti-jitter: ignore tiny opposite spikes right after a direction flip.
        if (
            self._wheel_accum_steps != 0
            and (self._wheel_accum_steps > 0) != (s > 0)
            and (now - float(self._wheel_last_event_ts)) < 0.012
            and abs(int(self._wheel_accum_steps)) <= 1
        ):
            self._perf["wheel_skipped"] = int(self._perf.get("wheel_skipped", 0)) + 1
            return

        self._wheel_accum_steps = int(self._wheel_accum_steps) + int(s)
        self._wheel_accum_modifiers = int(mod_i)
        self._wheel_last_event_ts = now

        if abs(int(self._wheel_accum_steps)) >= 4:
            self._flush_wheel_zoom()
            return
        self._wheel_debounce_timer.start(12)

    def _flush_wheel_zoom(self):
        steps = int(self._wheel_accum_steps)
        if steps == 0:
            return
        mod = int(self._wheel_accum_modifiers)
        self._wheel_accum_steps = 0
        self._wheel_accum_modifiers = 0
        self._apply_wheel_zoom(steps, mod)

    def _on_mouse_wheel_forward_vtk(self, obj, _ev):
        mod = 0
        try:
            from PySide6.QtWidgets import QApplication

            mod = QApplication.keyboardModifiers()
        except Exception:
            mod = 0
        self._queue_wheel_zoom(+1, mod)
        try:
            obj.SetAbortFlag(1)
        except Exception:
            pass

    def _on_mouse_wheel_backward_vtk(self, obj, _ev):
        mod = 0
        try:
            from PySide6.QtWidgets import QApplication

            mod = QApplication.keyboardModifiers()
        except Exception:
            mod = 0
        self._queue_wheel_zoom(-1, mod)
        try:
            obj.SetAbortFlag(1)
        except Exception:
            pass

    def _on_right_click_vtk(self, obj, _ev):
        picked_id, picked_point, picked_cell = self._vtk_pick()
        if not self.is_available():
            return
        iren = self.plotter.interactor
        x, y = iren.GetEventPosition()
        h = max(1, self.plotter.interactor.height())
        qpt = QPoint(int(x), int(h - y))
        ctx = ContextInfo(
            widget="viewport",
            mouse_pos=qpt,
            global_pos=QCursor.pos(),
            picked_object_id=picked_id,
            picked_cell_id=picked_cell,
            selected_ids=list(self._selection),
            tool_mode=str(self._tool_mode),
            picked_point_3d=picked_point,
        )
        self.contextRequested.emit(ctx)
        try:
            obj.SetAbortFlag(1)
        except Exception:
            pass

    # -------- view operations --------
    def set_tool_mode(self, mode: str):
        self._tool_mode = str(mode or "Select")
        self.statusMessage.emit(f"Tool mode: {self._tool_mode}")

    def reset_view(self):
        if self.is_available():
            self.plotter.reset_camera()
            self._safe_render()

    def fit_all(self):
        self.reset_view()

    def zoom_in(self, factor: float = 1.12):
        if not self.is_available():
            return
        try:
            cam = self.plotter.renderer.GetActiveCamera()
            if cam is None:
                return
            cam.Zoom(float(max(1.01, factor)))
            self._safe_render()
        except Exception:
            return

    def zoom_out(self, factor: float = 1.12):
        if not self.is_available():
            return
        try:
            cam = self.plotter.renderer.GetActiveCamera()
            if cam is None:
                return
            cam.Zoom(1.0 / float(max(1.01, factor)))
            self._safe_render()
        except Exception:
            return

    def fit_selection(self):
        if not self.focus_on_ids(self._selection, detail=False):
            self.fit_all()

    def focus_detail_selection(self):
        if not self.focus_on_ids(self._selection, detail=True):
            self.fit_selection()

    def toggle_grid(self):
        if not self.is_available():
            return
        self._show_grid = not self._show_grid
        self._apply_grid()
        self._sync_quick_toolbar_state()
        self._safe_render()

    def toggle_axes(self):
        self._show_axes = not self._show_axes
        if self.is_available():
            try:
                if self._show_axes:
                    self.plotter.show_axes()
                else:
                    self.plotter.hide_axes()
            except Exception:
                pass
            self._sync_quick_toolbar_state()
            self._safe_render()
        self.statusMessage.emit("Axes toggled.")

    def set_grid_config(self, *, enabled: Optional[bool] = None, step: Optional[float] = None, size: Optional[float] = None, color: Optional[str] = None):
        if enabled is not None:
            self._show_grid = bool(enabled)
        if step is not None:
            try:
                self._grid_step = max(1e-6, float(step))
            except Exception:
                pass
        if size is not None:
            try:
                self._grid_size = max(self._grid_step * 2.0, float(size))
            except Exception:
                pass
        if color is not None:
            token = str(color).strip()
            if token:
                self._grid_color = token
        if self.is_available():
            self._apply_grid()
            self._sync_quick_toolbar_state()
            self._safe_render()

    def set_background(self, mode: str):
        if not self.is_available():
            return
        if str(mode).lower() == "light":
            self.plotter.set_background("#f7f9fb")
        else:
            self.plotter.set_background("#1f2430")
        self._safe_render()

    def set_render_mode(self, mode: str):
        self._render_mode = str(mode or "solid_edges").strip().lower()
        if self._render_mode not in {"solid", "wireframe", "solid_edges"}:
            self._render_mode = "solid_edges"
        self._sync_quick_toolbar_state()
        if not self.is_available():
            return
        self._apply_actor_visuals()
        if self._lod_active:
            self._apply_navigation_lod_visuals()
        else:
            self._draw_selection_overlays()
        self._safe_render()

    def set_xray_mode(self, enabled: bool):
        self._xray_mode = bool(enabled)
        self._sync_quick_toolbar_state()
        self.statusMessage.emit(f"X-Ray {'enabled' if self._xray_mode else 'disabled'}.")
        if self.is_available():
            self._apply_actor_visuals()
            if self._lod_active:
                self._apply_navigation_lod_visuals()
            else:
                self._draw_selection_overlays()
            self._safe_render()

    def set_subselection(self, face: Optional[tuple] = None, edge: Optional[tuple] = None):
        self._sub_face = tuple(face) if isinstance(face, (list, tuple)) and len(face) == 2 else None
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            oid = str(edge[0])
            pair = edge[1]
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                self._sub_edge = (oid, (int(pair[0]), int(pair[1])))
            else:
                self._sub_edge = None
        else:
            self._sub_edge = None
        if self.is_available():
            if not self._lod_active:
                self._draw_subselection()
            self._safe_render()

    def clear_subselection(self):
        self._sub_face = None
        self._sub_edge = None
        if self.is_available():
            self._clear_subselection_actors()
            self._safe_render()

    def screenshot(self, path: str):
        if self.is_available():
            self.plotter.screenshot(str(path))

    def add_clipping_plane(self, orientation: str):
        token = str(orientation or "xy").lower()
        normal = (0, 0, 1)
        if token == "xz":
            normal = (0, 1, 0)
        elif token == "yz":
            normal = (1, 0, 0)
        self._clipping_planes.append({"normal": normal})
        self.statusMessage.emit(f"Clipping plane added: {token.upper()}")

    def clear_clipping(self):
        self._clipping_planes = []
        self.statusMessage.emit("All clipping planes removed.")

    def toggle_clipping(self):
        self._clipping_enabled = not self._clipping_enabled
        self.statusMessage.emit(f"Clipping {'enabled' if self._clipping_enabled else 'disabled'}.")

    def set_hover_object(self, object_id: str):
        oid = str(object_id or "")
        if oid == self._hover_object_id:
            return
        self._hover_object_id = oid
        if self.is_available():
            if self._lod_active:
                self._apply_navigation_lod_visuals()
            else:
                self._draw_selection_overlays()
            self._safe_render()

    def _on_quick_action(self, token: str):
        op = str(token or "").strip().lower()
        if not op:
            return
        if op == "view_iso":
            if self.is_available():
                try:
                    self.plotter.view_isometric()
                    self._safe_render()
                except Exception:
                    pass
            return
        if op == "view_top":
            if self.is_available():
                try:
                    self.plotter.view_xy()
                    self._safe_render()
                except Exception:
                    pass
            return
        if op == "view_front":
            if self.is_available():
                try:
                    self.plotter.view_xz()
                    self._safe_render()
                except Exception:
                    pass
            return
        if op == "view_right":
            if self.is_available():
                try:
                    self.plotter.view_yz()
                    self._safe_render()
                except Exception:
                    pass
            return
        if op == "zoom_in":
            self.zoom_in()
            return
        if op == "zoom_out":
            self.zoom_out()
            return
        if op == "fit_all":
            self.fit_all()
            return
        if op == "fit_selection":
            self.fit_selection()
            return
        if op == "detail_focus":
            self.focus_detail_selection()
            return
        if op == "selection_cycle":
            self.quickActionRequested.emit("selection_cycle")
            return
        if op == "toggle_snap":
            self.quickActionRequested.emit("toggle_snap")
            return
        if op == "toggle_lod":
            self.quickActionRequested.emit("toggle_lod")
            return
        if op == "toggle_grid":
            self.toggle_grid()
            return
        if op == "toggle_axes":
            self.toggle_axes()
            return
        if op == "render_solid_edges":
            self.set_xray_mode(False)
            self.set_render_mode("solid_edges")
            return
        if op == "render_wireframe":
            self.set_xray_mode(False)
            self.set_render_mode("wireframe")
            return
        if op == "render_xray":
            self.set_xray_mode(True)
            self.set_render_mode("solid")
            return
        if op == "section_xy":
            self.add_clipping_plane("xy")
            return
        if op == "toggle_panels":
            self.quickActionRequested.emit("toggle_panels")
            return
        if op == "screenshot":
            self.quickActionRequested.emit("screenshot")
