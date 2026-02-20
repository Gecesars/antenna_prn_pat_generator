from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from mech.engine.scene_object import SceneObject, Marker
from .context_menu import ContextInfo


class ViewportPyVista(QWidget):
    selectionRequested = Signal(list, str)  # ids, mode
    contextRequested = Signal(object)  # ContextInfo
    statusMessage = Signal(str)
    measurePointPicked = Signal(tuple)  # x,y,z

    def __init__(self, parent=None):
        super().__init__(parent)
        self.plotter = None
        self._pv = None
        self._selection = []
        self._actor_by_id: Dict[str, object] = {}
        self._marker_actors = []
        self._tool_mode = "Select"
        self._show_grid = True
        self._show_axes = True
        self._clipping_enabled = True
        self._clipping_planes = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor

            self._pv = pv
            self.plotter = QtInteractor(self)
            layout.addWidget(self.plotter.interactor)
            self.plotter.set_background("#1f2430")
            self.plotter.add_axes(line_width=2)
            self.plotter.show_grid(color="#888888")
            iren = self.plotter.interactor
            iren.AddObserver("LeftButtonPressEvent", self._on_left_click_vtk, 1.0)
            iren.AddObserver("RightButtonPressEvent", self._on_right_click_vtk, 1.0)
        except Exception as e:
            lbl = QLabel(f"Viewport 3D indisponivel.\nInstale: pyside6 pyvista pyvistaqt vtk\n\nDetalhe: {e}", self)
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)

    def is_available(self) -> bool:
        return self.plotter is not None

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

    def refresh_scene(self, objects: Dict[str, SceneObject], selection, markers: Dict[str, Marker]):
        self._selection = [str(x) for x in selection]
        if not self.is_available():
            return
        self.plotter.clear()
        self._actor_by_id = {}
        self._marker_actors = []

        for oid, obj in objects.items():
            if not obj.visible:
                continue
            poly = self._poly_from_object(obj)
            if poly is None:
                continue
            is_sel = oid in self._selection
            actor = self.plotter.add_mesh(
                poly,
                name=str(oid),
                color=str(obj.color),
                opacity=float(obj.opacity),
                show_edges=bool(is_sel),
                edge_color="#ff4d4f" if is_sel else "#222222",
                line_width=1.4 if is_sel else 0.2,
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

        if self._show_axes:
            self.plotter.add_axes(line_width=2)
        if self._show_grid:
            self.plotter.show_grid(color="#888888")
        for pl in self._clipping_planes:
            try:
                self.plotter.add_mesh_clip_plane(None, normal=pl["normal"], invert=False)
            except Exception:
                continue
        self.plotter.render()

    def _vtk_pick(self) -> Tuple[Optional[str], Optional[tuple]]:
        if not self.is_available():
            return None, None
        try:
            iren = self.plotter.interactor
            x, y = iren.GetEventPosition()
            picker = self._pv._vtk.vtkCellPicker()
            picker.SetTolerance(0.0005)
            picker.Pick(x, y, 0, self.plotter.renderer)
            actor = picker.GetActor()
            point = tuple(float(v) for v in picker.GetPickPosition())
            if actor is None:
                return None, point
            picked_id = None
            for oid, act in self._actor_by_id.items():
                try:
                    if actor == act:
                        picked_id = oid
                        break
                except Exception:
                    continue
            return picked_id, point
        except Exception:
            return None, None

    def _on_left_click_vtk(self, obj, _ev):
        picked_id, picked_point = self._vtk_pick()
        mod = int(self.keyboardModifiers())
        mode = "replace"
        if mod & int(Qt.ShiftModifier):
            mode = "add"
        if picked_id:
            self.selectionRequested.emit([picked_id], mode)
        else:
            self.selectionRequested.emit([], "replace")
        if str(self._tool_mode).lower() == "measure" and picked_point is not None:
            self.measurePointPicked.emit(tuple(picked_point))
        try:
            obj.SetAbortFlag(1)
        except Exception:
            pass

    def _on_right_click_vtk(self, obj, _ev):
        picked_id, picked_point = self._vtk_pick()
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
            self.plotter.render()

    def fit_all(self):
        self.reset_view()

    def toggle_grid(self):
        if not self.is_available():
            return
        self._show_grid = not self._show_grid
        if self._show_grid:
            self.plotter.show_grid(color="#888888")
        else:
            self.plotter.remove_bounds_axes()
        self.plotter.render()

    def toggle_axes(self):
        self._show_axes = not self._show_axes
        self.statusMessage.emit("Axes toggled. Refresh scene to apply.")

    def set_background(self, mode: str):
        if not self.is_available():
            return
        if str(mode).lower() == "light":
            self.plotter.set_background("#f7f9fb")
        else:
            self.plotter.set_background("#1f2430")
        self.plotter.render()

    def set_render_mode(self, mode: str):
        if not self.is_available():
            return
        token = str(mode or "solid").lower()
        for actor in self._actor_by_id.values():
            try:
                prop = actor.GetProperty()
                if token == "wireframe":
                    prop.SetRepresentationToWireframe()
                elif token == "solid_edges":
                    prop.SetRepresentationToSurface()
                    prop.EdgeVisibilityOn()
                else:
                    prop.SetRepresentationToSurface()
                    prop.EdgeVisibilityOff()
            except Exception:
                continue
        self.plotter.render()

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
