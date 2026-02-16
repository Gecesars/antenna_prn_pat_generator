from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from core.angles import ang_dist_deg, wrap_phi_deg
from core.math_engine import MarkerValue
from core.perf import DEFAULT_TRACER, PerfTracer


@dataclass
class _MarkerArtist:
    marker: MarkerValue
    point_artist: any
    text_artist: any
    order: int


class AdvancedPlotInteractor:
    """
    Marker interactor for 2D plots.
    - Left click: add marker at nearest sample
    - Drag marker: move marker
    - Right click: context menu callback
    """

    def __init__(
        self,
        ax,
        canvas,
        get_series_callable: Callable[[], Tuple[Optional[np.ndarray], Optional[np.ndarray], str]],
        is_polar: bool,
        on_change: Optional[Callable] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_context_menu: Optional[Callable] = None,
        tracer: Optional[PerfTracer] = None,
        drag_hz: float = 60.0,
        table_refresh_ms: float = 200.0,
    ):
        self.ax = ax
        self.canvas = canvas
        self.get_series = get_series_callable
        self.is_polar = bool(is_polar)
        self.on_change = on_change
        self.on_status = on_status
        self.on_context_menu = on_context_menu
        self.tracer = tracer or DEFAULT_TRACER

        self._markers: List[_MarkerArtist] = []
        self._drag_name: Optional[str] = None
        self._counter = 1
        self._last_motion_ts = 0.0
        self._last_emit_ts = 0.0
        self._drag_interval = 1.0 / max(float(drag_hz), 1.0)
        self._emit_interval = max(float(table_refresh_ms), 0.0) / 1000.0

        self._series_cache_key = None
        self._series_cache = None

        self._cid_press = self.canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_release = self.canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion = self.canvas.mpl_connect("motion_notify_event", self._on_motion)

    def disconnect(self):
        for cid in (self._cid_press, self._cid_release, self._cid_motion):
            try:
                self.canvas.mpl_disconnect(cid)
            except Exception:
                pass

    def set_mode(self, is_polar: bool):
        self.is_polar = bool(is_polar)

    def _focus_canvas(self):
        try:
            self.canvas.get_tk_widget().focus_set()
        except Exception:
            pass

    def _emit(self, reason: str, force: bool = False):
        now = time.perf_counter()
        if not force and (now - self._last_emit_ts) < self._emit_interval:
            return
        self._last_emit_ts = now
        if callable(self.on_change):
            try:
                self.on_change(self.markers(), reason)
            except TypeError:
                self.on_change(self.markers())

    def markers(self) -> List[MarkerValue]:
        return [m.marker for m in sorted(self._markers, key=lambda x: x.order)]

    def _event_ang_deg(self, event, kind: str) -> Optional[float]:
        if event.xdata is None:
            return None
        if self.is_polar:
            deg = float(np.rad2deg(event.xdata))
            if kind == "H":
                deg = float(wrap_phi_deg(deg))
            return deg
        return float(event.xdata)

    def _get_series_cache(self):
        a, v, _k = self.get_series()
        if a is None or v is None:
            return None
        arr_a = np.asarray(a, dtype=float).reshape(-1)
        arr_v = np.asarray(v, dtype=float).reshape(-1)
        if arr_a.size == 0 or arr_v.size == 0 or arr_a.size != arr_v.size:
            return None

        key = (id(a), id(v), int(arr_a.size), float(arr_a[0]), float(arr_a[-1]))
        if self._series_cache_key == key and self._series_cache is not None:
            return self._series_cache

        arr_a = arr_a.copy()
        if str(_k).upper() == "H":
            arr_a = np.asarray(wrap_phi_deg(arr_a), dtype=float)

        idx = np.argsort(arr_a)
        arr_a = arr_a[idx]
        arr_v = arr_v[idx]

        uniform = None
        if arr_a.size >= 3:
            d = np.diff(arr_a)
            step = float(np.median(np.abs(d)))
            if step > 1e-12:
                max_dev = float(np.max(np.abs(np.abs(d) - step)))
                if max_dev <= max(step * 0.05, 1e-6):
                    uniform = (float(arr_a[0]), step, int(arr_a.size))

        self._series_cache_key = key
        self._series_cache = {
            "a": arr_a,
            "v": arr_v,
            "kind": str(_k).upper(),
            "uniform": uniform,
        }
        return self._series_cache

    def _nearest_sample(self, ang_deg: float, kind: str):
        cache = self._get_series_cache()
        if cache is None:
            return None
        arr_a = cache["a"]
        arr_v = cache["v"]
        uniform = cache["uniform"]

        if uniform is not None:
            start, step, n = uniform
            x = float(ang_deg)
            if kind == "H":
                span = step * (n - 1)
                while x < start:
                    x += 360.0
                while x > start + span:
                    x -= 360.0
            idx = int(round((x - start) / step))
            idx = max(0, min(n - 1, idx))
            return float(arr_a[idx]), float(arr_v[idx])

        if kind == "H":
            d = ang_dist_deg(arr_a, ang_deg)
            idx = int(np.argmin(d))
            return float(arr_a[idx]), float(arr_v[idx])
        idx = int(np.argmin(np.abs(arr_a - ang_deg)))
        return float(arr_a[idx]), float(arr_v[idx])

    def _marker_xy_pixels(self, mk: _MarkerArtist):
        marker = mk.marker
        if self.is_polar:
            theta_deg = float(marker.ang_deg)
            if marker.cut == "HRP":
                theta_deg = float((theta_deg + 360.0) % 360.0)
            theta = float(np.deg2rad(theta_deg))
            r = float(max(marker.mag_lin, 0.0))
            x, y = self.ax.transData.transform((theta, r))
            return float(x), float(y)

        y_mid = float(np.mean(self.ax.get_ylim()))
        x, y = self.ax.transData.transform((float(marker.ang_deg), y_mid))
        return float(x), float(y)

    def _hit_test(self, ang_deg: float, kind: str, event=None, tol_deg: float = 2.0, tol_px: float = 10.0) -> Optional[str]:
        ex = float(getattr(event, "x", float("nan"))) if event is not None else float("nan")
        ey = float(getattr(event, "y", float("nan"))) if event is not None else float("nan")

        for mk in self._markers:
            ref = mk.marker.ang_deg
            d = float(ang_dist_deg(ref, ang_deg)) if kind == "H" else abs(float(ref) - float(ang_deg))
            if math.isfinite(ex) and math.isfinite(ey):
                try:
                    mx, my = self._marker_xy_pixels(mk)
                    if math.hypot(mx - ex, my - ey) <= tol_px:
                        return mk.marker.name
                except Exception:
                    pass
            if d <= tol_deg:
                return mk.marker.name
        return None

    def _draw_marker(self, marker: MarkerValue, order: int) -> _MarkerArtist:
        color = "#d62728"
        if self.is_polar:
            theta_deg = marker.ang_deg if marker.cut == "VRP" else float((marker.ang_deg + 360.0) % 360.0)
            theta = float(np.deg2rad(theta_deg))
            point = self.ax.plot([theta], [max(marker.mag_lin, 0.0)], marker="o", color=color, markersize=5)[0]
            text = self.ax.text(theta, max(marker.mag_lin, 0.0), marker.name, color=color, fontsize=8)
        else:
            point = self.ax.plot([marker.ang_deg], [max(marker.mag_lin, 0.0)], marker="o", color=color, markersize=5)[0]
            text = self.ax.text(marker.ang_deg, max(marker.mag_lin, 0.0), marker.name, color=color, fontsize=8)
        return _MarkerArtist(marker=marker, point_artist=point, text_artist=text, order=order)

    def _remove_artist(self, ma: _MarkerArtist):
        for artist in (ma.point_artist, ma.text_artist):
            try:
                artist.remove()
            except Exception:
                pass

    def clear_markers(self):
        for ma in self._markers:
            self._remove_artist(ma)
        self._markers = []
        self._counter = 1
        self.canvas.draw_idle()
        self._emit("clear", force=True)

    def delete_marker(self, name: str):
        keep: List[_MarkerArtist] = []
        for ma in self._markers:
            if ma.marker.name == name:
                self._remove_artist(ma)
            else:
                keep.append(ma)
        self._markers = keep
        self.canvas.draw_idle()
        self._emit("delete", force=True)

    def rename_marker(self, old_name: str, new_name: str):
        new_name = str(new_name).strip()
        if not new_name:
            return
        for ma in self._markers:
            if ma.marker.name == old_name:
                ma.marker.name = new_name
                try:
                    ma.text_artist.set_text(new_name)
                except Exception:
                    pass
                break
        self.canvas.draw_idle()
        self._emit("rename", force=True)

    def add_marker_value(self, marker: MarkerValue):
        ma = self._draw_marker(marker, order=len(self._markers))
        self._markers.append(ma)
        self.canvas.draw_idle()
        self._emit("add", force=True)

    def _update_marker_position(self, name: str, ang_deg: float, mag_lin: float, kind: str, emit: bool = False):
        for ma in self._markers:
            if ma.marker.name != name:
                continue
            ma.marker.ang_deg = float(ang_deg)
            ma.marker.mag_lin = float(mag_lin)
            ma.marker.mag_db = float(20.0 * math.log10(max(mag_lin, 1e-12)))
            if kind == "H":
                ma.marker.phi_deg = float(ang_deg)
                ma.marker.cut = "HRP"
                ma.marker.theta_deg = None
            else:
                ma.marker.cut = "VRP"
                ma.marker.theta_deg = float(90.0 - ang_deg)
                ma.marker.phi_deg = None

            if self.is_polar:
                theta = float(np.deg2rad((ang_deg + 360.0) % 360.0 if kind == "H" else ang_deg))
                ma.point_artist.set_data([theta], [max(mag_lin, 0.0)])
                ma.text_artist.set_position((theta, max(mag_lin, 0.0)))
            else:
                ma.point_artist.set_data([ang_deg], [max(mag_lin, 0.0)])
                ma.text_artist.set_position((ang_deg, max(mag_lin, 0.0)))
            break
        self.canvas.draw_idle()
        self._emit("motion" if not emit else "release", force=emit)

    def _on_press(self, event):
        t0 = self.tracer.start()
        if event.inaxes != self.ax:
            self.tracer.log_if_slow("BUTTON_PRESS", t0, extra="outside-axes")
            return

        self._focus_canvas()

        if event.button in (2, 3):
            if callable(self.on_context_menu):
                self.on_context_menu(event, self)
            self.tracer.log_if_slow("BUTTON_PRESS", t0, extra="context-menu")
            return

        if event.button != 1:
            self.tracer.log_if_slow("BUTTON_PRESS", t0, extra=f"button={event.button}")
            return

        _a, _v, kind = self.get_series()
        kind = str(kind).upper()
        ang = self._event_ang_deg(event, kind)
        if ang is None:
            self.tracer.log_if_slow("BUTTON_PRESS", t0, extra="ang-none")
            return

        hit = self._hit_test(ang, kind, event=event)
        if hit:
            self._drag_name = hit
            self.tracer.log_if_slow("BUTTON_PRESS", t0, extra=f"start-drag-{hit}")
            return

        pt = self._nearest_sample(ang, kind)
        if pt is None:
            self.tracer.log_if_slow("BUTTON_PRESS", t0, extra="nearest-none")
            return

        ang0, v0 = pt
        name = f"m{self._counter}"
        self._counter += 1

        marker = MarkerValue(
            name=name,
            kind="2D",
            cut="HRP" if kind == "H" else "VRP",
            theta_deg=(None if kind == "H" else float(90.0 - ang0)),
            phi_deg=(float(ang0) if kind == "H" else None),
            ang_deg=float(ang0),
            mag_lin=float(v0),
            mag_db=float(20.0 * math.log10(max(v0, 1e-12))),
        )
        self.add_marker_value(marker)
        if callable(self.on_status):
            self.on_status(f"Marker {name}: ang={ang0:.2f} deg | mag={marker.mag_db:.2f} dB")
        self.tracer.log_if_slow("BUTTON_PRESS", t0, extra=f"add-{name}")

    def _on_release(self, event):
        t0 = self.tracer.start()
        if event.button == 1 and self._drag_name is not None:
            self._drag_name = None
            self._emit("release", force=True)
        self.tracer.log_if_slow("BUTTON_RELEASE", t0)

    def _on_motion(self, event):
        t0 = self.tracer.start()
        if self._drag_name is None:
            self.tracer.log_if_slow("MOUSE_MOVE", t0, extra="not-dragging")
            return
        if event.inaxes != self.ax:
            self.tracer.log_if_slow("MOUSE_MOVE", t0, extra="outside-axes")
            return

        now = time.perf_counter()
        if (now - self._last_motion_ts) < self._drag_interval:
            self.tracer.log_if_slow("MOUSE_MOVE", t0, extra="throttled")
            return
        self._last_motion_ts = now

        _a, _v, kind = self.get_series()
        kind = str(kind).upper()
        ang = self._event_ang_deg(event, kind)
        if ang is None:
            self.tracer.log_if_slow("MOUSE_MOVE", t0, extra="ang-none")
            return
        t_drag = self.tracer.start()
        pt = self._nearest_sample(ang, kind)
        if pt is None:
            self.tracer.log_if_slow("MOUSE_MOVE", t0, extra="nearest-none")
            return
        ang0, v0 = pt
        self._update_marker_position(self._drag_name, ang0, v0, kind, emit=False)
        if callable(self.on_status):
            self.on_status(f"Marker {self._drag_name}: ang={ang0:.2f} deg | mag={20.0*math.log10(max(v0,1e-12)):.2f} dB")
        self.tracer.log_if_slow("DRAG_MARKER", t_drag, extra=f"active={self._drag_name}")
        self.tracer.log_if_slow("MOUSE_MOVE", t0, extra="drag-update")
