from __future__ import annotations

import math
import os
import tempfile
import webbrowser
from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from core.angles import wrap_phi_deg
from core.reconstruct3d import SphericalPattern, sample_spherical


def _grid_to_cartesian(pattern: SphericalPattern, r_lin: np.ndarray):
    th = np.deg2rad(np.asarray(pattern.theta_deg, dtype=float).reshape(-1))
    ph = np.deg2rad(np.asarray(pattern.phi_deg, dtype=float).reshape(-1))
    th2, ph2 = np.meshgrid(th, ph, indexing="ij")
    rr = np.asarray(r_lin, dtype=float)
    x = rr * np.sin(th2) * np.cos(ph2)
    y = rr * np.sin(th2) * np.sin(ph2)
    z = rr * np.cos(th2)
    return x, y, z


def _radius_from_mag(mag_lin: np.ndarray, gamma: float = 1.0, r0: float = 0.08, scale: float = 1.0):
    m = np.clip(np.asarray(mag_lin, dtype=float), 1e-12, None)
    return r0 + scale * np.power(m, float(gamma))


def _db_from_mag(mag_lin: np.ndarray, eps: float = 1e-12):
    m = np.clip(np.asarray(mag_lin, dtype=float), eps, None)
    return 20.0 * np.log10(m)


def open_3d_view(
    pattern: SphericalPattern,
    title: str = "3D Pattern",
    db_min: float = -40.0,
    db_max: float = 0.0,
    gamma: float = 1.0,
    r0: float = 0.08,
    scale: float = 1.0,
    wireframe: bool = False,
    on_pick: Optional[Callable[[float, float, float], None]] = None,
) -> str:
    """
    Open an interactive 3D viewer.

    Returns viewer backend: "pyvista" or "plotly".
    """
    mag = np.asarray(pattern.mag_lin, dtype=float)
    if mag.shape != (len(pattern.theta_deg), len(pattern.phi_deg)):
        raise ValueError("Invalid pattern grid shape.")

    r = _radius_from_mag(mag, gamma=gamma, r0=r0, scale=scale)
    db = _db_from_mag(mag)
    x, y, z = _grid_to_cartesian(pattern, r)

    try:
        import pyvista as pv  # type: ignore

        grid = pv.StructuredGrid(x, y, z)
        grid["mag_db"] = db.reshape(-1, order="F")

        p = pv.Plotter(title=title)
        p.add_mesh(
            grid,
            scalars="mag_db",
            cmap="turbo",
            clim=[float(db_min), float(db_max)],
            show_scalar_bar=True,
            scalar_bar_args={"title": "dB"},
            show_edges=bool(wireframe),
            smooth_shading=True,
        )

        # overlay phi=0 and phi=90 cuts
        ph = np.asarray(pattern.phi_deg, dtype=float)
        for target in (0.0, 90.0):
            j = int(np.argmin(np.abs(((ph - target + 180.0) % 360.0) - 180.0)))
            pts = np.c_[x[:, j], y[:, j], z[:, j]]
            p.add_lines(pts, color="white", width=2)

        def _pick_cb(point):
            try:
                xx, yy, zz = float(point[0]), float(point[1]), float(point[2])
                rr = math.sqrt(xx * xx + yy * yy + zz * zz)
                if rr <= 1e-12:
                    return
                theta = math.degrees(math.acos(max(-1.0, min(1.0, zz / rr))))
                phi = float(wrap_phi_deg(math.degrees(math.atan2(yy, xx))))
                mag_lin = sample_spherical(pattern, theta, phi)
                if callable(on_pick):
                    on_pick(theta, phi, mag_lin)
            except Exception:
                return

        p.enable_point_picking(callback=_pick_cb, show_message=False, show_point=False)
        p.show()
        return "pyvista"
    except Exception:
        pass

    # Tk + Matplotlib fallback (works without extra dependencies)
    try:
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure
        from matplotlib import cm, colors
        from mpl_toolkits.mplot3d import proj3d  # type: ignore

        root = tk._default_root
        if root is None:
            root = tk.Tk()
            root.withdraw()

        top = tk.Toplevel(root)
        top.title(title)
        top.geometry("980x720")

        fig = Figure(figsize=(9.0, 6.5), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        norm = colors.Normalize(vmin=float(db_min), vmax=float(db_max))
        cmap = cm.get_cmap("turbo")
        face = cmap(norm(db))
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=face,
            linewidth=0.2 if wireframe else 0.0,
            antialiased=True,
            shade=False,
            alpha=0.95,
        )

        # Colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(db)
        fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.08, label="dB")

        # overlay phi=0 and phi=90 lines
        ph = np.asarray(pattern.phi_deg, dtype=float)
        for target, color in ((0.0, "white"), (90.0, "yellow")):
            j = int(np.argmin(np.abs(((ph - target + 180.0) % 360.0) - 180.0)))
            ax.plot(x[:, j], y[:, j], z[:, j], color=color, linewidth=1.6)

        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()

        flat_x = x.reshape(-1)
        flat_y = y.reshape(-1)
        flat_z = z.reshape(-1)
        nt, np_ = db.shape

        def _on_click(event):
            if event.inaxes != ax or event.x is None or event.y is None:
                return
            try:
                xp, yp, _ = proj3d.proj_transform(flat_x, flat_y, flat_z, ax.get_proj())
                pix = ax.transData.transform(np.column_stack([xp, yp]))
                d2 = np.sum((pix - np.array([event.x, event.y])) ** 2, axis=1)
                idx = int(np.argmin(d2))
                i = idx // np_
                j = idx % np_
                theta = float(pattern.theta_deg[i])
                phi = float(pattern.phi_deg[j])
                mag_lin = float(pattern.mag_lin[i, j])
                if callable(on_pick):
                    on_pick(theta, phi, mag_lin)
            except Exception:
                return

        canvas.mpl_connect("button_press_event", _on_click)
        canvas.draw_idle()
        return "tk-mpl3d"
    except Exception:
        pass

    # Plotly fallback
    try:
        import plotly.graph_objects as go  # type: ignore

        fig = go.Figure(
            data=[
                go.Surface(
                    x=x,
                    y=y,
                    z=z,
                    surfacecolor=db,
                    colorscale="Turbo",
                    cmin=float(db_min),
                    cmax=float(db_max),
                    colorbar={"title": "dB"},
                )
            ]
        )
        fig.update_layout(title=title, scene_aspectmode="data")
        tmp = tempfile.NamedTemporaryFile(prefix="eftx_3d_", suffix=".html", delete=False)
        tmp.close()
        fig.write_html(tmp.name, auto_open=False)
        webbrowser.open(tmp.name)
        return "plotly"
    except Exception as e:
        raise RuntimeError(
            "No 3D backend available. Install pyvista or plotly for interactive 3D view."
        ) from e


def export_obj(pattern: SphericalPattern, path: str, gamma: float = 1.0, r0: float = 0.08, scale: float = 1.0):
    mag = np.asarray(pattern.mag_lin, dtype=float)
    r = _radius_from_mag(mag, gamma=gamma, r0=r0, scale=scale)
    x, y, z = _grid_to_cartesian(pattern, r)

    nt, np_ = mag.shape
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# EFTX reconstructed pattern OBJ\n")
        for i in range(nt):
            for j in range(np_):
                f.write(f"v {x[i,j]:.8f} {y[i,j]:.8f} {z[i,j]:.8f}\n")

        def idx(i, j):
            return i * np_ + j + 1

        for i in range(nt - 1):
            for j in range(np_ - 1):
                a = idx(i, j)
                b = idx(i + 1, j)
                c = idx(i + 1, j + 1)
                d = idx(i, j + 1)
                f.write(f"f {a} {b} {c}\n")
                f.write(f"f {a} {c} {d}\n")


def export_plotly_html(
    pattern: SphericalPattern,
    path: str,
    db_min: float = -40.0,
    db_max: float = 0.0,
    gamma: float = 1.0,
    r0: float = 0.08,
    scale: float = 1.0,
):
    import plotly.graph_objects as go  # type: ignore

    mag = np.asarray(pattern.mag_lin, dtype=float)
    r = _radius_from_mag(mag, gamma=gamma, r0=r0, scale=scale)
    db = _db_from_mag(mag)
    x, y, z = _grid_to_cartesian(pattern, r)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=db,
                colorscale="Turbo",
                cmin=float(db_min),
                cmax=float(db_max),
                colorbar={"title": "dB"},
            )
        ]
    )
    fig.update_layout(title="EFTX 3D Pattern", scene_aspectmode="data")
    fig.write_html(path, auto_open=False)
