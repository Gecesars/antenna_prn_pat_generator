"""Standalone power divider module.

This module runs independently from deep3.py.
"""

from __future__ import annotations

import csv
import math
import threading
import traceback
from dataclasses import dataclass
from typing import List, Optional
import tkinter as tk
from tkinter import filedialog, messagebox

try:
    import customtkinter as ctk
except Exception as e:  # pragma: no cover
    raise ImportError("customtkinter is required to use divisor.py") from e

from core.divider_aedt import (
    SUBSTRATE_MATERIALS,
    DividerGeometryError,
    compute_coaxial_divider_geometry,
    generate_hfss_divider_model,
)


@dataclass
class DividerResult:
    output_idx: int
    percent_norm: float
    power_w: float
    amp_rel_total: float
    amp_rel_max: float
    rel_db: float
    phase_deg: float


class DivisorTab(ctk.CTkFrame):
    def __init__(self, master, app=None):
        super().__init__(master)
        self.app = app
        self.total_power_var = tk.StringVar(value="100")
        self.outputs_var = tk.StringVar(value="2")
        self.f_start_var = tk.StringVar(value="800")
        self.f_stop_var = tk.StringVar(value="1200")
        self.d_ext_var = tk.StringVar(value="20.0")
        self.wall_thick_var = tk.StringVar(value="1.5")
        self.n_sections_var = tk.StringVar(value="4")
        self.geom_outputs_var = tk.StringVar(value=self.outputs_var.get() or "2")
        self.diel_material_var = tk.StringVar(value="Ar")
        self.percent_vars: List[tk.StringVar] = []
        self.phase_vars: List[tk.StringVar] = []
        self._results: List[DividerResult] = []
        self._last_geometry: Optional[dict] = None
        self._aedt_thread: Optional[threading.Thread] = None
        self._build_ui()
        self._build_output_rows(2, keep_existing=False)

    def _build_ui(self) -> None:
        top = ctk.CTkFrame(self, fg_color="transparent")
        top.pack(fill=ctk.X, padx=10, pady=(10, 6))

        ctk.CTkLabel(top, text="Total power (W):").pack(side=ctk.LEFT, padx=(0, 6))
        ctk.CTkEntry(top, textvariable=self.total_power_var, width=100).pack(side=ctk.LEFT, padx=(0, 10))
        ctk.CTkLabel(top, text="Outputs:").pack(side=ctk.LEFT, padx=(0, 6))
        ctk.CTkEntry(top, textvariable=self.outputs_var, width=70).pack(side=ctk.LEFT, padx=(0, 10))
        ctk.CTkButton(top, text="Build", width=80, command=self._on_build_outputs).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(top, text="Equal", width=80, command=self._set_equal_split).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(top, text="Normalize %", width=110, command=self._normalize_percentages).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(top, text="Calculate", width=90, fg_color="#2f7d32", command=self._calculate).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(top, text="Export CSV", width=100, command=self._export_csv).pack(side=ctk.LEFT, padx=3)

        self.summary_lbl = ctk.CTkLabel(self, text="Percent sum: 0.00%", anchor="w")
        self.summary_lbl.pack(fill=ctk.X, padx=10, pady=(0, 6))

        self.editor = ctk.CTkScrollableFrame(self, label_text="Outputs setup")
        self.editor.pack(fill=ctk.X, padx=10, pady=(0, 6))

        self._build_aedt_geometry_section()

        self.results_tabs = ctk.CTkTabview(self)
        self.results_tabs.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(0, 10))

        power_tab = self.results_tabs.add("Power Results")
        geometry_tab = self.results_tabs.add("AEDT Geometry")
        log_tab = self.results_tabs.add("AEDT Log")

        self.results_power_box = ctk.CTkTextbox(power_tab, height=300)
        self.results_power_box.pack(fill=ctk.BOTH, expand=True, padx=4, pady=4)
        self.results_geom_box = ctk.CTkTextbox(geometry_tab, height=300)
        self.results_geom_box.pack(fill=ctk.BOTH, expand=True, padx=4, pady=4)
        self.results_log_box = ctk.CTkTextbox(log_tab, height=300)
        self.results_log_box.pack(fill=ctk.BOTH, expand=True, padx=4, pady=4)

        self._set_power_results_text(
            "Use Build/Equal/Calculate to compute the divider results.\n"
            "Use the AEDT section to calculate coaxial geometry and create the divider model in HFSS."
        )
        self._set_geometry_results_text("AEDT geometry output will be shown here.")
        self._set_log_text("AEDT execution log will be shown here.")
        self.results_tabs.set("Power Results")

    def _build_aedt_geometry_section(self) -> None:
        box = ctk.CTkFrame(self)
        box.pack(fill=ctk.X, padx=10, pady=(0, 8))
        ctk.CTkLabel(box, text="AEDT Coaxial Divider Geometry", anchor="w", font=("Arial", 14, "bold")).pack(
            fill=ctk.X, padx=8, pady=(6, 2)
        )

        grid = ctk.CTkFrame(box, fg_color="transparent")
        grid.pack(fill=ctk.X, padx=8, pady=(0, 4))
        for col in range(6):
            grid.grid_columnconfigure(col, weight=1)

        def _cell(row: int, col: int, label: str, var: tk.StringVar, width: int = 110):
            ctk.CTkLabel(grid, text=label, anchor="w").grid(row=row, column=col, padx=4, pady=3, sticky="w")
            ctk.CTkEntry(grid, textvariable=var, width=width).grid(row=row + 1, column=col, padx=4, pady=3, sticky="ew")

        _cell(0, 0, "f_start (MHz)", self.f_start_var)
        _cell(0, 1, "f_stop (MHz)", self.f_stop_var)
        _cell(0, 2, "d_ext (mm)", self.d_ext_var)
        _cell(0, 3, "wall_thick (mm)", self.wall_thick_var)
        _cell(0, 4, "n_sections", self.n_sections_var, width=90)
        _cell(0, 5, "n_outputs", self.geom_outputs_var, width=90)

        row2 = ctk.CTkFrame(box, fg_color="transparent")
        row2.pack(fill=ctk.X, padx=8, pady=(2, 6))
        ctk.CTkLabel(row2, text="Dielectric").pack(side=ctk.LEFT, padx=(0, 6))
        ctk.CTkOptionMenu(row2, variable=self.diel_material_var, values=list(SUBSTRATE_MATERIALS.keys())).pack(
            side=ctk.LEFT, padx=(0, 8)
        )
        ctk.CTkButton(row2, text="Sync outputs", width=110, command=self._sync_geom_outputs).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(row2, text="Calc geometry", width=120, command=self._calculate_aedt_geometry).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(row2, text="Create AEDT model", width=140, fg_color="#2f7d32", command=self._create_aedt_model).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(row2, text="Save AEDT as...", width=110, command=self._create_aedt_model_save_as).pack(side=ctk.LEFT, padx=3)

        self.aedt_status_lbl = ctk.CTkLabel(box, text="AEDT: idle", anchor="w")
        self.aedt_status_lbl.pack(fill=ctk.X, padx=8, pady=(0, 6))
        self.aedt_params_lbl = ctk.CTkLabel(box, text="Design params loaded: -", anchor="w")
        self.aedt_params_lbl.pack(fill=ctk.X, padx=8, pady=(0, 6))

    def _set_power_results_text(self, text: str) -> None:
        self.results_power_box.delete("1.0", tk.END)
        self.results_power_box.insert("1.0", text)

    def _set_geometry_results_text(self, text: str) -> None:
        self.results_geom_box.delete("1.0", tk.END)
        self.results_geom_box.insert("1.0", text)

    def _set_log_text(self, text: str) -> None:
        self.results_log_box.delete("1.0", tk.END)
        self.results_log_box.insert("1.0", text)

    def _append_log(self, text: str) -> None:
        self.results_log_box.insert(tk.END, f"\n{text}")
        self.results_log_box.see(tk.END)

    def _parse_float(self, text: str, default: float = 0.0) -> float:
        raw = str(text or "").strip().replace(",", ".")
        if not raw:
            return default
        return float(raw)

    def _parse_int(self, text: str, default: int = 0) -> int:
        raw = str(text or "").strip().replace(",", ".")
        if not raw:
            return int(default)
        return int(round(float(raw)))

    def _set_status(self, text: str) -> None:
        msg = str(text or "").strip()
        if not msg:
            return
        if threading.current_thread() is not threading.main_thread():
            try:
                self.after(0, lambda m=msg: self._set_status(m))
            except Exception:
                pass
            return
        if hasattr(self, "aedt_status_lbl"):
            self.aedt_status_lbl.configure(text=f"AEDT: {msg}")
        if hasattr(self, "results_log_box"):
            self._append_log(msg)
        if self.app is not None and hasattr(self.app, "_set_status"):
            try:
                self.app._set_status(msg)
            except Exception:
                pass

    def _sync_geom_outputs(self) -> None:
        token = str(self.outputs_var.get() or "").strip()
        if token:
            self.geom_outputs_var.set(token)

    def _on_build_outputs(self) -> None:
        try:
            count = int(float(self.outputs_var.get().strip().replace(",", ".")))
        except Exception:
            messagebox.showerror("Invalid outputs", "Outputs must be an integer.")
            return
        if count < 1 or count > 64:
            messagebox.showerror("Invalid outputs", "Outputs must be between 1 and 64.")
            return
        self.geom_outputs_var.set(str(count))
        self._build_output_rows(count, keep_existing=True)

    def _build_output_rows(self, count: int, keep_existing: bool = True) -> None:
        old_pct = [v.get() for v in self.percent_vars]
        old_ph = [v.get() for v in self.phase_vars]
        for child in self.editor.winfo_children():
            child.destroy()
        self.percent_vars = []
        self.phase_vars = []

        header = ctk.CTkFrame(self.editor, fg_color="transparent")
        header.pack(fill=ctk.X, padx=4, pady=(2, 4))
        ctk.CTkLabel(header, text="Output", width=80, anchor="w").pack(side=ctk.LEFT, padx=2)
        ctk.CTkLabel(header, text="Percent (%)", width=100, anchor="w").pack(side=ctk.LEFT, padx=2)
        ctk.CTkLabel(header, text="Phase (deg)", width=100, anchor="w").pack(side=ctk.LEFT, padx=2)

        for i in range(count):
            row = ctk.CTkFrame(self.editor, fg_color="transparent")
            row.pack(fill=ctk.X, padx=4, pady=2)
            ctk.CTkLabel(row, text=f"OUT {i + 1}", width=80, anchor="w").pack(side=ctk.LEFT, padx=2)

            default_pct = str(100.0 / count)
            if keep_existing and i < len(old_pct) and old_pct[i].strip():
                default_pct = old_pct[i]
            pct_var = tk.StringVar(value=default_pct)
            pct_entry = ctk.CTkEntry(row, textvariable=pct_var, width=100)
            pct_entry.pack(side=ctk.LEFT, padx=2)
            self.percent_vars.append(pct_var)

            default_phase = "0"
            if keep_existing and i < len(old_ph) and old_ph[i].strip():
                default_phase = old_ph[i]
            ph_var = tk.StringVar(value=default_phase)
            ph_entry = ctk.CTkEntry(row, textvariable=ph_var, width=100)
            ph_entry.pack(side=ctk.LEFT, padx=2)
            self.phase_vars.append(ph_var)

        self._update_percent_sum()

    def _update_percent_sum(self) -> None:
        total = 0.0
        for var in self.percent_vars:
            try:
                total += self._parse_float(var.get(), 0.0)
            except Exception:
                continue
        self.summary_lbl.configure(text=f"Percent sum: {total:.2f}%")

    def _set_equal_split(self) -> None:
        n = max(1, len(self.percent_vars))
        value = 100.0 / float(n)
        for var in self.percent_vars:
            var.set(f"{value:.6f}")
        self._update_percent_sum()

    def _normalize_percentages(self) -> None:
        pcts: List[float] = []
        for var in self.percent_vars:
            try:
                pcts.append(self._parse_float(var.get(), 0.0))
            except Exception:
                pcts.append(0.0)
        total = sum(pcts)
        if total <= 0.0:
            messagebox.showwarning("Normalize", "Percent sum must be > 0.")
            return
        scale = 100.0 / total
        for i, var in enumerate(self.percent_vars):
            var.set(f"{pcts[i] * scale:.6f}")
        self._update_percent_sum()

    def _aedt_input_params(self) -> dict:
        return {
            "f_start": self._parse_float(self.f_start_var.get(), 800.0),
            "f_stop": self._parse_float(self.f_stop_var.get(), 1200.0),
            "d_ext": self._parse_float(self.d_ext_var.get(), 20.0),
            "wall_thick": self._parse_float(self.wall_thick_var.get(), 1.5),
            "n_sections": self._parse_int(self.n_sections_var.get(), 4),
            "n_outputs": self._parse_int(self.geom_outputs_var.get(), max(1, len(self.percent_vars))),
            "diel_material": str(self.diel_material_var.get() or "Ar"),
        }

    def _format_aedt_geometry(self, geo: dict) -> str:
        lines = [
            "AEDT geometry (coaxial divider)",
            "",
            f"Input f_start: {float(geo['f_start']):.6f} MHz",
            f"Input f_stop : {float(geo['f_stop']):.6f} MHz",
            f"Center f0    : {float(geo['f0_mhz']):.6f} MHz",
            f"n_sections   : {int(geo['n_sections'])}",
            f"n_outputs    : {int(geo['n_outputs'])}",
            f"d_ext        : {float(geo['d_ext']):.6f} mm",
            f"wall_thick   : {float(geo['wall_thick']):.6f} mm",
            f"d_int_tube   : {float(geo['d_int_tube']):.6f} mm",
            f"sec_len_mm   : {float(geo['sec_len_mm']):.6f}",
            f"len_inner_mm : {float(geo['len_inner_mm']):.6f}",
            f"len_outer_mm : {float(geo['len_outer_mm']):.6f}",
            f"d_out_50ohm  : {float(geo['d_out_50ohm']):.6f}",
            f"dia_saida_diel: {float(geo['dia_saida_diel']):.6f}",
            f"dia_saida_cond: {float(geo['dia_saida_cond']):.6f}",
        ]
        z_vals = [f"{float(v):.6f}" for v in geo.get("z_sects", [])]
        d_vals = [f"{float(v):.6f}" for v in geo.get("main_diams", [])]
        lines.append("z_sects    : " + ", ".join(z_vals))
        lines.append("main_diams : " + ", ".join(d_vals))
        return "\n".join(lines)

    def _render_aedt_params_loaded(self, geo: dict) -> None:
        self.aedt_params_lbl.configure(
            text=(
                "Design params loaded: "
                f"f_start={float(geo['f_start']):.6f} MHz | "
                f"f_stop={float(geo['f_stop']):.6f} MHz | "
                f"f0={float(geo['f0_mhz']):.6f} MHz | "
                f"n_sections={int(geo['n_sections'])} | "
                f"n_outputs={int(geo['n_outputs'])}"
            )
        )

    def _calculate_aedt_geometry(self, params: Optional[dict] = None) -> Optional[dict]:
        input_params = dict(params or self._aedt_input_params())
        try:
            geo = compute_coaxial_divider_geometry(input_params)
        except DividerGeometryError as exc:
            messagebox.showerror("AEDT geometry", str(exc))
            return None
        except Exception as exc:
            messagebox.showerror("AEDT geometry", f"Unexpected geometry error: {exc}")
            return None
        self._last_geometry = dict(geo)
        self._set_geometry_results_text(self._format_aedt_geometry(geo))
        self._render_aedt_params_loaded(geo)
        self.results_tabs.set("AEDT Geometry")
        self._set_status("coaxial geometry computed")
        return geo

    def _create_aedt_model(self) -> None:
        self._run_create_aedt_model(project_path=None)

    def _create_aedt_model_save_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save AEDT project as",
            defaultextension=".aedt",
            filetypes=[("AEDT project", "*.aedt"), ("All files", "*.*")],
            initialfile="Divisor_Coaxial.aedt",
        )
        if not path:
            return
        self._run_create_aedt_model(project_path=path)

    def _run_create_aedt_model(self, project_path: Optional[str]) -> None:
        if self._aedt_thread is not None and self._aedt_thread.is_alive():
            messagebox.showwarning("AEDT", "An AEDT generation is already running.")
            return
        params_snapshot = self._aedt_input_params()
        geo = self._calculate_aedt_geometry(params=params_snapshot)
        if not geo:
            return

        def _worker():
            try:
                out = generate_hfss_divider_model(
                    params_snapshot,
                    status_cb=self._set_status,
                    project_path=project_path,
                )
                if out:
                    self._set_status(f"HFSS project saved: {out}")
                    self.after(0, lambda p=str(out): messagebox.showinfo("AEDT", f"HFSS model created:\n{p}"))
                else:
                    self._set_status("HFSS model generation finished without project path.")
            except Exception as exc:
                self._set_status(f"HFSS generation error: {exc}")
                traceback.print_exc()
                self.after(0, lambda e=str(exc): messagebox.showerror("AEDT", e))

        self._set_status(
            "starting HFSS geometry generation "
            f"(f_start={params_snapshot['f_start']:.6f} MHz, "
            f"f_stop={params_snapshot['f_stop']:.6f} MHz)"
        )
        self._aedt_thread = threading.Thread(target=_worker, daemon=True)
        self._aedt_thread.start()

    def _calculate(self) -> Optional[List[DividerResult]]:
        try:
            total_power = self._parse_float(self.total_power_var.get(), 0.0)
        except Exception:
            messagebox.showerror("Invalid total power", "Total power must be numeric.")
            return None
        if total_power <= 0.0:
            messagebox.showerror("Invalid total power", "Total power must be > 0 W.")
            return None
        if not self.percent_vars:
            messagebox.showerror("No outputs", "Build at least one output.")
            return None

        pcts_raw: List[float] = []
        phases: List[float] = []
        for i, var in enumerate(self.percent_vars):
            try:
                val = self._parse_float(var.get(), 0.0)
            except Exception:
                messagebox.showerror("Invalid percent", f"Output {i + 1} has invalid percent.")
                return None
            if val < 0.0:
                messagebox.showerror("Invalid percent", f"Output {i + 1} percent must be >= 0.")
                return None
            pcts_raw.append(val)
        for i, var in enumerate(self.phase_vars):
            try:
                phases.append(self._parse_float(var.get(), 0.0))
            except Exception:
                messagebox.showerror("Invalid phase", f"Output {i + 1} has invalid phase.")
                return None

        pct_sum = sum(pcts_raw)
        if pct_sum <= 0.0:
            messagebox.showerror("Invalid split", "Percent sum must be > 0.")
            return None

        pcts_norm = [(p / pct_sum) * 100.0 for p in pcts_raw]
        amps = [math.sqrt(max(0.0, p / 100.0)) for p in pcts_norm]
        max_amp = max(amps) if amps else 1.0
        if max_amp <= 0.0:
            max_amp = 1.0

        results: List[DividerResult] = []
        for i, pct in enumerate(pcts_norm):
            frac = pct / 100.0
            pwr = total_power * frac
            amp_total = math.sqrt(max(0.0, frac))
            amp_max = amp_total / max_amp
            rel_db = (10.0 * math.log10(frac)) if frac > 0.0 else -120.0
            results.append(
                DividerResult(
                    output_idx=i + 1,
                    percent_norm=pct,
                    power_w=pwr,
                    amp_rel_total=amp_total,
                    amp_rel_max=amp_max,
                    rel_db=rel_db,
                    phase_deg=phases[i] if i < len(phases) else 0.0,
                )
            )

        self._results = results
        self._render_results(total_power, pct_sum, results)
        self._update_percent_sum()
        return results

    def _render_results(self, total_power: float, pct_sum_raw: float, results: List[DividerResult]) -> None:
        lines = [
            "Divider results",
            f"Total power: {total_power:.6f} W",
            f"Input percent sum (raw): {pct_sum_raw:.6f}%",
            "",
            "OUT | Percent(%) | Power(W)   | Amp(Total) | Amp(Max) | Rel dB   | Phase(deg)",
            "-" * 84,
        ]
        for row in results:
            lines.append(
                f"{row.output_idx:>3d} | "
                f"{row.percent_norm:>9.4f} | "
                f"{row.power_w:>9.4f} | "
                f"{row.amp_rel_total:>10.6f} | "
                f"{row.amp_rel_max:>8.6f} | "
                f"{row.rel_db:>8.3f} | "
                f"{row.phase_deg:>10.3f}"
            )
        self._set_power_results_text("\n".join(lines))
        self.results_tabs.set("Power Results")

    def _export_csv(self) -> None:
        if not self._results:
            computed = self._calculate()
            if not computed:
                return
        path = filedialog.asksaveasfilename(
            title="Export divider CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
            initialfile="divisor_outputs.csv",
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "output",
                    "percent_norm",
                    "power_w",
                    "amp_rel_total",
                    "amp_rel_max",
                    "rel_db",
                    "phase_deg",
                ]
            )
            for row in self._results:
                w.writerow(
                    [
                        row.output_idx,
                        f"{row.percent_norm:.6f}",
                        f"{row.power_w:.6f}",
                        f"{row.amp_rel_total:.8f}",
                        f"{row.amp_rel_max:.8f}",
                        f"{row.rel_db:.4f}",
                        f"{row.phase_deg:.4f}",
                    ]
                )
        messagebox.showinfo("Exported", f"CSV saved:\n{path}")


def register_divisor_tab(app, tabview: "ctk.CTkTabview", tab_name: str = "Divisor"):
    """Register Divisor tab in the main app tabview."""
    tab = tabview.add(tab_name)
    frame = DivisorTab(tab, app=app)
    frame.pack(fill="both", expand=True)
    return frame


class DivisorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("EFTX - Divisor de Potencia")
        self.geometry("980x680")
        self.minsize(860, 560)
        panel = DivisorTab(self, app=None)
        panel.pack(fill="both", expand=True, padx=8, pady=8)


def main() -> int:
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    app = DivisorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
