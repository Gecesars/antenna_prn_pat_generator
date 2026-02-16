from __future__ import annotations

import csv
from io import StringIO
from typing import Iterable, List, Optional

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk

from core.math_engine import MarkerValue


class MarkerTable(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        cols = ("name", "cut", "ang_deg", "theta_deg", "phi_deg", "mag_lin", "mag_db")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=9)
        self.tree.pack(fill="both", expand=True)

        headings = {
            "name": "Name",
            "cut": "Cut",
            "ang_deg": "Ang(deg)",
            "theta_deg": "Theta(deg)",
            "phi_deg": "Phi(deg)",
            "mag_lin": "Mag(lin)",
            "mag_db": "Mag(dB)",
        }
        for c in cols:
            self.tree.heading(c, text=headings[c])
            w = 90 if c in ("name", "cut") else 95
            self.tree.column(c, width=w, anchor="center", stretch=True)

    def set_markers(self, markers: Iterable[MarkerValue]):
        self.tree.delete(*self.tree.get_children())
        for m in markers:
            self.tree.insert(
                "",
                "end",
                values=(
                    m.name,
                    m.cut or "",
                    f"{m.ang_deg:.3f}",
                    "" if m.theta_deg is None else f"{float(m.theta_deg):.3f}",
                    "" if m.phi_deg is None else f"{float(m.phi_deg):.3f}",
                    f"{m.mag_lin:.6f}",
                    f"{m.mag_db:.3f}",
                ),
            )

    def selected_name(self) -> Optional[str]:
        sel = self.tree.selection()
        if not sel:
            return None
        vals = self.tree.item(sel[0], "values")
        if not vals:
            return None
        return str(vals[0])

    def table_rows(self) -> List[List[str]]:
        out: List[List[str]] = []
        for item in self.tree.get_children():
            vals = self.tree.item(item, "values")
            out.append([str(v) for v in vals])
        return out

    def to_csv_text(self) -> str:
        buf = StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["Name", "Cut", "Ang_deg", "Theta_deg", "Phi_deg", "Mag_lin", "Mag_dB"])
        for row in self.table_rows():
            w.writerow(row)
        return buf.getvalue()

    def copy_to_clipboard(self, root: tk.Misc):
        txt = self.to_csv_text()
        root.clipboard_clear()
        root.clipboard_append(txt)

    def export_csv(self, path: str):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(self.to_csv_text())
