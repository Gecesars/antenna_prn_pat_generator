from __future__ import annotations

import csv
from io import StringIO
from typing import Iterable

import customtkinter as ctk
from tkinter import ttk


class DerivedTable(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        cols = ("name", "expr", "value", "error")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=8)
        self.tree.pack(fill="both", expand=True)

        self.tree.heading("name", text="Function")
        self.tree.heading("expr", text="Expression")
        self.tree.heading("value", text="Value")
        self.tree.heading("error", text="Error")

        self.tree.column("name", width=120, anchor="w")
        self.tree.column("expr", width=260, anchor="w")
        self.tree.column("value", width=90, anchor="center")
        self.tree.column("error", width=180, anchor="w")

    def set_rows(self, rows: Iterable[dict]):
        self.tree.delete(*self.tree.get_children())
        for r in rows:
            val = r.get("value")
            if isinstance(val, float):
                val_txt = f"{val:.6g}"
            elif val is None:
                val_txt = ""
            else:
                val_txt = str(val)
            self.tree.insert(
                "",
                "end",
                values=(
                    str(r.get("name", "")),
                    str(r.get("expr", "")),
                    val_txt,
                    str(r.get("error", "")),
                ),
            )

    def to_csv_text(self) -> str:
        buf = StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(["Function", "Expression", "Value", "Error"])
        for item in self.tree.get_children():
            w.writerow([str(v) for v in self.tree.item(item, "values")])
        return buf.getvalue()
