from __future__ import annotations

import datetime
import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import customtkinter as ctk


ARTIFACT_EXTENSIONS = {
    ".geojson",
    ".kml",
    ".kmz",
    ".json",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".tif",
    ".tiff",
    ".pdf",
    ".txt",
    ".md",
    ".pat",
    ".prn",
    ".hgt",
    ".dem",
    ".srtm",
    ".shp",
    ".dbf",
    ".shx",
    ".prj",
    ".cpg",
    ".gpkg",
}


def _fmt_size(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes)))
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.2f} {units[idx]}"


def _kind_for_path(path: Path) -> str:
    name = path.name.lower()
    ext = path.suffix.lower()
    if ext in {".geojson", ".kml", ".kmz"}:
        return "Cobertura Vetorial"
    if ext in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
        if ("coverage" in name) or ("cobertura" in name) or ("heatmap" in name) or ("pycraf" in name):
            return "Mapa Cobertura"
        return "Imagem"
    if ext in {".pdf", ".md", ".txt"}:
        return "Relatorio"
    if ext in {".json", ".csv"}:
        return "Tabela/Metadata"
    if ext in {".pat", ".prn"}:
        return "Diagrama RF"
    if ext in {".hgt", ".dem", ".srtm", ".shp", ".dbf", ".shx", ".prj", ".cpg", ".gpkg"}:
        return "Terreno/Cartografia"
    return "Outros"


class CoverageViabilityTab(ctk.CTkFrame):
    def __init__(self, master, app, **kwargs):
        super().__init__(master, **kwargs)
        self.app = app
        default_root = str(getattr(app, "output_dir", "") or os.path.join(os.getcwd(), "out"))
        self.root_dir_var = tk.StringVar(value=default_root)
        self.filter_var = tk.StringVar(value="Todos")
        self.status_var = tk.StringVar(value="Cobertura/Viabilidade pronta.")
        self.items: List[Dict[str, Any]] = []
        self._build_ui()
        self.refresh_index()

    def _build_ui(self) -> None:
        top = ctk.CTkFrame(self)
        top.pack(fill=ctk.X, padx=8, pady=8)

        ctk.CTkLabel(top, text="Pasta de artefatos:", width=130).pack(side=ctk.LEFT, padx=(4, 2))
        ctk.CTkEntry(top, textvariable=self.root_dir_var, width=530).pack(side=ctk.LEFT, padx=2, fill=ctk.X, expand=True)
        ctk.CTkButton(top, text="Browse", width=80, command=self._browse_root_dir).pack(side=ctk.LEFT, padx=4)
        ctk.CTkButton(top, text="Abrir pasta", width=100, command=self._open_root_dir).pack(side=ctk.LEFT, padx=4)
        ctk.CTkButton(top, text="Abrir Modulo Cobertura", width=170, fg_color="#355f9a", command=self._launch_coverage_module).pack(
            side=ctk.LEFT, padx=4
        )

        actions = ctk.CTkFrame(self)
        actions.pack(fill=ctk.X, padx=8, pady=(0, 8))
        ctk.CTkButton(actions, text="Indexar Artefatos", width=140, command=self.refresh_index).pack(side=ctk.LEFT, padx=3)
        ctk.CTkButton(actions, text="Registrar no Projeto", width=145, fg_color="#1f8b4c", command=self._register_selected).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(actions, text="Gerar Relatorio de Indice", width=170, fg_color="#226699", command=self._export_index_report).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(actions, text="Relatorio PDF Projeto", width=150, fg_color="#2277bb", command=self._run_project_pdf_report).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkButton(actions, text="Gerar Artefatos Projeto", width=160, fg_color="#2a8a57", command=self._run_project_artifacts).pack(
            side=ctk.LEFT, padx=3
        )
        ctk.CTkLabel(actions, text="Filtro:", width=45).pack(side=ctk.RIGHT, padx=(8, 2))
        ctk.CTkOptionMenu(
            actions,
            variable=self.filter_var,
            values=[
                "Todos",
                "Mapa Cobertura",
                "Cobertura Vetorial",
                "Terreno/Cartografia",
                "Relatorio",
                "Tabela/Metadata",
                "Diagrama RF",
                "Imagem",
            ],
            width=180,
            command=lambda _v: self._refresh_tree(),
        ).pack(side=ctk.RIGHT, padx=2)

        split = ctk.CTkFrame(self)
        split.pack(fill=ctk.BOTH, expand=True, padx=8, pady=(0, 8))
        split.grid_columnconfigure(0, weight=2)
        split.grid_columnconfigure(1, weight=1)
        split.grid_rowconfigure(0, weight=1)

        tree_box = ctk.CTkFrame(split)
        tree_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=2)
        tree_box.grid_columnconfigure(0, weight=1)
        tree_box.grid_rowconfigure(0, weight=1)

        columns = ("kind", "size", "mtime", "name", "path")
        self.tree = ttk.Treeview(tree_box, columns=columns, show="headings", selectmode="extended")
        self.tree.heading("kind", text="Tipo")
        self.tree.heading("size", text="Tamanho")
        self.tree.heading("mtime", text="Modificado")
        self.tree.heading("name", text="Arquivo")
        self.tree.heading("path", text="Caminho")
        self.tree.column("kind", width=150, anchor="w")
        self.tree.column("size", width=90, anchor="center")
        self.tree.column("mtime", width=140, anchor="center")
        self.tree.column("name", width=220, anchor="w")
        self.tree.column("path", width=420, anchor="w")
        ysb = ttk.Scrollbar(tree_box, orient="vertical", command=self.tree.yview)
        xsb = ttk.Scrollbar(tree_box, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        self.tree.bind("<Double-1>", self._on_tree_open)
        self.tree.bind("<Button-3>", self._on_tree_context)
        self.tree.bind("<Button-2>", self._on_tree_context)

        side = ctk.CTkFrame(split)
        side.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=2)
        ctk.CTkLabel(side, text="Resumo de Viabilidade", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=6, pady=(6, 2))
        self.summary = ctk.CTkTextbox(side, wrap="word")
        self.summary.pack(fill=ctk.BOTH, expand=True, padx=6, pady=(0, 6))
        self.summary.configure(height=320)
        self.summary.configure(state="disabled")

        ctk.CTkLabel(self, textvariable=self.status_var, anchor="w").pack(fill=ctk.X, padx=10, pady=(0, 8))

    def _set_status(self, text: str) -> None:
        msg = str(text or "")
        self.status_var.set(msg)
        try:
            self.app._set_status(msg)
        except Exception:
            pass

    def set_root_dir(self, path: str) -> None:
        p = str(path or "").strip()
        if p:
            self.root_dir_var.set(p)

    def _browse_root_dir(self) -> None:
        d = filedialog.askdirectory(title="Escolha a pasta de artefatos de cobertura/viabilidade")
        if d:
            self.root_dir_var.set(d)
            self.refresh_index()

    def _open_root_dir(self) -> None:
        path = str(self.root_dir_var.get() or "").strip()
        if not path:
            messagebox.showwarning("Pasta", "Defina uma pasta de artefatos.")
            return
        if not os.path.isdir(path):
            messagebox.showwarning("Pasta", f"Pasta inexistente:\n{path}")
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            messagebox.showerror("Abrir pasta", str(e))

    def _launch_coverage_module(self) -> None:
        root = Path(__file__).resolve().parents[2]
        app_main = root / "analise_cobertura" / "Notebook_Cover" / "APP" / "main.py"
        if not app_main.exists():
            messagebox.showwarning("Modulo Cobertura", f"Arquivo nao encontrado:\n{app_main}")
            return
        cmd = [sys.executable, str(app_main)]
        try:
            kwargs: Dict[str, Any] = {"cwd": str(app_main.parent)}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            subprocess.Popen(cmd, **kwargs)
            self._set_status(f"Modulo de cobertura iniciado: {app_main}")
        except Exception as e:
            messagebox.showerror("Modulo Cobertura", str(e))

    def refresh_index(self) -> None:
        root = Path(str(self.root_dir_var.get() or "").strip())
        if not root.exists():
            self.items = []
            self._refresh_tree()
            self._refresh_summary()
            self._set_status("Pasta de artefatos inexistente.")
            return
        if not root.is_dir():
            self.items = []
            self._refresh_tree()
            self._refresh_summary()
            self._set_status("Caminho informado nao e uma pasta.")
            return

        indexed: List[Dict[str, Any]] = []
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in ARTIFACT_EXTENSIONS:
                continue
            try:
                st = path.stat()
                mtime = float(st.st_mtime)
                size = int(st.st_size)
            except Exception:
                mtime = 0.0
                size = 0
            indexed.append(
                {
                    "path": str(path.resolve()),
                    "name": path.name,
                    "kind": _kind_for_path(path),
                    "size": size,
                    "mtime": mtime,
                }
            )
        indexed.sort(key=lambda x: float(x.get("mtime", 0.0)), reverse=True)
        self.items = indexed
        self._refresh_tree()
        self._refresh_summary()
        self._set_status(f"Indice atualizado: {len(indexed)} artefato(s).")

    def _iter_filtered(self) -> List[Dict[str, Any]]:
        selected = str(self.filter_var.get() or "Todos").strip()
        if selected == "Todos":
            return list(self.items)
        return [item for item in self.items if str(item.get("kind", "")) == selected]

    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for idx, item in enumerate(self._iter_filtered(), start=1):
            mtime = "-"
            try:
                mtime = datetime.datetime.fromtimestamp(float(item.get("mtime", 0.0))).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            self.tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(
                    str(item.get("kind", "-")),
                    _fmt_size(int(item.get("size", 0))),
                    mtime,
                    str(item.get("name", "")),
                    str(item.get("path", "")),
                ),
            )

    def _refresh_summary(self) -> None:
        counts: Dict[str, int] = {}
        for item in self.items:
            kind = str(item.get("kind", "Outros"))
            counts[kind] = counts.get(kind, 0) + 1

        total = len(self.items)
        maps = counts.get("Mapa Cobertura", 0)
        vectors = counts.get("Cobertura Vetorial", 0)
        reports = counts.get("Relatorio", 0)
        tables = counts.get("Tabela/Metadata", 0)
        terrain = counts.get("Terreno/Cartografia", 0)
        rf = counts.get("Diagrama RF", 0)

        score = 0
        if maps > 0:
            score += 35
        if vectors > 0:
            score += 30
        if reports > 0:
            score += 20
        if (tables > 0) or (terrain > 0):
            score += 10
        if rf > 0:
            score += 5
        score = max(0, min(100, score))

        lines = [
            "INDICE DE COBERTURA E VIABILIDADE",
            "",
            f"Artefatos totais: {total}",
            f"Indice de viabilidade (artefatos): {score}/100",
            "",
            "Por categoria:",
        ]
        for kind in sorted(counts.keys()):
            lines.append(f"- {kind}: {counts[kind]}")
        if total == 0:
            lines.append("")
            lines.append("Nenhum artefato indexado na pasta atual.")

        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", "\n".join(lines))
        self.summary.configure(state="disabled")

    def _selected_items(self) -> List[Dict[str, Any]]:
        rows = self.tree.selection()
        filtered = self._iter_filtered()
        picked: List[Dict[str, Any]] = []
        for iid in rows:
            try:
                pos = int(iid) - 1
            except Exception:
                continue
            if 0 <= pos < len(filtered):
                picked.append(filtered[pos])
        return picked

    def _on_tree_open(self, _event=None) -> None:
        selected = self._selected_items()
        if not selected:
            return
        self._open_file(selected[0].get("path", ""))

    def _open_file(self, path: str) -> None:
        p = str(path or "").strip()
        if not p or not os.path.exists(p):
            messagebox.showwarning("Arquivo", "Arquivo nao encontrado.")
            return
        try:
            if hasattr(os, "startfile"):
                os.startfile(p)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", p])
        except Exception as e:
            messagebox.showerror("Abrir arquivo", str(e))

    def _copy_selected_path(self) -> None:
        selected = self._selected_items()
        if not selected:
            return
        path = str(selected[0].get("path", ""))
        self.clipboard_clear()
        self.clipboard_append(path)
        self._set_status(f"Caminho copiado: {path}")

    def _register_selected(self) -> None:
        rows = self._selected_items()
        if not rows:
            if not self.items:
                messagebox.showwarning("Registro", "Nao ha artefatos indexados.")
                return
            if not messagebox.askyesno("Registro", "Nenhum item selecionado. Registrar todos os itens indexados?"):
                return
            rows = list(self.items)

        app = self.app
        registry = getattr(app, "export_registry", [])
        existing = set()
        for rec in registry:
            p = str(rec.get("path", ""))
            try:
                existing.add(os.path.normcase(os.path.abspath(p)))
            except Exception:
                existing.add(p)

        added = 0
        now = datetime.datetime.now().isoformat(timespec="seconds")
        for item in rows:
            path = str(item.get("path", ""))
            if not path:
                continue
            try:
                key = os.path.normcase(os.path.abspath(path))
            except Exception:
                key = path
            if key in existing:
                continue
            kind = str(item.get("kind", "ARTIFACT")).upper().replace("/", "_").replace(" ", "_")
            registry.append({"timestamp": now, "kind": f"COVERAGE_{kind}", "path": path})
            existing.add(key)
            added += 1

        setattr(app, "export_registry", registry)
        try:
            app._refresh_project_overview()
        except Exception:
            pass
        self._set_status(f"Registro atualizado: {added} novo(s) item(ns) no projeto.")

    def _export_index_report(self) -> None:
        if not self.items:
            messagebox.showwarning("Relatorio", "Nenhum artefato indexado.")
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="Salvar relatorio de indice de cobertura/viabilidade",
            defaultextension=".md",
            initialfile=f"coverage_viability_index_{stamp}.md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        lines = [
            "# Relatorio de Indice - Cobertura e Viabilidade",
            "",
            f"- Gerado em: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"- Pasta indexada: `{self.root_dir_var.get()}`",
            f"- Total de artefatos: **{len(self.items)}**",
            "",
            "| # | Tipo | Tamanho | Modificado | Arquivo | Caminho |",
            "|---:|---|---:|---|---|---|",
        ]
        for idx, item in enumerate(self.items, start=1):
            mtime = "-"
            try:
                mtime = datetime.datetime.fromtimestamp(float(item.get("mtime", 0.0))).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
            lines.append(
                f"| {idx} | {item.get('kind','-')} | {_fmt_size(int(item.get('size',0)))} | {mtime} | "
                f"{item.get('name','-')} | {item.get('path','-')} |"
            )
        try:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines))
            try:
                self.app._register_export(path, "COVERAGE_VIABILITY_INDEX")
            except Exception:
                pass
            self._set_status(f"Relatorio de indice salvo: {path}")
        except Exception as e:
            messagebox.showerror("Relatorio", str(e))

    def _run_project_pdf_report(self) -> None:
        try:
            self.app.open_report_pdf_export()
        except Exception as e:
            messagebox.showerror("Relatorio PDF", str(e))

    def _run_project_artifacts(self) -> None:
        try:
            self.app.generate_all_project_artifacts()
        except Exception as e:
            messagebox.showerror("Artefatos", str(e))

    def _remove_from_index(self) -> None:
        rows = self._selected_items()
        if not rows:
            return
        keys = {str(x.get("path", "")) for x in rows}
        self.items = [it for it in self.items if str(it.get("path", "")) not in keys]
        self._refresh_tree()
        self._refresh_summary()
        self._set_status(f"Removidos {len(keys)} item(ns) do indice.")

    def _delete_files(self) -> None:
        rows = self._selected_items()
        if not rows:
            return
        if not messagebox.askyesno("Excluir arquivos", f"Excluir {len(rows)} arquivo(s) fisicamente?"):
            return
        deleted = 0
        for item in rows:
            p = str(item.get("path", ""))
            try:
                if os.path.exists(p):
                    os.remove(p)
                    deleted += 1
            except Exception:
                continue
        if deleted:
            self.refresh_index()
        self._set_status(f"Arquivos removidos: {deleted}.")

    def _on_tree_context(self, event) -> str:
        row = self.tree.identify_row(event.y)
        if row:
            self.tree.selection_set(row)
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Abrir", command=self._on_tree_open)
        menu.add_command(label="Copiar caminho", command=self._copy_selected_path)
        menu.add_separator()
        menu.add_command(label="Registrar no projeto", command=self._register_selected)
        menu.add_separator()
        menu.add_command(label="Remover do indice", command=self._remove_from_index)
        menu.add_command(label="Excluir arquivo fisico", command=self._delete_files)
        try:
            menu.tk_popup(int(event.x_root), int(event.y_root))
            menu.grab_release()
        except Exception:
            pass
        return "break"

