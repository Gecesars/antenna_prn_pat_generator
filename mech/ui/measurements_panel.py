from __future__ import annotations

from typing import List, Sequence

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout

from .context_menu import ContextInfo


class MeasurementsPanel(QWidget):
    contextRequested = Signal(object)  # ContextInfo

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: List[dict] = []
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(QLabel("Measurements / Markers"))
        self.table = QTableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Type", "Name", "Value", "Info", "Object"])
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        layout.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear)
        row.addWidget(self.btn_clear)
        row.addStretch(1)
        layout.addLayout(row)

    def _fmt(self, value) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def set_rows(self, rows: Sequence[dict]):
        self._rows = [dict(r) for r in rows]
        self.table.setRowCount(len(self._rows))
        for i, row in enumerate(self._rows):
            t = str(row.get("type", ""))
            name = str(row.get("name", ""))
            if t == "distance":
                val = self._fmt(row.get("value", ""))
                info = f"p1={row.get('p1')} p2={row.get('p2')}"
            elif t == "angle":
                val = self._fmt(row.get("value_deg", ""))
                info = f"p1={row.get('p1')} p2={row.get('p2')} p3={row.get('p3')}"
            elif t == "object":
                val = self._fmt(row.get("volume", ""))
                info = f"bbox=({self._fmt(row.get('bbox_dx', 0.0))},{self._fmt(row.get('bbox_dy', 0.0))},{self._fmt(row.get('bbox_dz', 0.0))})"
            else:
                val = self._fmt(row.get("value", ""))
                info = self._fmt(row.get("info", ""))
            obj = str(row.get("object_id", ""))
            for j, txt in enumerate([t, name, val, info, obj]):
                item = QTableWidgetItem(str(txt))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.table.setItem(i, j, item)
        self.table.resizeColumnsToContents()

    def clear(self):
        self._rows = []
        self.table.setRowCount(0)

    def selected_row_index(self) -> int:
        row = self.table.currentRow()
        return int(row) if row >= 0 else -1

    def row_data(self, row: int) -> dict:
        if row < 0 or row >= len(self._rows):
            return {}
        return dict(self._rows[row])

    def _on_context_menu(self, pos: QPoint):
        row = self.table.rowAt(pos.y())
        ctx = ContextInfo(
            widget="measurements",
            mouse_pos=pos,
            global_pos=self.table.viewport().mapToGlobal(pos),
            selected_ids=[],
            tool_mode="Measure",
            table_row=int(row) if row >= 0 else None,
        )
        self.contextRequested.emit(ctx)
