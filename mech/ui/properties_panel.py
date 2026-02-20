from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QTextEdit,
)

from .context_menu import ContextInfo


class PropertiesPanel(QWidget):
    applyTransformRequested = Signal(dict)
    contextRequested = Signal(object)  # ContextInfo

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_object_id: str = ""
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(6)
        root.addWidget(QLabel("Properties"))

        self.lbl_name = QLabel("Name: -")
        self.lbl_id = QLabel("ID: -")
        self.lbl_source = QLabel("Source: -")
        self.lbl_vis = QLabel("Visible: -")
        self.lbl_lock = QLabel("Locked: -")
        for lbl in [self.lbl_name, self.lbl_id, self.lbl_source, self.lbl_vis, self.lbl_lock]:
            lbl.setContextMenuPolicy(Qt.CustomContextMenu)
            lbl.customContextMenuRequested.connect(lambda pos, _l=lbl: self._on_context(pos, _l.objectName()))
            root.addWidget(lbl)

        form = QFormLayout()
        self.ed_tx = QLineEdit("0.0")
        self.ed_ty = QLineEdit("0.0")
        self.ed_tz = QLineEdit("0.0")
        self.ed_rx = QLineEdit("0.0")
        self.ed_ry = QLineEdit("0.0")
        self.ed_rz = QLineEdit("0.0")
        self.ed_scale = QLineEdit("1.0")
        for key, ed in [
            ("tx", self.ed_tx),
            ("ty", self.ed_ty),
            ("tz", self.ed_tz),
            ("rx", self.ed_rx),
            ("ry", self.ed_ry),
            ("rz", self.ed_rz),
            ("scale", self.ed_scale),
        ]:
            ed.setObjectName(key)
            ed.setContextMenuPolicy(Qt.CustomContextMenu)
            ed.customContextMenuRequested.connect(lambda pos, _e=ed: self._on_context(pos, _e.objectName()))
        form.addRow("Tx", self.ed_tx)
        form.addRow("Ty", self.ed_ty)
        form.addRow("Tz", self.ed_tz)
        form.addRow("Rx", self.ed_rx)
        form.addRow("Ry", self.ed_ry)
        form.addRow("Rz", self.ed_rz)
        form.addRow("Scale", self.ed_scale)
        root.addLayout(form)

        row = QHBoxLayout()
        btn_apply = QPushButton("Apply Transform")
        btn_reset = QPushButton("Reset Fields")
        btn_apply.clicked.connect(self._emit_transform)
        btn_reset.clicked.connect(self._reset_fields)
        row.addWidget(btn_apply)
        row.addWidget(btn_reset)
        root.addLayout(row)

        self.txt_metrics = QTextEdit(self)
        self.txt_metrics.setReadOnly(True)
        self.txt_metrics.setMinimumHeight(140)
        self.txt_metrics.setContextMenuPolicy(Qt.CustomContextMenu)
        self.txt_metrics.customContextMenuRequested.connect(lambda pos: self._on_context(pos, "metrics"))
        root.addWidget(self.txt_metrics, 1)

    def _float(self, text: str, default: float = 0.0) -> float:
        try:
            txt = str(text).strip().replace(",", ".")
            if not txt:
                return float(default)
            return float(txt)
        except Exception:
            return float(default)

    def _emit_transform(self):
        payload = {
            "tx": self._float(self.ed_tx.text(), 0.0),
            "ty": self._float(self.ed_ty.text(), 0.0),
            "tz": self._float(self.ed_tz.text(), 0.0),
            "rx_deg": self._float(self.ed_rx.text(), 0.0),
            "ry_deg": self._float(self.ed_ry.text(), 0.0),
            "rz_deg": self._float(self.ed_rz.text(), 0.0),
            "scale": self._float(self.ed_scale.text(), 1.0),
        }
        self.applyTransformRequested.emit(payload)

    def _reset_fields(self):
        self.ed_tx.setText("0.0")
        self.ed_ty.setText("0.0")
        self.ed_tz.setText("0.0")
        self.ed_rx.setText("0.0")
        self.ed_ry.setText("0.0")
        self.ed_rz.setText("0.0")
        self.ed_scale.setText("1.0")

    def _on_context(self, pos: QPoint, field: str):
        sender = self.sender()
        global_pos = None
        if hasattr(sender, "mapToGlobal"):
            global_pos = sender.mapToGlobal(pos)
        ctx = ContextInfo(
            widget="properties",
            mouse_pos=pos,
            global_pos=global_pos,
            picked_object_id=self._current_object_id or None,
            selected_ids=[self._current_object_id] if self._current_object_id else [],
            tool_mode="Select",
            field_name=str(field or ""),
        )
        self.contextRequested.emit(ctx)

    def set_object_info(self, obj_id: str, data: Optional[dict]):
        self._current_object_id = str(obj_id or "")
        if not data:
            self.lbl_name.setText("Name: -")
            self.lbl_id.setText("ID: -")
            self.lbl_source.setText("Source: -")
            self.lbl_vis.setText("Visible: -")
            self.lbl_lock.setText("Locked: -")
            self.txt_metrics.setPlainText("")
            return
        self.lbl_name.setText(f"Name: {data.get('name', '-')}")
        self.lbl_id.setText(f"ID: {obj_id}")
        self.lbl_source.setText(f"Source: {data.get('source', '-')}")
        self.lbl_vis.setText(f"Visible: {'Yes' if data.get('visible', True) else 'No'}")
        self.lbl_lock.setText(f"Locked: {'Yes' if data.get('locked', False) else 'No'}")
        metrics = data.get("metrics", {})
        lines = []
        for k in sorted(metrics.keys()):
            v = metrics[k]
            if isinstance(v, float):
                lines.append(f"{k}: {v:.6g}")
            else:
                lines.append(f"{k}: {v}")
        self.txt_metrics.setPlainText("\n".join(lines))
