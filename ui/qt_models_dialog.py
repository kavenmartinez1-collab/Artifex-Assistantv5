"""Models dialog for the Qt GUI: the model inventory, and "Add with defaults".

Shows every model Artifex can reach (core.model_inventory.scan) with its
status: ready, found in a model folder but not configured, or configured but
broken. For an unconfigured GGUF, "Add with defaults" shows the proposed
config entry (name and context editable, flags and reasons visible) and writes
it only on OK.
"""
import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QHeaderView, QLabel,
    QLineEdit, QMessageBox, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QTextBrowser,
    QVBoxLayout,
)

from core import model_inventory as mi

_STATUS = {"ready": ("ready", "#3fb950"), "unconfigured": ("not configured", "#d29922"),
           "broken": ("broken", "#f85149")}
_ORDER = {"broken": 0, "unconfigured": 1, "ready": 2}
_COLS = ["Status", "Name", "Backend", "Family", "Context", "Size", "Vision", "Notes"]


class ModelsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Models")
        self.resize(980, 520)
        self.changed = False
        self._models = []
        layout = QVBoxLayout(self)
        self._summary = QLabel("Scanning...")
        layout.addWidget(self._summary)
        self._table = QTableWidget(0, len(_COLS), self)
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setSectionResizeMode(len(_COLS) - 1, QHeaderView.ResizeMode.Stretch)
        self._table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self._table)
        row = QHBoxLayout()
        self._scan_btn = QPushButton("Scan")
        self._scan_btn.clicked.connect(self.refresh)
        self._add_btn = QPushButton("Add selected with defaults...")
        self._add_btn.clicked.connect(self._add_selected)
        row.addWidget(self._scan_btn)
        row.addWidget(self._add_btn)
        row.addStretch(1)
        close = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        close.rejected.connect(self.reject)
        row.addWidget(close)
        layout.addLayout(row)
        self.refresh()

    def refresh(self):
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            inv = mi.scan()
        finally:
            QApplication.restoreOverrideCursor()
        c = inv["counts"]
        self._summary.setText(f"{c['ready']} ready  -  {c['unconfigured']} not configured  -  "
                              f"{c['broken']} broken      (folders: {', '.join(inv['dirs'])})")
        self._models = sorted(inv["models"], key=lambda m: _ORDER.get(m["status"], 3))
        self._table.setRowCount(len(self._models))
        for r, m in enumerate(self._models):
            label, color = _STATUS.get(m["status"], (m["status"], "#8b949e"))
            ctx = m.get("num_ctx") or m.get("native_ctx")
            cells = [
                label, m.get("id") or os.path.basename(m.get("path") or ""), m.get("backend", ""),
                m.get("family") or "", (f"{ctx // 1024}k" + ("" if m.get("num_ctx") else " native")) if ctx else "",
                f"{m['size_bytes'] / 1e9:.1f} GB" if m.get("size_bytes") else "",
                "yes" if m.get("vision") else "", "; ".join((m.get("problems") or []) + (m.get("warnings") or [])),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 0:
                    item.setForeground(QColor(color))
                if m.get("path"):
                    item.setToolTip(m["path"])
                self._table.setItem(r, col, item)
        self._table.resizeColumnsToContents()
        self._update_buttons()

    def _selected(self):
        rows = self._table.selectionModel().selectedRows()
        return self._models[rows[0].row()] if rows else None

    def _update_buttons(self):
        m = self._selected()
        self._add_btn.setEnabled(bool(m and m["status"] == "unconfigured" and m.get("readable")))

    def _add_selected(self):
        m = self._selected()
        if not m:
            return
        try:
            prop = mi.propose_entry(m["path"])
        except ValueError as e:
            QMessageBox.warning(self, "Can't add this model", str(e))
            return
        dlg = _ProposalDialog(prop, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        prop["entry"]["num_ctx"] = dlg.ctx.value()
        try:
            backup = mi.add_entry(dlg.name.text().strip(), prop["entry"])
        except ValueError as e:
            QMessageBox.warning(self, "Not added", str(e))
            return
        self.changed = True
        QMessageBox.information(self, "Added", f"Added '{dlg.name.text().strip()}' to llama_cpp_config.json."
                                + (f"\nBackup of the previous file: {backup}" if backup else ""))
        self.refresh()


class _ProposalDialog(QDialog):
    def __init__(self, prop, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add with defaults")
        self.resize(720, 460)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name = QLineEdit(prop["name"])
        self.ctx = QSpinBox()
        self.ctx.setRange(2048, 1048576)
        self.ctx.setSingleStep(1024)
        self.ctx.setValue(int(prop["entry"]["num_ctx"]))
        self.ctx.setToolTip(prop.get("ctx_reason", ""))
        form.addRow("Entry name", self.name)
        form.addRow("Context (num_ctx)", self.ctx)
        layout.addLayout(form)
        info = QTextBrowser(self)
        notes = "".join(f"<li>{_esc(n)}</li>" for n in prop.get("notes", []))
        warns = "".join(f"<li style='color:#d29922'>{_esc(w)}</li>" for w in prop.get("warnings", []))
        info.setHtml(
            f"<p><b>{_esc(prop['family'])}</b> family. Context: {_esc(prop.get('ctx_reason', ''))}</p>"
            f"<ul>{notes}{warns}</ul><p><b>Flags</b></p>"
            f"<pre style='white-space:pre-wrap'>{_esc(' '.join(prop['entry']['extra_flags']))}</pre>"
            f"<p><b>Server</b>: {_esc(prop['entry']['server_path'])}</p>")
        layout.addWidget(info)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                                   parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
