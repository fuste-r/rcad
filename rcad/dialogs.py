"""Dialogs for features and sketch expressions."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from .units import MM
from .value_box import NumberField


class SweepDialog(QDialog):
    def __init__(self, parent, paths, path_ref=None, cut=False):
        super().__init__(parent)
        self.setWindowTitle("Sweep")
        self.path = QComboBox()
        for index, name in paths:
            self.path.addItem(name, index)
        current = self.path.findData(path_ref)
        if current >= 0:
            self.path.setCurrentIndex(current)
        self.mode = QComboBox()
        self.mode.addItems(["Add (join)", "Cut (subtract)"])
        self.mode.setCurrentIndex(1 if cut else 0)
        form = QFormLayout(self)
        form.addRow("Path sketch", self.path)
        form.addRow("Operation", self.mode)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    @property
    def path_ref(self) -> int:
        return int(self.path.currentData())

    @property
    def cut(self) -> bool:
        return self.mode.currentIndex() == 1


class LoftDialog(QDialog):
    def __init__(self, parent, sections, section_refs=None, cut=False):
        super().__init__(parent)
        self.setWindowTitle("Loft")
        self.sections = QListWidget()
        self.sections.setSelectionMode(QAbstractItemView.MultiSelection)
        selected = set(section_refs or [])
        for index, name in sections:
            item = QListWidgetItem(name)
            item.setData(Qt.UserRole, index)
            self.sections.addItem(item)
            item.setSelected(index in selected)
        self.mode = QComboBox()
        self.mode.addItems(["Add (join)", "Cut (subtract)"])
        self.mode.setCurrentIndex(1 if cut else 0)
        layout = QVBoxLayout(self)
        layout.addWidget(self.sections)
        form = QFormLayout()
        form.addRow("Operation", self.mode)
        layout.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    @property
    def section_refs(self) -> list[int]:
        return [int(item.data(Qt.UserRole)) for item in self.sections.selectedItems()]

    @property
    def cut(self) -> bool:
        return self.mode.currentIndex() == 1


class ExpressionDialog(QDialog):
    def __init__(self, parent, text: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Expression")
        self.text = QTextEdit()
        self.text.setPlainText(text)
        layout = QVBoxLayout(self)
        layout.addWidget(self.text)
        box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def expression(self) -> str:
        return self.text.toPlainText().strip()


class VariablesDialog(QDialog):
    def __init__(self, parent, variables: dict[str, float] | None = None):
        super().__init__(parent)
        self._vars = dict(variables or {})
        self.setWindowTitle("Expressions and Variables")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        # mm like every variable; units, maths and earlier variables work (e.g. `2in`, `wall * 2`)
        self.value_edit = NumberField(None, "length", MM, self._vars, allow_empty=True)
        add_btn = QPushButton("Add")
        remove_btn = QPushButton("Remove")
        form.addRow("Variable", self.name_edit)
        form.addRow("Value", self.value_edit)
        form.addRow(add_btn, remove_btn)
        layout.addLayout(form)
        self.list = QListWidget()
        layout.addWidget(self.list)
        add_btn.clicked.connect(self._add)
        remove_btn.clicked.connect(self._remove)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self.refresh()
        self.name_edit.setFocus()

    def refresh(self):
        self.list.clear()
        for name in self._vars:
            self.list.addItem(name)
        self.name_edit.clear()
        self.value_edit.clear()

    def _add(self):
        name = self.name_edit.text().strip()
        value = self.value_edit.value()
        if not name.isidentifier() or value is None:
            return
        new = name not in self._vars
        self._vars[name] = value
        if new:
            self.list.addItem(name)
        self.name_edit.clear()
        self.value_edit.clear()

    def _remove(self):
        item = self.list.currentItem()
        if item is None:
            return
        name = item.text()
        del self._vars[name]
        self.list.takeItem(self.list.row(item))
        self.refresh()

    def variables(self) -> dict[str, float]:
        return dict(self._vars)
