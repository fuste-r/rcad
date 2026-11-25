from __future__ import annotations

import math
import sys

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QFileDialog,
    QFormLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
    QPushButton, QTextEdit,
    QToolBar, QWidget,
    QVBoxLayout,
)

from .kernel import Document, Feature
from .camera import ray_plane
from .sketch import (
    CONSTRAINT_KINDS,
    Constraint,
    Sketch,
    SketchPlane,
    standard_plane,
    reflect_point_across_line,
)
from .units import MM, UNITS
from .theme import CHOICES, apply_choice
from .tool_bar import Choice, Number, ToolSettingsBar
from .quick_buttons import QuickButtons
from .tool_line import ToolLine
from .value_box import FieldSet, NumberField
from .viewport import Viewport

FILE_FILTER = "rCad project (*.rcad)"


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.doc = Document()
        self.path: str | None = None
        self.dirty = False
        self.mode = "model"  # or "sketch"
        self.sketch_index = -1
        self.tool = "select"
        self.pending = None  # first click of a two-click tool
        self.hover_region = None  # (sketch feature index, region index) under the cursor
        self.region_sketch = -1  # sketch whose regions were highlighted last
        # Sketch-drag state, driven by Viewport.sketch_mouse (left-button drag in sketch mode)
        self._sketch_press_uv = None    # (u, v) cursor position at the drag press
        self._sketch_press_hit = None   # (kind, index) of the geometry under the cursor at press
        self._sketch_drag_orig = None  # {point_index: [u, v]} restored if the post-drag solve fails
        self._sketch_drag_active = False
        self._sketch_drag_kind = None   # (kind, index) of the element currently being dragged
        self._sketch_drag_idx = None

        self.view = Viewport()
        self.setCentralWidget(self.view)
        self.view.clicked.connect(self.on_click)
        self.view.hovered.connect(self.on_hover)
        self.view.scrolled.connect(self.on_wheel_adjust)
        self.view.extrude_commit.connect(self.on_extrude_commit)
        self.view.fillet_commit.connect(self.on_fillet_commit)
        self.view.fillet_preview.connect(self.on_fillet_preview)
        self.view.fillet_cancel.connect(self.on_fillet_cancel)
        self.view.value_edit.connect(self.on_value_edit)
        self.view.variables_source = lambda: self.doc.variables
        self.view.value_scrub.connect(self.on_value_scrub)
        self._value_feature = None  # feature made by the last drag; its value box can retype it
        self._scrub_prev = None  # (feature, value before the scrub) while its value box is scrubbed
        # Sizes typed while drawing in a sketch (rectangle W/H, line length/angle, circle Ø)
        self.fields = FieldSet(self.view, lambda: self.unit, lambda: self.doc.variables)
        self.view.fields = self.fields
        self.fields.changed.connect(self._on_fields_changed)
        self.fields.accepted.connect(self._on_fields_accepted)
        self.fields.cancelled.connect(self.cancel)
        self._draw_mouse = None  # (snapped point index, (u, v), x, y) of the last drawing hover
        self._drawn = None  # shape just drawn whose size boxes stay on it (see _show_drawn)
        # Settings bar for the active feature tool; the feature previews live while it is open
        self.tool_bar = ToolSettingsBar(self.view, lambda: self.unit, lambda: self.doc.variables)
        self._session = None  # {"feature", "new", "snapshot", "apply", "done"} while the bar edits a feature
        self._last_extrude = 10.0  # extrude starts at the last distance used
        self._last_segment = None  # (start, end) points of the last line drawn in the current chain
        self.view.sketch_mouse.connect(self.on_sketch_mouse)

        self.features = QListWidget()
        self.features.itemDoubleClicked.connect(self.edit_feature)
        self.features.itemChanged.connect(self.toggle_layer)
        dock = QDockWidget("Layers", self)
        dock.setWidget(self.features)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        self.constraints = QListWidget()
        self.constraints.itemDoubleClicked.connect(self.edit_constraint)
        self.sketch_status = QLabel()
        holder = QWidget()
        lay = QVBoxLayout(holder)
        lay.addWidget(self.sketch_status)
        lay.addWidget(self.constraints)
        self.constraint_dock = QDockWidget("Constraints", self)
        self.constraint_dock.setWidget(holder)
        self.addDockWidget(Qt.RightDockWidgetArea, self.constraint_dock)

        self._build_actions()
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        self._restoring = False
        self._mirror_axis = None   # (a_idx, b_idx, line_idx) for symmetry tool
        self._mirroring = False
        self._sketch_undo: dict[int, list[dict]] = {}
        self._sketch_redo: dict[int, list[dict]] = {}
        self._sketch_restoring = False
        self._fillet_live_idx = None
        self._fillet_preview_r = None
        self._fillet_is_mod = False  # current drag edits an existing fillet
        self._fillet_mod_prev = None  # fillet radius before the modify drag
        self._fillet_mod_prev_kind = None  # its kind before the drag (a pull turns it into a chamfer)
        self._fillet_preview_kind = None  # kind at the last live rebuild
        QShortcut(QKeySequence(Qt.Key_Escape), self, self.cancel)
        self.resize(1300, 800)
        self.set_mode("model")
        self._apply_units()
        saved = str(QSettings("rcad", "rcad").value("theme", "system"))
        self.set_theme(saved if saved in CHOICES else "system", save=False)

    # ---- UI construction
    def _act(self, text, slot, tip="", shortcut=None, checkable=False):
        a = QAction(text, self)
        a.setToolTip(tip or text)
        a.setCheckable(checkable)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        return a

    def _build_actions(self):
        self._build_file_toolbar()
        self._build_tool_line()

    def _build_file_toolbar(self):
        file_tb = self.addToolBar("File")
        for text, slot, sc in [("📄", self.file_new, QKeySequence.New), ("📂", self.file_open, QKeySequence.Open),
                               ("💾", self.file_save, QKeySequence.Save),
                               ("📋", self.file_save_as, QKeySequence.SaveAs),
                               ("📤", self.file_export, None)]:
            a = self._act(text, slot, shortcut=sc)
            file_tb.addAction(a)
            self.addAction(a)
        file_tb.addSeparator()
        self.variables_tb = QToolBar("Variables")
        self.variables_tb.addSeparator()
        v_btn = self._act("Variables", self._edit_variables, "Edit parametric variables and expressions", shortcut="Ctrl+Shift+V")
        self.variables_tb.addAction(v_btn)
        self.variables_tb.addSeparator()
        file_tb.addWidget(self.variables_tb)
        file_tb.addWidget(QLabel(" Units: "))
        undo_a = self._act("Undo (Ctrl+Z)", self.undo, shortcut=QKeySequence.Undo)
        redo_a = self._act("Redo (Ctrl+Shift+Z)", self.redo, shortcut=QKeySequence.Redo)
        file_tb.addAction(undo_a)
        file_tb.addAction(redo_a)
        file_tb.addSeparator()
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(list(UNITS))
        self.unit_combo.setToolTip("Display and input units (geometry is stored in mm)")
        self.unit_combo.currentTextChanged.connect(self.set_units)
        file_tb.addWidget(self.unit_combo)
        file_tb.addWidget(QLabel(" Theme: "))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems([n.title() for n in CHOICES])
        self.theme_combo.currentTextChanged.connect(lambda t: self.set_theme(t.lower()))
        file_tb.addWidget(self.theme_combo)

    def _build_tool_line(self):
        """Every modelling tool in one grouped line along the bottom of the view."""
        line = self.tool_line = ToolLine(self.view)
        act = self._act

        # model page
        line.add_page("model")
        # The plane choice behind the Sketch list (also set directly by tests and scripts)
        self.plane_combo = QComboBox(self)
        self.plane_combo.addItems(["XY", "XZ", "YZ", "Selected face"])
        self.plane_combo.hide()

        def sketch_on(plane):
            return lambda _=False: (self.plane_combo.setCurrentText(plane), self.new_sketch())
        line.add_group("model", "Sketch", [
            act("✏️ On XY (floor)", sketch_on("XY"), "Start a sketch on the XY plane"),
            act("✏️ On XZ (front)", sketch_on("XZ"), "Start a sketch on the XZ plane"),
            act("✏️ On YZ (side)", sketch_on("YZ"), "Start a sketch on the YZ plane"),
            act("✏️ On selected face", sketch_on("Selected face"), "Start a sketch on the selected planar face"),
        ])
        line.add_group("model", "Create", [
            act("⇈ Extrude", self.extrude, "Extrude the selected (or last) sketch"),
            act("⟳ Revolve", self.revolve, "Revolve the selected (or last) sketch"),
            act("↝ Sweep", self.sweep, "Sweep a sketch profile along a sketch path"),
            act("⌁ Loft", self.loft, "Loft between sketch sections"),
        ])
        line.add_group("model", "Modify", [
            act("↕️ Push/Pull", self.pushpull, "Move the selected planar face along its normal"),
            act("◐ Fillet", self.fillet, "Round the selected edges"),
            act("◺ Chamfer", self.chamfer, "Bevel the selected edges"),
            None,
            act("🗑 Delete feature", self.delete_feature, "Delete the highlighted feature"),
        ])
        line.add_separator("model")
        line.add_action("model", act("⛶ Fit", self.view.frame_all, "Fit the model in view"))

        # sketch page
        line.add_page("sketch")
        group = QActionGroup(self)
        self.tool_actions = {}
        draw = []
        for name, text in [("select", "⌖ Select"), ("line", "╱ Line"), ("rect", "⬜ Rectangle"),
                           ("circle", "◯ Circle")]:
            a = act(text, lambda _=False, n=name: self.set_tool(n), checkable=True)
            group.addAction(a)
            a.setChecked(name == "select")
            self.tool_actions[name] = a
            draw.append(a)
        self.draw_button = line.add_group("sketch", "Draw", draw)
        tips = {"Symmetry": "Two-step: 1) click mirror line, 2) click points/lines to mirror, press Symmetry again."}
        constraints = []
        for item in [
                ("Horizontal", "horizontal"), ("Vertical", "vertical"), ("Coincident", "coincident"),
                ("Parallel", "parallel"), ("Perpendicular", "perpendicular"), ("Equal", "equal"), None,
                ("Distance", "distance"), ("Spacing", "spacing"), ("Diameter", "diameter"),
                ("Expression", "expression"), None,
                ("Symmetry", "symmetry"), ("Fixed", "fixed"), ("Point on line", "point_on_line")]:
            if item is None:
                constraints.append(None)
                continue
            text, kind = item
            constraints.append(act(text, lambda _=False, k=kind: self.add_constraint(k),
                                   tips.get(text, f"Add a {kind} constraint to the selection")))
        line.add_group("sketch", "Constrain", constraints)
        line.add_group("sketch", "Modify", [
            act("🗑 Delete selection", self.delete_selection, "Delete selected sketch items"),
            act("🗑 Delete constraint", self.delete_constraint, "Delete the constraint highlighted in the list"),
        ])
        line.add_separator("sketch")
        self.snap = NumberField(1.0, "length", MM, lo=0.0, hi=1e4)
        self.snap.setFixedWidth(80)
        self.snap.HINT = "Grid snap (0 = off). " + NumberField.HINT
        self.snap.setToolTip(self.snap.HINT)
        snap_label = QLabel(" Snap ")
        line.add_widget("sketch", snap_label)
        line.add_widget("sketch", self.snap)
        line.add_separator("sketch")
        line.add_action("sketch", act("⇈ Extrude", self.extrude_from_sketch,
                                      "Extrude the highlighted regions (Select tool: click a region to highlight it)"))
        line.add_action("sketch", act("✓ Exit sketch", self.finish_sketch, "Finish the sketch"))

        # a tool's settings bar takes the line's place while it is open
        self.tool_bar.active_changed.connect(line.set_suppressed)

        # quick buttons beside the selection
        self.quick = QuickButtons(self.view)
        self._region_anchor = None  # world centre of the highlighted region (set in _update_regions)
        self.view.paint_hooks.append(self._place_quick)
        self.tool_bar.active_changed.connect(lambda _active: self._refresh_quick())

    # ---- units
    @property
    def unit(self):
        return UNITS[self.doc.units]

    def _apply_units(self):
        u = self.unit
        self.unit_combo.blockSignals(True)
        self.unit_combo.setCurrentText(u.name)
        self.unit_combo.blockSignals(False)
        self.snap.unit = u
        self.snap.set_value(u.snap * u.mm)
        self.view.unit_mm = u.mm
        self.view.unit = u
        self.refresh()

    def set_units(self, name: str):
        if name in UNITS and name != self.doc.units:
            self.doc.units = name
            self.mark_dirty()
            self._apply_units()

    def set_theme(self, name: str, save: bool = True):
        self.view.theme = apply_choice(name)
        self.theme_combo.blockSignals(True)
        self.theme_combo.setCurrentText(name.title())
        self.theme_combo.blockSignals(False)
        if save:
            QSettings("rcad", "rcad").setValue("theme", name)
        self.view.update()

    # ---- variables / expressions
    def _edit_variables(self):
        dialog = VariablesDialog(self, self.doc.variables)
        if not dialog.exec():
            return
        variables = dialog.variables()
        solved_sketches = []
        try:
            for feature in self.doc.features:
                if feature.kind != "sketch" or feature.sketch is None:
                    continue
                sketch = Sketch.from_dict(feature.sketch.to_dict())
                sketch.variables = variables
                if not sketch.solve():
                    raise ValueError(sketch.status)
                solved_sketches.append((feature.sketch, sketch))
        except Exception as error:
            QMessageBox.warning(self, "Invalid variables", str(error))
            return

        self._push_undo()
        self.doc.variables = variables
        for sketch, solved in solved_sketches:
            sketch.variables = variables
            sketch.points = [list(point) for point in solved.points]
            sketch.circles = list(solved.circles)
            sketch.status = solved.status
            sketch.dof = solved.dof
            sketch.version += 1
        self.rebuild()

    # ---- state / refresh
    def set_mode(self, mode):
        self._settle_session()
        self.fields.stop()
        self.mode = mode
        sketching = mode == "sketch"
        self.tool_line.set_page("sketch" if sketching else "model")
        self.constraint_dock.setVisible(sketching)
        self.view.left_orbits = not sketching
        self.view.camera.ortho = sketching
        self.view.sketch = self.doc.features[self.sketch_index].sketch if sketching else None
        self.view.sketch_sel.clear()
        self.view.preview = []
        self.pending = None
        self.hover_region = None
        self._update_regions()

    def mark_dirty(self, dirty=True):
        self.dirty = dirty
        name = self.path or "untitled"
        self.setWindowTitle(f"rCad - {name}{' *' if dirty else ''}")

    def refresh(self):
        row = self.features.currentRow()
        self.features.blockSignals(True)
        self.features.clear()
        for i, f in enumerate(self.doc.features):
            text = f"{i}: {f.label(self.unit)}"
            if i in self.doc.errors:
                text += f"   !! {self.doc.errors[i]}"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if f.visible else Qt.Unchecked)
            item.setToolTip("Sketch layers show/hide their curves; other layers are suppressed when unchecked")
            self.features.addItem(item)
        self.features.blockSignals(False)
        if 0 <= row < self.features.count():
            self.features.setCurrentRow(row)
        self.view.overlay_sketches = [f.sketch for f in self.doc.features if f.kind == "sketch" and f.visible]
        self.view.set_mesh(self.doc.mesh)
        self.refresh_constraints()
        self.hover_region = None  # the geometry under the cursor may have changed
        self._update_regions()

    def refresh_constraints(self):
        self.constraints.clear()
        if self.mode != "sketch":
            return
        sk = self.sketch
        for i, c in enumerate(sk.constraints):
            self.constraints.addItem(f"{i}: {c.label(self.unit)}")
        self.sketch_status.setText(f"{sk.status}  |  {len(sk.points)} points, {len(sk.lines)} lines, "
                                   f"{len(sk.circles)} circles")

    @property
    def sketch(self) -> Sketch:
        return self.doc.features[self.sketch_index].sketch

    def _push_undo(self):
        if self._restoring:
            return
        self._settle_session()  # another change: keep (or drop, if failing) the open tool's feature
        self._undo.append(self.doc.to_dict())
        self._redo.clear()

    def _push_sketch_undo(self):
        self._settle_session()
        if self._sketch_restoring or self.mode != "sketch":
            return
        self._sketch_undo.setdefault(self.sketch_index, []).append(self.sketch.to_dict())
        self._sketch_redo.pop(self.sketch_index, None)

    def _restore_sketch(self, d, index):
        sk = self.doc.features[index].sketch
        plane = d["plane"]
        sk.plane = SketchPlane(tuple(plane["origin"]), tuple(plane["xdir"]), tuple(plane["normal"]))
        sk.points = [list(p) for p in d["points"]]
        sk.lines = [tuple(l) for l in d["lines"]]
        sk.circles = [tuple(c) for c in d["circles"]]
        sk.constraints = [Constraint(**c) for c in d["constraints"]]
        sk.selection = [tuple(p) for p in d.get("selection", [])]
        sk.solve()
        self.pending = None
        self.view.preview = []
        self.view.sketch_sel.clear()
        self.hover_region = None
        self._update_regions()
        self.refresh()

    def undo_sketch(self):
        self._settle_session()
        self.fields.stop()  # the shape the boxes were on may be gone
        history = self._sketch_undo.get(self.sketch_index, [])
        if not history:
            return False
        self._sketch_restoring = True
        self._sketch_redo.setdefault(self.sketch_index, []).append(self.sketch.to_dict())
        self._restore_sketch(history.pop(), self.sketch_index)
        self._sketch_restoring = False
        return True

    def redo_sketch(self):
        self._settle_session()
        self.fields.stop()
        history = self._sketch_redo.get(self.sketch_index, [])
        if not history:
            return False
        self._sketch_restoring = True
        self._sketch_undo.setdefault(self.sketch_index, []).append(self.sketch.to_dict())
        self._restore_sketch(history.pop(), self.sketch_index)
        self._sketch_restoring = False
        return True

    def _clear_sketch_history(self):
        self._sketch_undo.clear()
        self._sketch_redo.clear()

    def undo(self):
        self._settle_session()
        if self.mode == "sketch" and self.undo_sketch():
            return
        if not self._undo:
            return
        self._restoring = True
        self._redo.append(self.doc.to_dict())
        d = self._undo.pop()
        self.doc = Document([Feature.from_dict(f) for f in d["features"]], units=d.get("units", "mm"))
        self.doc.variables = d.get("variables", {})
        self._clear_sketch_history()
        self.sketch_index = -1
        self.set_mode("model")
        self.doc.rebuild()
        self.mark_dirty()
        self._apply_units()
        self.refresh()
        self._restoring = False

    def redo(self):
        self._settle_session()
        if self.mode == "sketch" and self.redo_sketch():
            return
        if not self._redo:
            return
        self._restoring = True
        self._undo.append(self.doc.to_dict())
        d = self._redo.pop()
        self.doc = Document([Feature.from_dict(f) for f in d["features"]], units=d.get("units", "mm"))
        self.doc.variables = d.get("variables", {})
        self._clear_sketch_history()
        self.sketch_index = -1
        self.set_mode("model")
        self.doc.rebuild()
        self.mark_dirty()
        self._apply_units()
        self.refresh()
        self._restoring = False

    def rebuild(self):
        self.doc.rebuild()
        self.mark_dirty()
        self.refresh()
        self._refresh_fillet_result_edges()
        if self.doc.errors:
            i, msg = next(iter(self.doc.errors.items()))
            self.statusBar().showMessage(f"Feature {i} failed: {msg}", 8000)

    def commit(self, feature: Feature) -> bool:
        """Append a feature; drop it again if the kernel rejects it."""
        self._push_undo()
        had_errors = set(self.doc.errors)
        self.doc.features.append(feature)
        self.rebuild()
        idx = len(self.doc.features) - 1
        if idx in self.doc.errors and idx not in had_errors:
            msg = self.doc.errors[idx]
            self.doc.features.pop()
            self.rebuild()
            QMessageBox.warning(self, "Operation failed", msg)
            return False
        return True

    # ---- sketch mode
    def new_sketch(self):
        choice = self.plane_combo.currentText()
        if choice == "Selected face":
            if len(self.view.sel_faces) != 1:
                QMessageBox.information(self, "Sketch on face", "Select exactly one planar face first.")
                return
            info = self.doc.mesh.faces[next(iter(self.view.sel_faces))]
            if info.normal is None:
                QMessageBox.information(self, "Sketch on face", "The selected face is not planar.")
                return
            n = info.normal
            helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
            x = np.cross(helper, n)
            x /= np.linalg.norm(x)
            plane = SketchPlane(tuple(map(float, info.center)), tuple(map(float, x)), tuple(map(float, n)))
        else:
            plane = standard_plane(choice)
        self._push_undo()
        n_sketch = sum(1 for f in self.doc.features if f.kind == "sketch") + 1
        self.doc.features.append(Feature("sketch", f"Sketch {n_sketch}", sketch=Sketch(plane)))
        self.enter_sketch(len(self.doc.features) - 1)

    def enter_sketch(self, index):
        self._push_undo()
        self.sketch_index = index
        self.set_mode("sketch")
        pl = self.sketch.plane
        self.view.camera.look_at_plane(pl.origin, pl.normal, pl.xdir, max(self.view.camera.dist, 60))
        self.sketch.solve()
        self.refresh()

    def finish_sketch(self):
        self.set_mode("model")
        self.doc.rebuild()
        self.mark_dirty()
        self.refresh()

    def set_tool(self, name):
        self.tool_actions[name].setChecked(True)
        self.draw_button.setText(f"Draw: {self.tool_actions[name].text().split(' ', 1)[1]} ▾")
        self.tool = name
        self.pending = None
        self.fields.stop()
        self.hover_region = None
        self.view.preview = []
        self.view.sketch_sel.clear()
        self._update_regions()

    def cancel(self):
        if self._session is not None:  # Esc: the open tool is cancelled and the feature put back
            self._cancel_session()
            return
        if self.mode == "sketch" and self.pending is None and self.tool != "select":
            self.set_tool("select")  # Esc with nothing in progress leaves the drawing tool
            return
        ending_chain = self.mode == "sketch" and self.tool == "line" and self.pending is not None
        self.pending = None
        self.fields.stop()
        self.view.preview = []
        self.view.sketch_sel.clear()
        self.view.sel_faces.clear()
        self.view.sel_edges.clear()
        for f in self.doc.features:
            if f.kind == "sketch" and (self.mode == "model" or f is self.doc.features[self.sketch_index]):
                f.sketch.selection = []
        self._mirror_axis = None
        self._mirroring = False
        # A held drag is reverted to the pre-drag positions before re-solving
        if self._sketch_drag_active and self._sketch_drag_orig is not None:
            for pi, (u, v) in self._sketch_drag_orig.items():
                self.sketch.points[pi] = [u, v]
            self._sketch_drag_active = False
            self._sketch_press_uv = None
            self._sketch_press_hit = None
            self._sketch_drag_orig = None
            self._solve_sketch()
        if ending_chain:  # Esc ends a line chain: keep the last segment's sizes on it
            self._show_last_segment()
        self._update_regions()

    # -- sketch drag
    def on_sketch_mouse(self, phase, x, y):
        """Left-button drag on sketch geometry (points, lines, circles).

        phase is "press" (track the geometry under the cursor), "move" (apply the
        cursor delta directly so the view updates live), or "release" (solve and
        either keep the move or revert it if the solve fails)."""
        if phase == "press":
            if self.tool == "select" and self.sketch is not None:
                self._sketch_press_uv = self.view.sketch_uv(x, y)
                self._sketch_press_hit = self.view.pick_sketch(x, y)
            else:
                self._sketch_press_uv = None
                self._sketch_press_hit = None
            self._sketch_drag_active = False
            self._sketch_drag_orig = None
        elif phase == "move":
            if not self._sketch_drag_active:
                if self._sketch_press_hit is None or self._sketch_press_uv is None:
                    return
                kind, idx = self._sketch_press_hit
                self._sketch_drag_active = True
                self._sketch_drag_kind = kind
                self._sketch_drag_idx = idx
                self._sketch_drag_orig = self._collect_drag_points(kind, idx)
                self._push_sketch_undo()  # snapshot so a failed solve can revert
            cur_uv = self.view.sketch_uv(x, y)
            if cur_uv is None or self._sketch_press_uv is None:
                return
            du = cur_uv[0] - self._sketch_press_uv[0]
            dv = cur_uv[1] - self._sketch_press_uv[1]
            du, dv = self._clamp_drag_delta(self._sketch_drag_kind, self._sketch_drag_idx, du, dv)
            sk = self.sketch
            orig = self._sketch_drag_orig
            if orig is not None:
                for pi, (u, v) in orig.items():
                    sk.points[pi] = [u + du, v + dv]
            # Show the silhouette only over the range this element is allowed to
            # sweep to (a fixed-angle line moves only along its legal direction).
            self._update_drag_silhouette(sk, du, dv)
            if self._sketch_press_hit is not None:
                self.view.sketch_sel = {self._sketch_press_hit}  # highlight while dragging
            self.view.update()
        elif phase == "release":
            if self._sketch_drag_active:
                self._finish_sketch_drag()
            self._sketch_drag_active = False
            self._sketch_press_uv = None
            self._sketch_press_hit = None
            self._sketch_drag_orig = None
            self._sketch_drag_kind = None
            self._sketch_drag_idx = None
            self.view.preview = []

    def _line_angle_fixed(self, idx):
        """Does the dragged line have a constraint fixing its angle?

        A horizontal/vertical line, or a line constrained parallel/perpendicular
        to another, can only translate perpendicular to its own direction.
        """
        for c in self.sketch.constraints:
            if c.kind in ("horizontal", "vertical") and c.refs == [idx]:
                return True
            if c.kind in ("parallel", "perpendicular") and idx in c.refs:
                return True
        return False

    def _clamp_drag_delta(self, kind, idx, du, dv):
        """Project (du, dv) onto the subspace the constraints allow.

        For a line with a fixed angle the legal motion is the direction
        perpendicular to the line; the perpendicular component of the cursor
        delta is dropped so the line never rotates off its angle. Returns the
        input unchanged when no restriction applies.
        """
        if kind != "l" or not self._line_angle_fixed(idx):
            return du, dv
        sk = self.sketch
        a, b = sk.lines[idx]
        pa = sk.points[a]
        pb = sk.points[b]
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return du, dv
        dx, dy = dx / L, dy / L          # unit direction of the line
        px, py = -dy, dx                # unit perpendicular = allowed translation
        s = du * px + dv * py           # scalar component along the allowed direction
        return s * px, s * py

    def _update_drag_silhouette(self, sk, du, dv):
        """Set the viewport preview to the range the dragged line is allowed to
        sweep over: for a fixed-angle line, the strip between its start and
        current positions along the allowed (perpendicular) direction."""
        kind, idx = self._sketch_drag_kind, self._sketch_drag_idx
        if kind != "l" or not self._line_angle_fixed(idx):
            self.view.preview = []
            return
        orig = self._sketch_drag_orig
        if orig is None:
            self.view.preview = []
            return
        a, b = sk.lines[idx]
        p0a = orig[a]
        p0b = orig[b]
        p1a = sk.points[a]
        p1b = sk.points[b]
        if all(math.hypot(x1 - x0, y1 - y0) < 1e-9
               for (x0, y0), (x1, y1) in ((p0a, p1a), (p0b, p1b))):
            self.view.preview = []   # no sweep yet; keep it clean
            return
        pl = sk.plane
        # Closed strip boundary (first point repeated so the LINE_STRIP loops):
        # start-A -> current-A -> current-B -> start-B -> start-A
        quad = [pl.to3d(*p0a), pl.to3d(*p1a), pl.to3d(*p1b),
               pl.to3d(*p0b), pl.to3d(*p0a)]
        self.view.preview = [np.array(quad, dtype=float)]

    def _collect_drag_points(self, kind, idx):
        """Indices of the sketch points that move with the dragged element."""
        sk = self.sketch
        if kind == "p":
            return {idx: list(sk.points[idx])}
        if kind == "l":
            a, b = sk.lines[idx]
            return {a: list(sk.points[a]), b: list(sk.points[b])}
        if kind == "c":
            center, _ = sk.circles[idx]
            return {center: list(sk.points[center])}
        return {}

    def _finish_sketch_drag(self):
        """Solve after a drag; revert to the pre-drag positions if the solve fails."""
        sk = self.sketch
        orig = self._sketch_drag_orig
        if not sk.solve():
            if orig is not None:
                for pi, (u, v) in orig.items():
                    sk.points[pi] = [u, v]
            sk.solve()  # restores the known-good state
            self.statusBar().showMessage("Drag rejected: sketch constraints unsatisfied.", 6000)
        else:
            sk.version += 1
            hit = self._sketch_press_hit
            if hit is not None:
                kind, idx = hit
                label = {"p": f"point {idx}", "l": f"line {idx}", "c": f"circle {idx}"}[kind]
                self.statusBar().showMessage(f"Dragged {label}.", 2000)
        self.refresh_constraints()
        self._update_regions()
        self.mark_dirty()

    def _snap(self, x, y):
        """-> (existing point index or None, (u, v)) for the cursor, or None if the ray misses the plane."""
        pi = self.view.nearest_sketch_point(x, y)
        if pi is not None:
            return pi, tuple(self.sketch.points[pi])
        uv = self.view.sketch_uv(x, y)
        if uv is None:
            return None
        g = self.snap.value()
        if g is None:
            g = self.snap.last_valid or 0.0  # keep snapping while the text is being edited
        return None, ((round(uv[0] / g) * g, round(uv[1] / g) * g) if g > 0 else uv)

    def _update_regions(self):
        """Refresh the fill for selected and hovered sketch regions (hover_region = (feature index, region))."""
        fills = []
        for i, f in enumerate(self.doc.features):
            if f.kind != "sketch":
                continue
            shown = i == self.sketch_index if self.mode == "sketch" else f.visible
            if not shown:
                continue
            chosen = f.sketch.selected_regions()
            chosen_fills = [f.sketch.region_triangles(r) for r in sorted(chosen)]
            fills += [(t, (1.0, 0.55, 0.15, 0.45)) for t in chosen_fills]
            if i == self.region_sketch and chosen_fills:
                self._region_anchor = chosen_fills[0].reshape(-1, 3).mean(axis=0)
            if self.hover_region is not None and self.hover_region[0] == i and self.hover_region[1] not in chosen:
                fills.append((f.sketch.region_triangles(self.hover_region[1]), (0.95, 0.9, 0.3, 0.35)))
        self.view.region_fill = fills
        self._refresh_quick()
        self.view.update()

    # ---- quick buttons beside the selection
    CONSTRAINT_LABELS = {
        "horizontal": "Horizontal", "vertical": "Vertical", "coincident": "Coincident", "parallel": "Parallel",
        "perpendicular": "Perpendicular", "equal": "Equal", "distance": "Distance", "spacing": "Spacing",
        "diameter": "Diameter", "expression": "Expression", "fixed": "Fixed", "point_on_line": "Point on line",
    }

    def _sketch_quick_kinds(self, pts, lines, circles) -> list[str]:
        """Constraints that fit a sketch selection, likeliest first (same rules as add_constraint)."""
        if lines and not pts and not circles:
            if len(lines) == 1:
                return ["distance", "horizontal", "vertical"]
            if len(lines) == 2:
                return ["parallel", "perpendicular", "equal", "spacing"]
        if pts and not lines and not circles:
            if len(pts) == 2:
                return ["coincident", "distance", "horizontal", "vertical", "expression", "fixed"]
            return ["fixed"]
        if len(pts) == 1 and len(lines) == 1 and not circles:
            return ["point_on_line"]
        if len(circles) == 1 and not pts and not lines:
            return ["diameter"]
        return []

    def _quick_spec(self):
        """(key, world anchor, likeliest tools, more tools, delete tool) for the selection, or None."""
        if self._session is not None or self.tool_bar.active:
            return None
        v = self.view
        if self.mode == "sketch":
            if self.tool != "select" or not v.sketch_sel or self.sketch is None:
                return None
            sk = self.sketch
            sel = sorted(v.sketch_sel)
            pts = [i for k, i in sel if k == "p"]
            lines = [i for k, i in sel if k == "l"]
            circles = [i for k, i in sel if k == "c"]
            try:
                uvs = [sk.points[i] for i in pts]
                uvs += [np.mean([sk.points[a], sk.points[b]], axis=0) for a, b in (sk.lines[i] for i in lines)]
                uvs += [sk.points[sk.circles[i][0]] for i in circles]
            except IndexError:
                return None
            if not uvs:
                return None
            anchor = np.asarray(sk.plane.to3d(*np.mean(uvs, axis=0)), float)
            tools = [(self.CONSTRAINT_LABELS[k], lambda k=k: self.add_constraint(k), f"Add a {k} constraint")
                     for k in self._sketch_quick_kinds(pts, lines, circles)]
            delete = ("🗑", self.delete_selection, "Delete the selected sketch items")
            return ("sketch", tuple(sel)), anchor, tools[:2], tools[2:], delete
        mesh = self.doc.mesh
        if v.sel_edges and not v.sel_faces and mesh is not None:
            edges = sorted(e for e in v.sel_edges if 0 <= e < len(mesh.edges))
            if not edges:
                return None
            owners = {self._edge_owner(e) for e in edges}
            delete = None
            if len(owners) == 1 and None not in owners:
                i = owners.pop()
                delete = ("🗑", lambda i=i: self._delete_feature_at(i),
                          f"Delete the {self.doc.features[i].kind} this edge belongs to")
            tools = [("◐ Fillet", self.fillet, "Round the selected edges"),
                     ("◺ Chamfer", self.chamfer, "Bevel the selected edges")]
            return ("edges", tuple(edges)), np.asarray(mesh.edges[edges[0]].mid, float), tools, [], delete
        if len(v.sel_faces) == 1 and not v.sel_edges and mesh is not None:
            face = next(iter(v.sel_faces))
            if not (0 <= face < len(mesh.faces)) or mesh.faces[face].normal is None:
                return None
            tools = [("✏️ Sketch", self._sketch_on_selected_face, "Start a sketch on this face"),
                     ("↕️ Push/Pull", self.pushpull, "Move this face along its normal")]
            return ("face", face), np.asarray(mesh.faces[face].center, float), tools, [], None
        feats = self.doc.features
        if (not v.sel_faces and not v.sel_edges and 0 <= self.region_sketch < len(feats)
                and feats[self.region_sketch].kind == "sketch"
                and feats[self.region_sketch].sketch.selected_regions() and self._region_anchor is not None):
            chosen = tuple(sorted(feats[self.region_sketch].sketch.selected_regions()))
            tools = [("⇈ Extrude", self.extrude, "Extrude the highlighted regions"),
                     ("⟳ Revolve", self.revolve, "Revolve the highlighted regions")]
            more = [("↝ Sweep", self.sweep, "Sweep this profile along another sketch"),
                    ("⌁ Loft", self.loft, "Loft between sketch sections")]
            return ("region", self.region_sketch, chosen), self._region_anchor, tools, more, None
        return None

    def _refresh_quick(self):
        spec = self._quick_spec()
        if spec is None:
            self.quick.clear()
            return
        key, anchor, primary, more, delete = spec
        self.quick.set_tools(key, anchor, primary, more, delete)
        self._place_quick()

    def _place_quick(self):
        """Keep the quick buttons beside the selection; hidden while the mouse drags."""
        q, v = self.quick, self.view
        if q.anchor is None or not q.buttons:
            q.hide()
            return
        if v._press is not None and v._dragged:
            q.hide()
            return
        s = v.camera.project(np.asarray(q.anchor, float).reshape(1, 3), v.width(), v.height())[0]
        if np.isnan(s[0]) or np.isnan(s[1]):
            q.hide()
            return
        q.place(float(s[0]), float(s[1]))

    def _edge_owner(self, edge: int) -> int | None:
        """Index of the fillet (by its rounded edges) or fillet/chamfer (by its picked edges) an edge belongs to."""
        i = self._find_fillet_by_edge(tuple(map(float, self.doc.mesh.edges[edge].mid)))
        return i if i is not None else self._feature_for_edge(edge)

    def _sketch_on_selected_face(self):
        self.plane_combo.setCurrentText("Selected face")
        self.new_sketch()

    def _delete_feature_at(self, i: int):
        if not 0 <= i < len(self.doc.features):
            return
        self._push_undo()
        self.doc.delete(i)
        self._clear_sketch_history()
        self.rebuild()

    def _pick_region(self, x, y):
        """Visible-sketch region under the cursor and not hidden behind a model face: (feature, region, (u, v))."""
        cam, w, h = self.view.camera, self.view.width(), self.view.height()
        o, d = cam.ray(x, y, w, h)
        face_t = self.view.pick_face_hit(x, y)[1]
        best = None
        for i, f in enumerate(self.doc.features):
            if f.kind != "sketch" or not f.visible:
                continue
            pl = f.sketch.plane
            p = ray_plane(o, d, pl.origin, np.array(pl.normal))
            if p is None:
                continue
            t = float(np.dot(p - o, d))
            uv = pl.to_uv(p)
            r = f.sketch.region_at(*uv)
            if r is not None and (face_t is None or t <= face_t + 1e-4 * cam.dist) and (best is None or t < best[0]):
                best = (t, i, r, uv)
        return None if best is None else best[1:]

    def _solve_sketch(self):
        self.sketch.variables = self.doc.variables
        ok = self.sketch.solve()
        if not ok:
            self.statusBar().showMessage(f"Solver: {self.sketch.status}", 6000)
        self.refresh_constraints()
        self.mark_dirty()
        self._update_regions()
        return ok

    def _sketch_click(self, x, y, mods):
        self._push_sketch_undo()
        sk = self.sketch
        if self.tool == "select":
            hit = self.view.pick_sketch(x, y)
            sel = self.view.sketch_sel
            multi = bool(mods & Qt.ControlModifier)
            # In mirror mode (self._mirror_axis is set) we always add hits to the selection,
            # without requiring Ctrl and without replacing the current selection.
            if self._mirror_axis is not None:
                if hit is not None:
                    sel.add(hit)
                # background clicks are ignored in mirror mode
                self._update_regions()
                return
            if hit is None:
                uv = self.view.sketch_uv(x, y)
                if not multi:
                    sel.clear()
                if uv is not None:
                    sk.toggle_region(uv[0], uv[1], multi)
                    self.region_sketch = self.sketch_index
                    self.mark_dirty()
            elif multi:
                sel.symmetric_difference_update({hit})
            else:
                sel.clear()
                sel.add(hit)
            self._update_regions()
            return
        snap = self._snap(x, y)
        if snap is None:
            return
        self._draw_at(*snap)

    # ---- sizes typed while drawing
    FIELD_SPECS = {
        "line": [("L", "L", "length"), ("A", "∠", "angle")],
        "rect": [("W", "W", "length"), ("H", "H", "length")],
        "circle": [("D", "Ø", "length")],
    }

    def _apply_locks(self, p0, uv):
        """Move the cursor point so the shape from p0 has the sizes typed into the fields."""
        f = self.fields
        du, dv = uv[0] - p0[0], uv[1] - p0[1]
        if self.tool == "line":
            length, angle = f.locked("L"), f.locked("A")
            if length is None and angle is None:
                return uv
            length = abs(length) if length is not None else math.hypot(du, dv)
            a = math.radians(angle) if angle is not None else (math.atan2(dv, du) if (du or dv) else 0.0)
            return p0[0] + length * math.cos(a), p0[1] + length * math.sin(a)
        if self.tool == "rect":
            w, h = f.locked("W"), f.locked("H")
            if w is not None:
                du = w if w < 0 else math.copysign(w, du)
            if h is not None:
                dv = h if h < 0 else math.copysign(h, dv)
            return p0[0] + du, p0[1] + dv
        if self.tool == "circle":
            d = f.locked("D")
            if d is None:
                return uv
            n = math.hypot(du, dv)
            ux, uy = (du / n, dv / n) if n > 1e-12 else (1.0, 0.0)
            return p0[0] + abs(d) / 2 * ux, p0[1] + abs(d) / 2 * uy
        return uv

    def _field_values(self, p0, uv) -> dict:
        du, dv = uv[0] - p0[0], uv[1] - p0[1]
        if self.tool == "line":
            return {"L": math.hypot(du, dv), "A": math.degrees(math.atan2(dv, du)) if (du or dv) else 0.0}
        if self.tool == "rect":
            return {"W": abs(du), "H": abs(dv)}
        if self.tool == "circle":
            return {"D": 2 * math.hypot(du, dv)}
        return {}

    # -- boxes left on a shape after it is drawn
    def _show_drawn(self, drawn: dict):
        """Keep the size boxes on the shape just drawn so it can be resized by typing or scrubbing.

        drawn: {"tool": "rect", "pts": [4 point indices]} | {"tool": "circle", "circle": index}
               | {"tool": "line", "a": start point, "b": end point}
        """
        self._drawn = dict(drawn, undo=False)
        self.fields.start(self.FIELD_SPECS[drawn["tool"]], post=True)
        self._refresh_drawn_fields()

    def _drawn_ok(self) -> bool:
        d = self._drawn
        if d is None or not self.fields.post or self.mode != "sketch":
            return False
        sk = self.sketch
        if d["tool"] == "rect":
            return all(i < len(sk.points) for i in d["pts"])
        if d["tool"] == "circle":
            return d["circle"] < len(sk.circles)
        return max(d["a"], d["b"]) < len(sk.points)

    def _drawn_geometry(self):
        """(start uv, end uv) for rect (opposite corners) / line, or (center uv, radius) for circle."""
        d, sk = self._drawn, self.sketch
        if d["tool"] == "rect":
            return tuple(sk.points[d["pts"][0]]), tuple(sk.points[d["pts"][2]])
        if d["tool"] == "circle":
            c, r = sk.circles[d["circle"]]
            return tuple(sk.points[c]), r
        return tuple(sk.points[d["a"]]), tuple(sk.points[d["b"]])

    def _refresh_drawn_fields(self):
        d = self._drawn
        p0, end = self._drawn_geometry()
        if d["tool"] == "circle":
            self.fields.set_values({"D": 2 * end})
            anchor = (p0[0] + end, p0[1])
        else:
            tool, self.tool = self.tool, d["tool"]  # _field_values reads the tool
            self.fields.set_values(self._field_values(p0, end))
            self.tool = tool
            anchor = end
        self.fields.anchor = self.sketch.plane.to3d(*anchor)
        self.view.update()

    def _apply_drawn(self):
        """Resize the shape just drawn to the sizes typed or scrubbed into its boxes."""
        if not self._drawn_ok():
            return
        d, sk, f = self._drawn, self.sketch, self.fields
        p0, end = self._drawn_geometry()
        if d["tool"] == "rect":
            du, dv = end[0] - p0[0], end[1] - p0[1]
            w, h = f.locked("W"), f.locked("H")
            if w is not None:
                du = w if w < 0 else math.copysign(w, du)
            if h is not None:
                dv = h if h < 0 else math.copysign(h, dv)
            if abs(du) < 1e-9 or abs(dv) < 1e-9:
                return
            u1, v1 = p0[0] + du, p0[1] + dv
            new = {d["pts"][1]: [u1, p0[1]], d["pts"][2]: [u1, v1], d["pts"][3]: [p0[0], v1]}
        elif d["tool"] == "circle":
            dia = f.locked("D")
            if dia is None or abs(dia) < 1e-9:
                return
            new = None
        else:
            length, angle = f.locked("L"), f.locked("A")
            du, dv = end[0] - p0[0], end[1] - p0[1]
            length = abs(length) if length is not None else math.hypot(du, dv)
            a = math.radians(angle) if angle is not None else math.atan2(dv, du)
            if length < 1e-9:
                return
            new = {d["b"]: [p0[0] + length * math.cos(a), p0[1] + length * math.sin(a)]}
        if not d["undo"]:  # one undo step per editing session
            self._push_sketch_undo()
            d["undo"] = True
        if new is None:
            c, _r = sk.circles[d["circle"]]
            sk.circles[d["circle"]] = (c, abs(dia) / 2)
        else:
            for i, uv in new.items():
                sk.points[i] = uv
        sk.version += 1
        self._solve_sketch()
        self._refresh_drawn_fields()

    def _show_last_segment(self):
        """At the end of a line chain, keep the last segment's boxes if its end point is free."""
        seg, self._last_segment = self._last_segment, None
        if seg is None or self.mode != "sketch":
            return
        a, b = seg
        if b >= len(self.sketch.points) or sum(b in ln for ln in self.sketch.lines) != 1:
            return  # closing onto other geometry: resizing would drag that along
        self._show_drawn({"tool": "line", "a": a, "b": b})

    def _start_fields(self):
        self._drawn = None
        specs = self.FIELD_SPECS.get(self.tool)
        if specs and self.pending is not None:
            self.fields.start(specs)
            self._draw_mouse = None
        else:
            self.fields.stop()

    def _on_fields_changed(self):
        if self.fields.post:
            self._apply_drawn()  # the shape follows typing and scrubbing
            return
        if self._draw_mouse is not None:
            self._update_preview(*self._draw_mouse[2:])
        elif self.pending is not None:
            self._show_draw_preview(self.pending[1], None, None)

    def _on_fields_accepted(self):
        """Enter in a drawing value box: place the shape at the typed sizes."""
        if self.fields.post:
            self._apply_drawn()
            if self._drawn is not None:
                self._drawn["undo"] = False  # the next edit is a new undo step
            self.fields.end_typing()
            if self._drawn_ok():
                self._refresh_drawn_fields()
            self.view.setFocus()
            return
        if self.mode != "sketch" or self.pending is None:
            return
        pi, uv = (self._draw_mouse[:2] if self._draw_mouse is not None else (None, self.pending[1]))
        self._push_sketch_undo()
        self._draw_at(pi, uv)
        self.view.setFocus()

    def _draw_at(self, pi, uv):
        """Place the next point of the active drawing tool at uv (pi: existing point there)."""
        sk = self.sketch
        if self.pending is not None and self.fields.active:
            fixed = self._apply_locks(self.pending[1], uv)
            if math.dist(fixed, uv) > 1e-9:
                pi, uv = None, fixed
        if self.tool == "line":
            if self.pending is None:
                self.pending = (pi, uv)
                self._last_segment = None
                self._start_fields()
                return
            a_idx, a_uv = self.pending
            if a_idx is None and pi is None and math.dist(a_uv, uv) < 1e-9:
                return
            a = a_idx if a_idx is not None else sk.add_point(*a_uv)
            b = pi if pi is not None else sk.add_point(*uv)
            if a != b:
                sk.add_line(a, b)
                sk.version += 1
                self._last_segment = (a, b)
            self.pending = (b, tuple(sk.points[b])) if pi is None else None
            self._start_fields()  # the next segment starts with nothing typed
            self._solve_sketch()
            if self.pending is None:  # the chain ended on an existing point
                self._show_last_segment()
        elif self.tool == "rect":
            if self.pending is None:
                self.pending = (pi, uv)
                self._start_fields()
                return
            (u0, v0), (u1, v1) = self.pending[1], uv
            if abs(u1 - u0) > 1e-9 and abs(v1 - v0) > 1e-9:
                sk.add_rect(u0, v0, u1, v1)
                sk.version += 1
                self._solve_sketch()
                n = len(sk.points)
                self.set_tool("select")  # so the new region can be highlighted right away
                self._show_drawn({"tool": "rect", "pts": list(range(n - 4, n))})
            else:
                self.set_tool("select")
        elif self.tool == "circle":
            if self.pending is None:
                self.pending = (pi, uv)
                self._start_fields()
                return
            c_idx, c_uv = self.pending
            r = math.dist(c_uv, uv)
            if r > 1e-9:
                c = c_idx if c_idx is not None else sk.add_point(*c_uv)
                ci = sk.add_circle(c, r)
                sk.version += 1
                self._solve_sketch()
                self.set_tool("select")
                self._show_drawn({"tool": "circle", "circle": ci})
            else:
                self.set_tool("select")

    def _update_preview(self, x, y):
        snap = self._snap(x, y)
        if snap is None or self.pending is None:
            self.view.preview = []
            self.view.update()
            return
        self._draw_mouse = (snap[0], snap[1], x, y)
        self._show_draw_preview(snap[1], x, y)

    def _show_draw_preview(self, uv, x, y):
        """Rubber-band the shape from the pending point to uv (after typed sizes) and show its sizes."""
        pl = self.sketch.plane
        p0 = self.pending[1]
        if self.fields.active:
            uv = self._apply_locks(p0, uv)
            self.fields.set_values(self._field_values(p0, uv))
            if x is None:  # no hover yet: put the boxes by the pending point
                x, y = self.view.camera.project(np.array([pl.to3d(*p0)], float),
                                                self.view.width(), self.view.height())[0][:2]
            self.fields.place(x, y)
        self.view.preview = []
        if self.tool == "line":
            self.view.preview = [np.array([pl.to3d(*p0), pl.to3d(*uv)])]
        elif self.tool == "rect":
            c = [(p0[0], p0[1]), (uv[0], p0[1]), (uv[0], uv[1]), (p0[0], uv[1]), (p0[0], p0[1])]
            self.view.preview = [np.array([pl.to3d(*p) for p in c])]
        elif self.tool == "circle":
            self.view.preview = [np.array(Viewport._circle_world(pl, p0, math.dist(p0, uv)))]
        self.view.update()
        u = self.unit
        self.statusBar().showMessage(f"u={u.from_mm(uv[0]):.{u.decimals}f}  v={u.from_mm(uv[1]):.{u.decimals}f} {u.name}")

    # ---- constraints
    def add_constraint(self, kind):
        self._push_sketch_undo()
        sk = self.sketch
        sel = sorted(self.view.sketch_sel)
        pts = [i for k, i in sel if k == "p"]
        lines = [i for k, i in sel if k == "l"]
        circles = [i for k, i in sel if k == "c"]
        new: list[Constraint] = []
        value = None
        try:
            if kind in ("horizontal", "vertical"):
                if len(lines) == 1 and not pts and not circles:
                    new = [Constraint(kind, lines)]
                elif len(pts) == 2 and not lines and not circles:
                    new = [Constraint("p" + kind, pts)]
            elif kind == "coincident" and len(pts) == 2 and not lines and not circles:
                new = [Constraint(kind, pts)]
            elif kind in ("parallel", "perpendicular", "equal") and len(lines) == 2 and not pts and not circles:
                new = [Constraint(kind, lines)]
            elif kind == "spacing" and len(lines) == 2 and not pts and not circles:
                a = sk.points[sk.lines[lines[0]][0]]
                b = sk.points[sk.lines[lines[1]][0]]
                c = sk.points[sk.lines[lines[1]][1]]
                ax, ay = a
                (x1, y1), (x2, y2) = b, c
                cur = abs((x2 - x1) * (y1 - ay) - (x1 - ax) * (y2 - y1)) / max(math.hypot(x2 - x1, y2 - y1), 1e-12)
                return self._constraint_session(
                    "Spacing", Constraint("spacing", [sk.lines[lines[0]][0], lines[1]], cur), new=True)
            elif kind == "expression" and len(pts) == 2 and not lines and not circles:
                dialog = ExpressionDialog(self, "(e.g. 2*a + 3)π")
                if not dialog.exec():
                    return
                expr = dialog.expression()
                if not expr:
                    return
                new = [Constraint("expression", [pts[0], pts[1]], expression=expr)]
                self.statusBar().showMessage(f"Expression: {expr}", 6000)
            elif kind == "distance":
                ends = pts if len(pts) == 2 and not lines else list(sk.lines[lines[0]]) if len(lines) == 1 and not pts else None
                if ends:
                    cur = math.dist(sk.points[ends[0]], sk.points[ends[1]])
                    return self._constraint_session("Distance", Constraint(kind, ends, cur), new=True)
            elif kind == "diameter" and len(circles) == 1 and not pts and not lines:
                cur = 2 * sk.circles[circles[0]][1]
                return self._constraint_session("Diameter", Constraint(kind, circles, cur), new=True)
            elif kind == "fixed" and pts and not lines and not circles:
                new = [Constraint(kind, [p]) for p in pts]
            elif kind == "point_on_line" and len(pts) == 1 and len(lines) == 1 and not circles:
                new = [Constraint(kind, [pts[0], lines[0]])]
            elif kind == "symmetry":
                # Two-step mirror: set mirror line
                if self._mirror_axis is None:
                    # Expect exactly one line in selection
                    lines = [i for k, i in sel if k == "l"]
                    if len(lines) == 1 and not pts and not circles:
                        line_idx = lines[0]
                        line_pts = sk.lines[line_idx]
                        a_idx, b_idx = line_pts
                        a = sk.points[a_idx]
                        b = sk.points[b_idx]
                        self._mirror_axis = (a_idx, b_idx, line_idx)
                        self._mirroring = True
                        self.statusBar().showMessage(
                            "Select points/lines to mirror across this line, then press Symmetry again.",
                            6000,
                        )
                        # Keep the selection (so the user can add more items)
                        return
                    else:
                        self.statusBar().showMessage(
                            "To set a mirror axis, select exactly one line first.", 6000
                        )
                        return
                else:
                    # We have a mirror axis; mirror the selected items
                    if not pts and not lines and not circles:
                        self.statusBar().showMessage(
                            "Select points, lines, or circles to mirror (or press Escape to cancel).", 6000
                        )
                        return
                    self._push_sketch_undo()
                    sk = self.sketch
                    a_idx, b_idx, mirror_line_idx = self._mirror_axis
                    # Exclude the mirror line and its endpoints from the items to mirror
                    lines = [i for i in lines if i != mirror_line_idx]
                    pts = [i for i in pts if i not in (a_idx, b_idx)]
                    circles = [i for i in circles if not (sk.circles[i][0] in (a_idx, b_idx))]
                    a = sk.points[a_idx]
                    b = sk.points[b_idx]
                    new_constraints = []
                    # Fix only the mirror line endpoints so the axis stays in place
                    new_constraints.append(Constraint("fixed", [a_idx]))
                    new_constraints.append(Constraint("fixed", [b_idx]))

                    mirrored_points = {}

                    def mirrored_point(point_idx):
                        if point_idx not in mirrored_points:
                            reflected = reflect_point_across_line(sk.points[point_idx], a, b)
                            mirrored_points[point_idx] = sk.add_point(reflected[0], reflected[1])
                            new_constraints.append(
                                Constraint("symmetry", [point_idx, mirrored_points[point_idx], mirror_line_idx])
                            )
                        return mirrored_points[point_idx]

                    for p in pts:
                        mirrored_point(p)

                    for l_idx in lines:
                        (p1, p2) = sk.lines[l_idx]
                        new_p1_idx = mirrored_point(p1)
                        new_p2_idx = mirrored_point(p2)
                        sk.add_line(new_p1_idx, new_p2_idx)

                    for ci in circles:
                        center_idx, r = sk.circles[ci]
                        new_center_idx = mirrored_point(center_idx)
                        sk.circles.append((new_center_idx, r))

                    if new_constraints:
                        sk.constraints += new_constraints
                        sk.version += 1
                        self._mirror_axis = None
                        self._mirroring = False
                        self.view.sketch_sel.clear()
                        self._solve_sketch()
                        self.statusBar().showMessage(
                            f"Mirrored {len(pts) + len(lines) + len(circles)} item(s).", 6000
                        )
                    else:
                        self.statusBar().showMessage(
                            "No items selected to mirror.", 6000
                        )
                    return
                # Fallback to old behavior (two points + one line)
                new = [Constraint("symmetry", [pts[0], pts[1], lines[0]])]
        except IndexError:
            new = []
        if not new:
            self.statusBar().showMessage(f"Select the right items for '{kind}' ({CONSTRAINT_KINDS[kind][0]}).", 6000)
            return
        sk.constraints += new
        sk.version += 1
        if not self._solve_sketch():
            del sk.constraints[-len(new):]
            self.sketch.solve()
            self.refresh_constraints()

    def edit_constraint(self, item):
        i = self.constraints.row(item)
        sk = self.sketch
        c = sk.constraints[i]
        if c.expression is not None:
            dialog = ExpressionDialog(self, c.expression)
            if dialog.exec():
                expr = dialog.expression()
                if expr:
                    self._push_sketch_undo()
                    old = c.expression
                    c.expression = expr
                    sk.version += 1
                    if not self._solve_sketch():
                        c.expression = old
                        self.sketch.solve()
                        self.refresh_constraints()
            return
        if c.value is None:
            return
        self._push_sketch_undo()
        self._constraint_session(f"Edit {c.kind}", c, new=False)

    def delete_constraint(self):
        i = self.constraints.currentRow()
        if i >= 0:
            self._push_sketch_undo()
            self.sketch.remove_constraint(i)
            self.sketch.version += 1
            self._solve_sketch()

    def delete_selection(self):
        self._push_sketch_undo()
        sk = self.sketch
        for kind in ("l", "c", "p"):  # deleting points last keeps the other indices stable
            for idx in sorted((i for k, i in self.view.sketch_sel if k == kind), reverse=True):
                sk.delete(kind, idx)
        sk.version += 1
        self.view.sketch_sel.clear()
        self._solve_sketch()

    # ---- model mode
    def on_click(self, x, y, button, mods):
        if button != Qt.LeftButton:
            return
        if self.mode == "sketch":
            self._sketch_click(x, y, mods)
            return
        v = self.view
        multi = bool(mods & Qt.ControlModifier)
        edge = v.pick_edge(x, y)
        hit = None if edge is not None else self._pick_region(x, y)
        face = None if edge is not None or hit is not None else v.pick_face(x, y)
        if not multi:
            v.sel_faces.clear()
            v.sel_edges.clear()
            for j, f in enumerate(self.doc.features):
                if f.kind == "sketch" and (hit is None or j != hit[0]):
                    f.sketch.selection = []
        if edge is not None:
            v.sel_edges.symmetric_difference_update({edge})
        elif hit is not None:
            i, _, uv = hit
            self.doc.features[i].sketch.toggle_region(uv[0], uv[1], multi)
            self.region_sketch = i
            self.mark_dirty()
        elif face is not None:
            v.sel_faces.symmetric_difference_update({face})
        self._update_regions()

    def on_wheel_adjust(self, x, y, delta, mods):
        edge = self.view.hover_edge
        if edge is None and self.view.sel_edges:
            edge = sorted(self.view.sel_edges)[0]
        feat_i = self._feature_for_edge(edge)
        if feat_i is None:
            row = self.features.currentRow()
            if 0 <= row < len(self.doc.features) and self.doc.features[row].kind in ("fillet", "chamfer"):
                feat_i = row
            else:
                candidates = [i for i, f in enumerate(self.doc.features) if f.kind in ("fillet", "chamfer")]
                feat_i = candidates[-1] if candidates else None
        if feat_i is None:
            return
        self._push_undo()
        f = self.doc.features[feat_i]
        step = 0.5 if not (mods & Qt.ShiftModifier) else 0.1
        f.value = max(1e-4, f.value + (delta / 120.0) * step)
        f.version += 1
        self.rebuild()
        self.statusBar().showMessage(f"{f.kind.title()} {self.unit.fmt(f.value)}", 2000)

    def _feature_for_edge(self, edge):
        if edge is None or self.doc.mesh is None or not (0 <= edge < len(self.doc.mesh.edges)):
            return None
        mid = np.asarray(self.doc.mesh.edges[edge].mid, float)
        for i, f in enumerate(self.doc.features):
            if f.kind not in ("fillet", "chamfer"):
                continue
            for m in f.edges:
                if np.linalg.norm(np.asarray(m, float) - mid) < 1e-3:
                    return i
        return None

    def on_hover(self, x, y):
        if self.mode == "sketch":
            if self.tool != "select":
                self._update_preview(x, y)
                return
            uv = self.view.sketch_uv(x, y)
            near = self.view.pick_sketch(x, y) is not None
            r = None if uv is None or near else self.sketch.region_at(*uv)
            region = None if r is None else (self.sketch_index, r)
            if region != self.hover_region:
                self.hover_region = region
                self._update_regions()
            return
        edge = self.view.pick_edge(x, y)
        hit = None if edge is not None else self._pick_region(x, y)
        region = None if hit is None else hit[:2]
        face = None if edge is not None or hit is not None else self.view.pick_face(x, y)
        if (edge, face) != (self.view.hover_edge, self.view.hover_face) or region != self.hover_region:
            self.view.hover_edge, self.view.hover_face = edge, face
            self.hover_region = region
            self._update_regions()

    def _sketch_for_extrude(self) -> int | None:
        feats = self.doc.features
        if 0 <= self.region_sketch < len(feats) and feats[self.region_sketch].kind == "sketch" \
                and feats[self.region_sketch].sketch.selected_regions():
            return self.region_sketch
        row = self.features.currentRow()
        if 0 <= row < len(feats) and feats[row].kind == "sketch":
            return row
        idx = [i for i, f in enumerate(feats) if f.kind == "sketch"]
        return idx[-1] if idx else None

    def extrude(self):
        i = self._sketch_for_extrude()
        if i is None:
            QMessageBox.information(self, "⇈", "Create a sketch with a closed profile first.")
            return
        sk = self.doc.features[i].sketch
        if len(sk.regions()) > 1 and not sk.selected_regions():
            self.enter_sketch(i)
            self.statusBar().showMessage("Click the regions to extrude (Ctrl+click for several), then press Extrude. "
                                         "With nothing highlighted, every region is extruded.", 15000)
            return
        self._extrude_sketch(i)

    def extrude_from_sketch(self):
        i = self.sketch_index
        self.finish_sketch()
        self._extrude_sketch(i)

    def _extrude_sketch(self, i: int):
        sk = self.doc.features[i].sketch
        if not sk.regions():
            QMessageBox.information(self, "⇈", "The sketch has no closed profile.")
            return
        chosen = sorted(sk.selected_regions())
        profiles = [list(next(p for p in sk.selection if sk.region_at(*p) == r)) for r in chosen] or None
        title = f"Extrude {len(profiles)} region(s)" if profiles else "Extrude"
        show_mode = self.doc.part is not None
        self._extrude_session(Feature("extrude", sketch_ref=i, value=self._last_extrude, auto=True,
                                      profiles=profiles), title, new=True, show_mode=show_mode)

    # ---- tool settings bar sessions
    EXTRUDE_MODES = [("auto", "Auto"), ("add", "Add"), ("cut", "Cut")]
    REVOLVE_AXES = [("u", "Sketch U axis"), ("v", "Sketch V axis")]
    ADD_CUT = [("add", "Add"), ("cut", "Cut")]

    def _extrude_session(self, f: Feature, title: str, new: bool, show_mode: bool = True):
        items = [Number("distance", "Distance", f.value, lo=-1e5, hi=1e5)]
        if show_mode:
            items.append(Choice("mode", "Operation", self.EXTRUDE_MODES,
                                "auto" if f.auto else "cut" if f.cut else "add"))

        def apply(f, v):
            f.value = v["distance"]
            mode = v.get("mode", "auto")
            f.auto, f.cut = mode == "auto", mode == "cut"

        def done(f):
            self._last_extrude = f.value

        self._begin_session(title, items, f, new, apply, done)

    def _revolve_session(self, f: Feature, title: str, new: bool):
        items = [Number("angle", "Angle", f.value, "angle", lo=1.0, hi=360.0),
                 Choice("axis", "Axis", self.REVOLVE_AXES, f.axis),
                 Choice("mode", "Operation", self.ADD_CUT, "cut" if f.cut else "add")]

        def apply(f, v):
            f.value, f.axis, f.cut = v["angle"], v["axis"], v["mode"] == "cut"

        self._begin_session(title, items, f, new, apply)

    def _value_session(self, f: Feature, title: str, label: str, new: bool, lo: float, hi: float,
                       nonzero: bool = False):
        """One length on a feature (push/pull distance, fillet radius, chamfer size)."""
        items = [Number("value", label, f.value, lo=lo, hi=hi)]

        def apply(f, v):
            f.value = v["value"]

        def validate(v):
            return f"{label} can't be 0" if nonzero and abs(v["value"]) < 1e-9 else None

        def done(f):
            if f.kind == "fillet":
                f.result_edges = self._fillet_result_edges(f)

        self._begin_session(title, items, f, new, apply, done, validate)

    def _begin_session(self, title, items, f: Feature, new: bool, apply, done=None, validate=None):
        """Open the settings bar on a feature: new ones are added to the history right away and
        every change rebuilds the model; ✓ keeps it, ✕ puts the history back. One undo step."""
        self._settle_session()
        self._push_undo()
        snapshot = {k: getattr(f, k) for k in ("value", "cut", "auto", "axis")}
        if new:
            self.doc.features.append(f)

        def index():
            return next((i for i, x in enumerate(self.doc.features) if x is f), None)

        def preview(values):
            if validate is not None:
                err = validate(values)
                if err:
                    return err
            apply(f, values)
            f.version += 1
            self.rebuild()
            i = index()
            return "The feature is gone" if i is None else self.doc.errors.get(i)

        def check():
            i = index()
            return "The feature is gone" if i is None else self.doc.errors.get(i)

        def restore():
            if new:
                self.doc.features[:] = [x for x in self.doc.features if x is not f]
            else:
                for k, v in snapshot.items():
                    setattr(f, k, v)
                f.version += 1
            if self._undo:
                self._undo.pop()  # the snapshot taken when the tool opened
            self.rebuild()

        self._open_session(title, items, preview, check, restore, (lambda: done(f)) if done else None)

    def _constraint_session(self, title: str, c: Constraint, new: bool):
        """A sketch dimension in the bar: the sketch re-solves as the value changes.
        The caller has pushed the sketch undo snapshot; new constraints are added here."""
        sk, index = self.sketch, self.sketch_index
        snapshot = sk.to_dict()
        if new:
            sk.constraints.append(c)

        def preview(values):
            c.value = values["value"]
            sk.version += 1
            return None if self._solve_sketch() else f"Can't solve the sketch ({sk.status})"

        def restore():
            self._sketch_restoring = True
            try:
                self._restore_sketch(snapshot, index)
            finally:
                self._sketch_restoring = False
            stack = self._sketch_undo.get(index)
            if stack:
                stack.pop()  # the snapshot taken when the dimension was started
            sk.version += 1
            self._solve_sketch()

        items = [Number("value", "Value", c.value, lo=1e-6, hi=1e6)]
        self._open_session(title, items, preview, lambda: None, restore)

    def _open_session(self, title, items, preview, check, restore, done=None):
        """preview(values) -> error or None; check() -> error or None; restore() undoes everything."""
        self._session = {"preview": preview, "check": check, "restore": restore, "done": done, "error": None}
        self.tool_bar.begin(title, items, self._session_changed, self._finish_session, self._cancel_session)
        self._session_changed(self.tool_bar.values())

    def _session_changed(self, values: dict | None):
        s = self._session
        if s is None or values is None:
            return
        s["error"] = s["preview"](values)
        self.tool_bar.set_error(s["error"])

    def _finish_session(self) -> bool:
        """✓ / Enter: keep the result as previewed. Returns False (bar stays open) if it fails."""
        s = self._session
        if s is None:
            return True
        err = s["error"] or s["check"]()
        if err:
            self.tool_bar.set_error(err)
            return False
        self._session = None
        self.tool_bar.end()
        if s["done"]:
            s["done"]()
        self.view.setFocus()
        return True

    def _cancel_session(self):
        """✕ / Esc: put everything back as it was before the tool opened."""
        s, self._session = self._session, None
        self.tool_bar.end()
        if s is None:
            return
        s["restore"]()
        self.view.setFocus()

    def _settle_session(self):
        """Something else is starting: keep the open tool's result, or drop it if it fails."""
        if self._session is not None and not self._finish_session():
            self._cancel_session()

    def revolve(self):
        i = self._sketch_for_extrude()
        if i is None:
            QMessageBox.information(self, "Revolve", "Create a sketch with a closed profile first.")
            return
        sketch = self.doc.features[i].sketch
        if len(sketch.regions()) > 1 and not sketch.selected_regions():
            self.enter_sketch(i)
            self.statusBar().showMessage("Click the regions to revolve (Ctrl+click for several), then finish the sketch.",
                                         12000)
            return
        chosen = sorted(sketch.selected_regions())
        profiles = [list(next(p for p in sketch.selection if sketch.region_at(*p) == r)) for r in chosen] or None
        self._revolve_session(Feature("revolve", sketch_ref=i, value=360.0, axis="v", profiles=profiles),
                              "Revolve", new=True)

    def sweep(self):
        profile_ref = self._sketch_for_extrude()
        if profile_ref is None:
            QMessageBox.information(self, "Sweep", "Create a sketch with one closed profile first.")
            return
        profile = self.doc.features[profile_ref].sketch
        if len(profile.regions()) != 1:
            QMessageBox.information(self, "Sweep", "Sweep currently requires exactly one closed profile region.")
            return
        paths = [(i, f"{i}: {f.name}") for i, f in enumerate(self.doc.features)
                 if f.kind == "sketch" and i != profile_ref and f.sketch.lines]
        if not paths:
            QMessageBox.information(self, "Sweep", "Create another sketch with connected path lines first.")
            return
        dialog = SweepDialog(self, paths)
        if dialog.exec():
            self.commit(Feature("sweep", sketch_ref=profile_ref, path_ref=dialog.path_ref, cut=dialog.cut))

    def loft(self):
        sections = [(i, f"{i}: {f.name}") for i, f in enumerate(self.doc.features)
                    if f.kind == "sketch" and len(f.sketch.regions()) == 1]
        if len(sections) < 2:
            QMessageBox.information(self, "Loft", "Create at least two sketches with one closed profile each.")
            return
        dialog = LoftDialog(self, sections)
        if dialog.exec():
            refs = dialog.section_refs
            if len(refs) < 2:
                QMessageBox.information(self, "Loft", "Select at least two section sketches.")
                return
            self.commit(Feature("loft", section_refs=refs, cut=dialog.cut))

    def pushpull(self):
        if len(self.view.sel_faces) != 1:
            QMessageBox.information(self, "↕️", "Select exactly one planar face.")
            return
        info = self.doc.mesh.faces[next(iter(self.view.sel_faces))]
        if info.normal is None:
            QMessageBox.information(self, "↕️", "Only planar faces are supported.")
            return
        f = Feature("pushpull", value=5.0, face=(tuple(map(float, info.center)), tuple(map(float, info.normal))))
        self._value_session(f, "Push/pull", "Distance (+ out)", new=True, lo=-1e5, hi=1e5, nonzero=True)

    def on_extrude_commit(self, dist: float, center: tuple, normal: tuple):
        """Drag-to-extrude: commit a pushpull feature from the drag distance."""
        self._value_feature = None
        if abs(dist) < 0.01:
            self.view.dismiss_value_box()
            return
        feature = Feature("pushpull", value=dist, face=(center, normal))
        if self.commit(feature):
            self._value_feature = feature
        else:
            self.view.dismiss_value_box()

    def on_value_edit(self, kind: str, value: float):
        """A size typed into the value box left on the model after a drag."""
        f = self._value_feature
        expected = {"extrude": "pushpull", "fillet": "fillet", "chamfer": "chamfer"}.get(kind)
        if f is None or f.kind != expected or not any(x is f for x in self.doc.features):
            self._value_feature = None  # gone (undo, delete, reload): nothing to edit
            self.view.dismiss_value_box()
            return
        scrub, self._scrub_prev = self._scrub_prev, None
        if scrub is not None and scrub[0] is not f:
            scrub = None
        prev = scrub[1] if scrub is not None else f.value
        if (kind in ("fillet", "chamfer") and value <= 1e-4) or abs(value) < 0.01:
            self.statusBar().showMessage("Value too small", 5000)
            if scrub is not None:  # put back the size from before the scrub
                f.value = prev
                f.version += 1
                self._undo.pop()
                self.rebuild()
            self.view.set_post_value(f.value)
            return
        if scrub is None:  # a scrub already saved the undo snapshot
            self._push_undo()
        f.value = float(value)
        f.version += 1
        self.rebuild()
        idx = next(i for i, x in enumerate(self.doc.features) if x is f)
        if idx in self.doc.errors:
            msg = self.doc.errors[idx]
            f.value = prev
            f.version += 1
            self._undo.pop()
            self.rebuild()
            self.view.set_post_value(prev)
            QMessageBox.warning(self, "Operation failed", msg)
        if f.kind == "fillet":
            f.result_edges = self._fillet_result_edges(f)

    def on_value_scrub(self, kind: str, value: float):
        """Live size while the value box left after a drag is scrubbed; on_value_edit finishes it."""
        f = self._value_feature
        expected = {"extrude": "pushpull", "fillet": "fillet", "chamfer": "chamfer"}.get(kind)
        if f is None or f.kind != expected or not any(x is f for x in self.doc.features):
            return
        if (kind in ("fillet", "chamfer") and value <= 1e-4) or abs(value) < 0.01:
            return
        if self._scrub_prev is None or self._scrub_prev[0] is not f:
            self._scrub_prev = (f, f.value)
            self._push_undo()
        f.value = float(value)
        f.version += 1
        self.rebuild()

    def on_fillet_cancel(self):
        """Esc during a fillet drag: put the history back as it was before the drag."""
        idx = self._fillet_live_idx
        if idx is None:
            return
        f = self.doc.features[idx]
        was_mod = self._fillet_is_mod
        if was_mod:
            f.value = float(self._fillet_mod_prev)
            f.kind = self._fillet_mod_prev_kind or "fillet"
            f.version += 1
        else:
            self.doc.features.pop(idx)
        if self._undo:
            self._undo.pop()  # the snapshot pushed when the live preview started
        self._fillet_live_idx = None
        self._fillet_preview_r = None
        self._fillet_is_mod = False
        self._fillet_mod_prev = None
        self._fillet_mod_prev_kind = None
        self._fillet_preview_kind = None
        self.rebuild()
        if was_mod:
            f.result_edges = self._fillet_result_edges(f)

    def _fillet_edge_dirs(self, mids: list) -> list:
        """Aligned unit directions of picked edges; curved edges have no direction."""
        mesh = self.doc.mesh
        if mesh is None:
            return []
        dirs = []
        for m in mids:
            m_arr = np.asarray(m, float)
            best_e, best_d = None, 1e-4  # edge mid within ~0.1mm is the picked edge
            for e in mesh.edges:
                d = float(np.linalg.norm(e.mid - m_arr))
                if d < best_d:
                    best_e, best_d = e, d
            if best_e is None:
                dirs.append(None)
                continue
            if len(best_e.polyline) > 2:
                dirs.append(None)
                continue
            p = np.asarray(best_e.polyline, float)
            v = p[-1] - p[0]
            n = float(np.linalg.norm(v))
            dirs.append(tuple((v / n).tolist()) if n > 1e-9 else None)
        return dirs

    def _fillet_edge_refs(self, mids: list) -> list:
        """Sampled path for each picked mesh edge, preserving selection order."""
        mesh = self.doc.mesh
        if mesh is None:
            return []
        refs = []
        for mid in mids:
            point = np.asarray(mid, float)
            best = min(mesh.edges, key=lambda edge: float(np.linalg.norm(edge.mid - point)), default=None)
            if best is None or float(np.linalg.norm(best.mid - point)) >= 1e-4:
                refs.append(None)
            else:
                refs.append([tuple(map(float, p)) for p in best.polyline])
        return refs

    @staticmethod
    def _point_path_distance(point: np.ndarray, path: list) -> float:
        points = np.asarray(path, float)
        if len(points) == 1:
            return float(np.linalg.norm(point - points[0]))
        starts, directions = points[:-1], points[1:] - points[:-1]
        lengths_sq = np.einsum("ij,ij->i", directions, directions)
        fractions = np.divide(np.einsum("ij,ij->i", point - starts, directions), lengths_sq,
                              out=np.zeros_like(lengths_sq), where=lengths_sq > 1e-18)
        closest = starts + np.clip(fractions, 0, 1)[:, None] * directions
        return float(np.min(np.linalg.norm(closest - point, axis=1)))

    def _fillet_result_edges(self, f: Feature) -> list:
        """Mids of the curve edges this fillet created in the current mesh.

        A fillet rounds the picked edge into arc geometry: the midpoints of
        those arcs sit a constant distance from the picked edge's original
        line (r(√2-1) for a 90° edge, growing toward r on acute angles).
        Matching each curve mid against every picked edge's line therefore
        separates one fillet's arcs from another's, so a drag on a rounded
        arc maps back to its own fillet feature.
        """
        if f.kind != "fillet" or self.doc.mesh is None:
            return []
        mids = [tuple(map(float, e.mid)) for e in self.doc.mesh.edges
                if len(e.polyline) > 2]
        if f.edge_refs:
            fillets = [(i, feature) for i, feature in enumerate(self.doc.features)
                       if feature.kind == "fillet" and feature.edge_refs]
            target_index = next((i for i, feature in fillets if feature is f), None)
            result = []
            for mid in mids:
                point = np.asarray(mid, float)
                owner, best_score = None, float("inf")
                for index, feature in fillets:
                    paths = [path for path in feature.edge_refs if path]
                    if not paths or feature.value <= 0:
                        continue
                    distance = min(self._point_path_distance(point, path) for path in paths)
                    score = distance / feature.value
                    if distance <= feature.value + 1e-3 and score < best_score:
                        owner, best_score = index, score
                if owner == target_index:
                    result.append(mid)
            return result
        if not f.edge_dirs:  # legacy feature (saved before edge_dirs existed)
            return mids
        res = []
        for m in mids:
            m_arr = np.asarray(m, float)
            best = 1e30
            for mid_e, dir_e in zip(f.edges, f.edge_dirs):
                mid_arr = np.asarray(mid_e, float)
                dir_arr = np.asarray(dir_e, float)
                t = float(np.dot(m_arr - mid_arr, dir_arr))
                d = float(np.linalg.norm((m_arr - mid_arr) - t * dir_arr))
                if d < best:
                    best = d
            if best <= f.value + 1e-3:
                res.append(m)
        return res

    def _refresh_fillet_result_edges(self):
        """Refresh every fillet's ``result_edges`` from the current mesh.

        Fillets are found by matching a grabbed edge mid against ``result_edges``.
        Only the edited feature used to be refreshed, so after a second fillet
        (or any rebuild) earlier fillets kept mids that no longer exist — and
        a saved model kept them forever.  Re-derive for all fillets on every
        rebuild and after load.
        """
        for f in self.doc.features:
            if f.kind == "fillet":
                f.result_edges = self._fillet_result_edges(f)

    def _find_fillet_by_edge(self, mid: tuple) -> int | None:
        """Index of the fillet feature whose rounded edges contain `mid`, else None."""
        m = np.asarray(mid, float)
        best, best_d = None, 1e-4  # require an exact-ish match (within ~0.1mm)
        for i, f in enumerate(self.doc.features):
            if f.kind != "fillet" or not f.result_edges:
                continue
            for re in f.result_edges:
                d = float(np.linalg.norm(np.asarray(re, float) - m))
                if d < best_d:
                    best, best_d = i, d
        return best

    def on_fillet_commit(self, radius: float, mids: list, kind: str = "fillet"):
        """Finalize an edge drag: commit a new fillet/chamfer, or edit an existing fillet.

        ``kind`` is "fillet" when the drag pushed toward the body and "chamfer" when it pulled
        away; dragging an existing fillet the other way turns it into a chamfer.

        - New fillet: the live temp feature is finalized in place; if the fillet
          is still failing at release the temp feature is rolled back so the model
          returns to the pre-drag state.
        - Edit (existing fillet): the feature's radius is changed in place; if the
          new radius makes the fillet fail the radius is reverted to its pre-drag
          value; dragging to ~0 removes the fillet.
        """
        mids = [tuple(map(float, m)) for m in mids]
        self._value_feature = None
        f = self._fillet_commit_feature(radius, mids, kind)
        # The value box stays on the model only if the feature exists at the dragged size.
        if (f is not None and any(x is f for x in self.doc.features) and abs(f.value - radius) < 1e-9
                and f.kind == kind):
            self._value_feature = f
        else:
            self.view.dismiss_value_box()

    def _fillet_commit_feature(self, radius: float, mids: list, kind: str = "fillet"):
        """on_fillet_commit's work; returns the fillet/chamfer feature it made or edited, if any."""
        if self._fillet_live_idx is not None:
            idx = self._fillet_live_idx
            f = self.doc.features[idx]
            if self._fillet_is_mod:
                if radius > 1e-4:
                    f.kind = kind
                    f.value = float(radius)
                    f.version += 1
                    self.rebuild()
                    if idx in self.doc.errors:
                        # New size invalid: back to the pre-drag fillet.
                        f.value = float(self._fillet_mod_prev or f.value)
                        f.kind = self._fillet_mod_prev_kind or "fillet"
                        f.version += 1
                        self.rebuild()
                        f.result_edges = self._fillet_result_edges(f)
                        QMessageBox.warning(self, f"Edit {kind}", "Reverted (it failed at the new size).")
                    else:
                        f.result_edges = self._fillet_result_edges(f)
                else:
                    # Dragged to ~0: remove the fillet.
                    self.doc.features.pop(idx)
                    self.rebuild()
            else:
                if radius > 1e-4:
                    f.kind = kind
                    f.value = float(radius)
                    f.version += 1
                    self.rebuild()
                    if idx in self.doc.errors:
                        self.doc.features.pop(idx)
                        self.rebuild()
                    else:
                        f.result_edges = self._fillet_result_edges(f)
                else:
                    self.doc.features.pop(idx)
                    self.rebuild()
            self._fillet_live_idx = None
            self._fillet_preview_r = None
            self._fillet_is_mod = False
            self._fillet_mod_prev = None
            self._fillet_mod_prev_kind = None
            self._fillet_preview_kind = None
            return f
        if radius > 1e-4:
            feature = Feature(kind, value=float(radius), edges=mids,
                              edge_dirs=self._fillet_edge_dirs(mids), edge_refs=self._fillet_edge_refs(mids))
            if self.commit(feature):
                feature.result_edges = self._fillet_result_edges(feature)
                return feature
        return None

    def on_fillet_preview(self, radius: float, mids: list, kind: str = "fillet"):
        """Live fillet/chamfer: keep the feature in the history while dragging.

        ``kind`` follows the drag direction (push: fillet, pull: chamfer) and may change
        mid-drag; the live feature switches with it.

        The first move decides whether the drag edits an existing fillet (the
        selected edge's mid matches a fillet feature's result_edges) or creates
        a new one.  Either way the radius is changed in place and the model is
        rebuilt (throttled to >= 0.05mm).  Editing also refreshes the fillet's
        result_edges so the next drag can find it again.
        """
        mids = [tuple(map(float, m)) for m in mids]
        if self._fillet_live_idx is None:
            self._push_undo()  # remember the pre-drag history
            mod_idx = self._find_fillet_by_edge(mids[0])
            if mod_idx is not None:
                self._fillet_live_idx = mod_idx
                self._fillet_is_mod = True
                self._fillet_mod_prev = self.doc.features[mod_idx].value
                self._fillet_mod_prev_kind = self.doc.features[mod_idx].kind
            else:
                self.doc.features.append(Feature(kind, value=float(radius), edges=mids,
                                                 edge_dirs=self._fillet_edge_dirs(mids),
                                                 edge_refs=self._fillet_edge_refs(mids)))
                self._fillet_live_idx = len(self.doc.features) - 1
            self._fillet_preview_r = None
        elif (self._fillet_preview_r is not None and abs(radius - self._fillet_preview_r) < 0.05
              and kind == self._fillet_preview_kind):
            return  # not worth a rebuild this frame
        f = self.doc.features[self._fillet_live_idx]
        f.kind = kind
        f.value = float(radius)
        f.version += 1
        sel_edges = set(self.view.sel_edges)
        sel_faces = set(self.view.sel_faces)
        self.rebuild()  # runs the kernel + refresh (clears selection); re-apply it
        self.view.sel_edges = sel_edges
        self.view.sel_faces = sel_faces
        if self._fillet_is_mod:
            f.result_edges = self._fillet_result_edges(f)
        self._fillet_preview_r = radius
        self._fillet_preview_kind = kind

    def _edge_feature(self, kind):
        if not self.view.sel_edges:
            QMessageBox.information(self, kind.title(), "Select one or more edges first (Ctrl+click for several).")
            return
        mids = [tuple(map(float, self.doc.mesh.edges[i].mid)) for i in sorted(self.view.sel_edges)]
        dirs = self._fillet_edge_dirs(mids) if kind == "fillet" else []
        refs = self._fillet_edge_refs(mids) if kind == "fillet" else []
        f = Feature(kind, value=1.0, edges=mids, edge_dirs=dirs, edge_refs=refs)
        self._value_session(f, kind.title(), "Radius" if kind == "fillet" else "Size", new=True, lo=1e-4, hi=1e4)

    def fillet(self):
        self._edge_feature("fillet")

    def chamfer(self):
        self._edge_feature("chamfer")

    def edit_feature(self, item):
        i = self.features.row(item)
        f = self.doc.features[i]
        if f.kind == "sketch":
            self.enter_sketch(i)
            return
        if f.kind == "extrude":
            self._extrude_session(f, "Edit extrude", new=False)
            return
        if f.kind == "revolve":
            self._revolve_session(f, "Edit revolve", new=False)
            return
        if f.kind == "pushpull":
            self._value_session(f, "Edit push/pull", "Distance (+ out)", new=False, lo=-1e5, hi=1e5, nonzero=True)
            return
        if f.kind in ("fillet", "chamfer"):
            self._value_session(f, f"Edit {f.kind}", "Radius" if f.kind == "fillet" else "Size",
                                new=False, lo=1e-4, hi=1e4)
            return
        self._push_undo()
        if f.kind == "sweep":
            paths = [(j, f"{j}: {g.name}") for j, g in enumerate(self.doc.features)
                     if g.kind == "sketch" and j != f.sketch_ref and g.sketch.lines]
            dlg = SweepDialog(self, paths, f.path_ref, f.cut)
            if not dlg.exec():
                return
            old = (f.path_ref, f.cut)
            f.path_ref, f.cut = dlg.path_ref, dlg.cut
            f.version += 1
            self.rebuild()
            if i in self.doc.errors:
                QMessageBox.warning(self, "Operation failed", self.doc.errors[i])
                f.path_ref, f.cut = old
                f.version += 1
                self.rebuild()
            return
        elif f.kind == "loft":
            sections = [(j, f"{j}: {g.name}") for j, g in enumerate(self.doc.features)
                        if g.kind == "sketch" and len(g.sketch.regions()) == 1]
            dlg = LoftDialog(self, sections, f.section_refs, f.cut)
            if not dlg.exec():
                return
            new_refs = dlg.section_refs
            if len(new_refs) < 2:
                QMessageBox.information(self, "Loft", "Select at least two section sketches.")
                return
            old = (f.section_refs, f.cut)
            f.section_refs, f.cut = new_refs, dlg.cut
            f.version += 1
            self.rebuild()
            if i in self.doc.errors:
                QMessageBox.warning(self, "Operation failed", self.doc.errors[i])
                f.section_refs, f.cut = old
                f.version += 1
                self.rebuild()

    def toggle_layer(self, item):
        i = self.features.row(item)
        f = self.doc.features[i]
        self._push_undo()
        f.visible = item.checkState() == Qt.Checked
        if f.kind == "sketch":
            self.view.overlay_sketches = [g.sketch for g in self.doc.features if g.kind == "sketch" and g.visible]
            self.mark_dirty()
            self._update_regions()
        else:
            f.version += 1
            self.rebuild()

    def delete_feature(self):
        i = self.features.currentRow()
        if i < 0:
            return
        self._push_undo()
        self.doc.delete(i)
        self._clear_sketch_history()
        self.rebuild()

    # ---- files
    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Discard unsaved changes?")
        return r == QMessageBox.Yes

    def _load(self, doc, path):
        self._cancel_session()
        self.doc, self.path = doc, path
        self.sketch_index = -1
        self._undo.clear()
        self._redo.clear()
        self._clear_sketch_history()
        self.set_mode("model")
        self.mark_dirty(False)
        self._apply_units()
        self.view.frame_all()
        self._refresh_fillet_result_edges()

    def file_new(self):
        if self._confirm_discard():
            self._load(Document(), None)

    def file_open(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", FILE_FILTER)
        if path:
            try:
                self._load(Document.load(path), path)
            except (OSError, ValueError, KeyError) as e:
                QMessageBox.warning(self, "Open failed", str(e))

    def file_save(self):
        if self.path is None:
            return self.file_save_as()
        try:
            self.doc.save(self.path)
            self.mark_dirty(False)
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def file_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self.path or "model.rcad", FILE_FILTER)
        if path:
            self.path = path if path.endswith(".rcad") else path + ".rcad"
            self.file_save()

    def file_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "model.step", "STEP (*.step *.stp);;STL (*.stl)")
        if path:
            try:
                self.doc.export(path)
            except Exception as e:  # kernel or I/O error
                QMessageBox.warning(self, "Export failed", str(e))

    def closeEvent(self, e):
        e.accept() if self._confirm_discard() else e.ignore()


def main():
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
