"""rCad's main window: layout, menus, sketch and model modes, undo and redo."""
from __future__ import annotations

import sys

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .constraints import ConstraintsMixin
from .dialogs import ExpressionDialog, LoftDialog, SweepDialog, VariablesDialog
from .files import FilesMixin
from .kernel import Document, Feature
from .model_tools import ModelToolsMixin
from .quick_buttons import QuickButtons
from .recovery import Autosaver
from .selection import SelectionMixin
from .sessions import SessionsMixin
from .sketch import end_direction, Sketch, SketchPlane, standard_plane
from .sketch_tools import SketchToolsMixin
from .theme import apply_choice, CHOICES
from .tool_bar import ToolSettingsBar
from .tool_line import ToolLine
from .units import MM, UNITS
from .value_box import FieldSet, NumberField
from .viewport import Viewport

# names other code and tests reach through rcad.app
__all__ = ['MainWindow', 'main', 'Document', 'SweepDialog', 'LoftDialog', 'ExpressionDialog', 'VariablesDialog']


class MainWindow(FilesMixin, SelectionMixin, ConstraintsMixin, SketchToolsMixin, ModelToolsMixin, SessionsMixin, QMainWindow):
    def __init__(self):
        super().__init__()
        self.doc = Document()
        self.path: str | None = None
        self.autosaver = Autosaver()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(self.AUTOSAVE_MS)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()
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
        self.view.double_clicked.connect(self.on_double_click)
        self.view.box_selected.connect(self.on_box_select)
        self.view.hovered.connect(self.on_hover)
        self.view.scrolled.connect(self.on_wheel_adjust)
        self.view.extrude_commit.connect(self.on_extrude_commit)
        self.view.region_extrude.connect(self.on_region_extrude)
        self.view.region_drag = self._region_drag_target
        self.view.planes_toggled.connect(self._on_planes_toggled)
        self.view.construction_key.connect(self.toggle_construction)
        self.view.dim_clicked.connect(self.edit_dimension)
        self.view.dim_menu.connect(self._dimension_menu)
        self.view.mark_clicked.connect(self.select_constraint)
        self.view.delete_key.connect(self.delete_selection)
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
        self._draw_snap = None  # the Snap behind _draw_mouse
        self._snap_text = ""  # what the pointer snaps to, for the status bar
        self._pending_snap = None  # the Snap the pending first point was placed with
        self._arc_start = None  # arc tool: (point index or None, uv, Snap) of the start, after the centre
        self._tarc_from = None  # tangent arc tool: (("l" | "a", index), unit direction) it leaves along
        self._chain_item = None  # the newest line or arc of the chain being drawn
        self._trim_stroke = []  # trim tool: sketch points along the current drag
        self._arc_sweep = 0.0  # arc tool: degrees swept by the pointer from the start (+ counter-clockwise)
        self._arc_angle = 0.0  # arc tool: the pointer's last angle around the centre (degrees)
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
        self.selection_menu = None  # the last right-click tools menu
        self.dimension_menu = None  # the last right-click menu of a dimension on the sketch
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
        QShortcut(QKeySequence.SelectAll, self, self.select_all)
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
            act("✥ Move body", self.move_bodies, "Move or rotate the selected bodies (double-click selects one)"),
            act("⇋ Mirror body", self.mirror_bodies, "Mirror the selected bodies across a plane"),
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
                           ("circle", "◯ Circle"), ("arc", "◠ Arc"),
                           ("tarc", "◡ Tangent arc"), ("trim", "✂ Trim")]:
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
                ("Tangent", "tangent"), ("Concentric", "concentric"), None,
                ("Distance", "distance"), ("Spacing", "spacing"), ("Diameter", "diameter"), ("Radius", "radius"),
                ("Expression", "expression"), None,
                ("Symmetry", "symmetry"), ("Fixed", "fixed"), ("Point on line", "point_on_line")]:
            if item is None:
                constraints.append(None)
                continue
            text, kind = item
            constraints.append(act(text, lambda _=False, k=kind: self.add_constraint(k),
                                   tips.get(text, f"Add a {kind} constraint to the selection")))
        self.marks_action = act("👁 Show constraint marks", self._toggle_marks,
                                "Small symbols beside the geometry each constraint holds", checkable=True)
        self.marks_action.setChecked(True)
        constraints += [None, self.marks_action]
        line.add_group("sketch", "Constrain", constraints)
        line.add_group("sketch", "Modify", [
            act("┄ Construction (Q)", self.toggle_construction,
                "Make the selected lines, circles and arcs reference geometry (or normal again): dashed, "
                "used for snapping and constraints, never part of a region"),
            act("🗑 Delete selection", self.delete_selection, "Delete selected sketch items"),
            act("🗑 Delete constraint", self.delete_constraint, "Delete the constraint highlighted in the list"),
        ])
        line.add_separator("sketch")
        self.snap = NumberField(1.0, "length", MM, lo=0.0, hi=1e4)
        self.snap.setFixedWidth(80)
        self.snap.HINT = "Grid snap step when zoomed in; zooming out scales it up (0 = off). " + NumberField.HINT
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
        # a tag at the top of the view naming what is selected
        self.sel_tag = QLabel(self.view)
        self.sel_tag.setObjectName("selTag")
        self.sel_tag.setStyleSheet("#selTag { color: #ffffff; background: rgba(32,34,40,200); "
                                   "border-radius: 10px; padding: 2px 10px; }")
        self.sel_tag.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.sel_tag.hide()
        self.view.paint_hooks.append(self._place_tag)

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
        self._pending_snap = None
        self._arc_start = None
        self.view.snap_mark = None
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
                                   f"{len(sk.circles)} circles, {len(sk.arcs)} arcs")

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
        restored = Sketch.from_dict(d)
        for name in ("plane", "points", "lines", "circles", "arcs", "construction", "constraints", "selection"):
            setattr(sk, name, getattr(restored, name))
        sk.solve()
        self.pending = None
        self._pending_snap = None
        self._arc_start = None
        self.view.snap_mark = None
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
        # switching between Line and Tangent arc mid-chain carries on from the chain's end
        chain = self.pending[0] if (self.pending is not None and self.mode == "sketch" and name != self.tool
                                    and {self.tool, name} <= {"line", "tarc"}) else None
        if chain is None:
            self._chain_item = None
        self.tool_actions[name].setChecked(True)
        self.draw_button.setText(f"Draw: {self.tool_actions[name].text().split(' ', 1)[1]} ▾")
        self.tool = name
        self.view.marks_clickable = name == "select"
        self.view.trim_preview = []
        self._trim_stroke = []
        self.pending = None
        self._pending_snap = None
        self._arc_start = None
        self.view.snap_mark = None
        self.fields.stop()
        self.hover_region = None
        self.view.preview = []
        self.view.sketch_sel.clear()
        self._update_regions()
        if chain is not None:
            self._resume_chain(chain)

    def _resume_chain(self, p: int):
        """Go on drawing from point p with the current tool (a tangent arc follows the chain's last curve)."""
        if self.tool == "tarc":
            self._tarc_from = end_direction(self.sketch, p, prefer=self._chain_item)
            if self._tarc_from is None:
                return
        self.pending = (p, tuple(self.sketch.points[p]))
        self._start_fields()
        self.statusBar().showMessage("Carrying on from the end of the chain.", 3000)

    def cancel(self):
        if self._session is not None:  # Esc: the open tool is cancelled and the feature put back
            self._cancel_session()
            return
        if self.mode == "sketch" and self.pending is None and self.tool != "select":
            self.set_tool("select")  # Esc with nothing in progress leaves the drawing tool
            return
        ending_chain = self.mode == "sketch" and self.tool == "line" and self.pending is not None
        self.pending = None
        self._pending_snap = None
        self._arc_start = None
        self.view.snap_mark = None
        self.fields.stop()
        self.view.preview = []
        self.view.sketch_sel.clear()
        self.view.sel_faces.clear()
        self.view.sel_edges.clear()
        self.view.sel_plane = None
        self.view.sel_constraint = None
        self.view.conflicts = set()
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

    def _delete_feature_at(self, i: int):
        if not 0 <= i < len(self.doc.features):
            return
        self._push_undo()
        self.doc.delete(i)
        self._clear_sketch_history()
        self.rebuild()

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
        if f.kind == "move":
            self._move_session(f, "Edit move", new=False)
            return
        if f.kind == "mirror":
            self._mirror_session(f, "Edit mirror", new=False)
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
                        if g.kind == "sketch" and len(g.sketch.profile_faces()) == 1]
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


def main():
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    QTimer.singleShot(0, w.offer_recovery)
    sys.exit(app.exec())
