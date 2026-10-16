from __future__ import annotations

import math
import sys

import numpy as np
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox, QFileDialog,
    QFormLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QToolBar, QWidget,
    QVBoxLayout,
)

from .kernel import Document, Feature
from .camera import ray_plane
from .sketch import CONSTRAINT_KINDS, Constraint, Sketch, SketchPlane, standard_plane
from .units import MM, UNITS
from .theme import CHOICES, apply_choice
from .viewport import Viewport

FILE_FILTER = "rCad project (*.rcad)"


class ExtrudeDialog(QDialog):
    def __init__(self, parent, distance=10.0, auto=True, cut=False, show_mode=True, unit=MM):
        super().__init__(parent)
        self.unit = unit
        self.setWindowTitle("⇈")
        self.dist = QDoubleSpinBox()
        self.dist.setRange(-1e5, 1e5)
        self.dist.setDecimals(unit.decimals)
        self.dist.setSuffix(f" {unit.name}")
        self.dist.setValue(unit.from_mm(distance))
        self._orig, self._shown = distance, self.dist.value()
        self.mode = QComboBox()
        self.mode.addItems(["Auto (into the body = cut, away = add)", "Add (join)", "Cut (subtract)"])
        self.mode.setCurrentIndex(0 if auto else 2 if cut else 1)
        form = QFormLayout(self)
        form.addRow("Distance (negative = opposite side)", self.dist)
        if show_mode:
            form.addRow("Operation", self.mode)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    @property
    def distance_mm(self) -> float:
        if self.dist.value() == self._shown:
            return self._orig  # untouched: avoid display-rounding drift
        return self.unit.to_mm(self.dist.value())

    @property
    def auto(self) -> bool:
        return self.mode.currentIndex() == 0

    @property
    def cut(self) -> bool:
        return self.mode.currentIndex() == 2


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

        self.view = Viewport()
        self.setCentralWidget(self.view)
        self.view.clicked.connect(self.on_click)
        self.view.hovered.connect(self.on_hover)
        self.view.scrolled.connect(self.on_wheel_adjust)

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
        self._build_model_toolbar()
        self._build_sketch_toolbar()
        self._build_constraint_toolbar()

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
        file_tb.addWidget(QLabel(" Units: "))
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

    def _build_model_toolbar(self):
        self.model_tb = QToolBar("Model")
        self.addToolBar(self.model_tb)
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["XY", "XZ", "YZ", "Selected face"])
        self.model_tb.addWidget(QLabel(" Sketch on: "))
        self.model_tb.addWidget(self.plane_combo)
        for text, slot, tip in [
            ("✏️", self.new_sketch, "Start a sketch on the chosen plane or the selected planar face"),
            ("⇈", self.extrude, "Extrude the selected (or last) sketch"),
            ("↕️", self.pushpull, "Move the selected planar face along its normal"),
            ("◐", self.fillet, "Round the selected edges"),
            ("◺", self.chamfer, "Bevel the selected edges"),
            ("🗑", self.delete_feature, "Delete the highlighted feature"),
            ("⛶", self.view.frame_all, "Fit the model in view"),
        ]:
            self.model_tb.addAction(self._act(text, slot, tip))

    def _build_sketch_toolbar(self):
        self.sketch_tb = QToolBar("Sketch")
        self.addToolBar(Qt.TopToolBarArea, self.sketch_tb)
        group = QActionGroup(self)
        self.tool_actions = {}
        for name, text in [("select", "⌖"), ("line", "╱"), ("rect", "⬜"), ("circle", "◯")]:
            a = self._act(text, lambda _=False, n=name: self.set_tool(n), checkable=True)
            group.addAction(a)
            a.setChecked(name == "select")
            self.tool_actions[name] = a
            self.sketch_tb.addAction(a)
        self.sketch_tb.addSeparator()
        self.snap = QDoubleSpinBox()
        self.snap.setRange(0, 1000)
        self.snap.setValue(1.0)
        self.snap.setPrefix("snap ")
        self.snap.setToolTip("Grid snap (0 = off)")
        self.sketch_tb.addWidget(self.snap)
        self.sketch_tb.addSeparator()
        self.sketch_tb.addAction(self._act("🗑 sel", self.delete_selection, "Delete selected sketch items"))
        self.sketch_tb.addAction(self._act("🗑 ct", self.delete_constraint))
        self.sketch_tb.addSeparator()
        self.sketch_tb.addAction(self._act("⇈", self.extrude_from_sketch,
                                           "Extrude the highlighted regions (Select tool: click a region to highlight it)"))
        self.sketch_tb.addAction(self._act("✓ Done", self.finish_sketch))

    def _build_constraint_toolbar(self):
        self.addToolBarBreak(Qt.TopToolBarArea)
        self.constraint_tb = QToolBar("Constraints")
        self.addToolBar(Qt.TopToolBarArea, self.constraint_tb)
        for text, kind in [("Horizontal", "horizontal"), ("Vertical", "vertical"), ("Coincident", "coincident"),
                           ("Parallel", "parallel"), ("Perpend.", "perpendicular"), ("Equal", "equal"),
                           ("Distance", "distance"), ("Diameter", "diameter"), ("Fixed", "fixed"),
                           ("Point on line", "point_on_line")]:
            self.constraint_tb.addAction(self._act(text, lambda _=False, k=kind: self.add_constraint(k),
                                                   f"Add a {kind} constraint to the selection"))

    # ---- units
    @property
    def unit(self):
        return UNITS[self.doc.units]

    def _apply_units(self):
        u = self.unit
        self.unit_combo.blockSignals(True)
        self.unit_combo.setCurrentText(u.name)
        self.unit_combo.blockSignals(False)
        self.snap.setDecimals(u.decimals)
        self.snap.setSuffix(f" {u.name}")
        self.snap.setValue(u.snap)
        self.view.unit_mm = u.mm
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

    def ask_length(self, title, label, mm_value, lo_mm=1e-6, hi_mm=1e6):
        """Prompt for a length in the current units; returns (value in mm, accepted)."""
        u = self.unit
        v, ok = QInputDialog.getDouble(self, title, f"{label} ({u.name})", u.from_mm(mm_value),
                                       u.from_mm(lo_mm), u.from_mm(hi_mm), u.decimals)
        if v == round(u.from_mm(mm_value), u.decimals):
            return mm_value, ok  # untouched: avoid display-rounding drift
        return u.to_mm(v), ok

    # ---- state / refresh
    def set_mode(self, mode):
        self.mode = mode
        sketching = mode == "sketch"
        self.model_tb.setVisible(not sketching)
        self.sketch_tb.setVisible(sketching)
        self.constraint_tb.setVisible(sketching)
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

    def rebuild(self):
        self.doc.rebuild()
        self.mark_dirty()
        self.refresh()
        if self.doc.errors:
            i, msg = next(iter(self.doc.errors.items()))
            self.statusBar().showMessage(f"Feature {i} failed: {msg}", 8000)

    def commit(self, feature: Feature) -> bool:
        """Append a feature; drop it again if the kernel rejects it."""
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
        n_sketch = sum(1 for f in self.doc.features if f.kind == "sketch") + 1
        self.doc.features.append(Feature("sketch", f"Sketch {n_sketch}", sketch=Sketch(plane)))
        self.enter_sketch(len(self.doc.features) - 1)

    def enter_sketch(self, index):
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
        self.tool = name
        self.pending = None
        self.hover_region = None
        self.view.preview = []
        self.view.sketch_sel.clear()
        self._update_regions()

    def cancel(self):
        if self.mode == "sketch" and self.pending is None and self.tool != "select":
            self.set_tool("select")  # Esc with nothing in progress leaves the drawing tool
            return
        self.pending = None
        self.view.preview = []
        self.view.sketch_sel.clear()
        self.view.sel_faces.clear()
        self.view.sel_edges.clear()
        for f in self.doc.features:
            if f.kind == "sketch" and (self.mode == "model" or f is self.doc.features[self.sketch_index]):
                f.sketch.selection = []
        self._update_regions()

    def _snap(self, x, y):
        """-> (existing point index or None, (u, v)) for the cursor, or None if the ray misses the plane."""
        pi = self.view.nearest_sketch_point(x, y)
        if pi is not None:
            return pi, tuple(self.sketch.points[pi])
        uv = self.view.sketch_uv(x, y)
        if uv is None:
            return None
        g = self.snap.value() * self.unit.mm
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
            fills += [(f.sketch.region_triangles(r), (1.0, 0.55, 0.15, 0.45)) for r in sorted(chosen)]
            if self.hover_region is not None and self.hover_region[0] == i and self.hover_region[1] not in chosen:
                fills.append((f.sketch.region_triangles(self.hover_region[1]), (0.95, 0.9, 0.3, 0.35)))
        self.view.region_fill = fills
        self.view.update()

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
        ok = self.sketch.solve()
        if not ok:
            self.statusBar().showMessage(f"Solver: {self.sketch.status}", 6000)
        self.refresh_constraints()
        self.mark_dirty()
        self._update_regions()
        return ok

    def _sketch_click(self, x, y, mods):
        sk = self.sketch
        if self.tool == "select":
            hit = self.view.pick_sketch(x, y)
            sel = self.view.sketch_sel
            multi = bool(mods & Qt.ControlModifier)
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
        pi, uv = snap
        if self.tool == "line":
            if self.pending is None:
                self.pending = (pi, uv)
                return
            a_idx, a_uv = self.pending
            if a_idx is None and pi is None and math.dist(a_uv, uv) < 1e-9:
                return
            a = a_idx if a_idx is not None else sk.add_point(*a_uv)
            b = pi if pi is not None else sk.add_point(*uv)
            if a != b:
                sk.add_line(a, b)
            self.pending = (b, tuple(sk.points[b])) if pi is None else None
            self._solve_sketch()
        elif self.tool == "rect":
            if self.pending is None:
                self.pending = (pi, uv)
                return
            (u0, v0), (u1, v1) = self.pending[1], uv
            if abs(u1 - u0) > 1e-9 and abs(v1 - v0) > 1e-9:
                sk.add_rect(u0, v0, u1, v1)
                self._solve_sketch()
            self.set_tool("select")  # so the new region can be highlighted right away
        elif self.tool == "circle":
            if self.pending is None:
                self.pending = (pi, uv)
                return
            c_idx, c_uv = self.pending
            r = math.dist(c_uv, uv)
            if r > 1e-9:
                c = c_idx if c_idx is not None else sk.add_point(*c_uv)
                sk.add_circle(c, r)
                self._solve_sketch()
            self.set_tool("select")

    def _update_preview(self, x, y):
        pl = self.sketch.plane
        snap = self._snap(x, y)
        self.view.preview = []
        if snap is None or self.pending is None:
            self.view.update()
            return
        uv = snap[1]
        p0 = self.pending[1]
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
            elif kind == "distance":
                ends = pts if len(pts) == 2 and not lines else list(sk.lines[lines[0]]) if len(lines) == 1 and not pts else None
                if ends:
                    cur = math.dist(sk.points[ends[0]], sk.points[ends[1]])
                    value, ok = self.ask_length("Distance", "Distance", cur)
                    if not ok:
                        return
                    new = [Constraint(kind, ends, value)]
            elif kind == "diameter" and len(circles) == 1 and not pts and not lines:
                cur = 2 * sk.circles[circles[0]][1]
                value, ok = self.ask_length("Diameter", "Diameter", cur)
                if not ok:
                    return
                new = [Constraint(kind, circles, value)]
            elif kind == "fixed" and pts and not lines and not circles:
                new = [Constraint(kind, [p]) for p in pts]
            elif kind == "point_on_line" and len(pts) == 1 and len(lines) == 1 and not circles:
                new = [Constraint(kind, [pts[0], lines[0]])]
        except IndexError:
            new = []
        if not new:
            self.statusBar().showMessage(f"Select the right items for '{kind}' ({CONSTRAINT_KINDS[kind][0]}).", 6000)
            return
        sk.constraints += new
        if not self._solve_sketch():
            del sk.constraints[-len(new):]
            self.sketch.solve()
            self.refresh_constraints()

    def edit_constraint(self, item):
        i = self.constraints.row(item)
        c = self.sketch.constraints[i]
        if c.value is None:
            return
        v, ok = self.ask_length("Dimension", "Value", c.value)
        if ok:
            old, c.value = c.value, v
            if not self._solve_sketch():
                c.value = old
                self.sketch.solve()
                self.refresh_constraints()

    def delete_constraint(self):
        i = self.constraints.currentRow()
        if i >= 0:
            self.sketch.remove_constraint(i)
            self._solve_sketch()

    def delete_selection(self):
        sk = self.sketch
        for kind in ("l", "c", "p"):  # deleting points last keeps the other indices stable
            for idx in sorted((i for k, i in self.view.sketch_sel if k == kind), reverse=True):
                sk.delete(kind, idx)
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
        f = self.doc.features[feat_i]
        step = 0.5 if not (mods & Qt.ShiftModifier) else 0.1
        f.value = max(1e-4, f.value + (delta / 120.0) * step)
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
        dlg = ExtrudeDialog(self, show_mode=self.doc.part is not None, unit=self.unit)
        dlg.setWindowTitle(f"Extrude {len(profiles)} highlighted region(s)" if profiles else "Extrude all regions")
        if dlg.exec():
            self.commit(Feature("extrude", sketch_ref=i, value=dlg.distance_mm, cut=dlg.cut, auto=dlg.auto,
                                profiles=profiles))

    def pushpull(self):
        if len(self.view.sel_faces) != 1:
            QMessageBox.information(self, "↕️", "Select exactly one planar face.")
            return
        info = self.doc.mesh.faces[next(iter(self.view.sel_faces))]
        if info.normal is None:
            QMessageBox.information(self, "↕️", "Only planar faces are supported.")
            return
        v, ok = self.ask_length("↕️", "Distance, positive = outward", 5.0, -1e5, 1e5)
        if ok and v != 0:
            self.commit(Feature("pushpull", value=v, face=(tuple(map(float, info.center)),
                                                           tuple(map(float, info.normal)))))

    def _edge_feature(self, kind):
        if not self.view.sel_edges:
            QMessageBox.information(self, kind.title(), "Select one or more edges first (Ctrl+click for several).")
            return
        mids = [tuple(map(float, self.doc.mesh.edges[i].mid)) for i in sorted(self.view.sel_edges)]
        v, ok = self.ask_length(kind.title(), "Size", 1.0, 1e-4, 1e4)
        if ok:
            self.commit(Feature(kind, value=v, edges=mids))

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
            dlg = ExtrudeDialog(self, f.value, f.auto, f.cut, unit=self.unit)
            if not dlg.exec():
                return
            new = (dlg.distance_mm, dlg.cut, dlg.auto)
        else:
            v, ok = self.ask_length(f.kind.title(), "Value", f.value, -1e5, 1e5)
            if not ok or v == 0:
                return
            new = (v, f.cut, f.auto)
        old = (f.value, f.cut, f.auto)
        f.value, f.cut, f.auto = new
        self.rebuild()
        if i in self.doc.errors:
            QMessageBox.warning(self, "Operation failed", self.doc.errors[i])
            f.value, f.cut, f.auto = old
            self.rebuild()

    def toggle_layer(self, item):
        i = self.features.row(item)
        f = self.doc.features[i]
        f.visible = item.checkState() == Qt.Checked
        if f.kind == "sketch":
            self.view.overlay_sketches = [g.sketch for g in self.doc.features if g.kind == "sketch" and g.visible]
            self.mark_dirty()
            self._update_regions()
        else:
            self.rebuild()

    def delete_feature(self):
        i = self.features.currentRow()
        if i < 0:
            return
        self.doc.delete(i)
        self.rebuild()

    # ---- files
    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Discard unsaved changes?")
        return r == QMessageBox.Yes

    def _load(self, doc, path):
        self.doc, self.path = doc, path
        self.sketch_index = -1
        self.set_mode("model")
        self.mark_dirty(False)
        self._apply_units()
        self.view.frame_all()

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
