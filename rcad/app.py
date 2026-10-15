from __future__ import annotations

import math
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence, QShortcut, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox, QFileDialog,
    QFormLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QToolBar, QWidget,
    QVBoxLayout,
)

from .kernel import Document, Feature
from .sketch import CONSTRAINT_KINDS, Constraint, Sketch, SketchPlane, standard_plane
from .viewport import Viewport

FILE_FILTER = "rCad project (*.rcad)"


class ExtrudeDialog(QDialog):
    def __init__(self, parent, distance=10.0, cut=False, show_cut=True):
        super().__init__(parent)
        self.setWindowTitle("Extrude")
        self.dist = QDoubleSpinBox()
        self.dist.setRange(-1e5, 1e5)
        self.dist.setDecimals(3)
        self.dist.setValue(distance)
        self.cut = QCheckBox("Cut (subtract from body)")
        self.cut.setChecked(cut)
        form = QFormLayout(self)
        form.addRow("Distance (negative = opposite side)", self.dist)
        if show_cut:
            form.addRow(self.cut)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)


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

        self.view = Viewport()
        self.setCentralWidget(self.view)
        self.view.clicked.connect(self.on_click)
        self.view.hovered.connect(self.on_hover)

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
        self.refresh()

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
        file_tb = self.addToolBar("File")
        for text, slot, sc in [("New", self.file_new, QKeySequence.New), ("Open", self.file_open, QKeySequence.Open),
                               ("Save", self.file_save, QKeySequence.Save),
                               ("Save As", self.file_save_as, QKeySequence.SaveAs),
                               ("Export STEP/STL", self.file_export, None)]:
            a = self._act(text, slot, shortcut=sc)
            file_tb.addAction(a)
            self.addAction(a)

        self.model_tb = QToolBar("Model")
        self.addToolBar(self.model_tb)
        self.plane_combo = QComboBox()
        self.plane_combo.addItems(["XY", "XZ", "YZ", "Selected face"])
        self.model_tb.addWidget(QLabel(" Sketch on: "))
        self.model_tb.addWidget(self.plane_combo)
        for text, slot, tip in [
            ("New Sketch", self.new_sketch, "Start a sketch on the chosen plane or the selected planar face"),
            ("Extrude", self.extrude, "Extrude the selected (or last) sketch"),
            ("Push/Pull", self.pushpull, "Move the selected planar face along its normal"),
            ("Fillet", self.fillet, "Round the selected edges"),
            ("Chamfer", self.chamfer, "Bevel the selected edges"),
            ("Delete Feature", self.delete_feature, "Delete the highlighted feature"),
            ("Frame", self.view.frame_all, "Fit the model in view"),
        ]:
            self.model_tb.addAction(self._act(text, slot, tip))

        self.sketch_tb = QToolBar("Sketch")
        self.addToolBar(Qt.TopToolBarArea, self.sketch_tb)
        group = QActionGroup(self)
        for name, text in [("select", "Select"), ("line", "Line"), ("rect", "Rectangle"), ("circle", "Circle")]:
            a = self._act(text, lambda _=False, n=name: self.set_tool(n), checkable=True)
            group.addAction(a)
            a.setChecked(name == "select")
            self.sketch_tb.addAction(a)
        self.sketch_tb.addSeparator()
        self.snap = QDoubleSpinBox()
        self.snap.setRange(0, 1000)
        self.snap.setValue(1.0)
        self.snap.setPrefix("snap ")
        self.snap.setToolTip("Grid snap (0 = off)")
        self.sketch_tb.addWidget(self.snap)
        self.sketch_tb.addSeparator()
        for text, kind in [("Horizontal", "horizontal"), ("Vertical", "vertical"), ("Coincident", "coincident"),
                           ("Parallel", "parallel"), ("Perpend.", "perpendicular"), ("Equal", "equal"),
                           ("Distance", "distance"), ("Diameter", "diameter"), ("Fixed", "fixed"),
                           ("Point on line", "point_on_line")]:
            self.sketch_tb.addAction(self._act(text, lambda _=False, k=kind: self.add_constraint(k),
                                               f"Add a {kind} constraint to the selection"))
        self.sketch_tb.addSeparator()
        self.sketch_tb.addAction(self._act("Delete selected", self.delete_selection, "Delete selected sketch items"))
        self.sketch_tb.addAction(self._act("Delete constraint", self.delete_constraint))
        self.sketch_tb.addAction(self._act("Finish Sketch", self.finish_sketch))

    # ---- state / refresh
    def set_mode(self, mode):
        self.mode = mode
        sketching = mode == "sketch"
        self.model_tb.setVisible(not sketching)
        self.sketch_tb.setVisible(sketching)
        self.constraint_dock.setVisible(sketching)
        self.view.left_orbits = not sketching
        self.view.camera.ortho = sketching
        self.view.sketch = self.doc.features[self.sketch_index].sketch if sketching else None
        self.view.sketch_sel.clear()
        self.view.preview = []
        self.pending = None

    def mark_dirty(self, dirty=True):
        self.dirty = dirty
        name = self.path or "untitled"
        self.setWindowTitle(f"rCad - {name}{' *' if dirty else ''}")

    def refresh(self):
        row = self.features.currentRow()
        self.features.blockSignals(True)
        self.features.clear()
        for i, f in enumerate(self.doc.features):
            text = f"{i}: {f.label()}"
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
        self.view.update()

    def refresh_constraints(self):
        self.constraints.clear()
        if self.mode != "sketch":
            return
        sk = self.sketch
        for i, c in enumerate(sk.constraints):
            self.constraints.addItem(f"{i}: {c.label()}")
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
        self.tool = name
        self.cancel()

    def cancel(self):
        self.pending = None
        self.view.preview = []
        self.view.sketch_sel.clear()
        self.view.sel_faces.clear()
        self.view.sel_edges.clear()
        self.view.update()

    def _snap(self, x, y):
        """-> (existing point index or None, (u, v)) for the cursor, or None if the ray misses the plane."""
        pi = self.view.nearest_sketch_point(x, y)
        if pi is not None:
            return pi, tuple(self.sketch.points[pi])
        uv = self.view.sketch_uv(x, y)
        if uv is None:
            return None
        g = self.snap.value()
        return None, ((round(uv[0] / g) * g, round(uv[1] / g) * g) if g > 0 else uv)

    def _solve_sketch(self):
        ok = self.sketch.solve()
        if not ok:
            self.statusBar().showMessage(f"Solver: {self.sketch.status}", 6000)
        self.refresh_constraints()
        self.mark_dirty()
        self.view.update()
        return ok

    def _sketch_click(self, x, y, mods):
        sk = self.sketch
        if self.tool == "select":
            hit = self.view.pick_sketch(x, y)
            sel = self.view.sketch_sel
            if hit is None:
                if not mods & Qt.ControlModifier:
                    sel.clear()
            elif mods & Qt.ControlModifier:
                sel.symmetric_difference_update({hit})
            else:
                sel.clear()
                sel.add(hit)
            self.view.update()
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
            else:
                (u0, v0), (u1, v1) = self.pending[1], uv
                if abs(u1 - u0) > 1e-9 and abs(v1 - v0) > 1e-9:
                    sk.add_rect(u0, v0, u1, v1)
                    self._solve_sketch()
                self.pending = None
                self.view.preview = []
        elif self.tool == "circle":
            if self.pending is None:
                self.pending = (pi, uv)
            else:
                c_idx, c_uv = self.pending
                r = math.dist(c_uv, uv)
                if r > 1e-9:
                    c = c_idx if c_idx is not None else sk.add_point(*c_uv)
                    sk.add_circle(c, r)
                    self._solve_sketch()
                self.pending = None
                self.view.preview = []

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
        self.statusBar().showMessage(f"u={uv[0]:.3f}  v={uv[1]:.3f}")

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
                    value, ok = QInputDialog.getDouble(self, "Distance", "Distance:", round(cur, 4), 1e-6, 1e6, 4)
                    if not ok:
                        return
                    new = [Constraint(kind, ends, value)]
            elif kind == "diameter" and len(circles) == 1 and not pts and not lines:
                cur = 2 * sk.circles[circles[0]][1]
                value, ok = QInputDialog.getDouble(self, "Diameter", "Diameter:", round(cur, 4), 1e-6, 1e6, 4)
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
        v, ok = QInputDialog.getDouble(self, "Dimension", "Value:", c.value, 1e-6, 1e6, 4)
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
        face = None if edge is not None else v.pick_face(x, y)
        if not multi:
            v.sel_faces.clear()
            v.sel_edges.clear()
        if edge is not None:
            v.sel_edges.symmetric_difference_update({edge})
        elif face is not None:
            v.sel_faces.symmetric_difference_update({face})
        v.update()

    def on_hover(self, x, y):
        if self.mode == "sketch":
            if self.tool != "select":
                self._update_preview(x, y)
            return
        edge = self.view.pick_edge(x, y)
        face = None if edge is not None else self.view.pick_face(x, y)
        if (edge, face) != (self.view.hover_edge, self.view.hover_face):
            self.view.hover_edge, self.view.hover_face = edge, face
            self.view.update()

    def _sketch_for_extrude(self) -> int | None:
        row = self.features.currentRow()
        if 0 <= row < len(self.doc.features) and self.doc.features[row].kind == "sketch":
            return row
        idx = [i for i, f in enumerate(self.doc.features) if f.kind == "sketch"]
        return idx[-1] if idx else None

    def extrude(self):
        i = self._sketch_for_extrude()
        if i is None:
            QMessageBox.information(self, "Extrude", "Create a sketch with a closed profile first.")
            return
        dlg = ExtrudeDialog(self, show_cut=self.doc.part is not None)
        if dlg.exec():
            self.commit(Feature("extrude", sketch_ref=i, value=dlg.dist.value(), cut=dlg.cut.isChecked()))

    def pushpull(self):
        if len(self.view.sel_faces) != 1:
            QMessageBox.information(self, "Push/Pull", "Select exactly one planar face.")
            return
        info = self.doc.mesh.faces[next(iter(self.view.sel_faces))]
        if info.normal is None:
            QMessageBox.information(self, "Push/Pull", "Only planar faces are supported.")
            return
        v, ok = QInputDialog.getDouble(self, "Push/Pull", "Distance (positive = outward):", 5.0, -1e5, 1e5, 3)
        if ok and v != 0:
            self.commit(Feature("pushpull", value=v, face=(tuple(map(float, info.center)),
                                                           tuple(map(float, info.normal)))))

    def _edge_feature(self, kind):
        if not self.view.sel_edges:
            QMessageBox.information(self, kind.title(), "Select one or more edges first (Ctrl+click for several).")
            return
        mids = [tuple(map(float, self.doc.mesh.edges[i].mid)) for i in sorted(self.view.sel_edges)]
        v, ok = QInputDialog.getDouble(self, kind.title(), "Size:", 1.0, 1e-4, 1e4, 3)
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
            dlg = ExtrudeDialog(self, f.value, f.cut)
            if not dlg.exec():
                return
            new = (dlg.dist.value(), dlg.cut.isChecked())
        else:
            v, ok = QInputDialog.getDouble(self, f.kind, "Value:", f.value, -1e5, 1e5, 3)
            if not ok or v == 0:
                return
            new = (v, f.cut)
        old = (f.value, f.cut)
        f.value, f.cut = new
        self.rebuild()
        if i in self.doc.errors:
            QMessageBox.warning(self, "Operation failed", self.doc.errors[i])
            f.value, f.cut = old
            self.rebuild()

    def toggle_layer(self, item):
        i = self.features.row(item)
        f = self.doc.features[i]
        f.visible = item.checkState() == Qt.Checked
        if f.kind == "sketch":
            self.view.overlay_sketches = [g.sketch for g in self.doc.features if g.kind == "sketch" and g.visible]
            self.mark_dirty()
            self.view.update()
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
        self.refresh()
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
