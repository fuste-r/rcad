import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _px(w, p):
    return tuple(w.view.camera.project(np.array([p], float), w.view.width(), w.view.height())[0][:2])


def _box(monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.sketch.add_rect(0, 0, 40, 30)
    w.sketch.solve()
    w.finish_sketch()
    w.extrude()
    w.tool_bar.finish()
    w.view.camera.look_at_plane((20, 15, 5), (0.3, -0.6, 0.74), (1, 0, 0), 150)
    return w


def _click_button(w, label):
    b = next(b for b in w.quick.buttons if b.text() == label)
    b.click()


def test_face_offers_sketch_and_pushpull_beside_it(qapp, monkeypatch):
    w = _box(monkeypatch)
    assert not w.quick.buttons  # nothing selected
    w.on_click(*_px(w, (20, 15, 10)), Qt.LeftButton, Qt.NoModifier)  # the top face
    assert len(w.view.sel_faces) == 1
    assert w.quick.labels() == ["✏️ Sketch", "↕️ Push/Pull"]
    ax, ay = _px(w, w.quick.anchor)
    assert w.quick.x() > ax and abs(w.quick.y() + w.quick.height() / 2 - ay) <= w.quick.height()
    _click_button(w, "↕️ Push/Pull")
    assert w.tool_bar.active and not w.quick.buttons  # the settings bar takes over
    w.tool_bar.finish()
    assert w.doc.features[-1].kind == "pushpull"


def test_face_sketch_button_starts_a_sketch_on_it(qapp, monkeypatch):
    w = _box(monkeypatch)
    w.on_click(*_px(w, (20, 15, 10)), Qt.LeftButton, Qt.NoModifier)
    _click_button(w, "✏️ Sketch")
    assert w.mode == "sketch"
    assert np.allclose(w.sketch.plane.origin[2], 10) and np.allclose(np.abs(w.sketch.plane.normal), (0, 0, 1))


def test_edges_offer_fillet_chamfer_and_delete_for_a_fillets_edge(qapp, monkeypatch):
    w = _box(monkeypatch)
    mesh = w.doc.mesh
    edge = min(range(len(mesh.edges)), key=lambda i: np.linalg.norm(mesh.edges[i].mid - (20, 0, 10)))
    w.view.sel_edges = {edge}
    w._update_regions()
    assert w.quick.labels() == ["◐ Fillet", "◺ Chamfer"]  # a plain edge: nothing to delete
    _click_button(w, "◐ Fillet")
    w.tool_bar.fields["value"].setText("2")
    w.tool_bar.finish()
    fillet = w.doc.features[-1]
    rounded = min(range(len(w.doc.mesh.edges)),
                  key=lambda i: np.linalg.norm(w.doc.mesh.edges[i].mid - np.asarray(fillet.result_edges[0])))
    w.view.sel_edges = {rounded}
    w._update_regions()
    assert w.quick.labels() == ["◐ Fillet", "◺ Chamfer", "🗑"]
    _click_button(w, "🗑")
    assert all(f.kind != "fillet" for f in w.doc.features)
    w.undo()
    assert w.doc.features[-1].kind == "fillet"


def test_region_offers_extrude_revolve_and_more(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.sketch.add_rect(0, 0, 40, 30)
    w.sketch.solve()
    w.finish_sketch()
    w.view.camera.look_at_plane((20, 15, 0), (0, 0, 1), (1, 0, 0), 120)
    w.on_click(*_px(w, (20, 15, 0)), Qt.LeftButton, Qt.NoModifier)
    assert w.quick.labels() == ["⇈ Extrude", "⟳ Revolve", "⋯"]
    assert [a.text() for a in w.quick.more_menu.actions()] == ["↝ Sweep", "⌁ Loft"]
    _click_button(w, "⇈ Extrude")
    assert w.tool_bar.active and w.doc.features[-1].kind == "extrude"
    w.tool_bar.finish()


def _sketch_with_rect_and_circle(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.sketch.add_rect(0, 0, 40, 30)
    c = w.sketch.add_point(20, 15)
    w.sketch.add_circle(c, 5)
    w.sketch.solve()
    return w


@pytest.mark.parametrize("selection, labels, more", [
    ({("l", 0)}, ["Distance", "Horizontal", "🗑", "⋯"], ["Vertical", "┄ Construction"]),
    ({("l", 0), ("l", 1)}, ["Parallel", "Perpendicular", "🗑", "⋯"], ["Equal", "Spacing", "┄ Construction"]),
    ({("p", 0), ("p", 2)}, ["Coincident", "Distance", "🗑", "⋯"],
     ["Horizontal", "Vertical", "Expression", "Fixed"]),
    ({("p", 0), ("l", 1)}, ["Point on line", "🗑", "⋯"], ["┄ Construction"]),
    ({("c", 0)}, ["Diameter", "🗑", "⋯"], ["┄ Construction"]),
    ({("p", 0)}, ["Fixed", "🗑"], None),  # points alone can't be construction geometry
    ({("c", 0), ("l", 0)}, ["🗑", "⋯"], ["┄ Construction"]),  # no constraint fits: still deletable
])
def test_sketch_selection_offers_fitting_constraints(qapp, monkeypatch, selection, labels, more):
    w = _sketch_with_rect_and_circle(monkeypatch)
    w.view.sketch_sel = set(selection)
    w._update_regions()
    assert w.quick.labels() == labels
    if more is not None:
        assert [a.text() for a in w.quick.more_menu.actions()] == more


def test_sketch_quick_buttons_act_and_hide_while_drawing(qapp, monkeypatch):
    w = _sketch_with_rect_and_circle(monkeypatch)
    a, b, c, d = (w.sketch.add_point(*p) for p in [(50, 0), (60, 2), (50, 10), (60, 15)])
    l1, l2 = w.sketch.add_line(a, b), w.sketch.add_line(c, d)  # two free lines
    w.view.sketch_sel = {("l", l1), ("l", l2)}
    w._update_regions()
    n = len(w.sketch.constraints)
    _click_button(w, "Parallel")
    assert w.sketch.constraints[-1].kind == "parallel" and len(w.sketch.constraints) == n + 1
    w.view.sketch_sel = {("c", 0)}
    w._update_regions()
    _click_button(w, "Diameter")
    assert w.tool_bar.active and not w.quick.buttons  # the dimension opens the settings bar
    w.tool_bar.finish()
    w.view.sketch_sel = {("l", 0)}
    w._update_regions()
    assert w.quick.buttons
    w.set_tool("line")
    assert not w.quick.buttons  # drawing: no quick buttons
    w.set_tool("select")
    w.view.sketch_sel = {("c", 0)}
    w._update_regions()
    n_circles = len(w.sketch.circles)
    _click_button(w, "🗑")
    assert len(w.sketch.circles) == n_circles - 1


def test_quick_buttons_hide_while_dragging(qapp, monkeypatch):
    w = _box(monkeypatch)
    w.on_click(*_px(w, (20, 15, 10)), Qt.LeftButton, Qt.NoModifier)
    assert not w.quick.isHidden()
    w.view._press, w.view._dragged = (None, Qt.LeftButton, Qt.NoModifier), True
    w._place_quick()
    assert w.quick.isHidden()
    w.view._press, w.view._dragged = None, False
    w._place_quick()
    assert not w.quick.isHidden()
