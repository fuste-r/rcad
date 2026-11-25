"""Selection shortcuts: double-click for a whole body (or a sketch chain), Ctrl+A for everything,
a box for several things, and a tag naming what is picked."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow
from rcad.kernel import Document, Feature
from rcad.sketch import Sketch, standard_plane


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)


def _pt(w, p):
    q = w.view.camera.project(np.array([p], float), w.view.width(), w.view.height())[0]
    return QPoint(int(round(q[0])), int(round(q[1])))


def _dclick(w, mods, p):
    """A real double-click: press, release, then a double-click press and its release
    (QTest.mouseDClick alone sends only the double-click event)."""
    QTest.mouseClick(w.view, Qt.LeftButton, mods, p)
    QTest.mouseDClick(w.view, Qt.LeftButton, mods, p)
    QTest.mouseRelease(w.view, Qt.LeftButton, mods, p)


def _two_blocks():
    """Two separate blocks: 0..20 and 40..60 in x, 20 deep, 10 tall."""
    w = MainWindow()
    w.view.resize(1000, 700)
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 20)
    sk.add_rect(40, 0, 60, 20)
    w.doc.features = [Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=10)]
    w.doc.rebuild(0)
    w.refresh()
    w.view.camera.look_at_plane((30, 10, 5), (0.3, -0.6, 0.74), (1, 0, 0), 160)
    return w


def _sketch():
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.view.camera.look_at_plane((20, 15, 0), (0, 0, 1), (1, 0, 0), 120)
    sk = w.sketch
    sk.add_rect(0, 0, 20, 10)  # lines 0-3, closed chain
    a, b, c = sk.add_point(30, 0), sk.add_point(40, 5), sk.add_point(45, 0)
    sk.add_line(a, b)  # line 4
    sk.add_line(b, c)  # line 5: an open chain of two
    sk.add_circle(sk.add_point(10, 25), 3)  # circle 0
    w.set_tool("select")
    return w


def test_mesh_knows_which_body_each_face_and_edge_is_on():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 20)
    sk.add_rect(40, 0, 60, 20)
    doc = Document([Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=10)])
    doc.rebuild(0)
    m = doc.mesh
    assert sorted(set(m.face_body)) == [0, 1] and m.face_body.count(0) == 6 and m.face_body.count(1) == 6
    assert len(m.edge_body) == len(m.edges) == 24
    for f, b in zip(m.faces, m.face_body):
        assert (f.center[0] < 30) == (b == m.face_body[0])
    for e, b in zip(m.edges, m.edge_body):
        assert (e.mid[0] < 30) == (b == m.face_body[0])


def test_double_click_selects_the_whole_body(qapp, quiet):
    w = _two_blocks()
    _dclick(w, Qt.NoModifier, _pt(w, (50, 10, 10)))  # right block's top
    body = w.doc.mesh.face_body[next(iter(w.view.sel_faces))]
    assert len(w.view.sel_faces) == 6 and {w.doc.mesh.face_body[f] for f in w.view.sel_faces} == {body}
    assert w.sel_tag.text() == f"Body {body + 1}" and not w.sel_tag.isHidden()
    # Ctrl adds the other body; a plain double-click on an edge picks its body alone
    _dclick(w, Qt.ControlModifier, _pt(w, (10, 10, 10)))
    assert len(w.view.sel_faces) == 12 and w.sel_tag.text() == "2 bodies"
    _dclick(w, Qt.NoModifier, _pt(w, (10, 0, 10)))  # left block's top front edge
    assert len(w.view.sel_faces) == 6 and not w.view.sel_edges


def test_ctrl_a_selects_everything(qapp, quiet):
    w = _two_blocks()
    w.select_all()
    assert len(w.view.sel_faces) == 12 and w.sel_tag.text() == "2 bodies"
    s = _sketch()
    s.set_tool("line")
    s.select_all()
    assert s.tool == "select"
    assert s.view.sketch_sel == ({("p", i) for i in range(len(s.sketch.points))}
                                 | {("l", i) for i in range(6)} | {("c", 0)})
    assert s.sel_tag.text() == "8 points, 6 lines, 1 circle"


def test_ctrl_drag_boxes_the_visible_edges_in_the_model(qapp, quiet):
    w = _two_blocks()
    QTest.mousePress(w.view, Qt.LeftButton, Qt.ControlModifier, _pt(w, (36, -12, 0)))
    QTest.mouseMove(w.view, _pt(w, (50, 10, 5)))
    assert w.view.rubber.isVisible() or not w.view.isVisible()  # the box shows while dragging
    QTest.mouseMove(w.view, _pt(w, (66, 30, 10)))
    QTest.mouseRelease(w.view, Qt.LeftButton, Qt.ControlModifier, _pt(w, (66, 30, 10)))
    mesh = w.doc.mesh
    picked = {tuple(np.round(mesh.edges[i].mid, 3)) for i in w.view.sel_edges}
    right = {tuple(np.round(e.mid, 3)) for e, b in zip(mesh.edges, mesh.edge_body)
             if e.mid[0] > 30}
    assert picked and picked <= right  # nothing from the left block
    assert (50.0, 0.0, 10.0) in picked and (50.0, 20.0, 0.0) not in picked  # the back bottom edge is hidden
    assert w.sel_tag.text() == f"{len(picked)} edges"
    assert w.doc.features[-1].kind == "extrude"  # the drag didn't fillet or orbit anything into a feature


def test_plain_drag_in_the_model_still_orbits(qapp, quiet):
    w = _two_blocks()
    before = w.view.camera.eye.copy()
    QTest.mousePress(w.view, Qt.LeftButton, Qt.NoModifier, QPoint(900, 650))
    QTest.mouseMove(w.view, QPoint(850, 600))
    QTest.mouseRelease(w.view, Qt.LeftButton, Qt.NoModifier, QPoint(850, 600))
    assert not np.allclose(before, w.view.camera.eye) and not w.view.sel_edges


def test_dragging_from_empty_sketch_space_boxes_whole_items(qapp, quiet):
    w = _sketch()
    QTest.mousePress(w.view, Qt.LeftButton, Qt.NoModifier, _pt(w, (-5, -5, 0)))
    QTest.mouseMove(w.view, _pt(w, (10, 10, 0)))
    QTest.mouseMove(w.view, _pt(w, (25, 30, 0)))
    QTest.mouseRelease(w.view, Qt.LeftButton, Qt.NoModifier, _pt(w, (25, 30, 0)))
    pts = {("p", i) for i in range(4)} | {("p", 7)}  # the rectangle corners and the circle centre
    assert w.view.sketch_sel == pts | {("l", i) for i in range(4)} | {("c", 0)}
    assert w.sketch.points[0] == [0, 0]  # nothing was dragged
    # a box catching part of the open chain takes only what is wholly inside; Ctrl adds
    QTest.mousePress(w.view, Qt.LeftButton, Qt.ControlModifier, _pt(w, (28, -2, 0)))
    QTest.mouseMove(w.view, _pt(w, (35, 5, 0)))
    QTest.mouseMove(w.view, _pt(w, (42, 7, 0)))
    QTest.mouseRelease(w.view, Qt.LeftButton, Qt.ControlModifier, _pt(w, (42, 7, 0)))
    assert {("l", 4), ("p", 4), ("p", 5)} <= w.view.sketch_sel and ("l", 5) not in w.view.sketch_sel
    assert ("l", 0) in w.view.sketch_sel


def test_dragging_sketch_geometry_still_moves_it(qapp, quiet):
    w = _sketch()
    QTest.mousePress(w.view, Qt.LeftButton, Qt.NoModifier, _pt(w, (35, 2.5, 0)))  # on line 4
    QTest.mouseMove(w.view, _pt(w, (35, 6, 0)))
    QTest.mouseMove(w.view, _pt(w, (35, 8.5, 0)))
    QTest.mouseRelease(w.view, Qt.LeftButton, Qt.NoModifier, _pt(w, (35, 8.5, 0)))
    assert w.sketch.points[4][1] == pytest.approx(6, abs=0.3)
    assert not w.view.rubber.isVisible()


def test_double_click_in_a_sketch_selects_the_connected_chain(qapp, quiet):
    w = _sketch()
    _dclick(w, Qt.NoModifier, _pt(w, (10, 0, 0)))  # bottom of the rectangle
    assert w.view.sketch_sel == {("l", i) for i in range(4)} and w.sel_tag.text() == "4 lines"
    _dclick(w, Qt.ControlModifier, _pt(w, (42.5, 2.5, 0)))
    assert w.view.sketch_sel == {("l", i) for i in range(6)}
    _dclick(w, Qt.NoModifier, _pt(w, (13, 25, 0)))  # the circle
    assert w.view.sketch_sel == {("c", 0)} and w.sel_tag.text() == "Circle"


def test_double_click_while_drawing_is_two_clicks(qapp, quiet):
    def state(w):
        return w.tool, w.pending, len(w.sketch.circles), set(w.view.sketch_sel)
    twice = _sketch()
    twice.set_tool("circle")
    for _ in range(2):
        QTest.mouseClick(twice.view, Qt.LeftButton, Qt.NoModifier, _pt(twice, (30, 25, 0)))
    w = _sketch()
    w.set_tool("circle")
    _dclick(w, Qt.NoModifier, _pt(w, (30, 25, 0)))
    assert state(w) == state(twice)


@pytest.mark.parametrize("pick, text", [
    ("face", "Flat face"), ("edge", "Edge"), ("edges", "2 edges"), ("mixed", "Flat face, 1 edge"),
])
def test_tag_names_model_picks(qapp, quiet, pick, text):
    w = _two_blocks()
    mesh = w.doc.mesh
    edge = lambda m: min(range(len(mesh.edges)), key=lambda i: np.linalg.norm(mesh.edges[i].mid - m))
    top = next(i for i, f in enumerate(mesh.faces) if f.normal is not None and f.normal[2] > 0.99)
    if pick in ("face", "mixed"):
        w.view.sel_faces = {top}
    if pick in ("edge", "edges", "mixed"):
        w.view.sel_edges = {edge((10, 0, 10))}
    if pick == "edges":
        w.view.sel_edges.add(edge((50, 0, 10)))
    w._update_regions()
    assert w.sel_tag.text() == text
    w.cancel()
    assert w.sel_tag.isHidden()


def test_tag_names_a_fillet_edge_and_a_region(qapp, quiet):
    w = _two_blocks()
    assert w.commit(Feature("fillet", value=2.0, edges=[(10.0, 0.0, 10.0)]))
    rounded = min(range(len(w.doc.mesh.edges)), key=lambda i: np.linalg.norm(
        w.doc.mesh.edges[i].mid - np.asarray(w.doc.features[-1].result_edges[0])))
    w.view.sel_edges = {rounded}
    w._update_regions()
    assert w.sel_tag.text() == "Fillet edge"
    w.cancel()
    s = MainWindow()
    s.view.resize(1000, 700)
    s.new_sketch()
    s.sketch.add_rect(0, 0, 40, 30)
    s.sketch.solve()
    s.finish_sketch()
    s.view.camera.look_at_plane((20, 15, 0), (0, 0, 1), (1, 0, 0), 120)
    s.on_click(*_pt(s, (20, 15, 0)).toTuple(), Qt.LeftButton, Qt.NoModifier)
    assert s.sel_tag.text() == f"Region of {s.doc.features[0].name}"
