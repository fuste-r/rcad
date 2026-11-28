import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow
from rcad.kernel import Document, Feature
from rcad.sketch import Sketch, end_direction, standard_plane, tangent_arc
from rcad.snapping import find_snap


# -- geometry

def test_direction_at_the_end_of_a_line_and_an_arc():
    sk = Sketch(standard_plane("XY"))
    a, b = sk.add_point(0, 0), sk.add_point(10, 0)
    sk.add_line(a, b)
    assert end_direction(sk, b) == (("l", 0), pytest.approx((1, 0)))
    assert end_direction(sk, a) == (("l", 0), pytest.approx((-1, 0)))
    c, e = sk.add_point(10, 5), sk.add_point(15, 5)
    sk.add_arc(c, b, e)  # a left quarter turn from b, counter-clockwise
    assert end_direction(sk, e) == (("a", 0), pytest.approx((0, 1)))  # heading up as it arrives at e
    assert end_direction(sk, b)[0] == ("a", 0)  # an arc wins over the line at a shared point
    assert end_direction(sk, b)[1] == pytest.approx((-1, 0))  # going backwards round the arc
    assert end_direction(sk, b, prefer=("l", 0))[0] == ("l", 0)
    assert end_direction(sk, c) is None


def test_tangent_arc_bends_toward_the_pointer():
    left = tangent_arc((10, 0), (1, 0), (15, 5))
    assert left[0] == pytest.approx((10, 5)) and left[1] == pytest.approx((15, 5))
    assert left[2] == pytest.approx(5) and left[3] == pytest.approx(90)
    right = tangent_arc((10, 0), (1, 0), (15, -5))
    assert right[2] == pytest.approx(-5) and right[1] == pytest.approx((15, -5))
    back = tangent_arc((10, 0), (1, 0), (10, 10))  # straight above the start: a half circle
    assert back[3] == pytest.approx(180) and back[1] == pytest.approx((10, 10))
    assert tangent_arc((10, 0), (1, 0), (30, 0)) is None  # straight on
    typed = tangent_arc((10, 0), (1, 0), (15, 5), radius=3, sweep=180)
    assert typed[2] == pytest.approx(3) and typed[1] == pytest.approx((10, 6))


def test_line_from_an_arc_end_snaps_along_its_tangent():
    sk = Sketch(standard_plane("XY"))
    b, c, e = sk.add_point(10, 0), sk.add_point(10, 5), sk.add_point(15, 5)
    sk.add_arc(c, b, e)
    hit = find_snap(sk, (15.4, 12.1), tol=1, grid=1, start=(e, (15, 5)), tangent=(0, (0, 1)))
    assert hit.kind == "tangent" and hit.tangent == 0 and hit.uv == pytest.approx((15, 12))
    assert hit.guides(sk, (15, 5)) == [((15, 5), hit.uv)]
    off = find_snap(sk, (18, 12), tol=1, grid=1, start=(e, (15, 5)), tangent=(0, (0, 1)))
    assert off.kind != "tangent"
    behind = find_snap(sk, (15.2, -3), tol=1, grid=1, start=(e, (15, 5)), tangent=(0, (0, 1)))
    assert behind.kind != "tangent"  # only straight on, not back into the arc


# -- drawing

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.view.camera.dist = 150
    return w


def _click(w, uv, mods=Qt.NoModifier):
    x, y = w.view.camera.project(np.array([(uv[0], uv[1], 0)], float), 1000, 700)[0][:2]
    w.on_click(x, y, Qt.LeftButton, mods)


def _hover(w, uv):
    x, y = w.view.camera.project(np.array([(uv[0], uv[1], 0)], float), 1000, 700)[0][:2]
    w.on_hover(x, y)


def test_line_then_tangent_arc_then_line_makes_a_smooth_chain(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    w.set_tool("line")
    _click(w, (0, 0))
    _click(w, (20, 0))
    w.set_tool("tarc")  # carries on from the chain's end
    assert w.tool == "tarc" and w.pending is not None and w.pending[0] == 1
    _hover(w, (30, 10.3))
    _click(w, (30, 10.3), Qt.ControlModifier)  # free: exactly where it is clicked
    assert len(sk.arcs) == 1
    assert [c.kind for c in sk.constraints if c.kind.startswith("tangent")] == ["tangent"]
    (cu, cv), r, a0, sweep = sk.arc_geometry(0)
    assert cu == pytest.approx(20, abs=1e-6)  # the centre straight above the joint: smooth
    w.set_tool("line")  # back to lines from the arc's end
    assert w.pending is not None
    end = w.pending[1]
    _hover(w, (end[0] + 0.3, end[1] + 15))
    assert w.view.snap_mark is not None and w.view.snap_mark["kind"] == "tangent"
    _click(w, (end[0] + 0.3, end[1] + 15))
    assert [c.kind for c in sk.constraints if c.kind.startswith("tangent")] == ["tangent", "tangent"]
    assert sk.solve()
    la, lb = sk.lines[-1]
    d = np.subtract(sk.points[lb], sk.points[la])
    ex, ey = end_direction(sk, sk.lines[-1][0], prefer=("a", 0))[1]
    assert abs(d[0] * ey - d[1] * ex) < 1e-6  # the new line runs straight on from the arc


def test_tangent_arc_needs_a_curve_end(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.set_tool("tarc")
    _click(w, (5, 5))
    assert w.pending is None and "end of a line or arc" in w.statusBar().currentMessage()


def test_tangent_arcs_chain_and_close_a_slot(qapp, monkeypatch):
    """Line, half-circle tangent arc, line back, half-circle tangent arc closing on the start: a slot."""
    w = _window(monkeypatch)
    sk = w.sketch
    w.set_tool("line")
    _click(w, (0, 0))
    _click(w, (30, 0))
    w.set_tool("tarc")
    w.fields.boxes["S"].setText("180")
    w.fields.boxes["S"]._on_text("180")
    _hover(w, (30, 10))
    _click(w, (30, 10))
    assert sk.points[-1] == pytest.approx([30, 10], abs=1e-6)
    w.set_tool("line")
    _click(w, (0, 10))
    w.set_tool("tarc")
    _hover(w, (0.1, 0.2))
    _click(w, (0.1, 0.2))  # onto the first point: the slot closes and the chain ends
    assert w.pending is None
    assert len(sk.arcs) == 2 and len(sk.lines) == 2
    # smooth at all four joints, including where the last arc closes onto the first line
    assert sorted(c.refs for c in sk.constraints if c.kind == "tangent") == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert sk.solve() and len(sk.regions()) == 1
    doc = Document([Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=2)])
    doc.rebuild()
    assert not doc.errors and doc.part.volume == pytest.approx((30 * 10 + math.pi * 25) * 2, rel=1e-3)


def test_typed_radius_and_angle(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    w.set_tool("line")
    _click(w, (0, 0))
    _click(w, (10, 0))
    w.set_tool("tarc")
    for k, v in (("R", "4"), ("S", "90")):
        w.fields.boxes[k].setText(v)
        w.fields.boxes[k]._on_text(v)
    _hover(w, (12, -6))  # below: it bends right
    w.fields.boxes["S"].returnPressed.emit()
    (cu, cv), r, a0, sweep = sk.arc_geometry(0)
    assert (cu, cv) == pytest.approx((10, -4)) and r == pytest.approx(4)
    assert math.degrees(sweep) == pytest.approx(90)
    assert sk.points[-1] == pytest.approx([14, -4])
