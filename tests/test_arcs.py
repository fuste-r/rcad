import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow
from rcad.kernel import Document, Feature
from rcad.sketch import Constraint, Sketch, standard_plane
from rcad.snapping import find_snap


def slot(r=5.0, length=30.0):
    """Two horizontal lines joined by two tangent arcs (an obround), fully sized."""
    sk = Sketch(standard_plane("XY"))
    p = [sk.add_point(*q) for q in [(0, 0), (length, 0), (length, 2 * r), (0, 2 * r), (length, r), (0, r)]]
    sk.add_line(p[0], p[1])
    sk.add_line(p[2], p[3])
    sk.add_arc(p[4], p[1], p[2])  # right end, counter-clockwise from the bottom
    sk.add_arc(p[5], p[3], p[0])  # left end
    sk.constraints += [Constraint("tangent", [0, 0]), Constraint("tangent", [0, 1]),
                       Constraint("tangent", [1, 0]), Constraint("tangent", [1, 1]),
                       Constraint("radius", [0], r), Constraint("radius", [1], r),
                       Constraint("horizontal", [0]), Constraint("distance", [0, 1], length)]
    assert sk.solve(), sk.status
    return sk


# -- the sketch

def test_slot_is_one_region_and_extrudes_exactly():
    sk = slot()
    assert len(sk.regions()) == 1
    area = 30 * 10 + math.pi * 25
    assert sk.profile_faces()[0].area == pytest.approx(area)
    doc = Document([Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=4)])
    doc.rebuild()
    assert not doc.errors and doc.part.volume == pytest.approx(area * 4)


def test_tangent_and_radius_hold_when_the_sketch_changes():
    sk = slot()
    sk.constraints[-1].value = 50  # lengthen it
    assert sk.solve()
    (c, r, a0, sweep) = sk.arc_geometry(0)
    assert r == pytest.approx(5) and sweep == pytest.approx(math.pi)  # still a half circle
    assert sk.points[1][0] - sk.points[0][0] == pytest.approx(50)


def test_tangent_needs_a_shared_end():
    sk = Sketch(standard_plane("XY"))
    a, b, c, s, e = (sk.add_point(*q) for q in [(0, 0), (10, 0), (20, 5), (25, 5), (20, 10)])
    sk.add_line(a, b)
    sk.add_arc(c, s, e)
    sk.constraints.append(Constraint("tangent", [0, 0]))
    with pytest.raises(ValueError, match="share an end"):
        sk.solve()


def test_construction_geometry_makes_no_regions():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 10)
    sk.add_line(sk.add_point(10, -5), sk.add_point(10, 15))  # would split the rectangle
    sk.add_circle(sk.add_point(40, 5), 4)  # would be a region of its own
    assert len(sk.regions()) == 3  # two halves and the circle
    sk.set_construction([("l", 4), ("c", 0)], True)
    assert len(sk.regions()) == 1 and sk.region_at(5, 5) == sk.region_at(15, 5)
    sk.set_construction([("c", 0)], False)
    assert len(sk.regions()) == 2


def test_arcs_and_construction_round_trip(tmp_path):
    sk = slot()
    sk.set_construction([("a", 1), ("l", 0)], True)
    doc = Document([Feature("sketch", "S", sketch=sk)])
    path = str(tmp_path / "slot.rcad")
    doc.save(path)
    back = Document.load(path).features[0].sketch
    assert back.arcs == sk.arcs and back.construction == {("a", 1), ("l", 0)}
    assert [c.kind for c in back.constraints] == [c.kind for c in sk.constraints]
    assert back.solve()


def test_deleting_renumbers_arcs_and_construction():
    sk = slot()
    sk.set_construction([("a", 1), ("l", 1)], True)
    sk.delete("a", 0)
    assert sk.arcs == [(5, 3, 0)] and sk.construction == {("a", 0), ("l", 1)}
    assert all(c.kind != "radius" or c.refs == [0] for c in sk.constraints)
    sk.delete("p", 0)  # the left arc and the bottom line end there
    assert sk.arcs == [] and sk.construction == {("l", 0)} and len(sk.lines) == 1


def test_snapping_onto_an_arc_but_not_its_gap():
    sk = Sketch(standard_plane("XY"))
    c, s, e = sk.add_point(0, 0), sk.add_point(10, 0), sk.add_point(0, 10)  # a quarter, counter-clockwise
    sk.add_arc(c, s, e)
    hit = find_snap(sk, (7.3, 7.0), tol=1, grid=0)
    assert hit.kind == "on_arc" and hit.arc == 0 and math.dist(hit.uv, (0, 0)) == pytest.approx(10)
    assert [k.kind for k in hit.constraints(9)] == ["point_on_arc"]
    assert find_snap(sk, (-7.2, -7.0), tol=1, grid=0).kind != "on_arc"  # where the arc is not


# -- drawing and editing

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.view.camera.dist = 150
    return w


def _xy(w, uv):
    return w.view.camera.project(np.array([(uv[0], uv[1], 0)], float), 1000, 700)[0][:2]


def _click(w, uv, mods=Qt.NoModifier):
    w.on_click(*_xy(w, uv), Qt.LeftButton, mods)


def _hover_around(w, center, r, degrees):
    for d in degrees:
        a = math.radians(d)
        w.on_hover(*_xy(w, (center[0] + r * math.cos(a), center[1] + r * math.sin(a))))


def test_draw_an_arc_counter_clockwise(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.set_tool("arc")
    _click(w, (0, 0))
    _click(w, (20, 0))
    _hover_around(w, (0, 0), 20, range(10, 100, 10))
    _click(w, (0.3, 20.4))
    sk = w.sketch
    assert len(sk.arcs) == 1 and w.tool == "select"
    (cu, cv), r, a0, sweep = sk.arc_geometry(0)
    assert (cu, cv) == pytest.approx((0, 0)) and r == pytest.approx(20)
    assert a0 == pytest.approx(0) and sweep == pytest.approx(math.pi / 2, abs=0.03)


def test_draw_an_arc_clockwise(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.set_tool("arc")
    _click(w, (0, 0))
    _click(w, (20, 0))
    _hover_around(w, (0, 0), 20, range(-10, -200, -10))  # round the other way, past the half
    _click(w, (-20, -7))
    sk = w.sketch
    c, s, e = sk.arcs[0]
    assert sk.points[e] == pytest.approx([20, 0])  # stored counter-clockwise: it ends where it was started
    sweep = sk.arc_geometry(0)[3]
    assert math.degrees(sweep) == pytest.approx(360 - math.degrees(math.atan2(7, 20)) - 180, abs=2)


def test_typed_radius_and_angle(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.set_tool("arc")
    _click(w, (5, 5))
    w.fields.boxes["R"].setText("12")
    w.fields.boxes["R"]._on_text("12")
    _click(w, (40, 5))  # the direction counts; the radius is typed
    w.fields.boxes["S"].setText("-90")
    w.fields.boxes["S"]._on_text("-90")
    w.fields.boxes["S"].returnPressed.emit()
    sk = w.sketch
    assert len(sk.arcs) == 1
    (cu, cv), r, a0, sweep = sk.arc_geometry(0)
    assert r == pytest.approx(12) and math.degrees(sweep) == pytest.approx(90)
    c, s, e = sk.arcs[0]
    assert sk.points[e] == pytest.approx([17, 5])  # clockwise from (17, 5) down to (5, -7)
    assert sk.points[s] == pytest.approx([5, -7])


def test_arc_from_a_line_end_shares_its_point(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.set_tool("line")
    _click(w, (-20, 0))
    _click(w, (0, 0))
    w.cancel()
    w.set_tool("arc")
    _click(w, (6, 9), Qt.ControlModifier)  # a free centre: the arc does not start out tangent
    _click(w, (0.2, -0.3))  # the line's end: reused as the arc's start
    r = math.hypot(6, 9)
    _hover_around(w, (6, 9), r, range(-120, 10, 10))
    _click(w, (6 + r, 9))
    sk = w.sketch
    c, s, e = sk.arcs[0]
    assert s == 1 and len(sk.points) == 4
    w.view.sketch_sel = {("l", 0), ("a", 0)}
    w._update_regions()
    assert w.quick.labels()[0] == "Tangent"
    w.add_constraint("tangent")
    assert sk.constraints[-1].kind == "tangent" and sk.solve()
    cu, cv = sk.points[c]
    assert cu == pytest.approx(sk.points[1][0], abs=1e-6)  # the centre is now straight above the joint


def test_radius_and_concentric_constraints(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    sk.add_arc(sk.add_point(0, 0), sk.add_point(10, 0), sk.add_point(0, 10))
    sk.add_circle(sk.add_point(1, 1), 4)
    sk.solve()
    w.view.sketch_sel = {("a", 0)}
    w.add_constraint("radius")
    assert w.tool_bar.active
    w.tool_bar.fields["value"].setText("15")
    w.tool_bar.finish()
    assert sk.constraints[-1].kind == "radius" and sk.constraints[-1].value == pytest.approx(15)
    assert sk.arc_geometry(0)[1] == pytest.approx(15)
    w.view.sketch_sel = {("a", 0), ("c", 0)}
    w.add_constraint("concentric")
    assert sk.constraints[-1].kind == "coincident" and sk.solve()
    assert sk.points[sk.circles[0][0]] == pytest.approx(sk.points[sk.arcs[0][0]])


def test_q_toggles_construction(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    sk.add_rect(0, 0, 20, 10)
    sk.add_line(sk.add_point(10, -5), sk.add_point(10, 15))
    sk.solve()
    assert len(sk.regions()) == 2
    w.view.sketch_sel = {("l", 4), ("p", 4)}  # points are left alone
    QApplication.sendEvent(w.view, QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Q, Qt.NoModifier, "q"))
    assert sk.construction == {("l", 4)} and len(sk.regions()) == 1
    w.undo()
    assert w.sketch.construction == set()
    w.view.sketch_sel = {("l", 4)}
    w.toggle_construction()
    w.toggle_construction()  # all of them were: back to normal
    assert sk.construction == set()


def test_double_click_selects_the_whole_outline(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    s = slot()
    sk.points, sk.lines, sk.arcs, sk.constraints = s.points, s.lines, s.arcs, s.constraints
    sk.solve()
    w.on_double_click(*_xy(w, (15, 0)), Qt.NoModifier)
    assert w.view.sketch_sel == {("l", 0), ("l", 1), ("a", 0), ("a", 1)}
    assert w.sel_tag.text() == "2 lines, 2 arcs"


def test_box_select_and_drag_an_arc(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    sk.add_arc(sk.add_point(0, 0), sk.add_point(10, 0), sk.add_point(0, 10))
    sk.solve()
    (x0, y1), (x1, y0) = _xy(w, (-2, -2)), _xy(w, (12, 12))
    w.on_box_select(x0, y0, x1, y1, Qt.NoModifier)
    assert ("a", 0) in w.view.sketch_sel
    assert w.view.pick_sketch(*_xy(w, (7.07, 7.07))) == ("a", 0)
    w.on_sketch_mouse("press", *_xy(w, (7.07, 7.07)))
    w.on_sketch_mouse("move", *_xy(w, (12.07, 7.07)))
    w.on_sketch_mouse("release", *_xy(w, (12.07, 7.07)))
    assert sk.points[0] == pytest.approx([5, 0], abs=0.2)  # the whole arc moved over
    assert sk.arc_geometry(0)[1] == pytest.approx(10)


def test_mirror_an_arc(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    axis = sk.add_line(sk.add_point(0, -20), sk.add_point(0, 20))
    sk.add_arc(sk.add_point(10, 0), sk.add_point(15, 0), sk.add_point(10, 5))
    sk.solve()
    w.view.sketch_sel = {("l", axis)}
    w.add_constraint("symmetry")
    w.view.sketch_sel = {("a", 0)}
    w.add_constraint("symmetry")
    assert len(sk.arcs) == 2
    (cu, cv), r, a0, sweep = sk.arc_geometry(1)
    assert (cu, cv) == pytest.approx((-10, 0), abs=1e-6) and r == pytest.approx(5)
    assert math.degrees(sweep) == pytest.approx(90)  # the same quarter, mirrored
    mid = sk.arc_polyline(1, 3)[1]
    assert mid[0] < -10 and mid[1] > 0  # bulging up and to the left
