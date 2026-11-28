import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow
from rcad.sketch import Constraint, Sketch, standard_plane
from rcad.trim import cuts, piece_at, trim


def _do(sk, uv, tol=0.5):
    """Trim the piece at uv the way the tool does; returns the constraints it reported dropped."""
    result = trim(sk, piece_at(sk, uv, tol))
    for c in result.keep:  # (the tool adds these one by one, dropping any the sketch refuses)
        sk.constraints.append(c)
        if not sk.solve():
            sk.constraints.pop()
    assert sk.solve(), sk.status
    return result.dropped


def rect_with_cross():
    """A 40 x 20 rectangle (points 0-3, lines 0-3) and a vertical line through it at x = 15 (line 4)."""
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 40, 20)
    sk.add_line(sk.add_point(15, -10), sk.add_point(15, 30))
    sk.constraints.append(Constraint("vertical", [4]))
    assert sk.solve()
    return sk


# -- finding pieces

def test_cuts_are_where_other_curves_cross():
    sk = rect_with_cross()
    assert [c.uv for c in cuts(sk, "l", 4)] == [pytest.approx((15, 0)), pytest.approx((15, 20))]
    assert [c.cutter for c in cuts(sk, "l", 4)] == [("l", 0), ("l", 2)]
    piece = piece_at(sk, (15, 25), 0.5)
    assert (piece.kind, piece.index) == ("l", 4) and piece.lo.uv == pytest.approx((15, 20)) and piece.hi is None
    middle = piece_at(sk, (15, 10), 0.5)
    assert middle.lo.uv == pytest.approx((15, 0)) and middle.hi.uv == pytest.approx((15, 20))
    assert piece_at(sk, (30, 10), 0.5) is None  # nothing there


# -- lines

def test_trim_an_overhang_keeps_the_new_end_on_what_cut_it():
    sk = rect_with_cross()
    assert _do(sk, (15, 25)) == []
    a, b = sk.lines[4]
    assert sk.points[b] == pytest.approx([15, 20]) and len(sk.points) == 6  # the old top end is gone
    assert ("point_on_line", [b, 2]) in [(c.kind, c.refs) for c in sk.constraints]
    sk.constraints.append(Constraint("distance", [0, 1], 60.0))  # stretch the rectangle up? no: wider
    sk.constraints.append(Constraint("distance", [1, 2], 35.0))  # taller: the cut end follows the top
    assert sk.solve() and sk.points[b][1] == pytest.approx(sk.points[2][1])


def test_trim_the_middle_splits_a_line_that_stays_straight():
    sk = Sketch(standard_plane("XY"))
    sk.add_line(sk.add_point(0, 0), sk.add_point(40, 0))
    for x in (10, 30):
        sk.add_line(sk.add_point(x, -5), sk.add_point(x, 5))
    sk.solve()
    _do(sk, (20, 0))
    assert len(sk.lines) == 4
    left, right = sk.lines[0], sk.lines[3]
    assert sk.points[left[1]] == pytest.approx([10, 0]) and sk.points[right[0]] == pytest.approx([30, 0])
    sk.points[0] = [0, -8]  # tilt the left piece: the right piece follows in line with it
    assert sk.solve()
    (ax, ay), (bx, by) = sk.points[left[0]], sk.points[left[1]]
    (cx, cy), (dx, dy) = sk.points[right[0]], sk.points[right[1]]
    assert abs((bx - ax) * (dy - ay) - (by - ay) * (dx - ax)) < 1e-6
    assert abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) < 1e-6


def test_a_line_ending_on_another_reuses_its_point():
    sk = Sketch(standard_plane("XY"))
    sk.add_line(sk.add_point(0, 0), sk.add_point(40, 0))
    joint = sk.add_point(20, 0)
    sk.add_line(joint, sk.add_point(20, 15))  # a T: it ends on the first line
    sk.solve()
    _do(sk, (30, 0))
    a, b = sk.lines[0]
    assert sk.points[b] == pytest.approx([20, 0]) and b in sk.lines[1]  # ends on the T's own point
    assert len(sk.points) == 3  # no new point (and the cut-off end is gone)
    assert all(c.kind != "point_on_line" for c in sk.constraints)  # nothing to hold


def test_constraints_that_no_longer_apply_are_reported():
    sk = Sketch(standard_plane("XY"))
    sk.add_line(sk.add_point(0, 0), sk.add_point(40, 0))
    top = sk.add_line(sk.add_point(0, 20), sk.add_point(40, 20))
    sk.add_line(sk.add_point(15, 10), sk.add_point(15, 30))  # crosses the top line
    mid = sk.add_point(20, 20)
    sk.constraints += [Constraint("equal", [0, top]), Constraint("midpoint", [mid, top])]
    assert sk.solve(), sk.status
    assert sorted(_do(sk, (5, 20))) == ["equal", "midpoint"]  # the top line's length changed


def test_trim_a_rectangle_down_to_a_smaller_one():
    sk = rect_with_cross()
    for uv in [(15, 25), (5, 20), (15, -5), (5, 0), (0, 10)]:
        _do(sk, uv)
    assert len(sk.lines) == 4 and len(sk.points) == 4 and len(sk.regions()) == 1
    assert sk.profile_faces()[0].area == pytest.approx(25 * 20)


# -- circles and arcs

def test_a_circle_cut_by_a_line_becomes_an_arc():
    sk = Sketch(standard_plane("XY"))
    sk.add_circle(sk.add_point(0, 0), 10)
    sk.constraints.append(Constraint("diameter", [0], 20.0))
    sk.add_line(sk.add_point(-20, 3), sk.add_point(20, 3))
    p = sk.add_point(0, -10)
    sk.constraints.append(Constraint("point_on_circle", [p, 0]))
    sk.solve()
    assert _do(sk, (0, 10)) == []  # the cap above the line
    assert sk.circles == [] and len(sk.arcs) == 1
    kinds = {c.kind: c for c in sk.constraints}
    assert kinds["radius"].value == pytest.approx(10) and "point_on_arc" in kinds
    mid = sk.arc_polyline(0, 3)[1]
    assert mid[1] < 0  # what is left is the part below the line
    assert len(sk.regions()) == 1


def test_a_circle_nothing_crosses_goes_entirely():
    sk = Sketch(standard_plane("XY"))
    sk.add_circle(sk.add_point(0, 0), 5)
    sk.solve()
    _do(sk, (5, 0))
    assert sk.circles == [] and sk.points == []


def test_split_an_arc_and_both_parts_keep_one_radius():
    sk = Sketch(standard_plane("XY"))
    sk.add_arc(sk.add_point(0, 0), sk.add_point(10, 0), sk.add_point(-10, 0))  # the upper half
    sk.constraints.append(Constraint("radius", [0], 10.0))
    for x in (-4, 4):
        sk.add_line(sk.add_point(x, 0), sk.add_point(x, 15))
    sk.solve()
    _do(sk, (0, 10))
    assert len(sk.arcs) == 2
    sk.constraints[0].value = 14
    assert sk.solve()
    assert [round(sk.arc_geometry(i)[1], 6) for i in range(2)] == [14, 14]


def test_tangent_moves_to_the_piece_that_keeps_the_joint():
    sk = Sketch(standard_plane("XY"))
    a, b = sk.add_point(0, 0), sk.add_point(30, 0)
    sk.add_line(a, b)
    sk.add_arc(sk.add_point(30, 5), b, sk.add_point(35, 5))  # tangent at b
    sk.constraints.append(Constraint("tangent", [0, 0]))
    sk.add_line(sk.add_point(10, -5), sk.add_point(10, 5))
    sk.add_line(sk.add_point(20, -5), sk.add_point(20, 5))
    sk.solve()
    _do(sk, (15, 0))  # split the line: the right piece now meets the arc
    tangent = next(c for c in sk.constraints if c.kind == "tangent")
    assert b in sk.lines[tangent.refs[0]] and tangent.refs[0] != 0
    _do(sk, (25, 0))  # cut that piece away: the tangent has nothing left to hold
    assert all(c.kind != "tangent" for c in sk.constraints)


# -- the tool

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.view.camera.dist = 150
    s = rect_with_cross()
    sk = w.sketch
    sk.points, sk.lines, sk.constraints = s.points, s.lines, s.constraints
    sk.solve()
    w.set_tool("trim")
    return w


def _xy(w, uv):
    return w.view.camera.project(np.array([(uv[0], uv[1], 0)], float), 1000, 700)[0][:2]


def test_hover_shows_the_piece_and_a_click_trims_it(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.on_hover(*_xy(w, (15, 25)))
    assert len(w.view.trim_preview) == 1
    assert w.view.trim_preview[0][:, 1].max() == pytest.approx(30)  # the overhang, up to its free end
    w.on_click(*_xy(w, (15, 25)), Qt.LeftButton, Qt.NoModifier)
    assert w.view.trim_preview == [] and len(w.sketch.points) == 6
    assert w.statusBar().currentMessage() == "Trimmed 1 piece."
    w.undo()
    assert len(w.sketch.points) == 6 and w.sketch.points[5] == pytest.approx([15, 30])  # the overhang is back


def test_a_click_on_nothing_adds_no_undo_step(qapp, monkeypatch):
    w = _window(monkeypatch)
    n = len(w._sketch_undo.get(w.sketch_index, []))
    w.on_click(*_xy(w, (30, 10)), Qt.LeftButton, Qt.NoModifier)
    assert len(w._sketch_undo.get(w.sketch_index, [])) == n
    assert "Nothing to trim" in w.statusBar().currentMessage()


def test_drag_across_pieces_trims_them_all_in_one_step(qapp, monkeypatch):
    w = _window(monkeypatch)
    n = len(w._sketch_undo.get(w.sketch_index, []))
    # a stroke from above the rectangle, down across the top-left edge and the overhang
    w.on_sketch_mouse("press", *_xy(w, (5, 24)))
    for uv in [(8, 22), (10, 20), (12, 22), (14, 25), (17, 25)]:
        w.on_sketch_mouse("move", *_xy(w, uv))
    assert len(w.view.trim_preview) == 2
    w.on_sketch_mouse("release", *_xy(w, (17, 25)))
    assert w.statusBar().currentMessage().startswith("Trimmed 2 pieces.")
    assert len(w._sketch_undo.get(w.sketch_index, [])) == n + 1
    sk = w.sketch
    assert not any(math.dist(sk.points[a], (15, 30)) < 1e-6 or math.dist(sk.points[b], (15, 30)) < 1e-6
                   for a, b in sk.lines)
    w.undo()
    assert len(w.sketch.lines) == 5


def test_a_trim_the_sketch_cannot_solve_is_undone(qapp, monkeypatch):
    w = _window(monkeypatch)
    before = w.sketch.to_dict()
    real = type(w.sketch).solve
    monkeypatch.setattr(type(w.sketch), "solve", lambda self: False)  # whatever is trimmed, it can't solve
    w.on_click(*_xy(w, (15, 25)), Qt.LeftButton, Qt.NoModifier)
    monkeypatch.setattr(type(w.sketch), "solve", real)
    assert "Trim undone" in w.statusBar().currentMessage()
    assert w.sketch.to_dict()["lines"] == before["lines"]
