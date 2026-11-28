import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow
from rcad.sketch import Constraint, Sketch, standard_plane
from rcad.snapping import START, find_snap, grid_step


def base():
    """A 40 x 20 rectangle (points 0-3, lines 0-3) and a circle of radius 5 at (60, 10) (centre point 4)."""
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 40, 20)
    sk.add_circle(sk.add_point(60, 10), 5)
    return sk


# -- where points land

def test_existing_point_beats_everything():
    s = find_snap(base(), (40.4, 19.7), tol=1, grid=1)
    assert s.kind == "point" and s.point == 2 and s.uv == (40, 20)


def test_midpoint_of_a_line():
    s = find_snap(base(), (20.3, -0.4), tol=1, grid=1)
    assert s.kind == "midpoint" and s.line == 0 and s.uv == (20, 0)
    assert [c.kind for c in s.constraints(9)] == ["midpoint"]


def test_on_a_line_and_on_a_circle():
    s = find_snap(base(), (40.3, 7.3), tol=1, grid=1)  # (not level with any point)
    assert s.kind == "on_line" and s.line == 1 and s.uv == pytest.approx((40, 7.3))
    s = find_snap(base(), (63.2, 13.9), tol=1, grid=1)  # (off the centre's vertical)
    assert s.kind == "on_circle" and s.circle == 0 and math.dist(s.uv, (60, 10)) == pytest.approx(5)
    assert [c.kind for c in s.constraints(9)] == ["point_on_circle"]


def test_alignment_with_points_on_one_or_both_axes():
    sk = base()
    s = find_snap(sk, (25.3, 30.4), tol=1, grid=1)  # level with nothing: grid
    assert s.kind == "grid" and s.uv == (25, 30)
    s = find_snap(sk, (60.3, 30.4), tol=1, grid=1)  # straight above the circle's centre
    assert s.kind == "align" and s.align == [("v", 4)] and s.uv == (60, 30)
    assert [(c.kind, c.refs) for c in s.constraints(9)] == [("pvertical", [4, 9])]
    s = find_snap(sk, (60.4, 19.6), tol=1, grid=1)  # above the centre and level with the top corners
    assert s.kind == "align" and {a for a, _ in s.align} == {"h", "v"} and s.uv == (60, 20)


def test_alignment_with_the_start_point_crossing_a_line():
    sk = base()
    s = find_snap(sk, (39.6, 7.5), tol=1, grid=1, start=(None, (-30, 7)))
    assert s.kind == "on_line" and s.uv == pytest.approx((40, 7)) and s.align == [("h", START)]
    assert [(c.kind, c.refs) for c in s.constraints(9, start=8)] == [("point_on_line", [9, 1]),
                                                                      ("phorizontal", [8, 9])]


def test_free_placement_and_out_of_reach():
    sk = base()
    assert find_snap(sk, (40.3, 19.6), tol=1, grid=1, smart=False).uv == (40, 20)  # grid, but no point
    assert find_snap(sk, (40.3, 19.6), tol=1, grid=1, smart=False).point is None
    assert find_snap(sk, (20.3, -0.4), tol=0.2, grid=0).kind == "free"


def test_grid_step_follows_zoom():
    assert grid_step(1, 0.05) == 1  # 1 mm is 20 px
    assert grid_step(1, 0.3) == 2
    assert grid_step(1, 0.5) == 5
    assert grid_step(1, 3) == 20
    assert grid_step(0, 3) == 0  # off stays off


# -- the constraints drawing adds

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


def _line(w, *pts, mods=Qt.NoModifier):
    w.set_tool("line")
    for p in pts:
        _click(w, p, mods)
    w.cancel()


def _kinds(sk):
    return sorted(c.kind for c in sk.constraints)


def test_line_drawn_level_with_its_start_is_horizontal(qapp, monkeypatch):
    w = _window(monkeypatch)
    _line(w, (-20, 5), (20, 5.4))
    sk = w.sketch
    assert _kinds(sk) == ["horizontal"]
    assert sk.points[0][1] == pytest.approx(sk.points[1][1])


def test_line_from_a_midpoint_stays_on_it(qapp, monkeypatch):
    w = _window(monkeypatch)
    _line(w, (-20, 0), (20, 0))
    _line(w, (0.3, 0.2), (8, 25))
    sk = w.sketch
    assert "midpoint" in _kinds(sk)
    sk.points[1][0] = 40  # stretch the first line: the second one's start follows its middle
    assert sk.solve()
    mid = np.mean([sk.points[0], sk.points[1]], axis=0)
    assert sk.points[1][0] != pytest.approx(20) and sk.points[2] == pytest.approx(list(mid), abs=1e-6)


def test_line_ending_on_a_circle(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.set_tool("circle")
    _click(w, (0, 0))
    _click(w, (10, 0))
    _line(w, (-30, 25), (6.2, 7.8))
    sk = w.sketch
    assert "point_on_circle" in _kinds(sk)
    end = sk.lines[-1][1]
    assert math.dist(sk.points[end], sk.points[sk.circles[0][0]]) == pytest.approx(10)


def test_rectangle_on_existing_points_stays_attached(qapp, monkeypatch):
    w = _window(monkeypatch)
    _line(w, (-30, -10), (-10, -25))  # points 0 and 1
    w.set_tool("rect")
    _click(w, (-10.2, -25.3))  # on point 1
    _click(w, (15, 5))
    sk = w.sketch
    assert ("coincident", [1, 2]) in [(c.kind, c.refs) for c in sk.constraints]
    sk.points[2] = [-12, -27]  # move the corner: the line's end comes with it
    assert sk.solve() and sk.points[1] == pytest.approx(sk.points[2], abs=1e-6)
    assert sk.points[1] != pytest.approx([-10, -25])


def test_circle_drawn_to_a_point_passes_through_it(qapp, monkeypatch):
    w = _window(monkeypatch)
    _line(w, (20, 20), (30, 30))
    w.set_tool("circle")
    _click(w, (0, 0))
    _click(w, (20.2, 19.7))
    sk = w.sketch
    assert ("point_on_circle", [0, 0]) in [(c.kind, c.refs) for c in sk.constraints]


def test_ctrl_places_freely(qapp, monkeypatch):
    w = _window(monkeypatch)
    _line(w, (-20, 0), (20, 0))
    _line(w, (0.3, 0.2), (8, 0.3), mods=Qt.ControlModifier)
    sk = w.sketch
    assert _kinds(sk) == ["horizontal"]  # only the first line's
    assert len(sk.points) == 4


def test_contradicting_snaps_are_dropped(qapp, monkeypatch):
    w = _window(monkeypatch)
    _line(w, (0, 0), (20, 7))
    sk = w.sketch
    sk.constraints.append(Constraint("distance", [0, 1], 20.0))
    assert sk.solve()
    w._auto_constrain([Constraint("distance", [0, 1], 30.0),  # contradicts: dropped
                       Constraint("pvertical", [0, 1])])  # fits: kept
    assert [c.kind for c in sk.constraints] == ["distance", "pvertical"]
    assert sk.solve() and sk.points[1][0] == pytest.approx(sk.points[0][0])


def test_hover_shows_the_snap(qapp, monkeypatch):
    w = _window(monkeypatch)
    _line(w, (-20, 0), (20, 0))
    w.set_tool("line")
    x, y = w.view.camera.project(np.array([(0.2, 0.3, 0)], float), 1000, 700)[0][:2]
    w.on_hover(x, y)
    assert w.view.snap_mark is not None and w.view.snap_mark["kind"] == "midpoint"
    w.set_tool("select")
    assert w.view.snap_mark is None
