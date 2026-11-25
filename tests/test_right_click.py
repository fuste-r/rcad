"""Right-click steps back: the open tool, then a half-drawn shape, then the drawing tool, then out
of the sketch. With something selected (or under the pointer), it opens its tools there."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)


def _px(w, p):
    return tuple(w.view.camera.project(np.array([p], float), w.view.width(), w.view.height())[0][:2])


def _left(w, p):
    w.on_click(*_px(w, p), Qt.LeftButton, Qt.NoModifier)


def _right(w, p):
    w.on_click(*_px(w, p), Qt.RightButton, Qt.NoModifier)


def _menu(w):
    return [a.text() for a in w.selection_menu.actions() if not a.isSeparator()]


def _sketching():
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.view.camera.look_at_plane((20, 15, 0), (0, 0, 1), (1, 0, 0), 120)
    return w


def _box():
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


def test_steps_back_through_a_half_drawn_line_the_tool_and_the_sketch(qapp, quiet):
    w = _sketching()
    w.set_tool("line")
    _left(w, (0, 0, 0))
    _left(w, (20, 0, 0))
    _left(w, (20, 10, 0))  # a chain of two lines, the third under way
    assert w.pending is not None and len(w.sketch.lines) == 2
    _right(w, (30, 20, 0))
    assert w.pending is None and w.tool == "line" and len(w.sketch.lines) == 2  # drawn lines kept
    _right(w, (30, 20, 0))
    assert w.tool == "select" and w.mode == "sketch"
    _right(w, (35, 25, 0))  # nothing selected, nothing under the pointer
    assert w.mode == "model"
    assert len(w.doc.features[0].sketch.lines) == 2


def test_right_click_with_a_rectangle_started_drops_only_that(qapp, quiet):
    w = _sketching()
    w.set_tool("rect")
    _left(w, (0, 0, 0))
    _right(w, (10, 10, 0))
    assert w.pending is None and not w.sketch.lines and w.tool == "rect" and w.mode == "sketch"


def test_right_click_on_sketch_geometry_offers_its_tools(qapp, quiet):
    w = _sketching()
    w.sketch.add_rect(0, 0, 40, 30)
    w.sketch.solve()
    w.set_tool("select")
    _right(w, (20, 0, 0))  # the bottom line, not selected yet: it is selected and its tools open
    assert w.view.sketch_sel == {("l", 0)} and w.mode == "sketch"
    assert _menu(w) == ["Distance", "Horizontal", "Vertical", "🗑 Delete"]
    n = len(w.sketch.lines)
    next(a for a in w.selection_menu.actions() if a.text() == "🗑 Delete").trigger()
    assert len(w.sketch.lines) == n - 1


def test_right_click_keeps_a_multi_selection_it_is_part_of(qapp, quiet):
    w = _sketching()
    a, b, c, d = (w.sketch.add_point(*p) for p in [(0, 0), (20, 2), (0, 10), (20, 15)])
    l1, l2 = w.sketch.add_line(a, b), w.sketch.add_line(c, d)
    w.view.sketch_sel = {("l", l1), ("l", l2)}
    _right(w, (10, 1, 0))  # on the first line
    assert w.view.sketch_sel == {("l", l1), ("l", l2)}
    assert _menu(w)[:2] == ["Parallel", "Perpendicular"]
    _right(w, (40, 40, 0))  # empty space, still with a selection: its tools, not out of the sketch
    assert w.mode == "sketch" and w.view.sketch_sel == {("l", l1), ("l", l2)}


def test_right_click_cancels_an_open_tool_and_puts_the_model_back(qapp, quiet):
    w = _box()
    vol = w.doc.part.volume
    _left(w, (20, 15, 10))
    w.pushpull()
    w.tool_bar.fields["value"].setText("5")
    assert w.tool_bar.active and w.doc.part.volume > vol
    _right(w, (20, 15, 10))
    assert not w.tool_bar.active and w.doc.features[-1].kind == "extrude"
    assert w.doc.part.volume == pytest.approx(vol)


def test_right_click_on_the_model_selects_and_offers_tools(qapp, quiet):
    w = _box()
    _right(w, (20, 15, 10))  # the top face
    assert len(w.view.sel_faces) == 1
    assert _menu(w) == ["✏️ Sketch", "↕️ Push/Pull"]
    next(a for a in w.selection_menu.actions() if "Push/Pull" in a.text()).trigger()
    assert w.tool_bar.active
    w.tool_bar.finish()
    assert w.doc.features[-1].kind == "pushpull"
    w.view.sel_faces.clear()
    w.view.sel_edges.clear()
    w._update_regions()
    w.selection_menu = None
    _right(w, (200, 200, 200))  # empty space, nothing selected: nothing happens
    assert w.selection_menu is None and w.mode == "model"


def test_right_click_on_an_edge_offers_fillet_and_chamfer(qapp, quiet):
    w = _box()
    _right(w, (20, 0, 10))
    assert len(w.view.sel_edges) == 1
    assert _menu(w) == ["◐ Fillet", "◺ Chamfer"]


def test_right_click_ends_a_mirror_pick(qapp, quiet):
    w = _sketching()
    w.sketch.add_rect(0, 0, 40, 30)
    w.sketch.solve()
    w.view.sketch_sel = {("l", 0)}
    w.add_constraint("symmetry")
    assert w._mirror_axis is not None
    _right(w, (40, 40, 0))
    assert w._mirror_axis is None and w.mode == "sketch"
