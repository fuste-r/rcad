import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QResizeEvent
from PySide6.QtWidgets import QApplication, QMessageBox, QToolBar

from rcad.app import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    return w


def _menu_action(w, page, group, text):
    menu = w.tool_line.groups[(page, group)].menu()
    return next(a for a in menu.actions() if text in a.text())


def test_one_line_at_the_bottom_replaces_the_model_toolbars(qapp, monkeypatch):
    w = _window(monkeypatch)
    line = w.tool_line
    assert line.page == "model" and not line.isHidden()
    assert [k[1] for k in line.groups if k[0] == "model"] == ["Sketch", "Create", "Modify"]
    names = {t.windowTitle() for t in w.findChildren(QToolBar)}
    assert not names & {"Model", "Sketch", "Constraints"}  # the old tool toolbars are gone
    # hidden widgets get resize events late (and a GL repaint fails headless): hand them to the line
    line.eventFilter(w.view, QResizeEvent(QSize(1000, 700), QSize(640, 480)))
    assert abs((line.x() + line.width() / 2) - 500) <= 1 and line.y() + line.height() <= 700
    w.view.resize(800, 500)
    line.eventFilter(w.view, QResizeEvent(QSize(800, 500), QSize(1000, 700)))
    assert abs((line.x() + line.width() / 2) - 400) <= 1 and line.y() + line.height() <= 500


def test_sketch_list_starts_a_sketch_on_the_chosen_plane(qapp, monkeypatch):
    w = _window(monkeypatch)
    _menu_action(w, "model", "Sketch", "XZ").trigger()
    assert w.mode == "sketch" and w.tool_line.page == "sketch"
    assert np.allclose(np.abs(w.sketch.plane.normal), (0, 1, 0))
    _menu_action(w, "sketch", "Draw", "Rectangle").trigger()
    assert w.tool == "rect" and w.draw_button.text() == "Draw: Rectangle ▾"
    w.sketch.add_rect(0, 0, 10, 10)
    w.sketch.solve()
    exit_button = next(b for b in w.tool_line.pages["sketch"].findChildren(type(w.draw_button))
                       if b.defaultAction() is not None and "Exit" in b.defaultAction().text())
    exit_button.click()
    assert w.mode == "model" and w.tool_line.page == "model"


def test_settings_bar_takes_the_lines_place(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.new_sketch()
    w.sketch.add_rect(0, 0, 40, 30)
    w.sketch.solve()
    w.finish_sketch()
    _menu_action(w, "model", "Create", "Extrude").trigger()
    assert w.tool_bar.active and w.tool_line.isHidden()
    w.tool_bar.finish()
    assert not w.tool_line.isHidden() and w.tool_line.page == "model"
    _menu_action(w, "model", "Create", "Extrude").trigger()
    w.tool_bar.cancel()
    assert not w.tool_line.isHidden()


def test_constrain_and_modify_lists_reach_the_tools(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.new_sketch()
    w.sketch.add_rect(0, 0, 40, 30)
    w.sketch.solve()
    w.view.sketch_sel = {("l", 0)}
    _menu_action(w, "sketch", "Constrain", "Distance").trigger()
    assert w.tool_bar.active and w.tool_line.isHidden()  # dimensions use the settings bar too
    w.tool_bar.finish()
    assert w.sketch.constraints[-1].kind == "distance"
    n = len(w.sketch.lines)
    w.view.sketch_sel = {("l", 0)}
    _menu_action(w, "sketch", "Modify", "Delete selection").trigger()
    assert len(w.sketch.lines) == n - 1
