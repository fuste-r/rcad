import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from build123d import Box, Cylinder, Pos
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow
from rcad.kernel import build_mesh


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


MID = (20, 0, 10)  # top front edge of the 40 x 30 x 10 box
VIEW = (0.15, -0.3, 0.94)  # from above and a little in front: pulling that edge away is down the screen
AWAY = np.array([0, -1, 1]) / np.sqrt(2)


def _edge_at(mesh, mid):
    return min(range(len(mesh.edges)), key=lambda i: np.linalg.norm(mesh.edges[i].mid - np.asarray(mid, float)))


def test_edge_away_direction_points_out_of_the_material():
    mesh = build_mesh(Box(10, 10, 10) - Pos(0, 0, 5) * Cylinder(2, 4))  # box with a blind hole on top
    s = 1 / np.sqrt(2)
    top_front = mesh.edge_away(_edge_at(mesh, (0, -5, 5)))
    assert np.allclose(top_front, (0, -s, s), atol=1e-6)  # convex: diagonally out of the box
    rim = mesh.edge_away(_edge_at(mesh, (-2, 0, 5)))
    assert np.allclose(rim, (s, 0, s), atol=1e-6)  # hole rim: up and into the hole
    bottom = mesh.edge_away(_edge_at(mesh, (-2, 0, 3)))
    assert np.allclose(bottom, (s, 0, s), atol=1e-6)  # concave hole bottom: into the empty hole


def _box_window(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.sketch.add_rect(0, 0, 40, 30)
    w.sketch.solve()
    w.finish_sketch()
    w.extrude()
    w.tool_bar.finish()
    w.view.camera.look_at_plane((20, 15, 5), VIEW, (1, 0, 0), 150)
    return w


def _screen(v, p):
    s = v.camera.project(np.array([p], float), v.width(), v.height())[0]
    return QPoint(int(round(s[0])), int(round(s[1])))


def _drag(w, mid, offsets, release=True):
    """Press on the edge at mid and move by the given world offsets from it (in order)."""
    v = w.view
    start = _screen(v, mid)
    QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, start)
    for off in offsets:
        QTest.mouseMove(v, _screen(v, np.asarray(mid, float) + np.asarray(off, float)))
    if release:
        end = _screen(v, np.asarray(mid, float) + np.asarray(offsets[-1], float))
        QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, end)




def test_pull_away_from_body_chamfers(qapp, monkeypatch):
    w = _box_window(monkeypatch)
    assert w.view.pick_edge(*_screen(w.view, MID).toTuple()) == _edge_at(w.doc.mesh, MID)
    vol = w.doc.part.volume
    _drag(w, MID, [AWAY * 1, AWAY * 3])
    f = w.doc.features[-1]
    assert f.kind == "chamfer" and f.value == pytest.approx(3.0, abs=0.3)
    assert w.doc.part.volume == pytest.approx(vol - 40 * f.value ** 2 / 2, rel=1e-3)
    assert w.view._value_post["kind"] == "chamfer"


def test_push_into_body_fillets(qapp, monkeypatch):
    w = _box_window(monkeypatch)
    _drag(w, MID, [-AWAY * 1, -AWAY * 3])
    f = w.doc.features[-1]
    assert f.kind == "fillet" and f.value == pytest.approx(3.0, abs=0.3)
    assert f.result_edges  # can be grabbed and dragged again


def test_direction_can_change_mid_drag(qapp, monkeypatch):
    w = _box_window(monkeypatch)
    n = len(w.doc.features)
    _drag(w, MID, [AWAY * 2, AWAY * 0.5, -AWAY * 2], release=False)
    assert len(w.doc.features) == n + 1 and w.doc.features[-1].kind == "fillet"
    QTest.mouseMove(w.view, _screen(w.view, np.asarray(MID) + AWAY * 2))
    assert w.doc.features[-1].kind == "chamfer"  # one live feature, switched
    QTest.mouseRelease(w.view, Qt.LeftButton, Qt.NoModifier, _screen(w.view, np.asarray(MID) + AWAY * 2))
    assert len(w.doc.features) == n + 1 and w.doc.features[-1].kind == "chamfer"


def test_value_box_retypes_a_dragged_chamfer(qapp, monkeypatch):
    w = _box_window(monkeypatch)
    _drag(w, MID, [AWAY * 2])
    f = w.doc.features[-1]
    assert f.kind == "chamfer" and w._value_feature is f
    w.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, 0, Qt.NoModifier, "4"))
    w.view.value_box.returnPressed.emit()
    assert f.kind == "chamfer" and f.value == pytest.approx(4.0) and not w.doc.errors


def test_pulling_an_existing_fillet_turns_it_into_a_chamfer_and_esc_restores(qapp, monkeypatch):
    w = _box_window(monkeypatch)
    _drag(w, MID, [-AWAY * 2])
    fillet = w.doc.features[-1]
    assert fillet.kind == "fillet"
    n = len(w.doc.features)
    w.view.dismiss_value_box()
    rmid = np.asarray(min(fillet.result_edges, key=lambda m: m[2]), float)  # the edge on the front face
    w.view.camera.look_at_plane((20, 15, 5), VIEW, (1, 0, 0), 150)
    edge = w.view.pick_edge(*_screen(w.view, rmid).toTuple())
    assert edge is not None
    away = w.doc.mesh.edge_away(edge)
    _drag(w, rmid, [away * 1, away * 3], release=False)
    assert len(w.doc.features) == n and fillet.kind == "chamfer"  # same feature, switched
    w.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, 0, Qt.NoModifier, "5"))
    w.view.value_box.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert fillet.kind == "fillet" and fillet.value == pytest.approx(2.0, abs=0.3)
    assert len(w.doc.features) == n


def test_pull_is_down_the_screen_from_above_and_along_the_edge_does_nothing(qapp, monkeypatch):
    w = _box_window(monkeypatch)
    v = w.view
    start = _screen(v, MID)
    QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(v, _screen(v, (30, 0, 10)))  # along the edge
    assert v._fillet_state["radius"] < 0.5  # only pixel rounding (1 px is ~0.5 mm here)
    QTest.mouseMove(v, start + QPoint(0, 20))  # down: away from the box
    assert v._fillet_state["kind"] == "chamfer" and v._fillet_state["radius"] > 1
    QTest.mouseMove(v, start + QPoint(0, -20))  # up: into the box
    assert v._fillet_state["kind"] == "fillet" and v._fillet_state["radius"] > 1
    QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, start + QPoint(0, -20))
    assert w.doc.features[-1].kind == "fillet"


def test_edge_facing_the_viewer_uses_screen_up_as_pull(qapp, monkeypatch):
    w = _box_window(monkeypatch)
    v = w.view
    # look straight down the edge's away direction: it doesn't show on screen at all
    v.camera.look_at_plane((20, 0, 10), tuple(AWAY), (1, 0, 0), 150)
    start = _screen(v, MID)
    assert v.pick_edge(start.x(), start.y()) == _edge_at(w.doc.mesh, MID)
    QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, start)
    QTest.mouseMove(v, start + QPoint(0, -30))
    assert v._fillet_state["kind"] == "chamfer"
    QTest.mouseMove(v, start + QPoint(0, 30))
    assert v._fillet_state["kind"] == "fillet"
    QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, start + QPoint(0, 30))
