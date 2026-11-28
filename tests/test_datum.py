import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad import datum
from rcad.app import MainWindow
from rcad.kernel import Feature
from rcad.sketch import Sketch, standard_plane


def test_planes_follow_the_view_cube_names():
    assert {n: datum.PLANES[n][0] for n in datum.PLANES} == {"Top": "XY", "Front": "XZ", "Right": "YZ"}
    for name in datum.PLANES:
        n, u, v = datum.axes(name)
        assert np.allclose(np.cross(u, v), n)  # right x up points out of the plane, towards the viewer
        sp = standard_plane(datum.PLANES[name][0])
        assert abs(np.dot(sp.normal, n)) == pytest.approx(1)


def test_label_sits_inside_the_top_left_corner():
    a = datum.label_anchor("Front", 25)
    assert a == pytest.approx((-25 + 1.25, 0, 25 - 1.25))
    assert datum.label_anchor("Right", 25) == pytest.approx((0, -23.75, 23.75))


def test_plane_size_reaches_past_the_model():
    assert datum.plane_half(None) == datum.MIN_HALF
    assert datum.plane_half(np.array([[10, 5, 3]])) == datum.MIN_HALF
    assert datum.plane_half(np.array([[0, 0, 0], [70, -20, 10]])) == 100  # 84 rounded up to 1-2-5
    assert datum.plane_half(np.array([[0, 0, 0], [140, 0, 0]])) == 200


def test_pick_nearest_plane_inside_its_square():
    assert datum.pick((10, -5, 100), (0, 0, -1), 25) == ("Top", pytest.approx(100))
    assert datum.pick((10, -100, 5), (0, 1, 0), 25) == ("Front", pytest.approx(100))
    assert datum.pick((40, -5, 100), (0, 0, -1), 25) is None  # outside the square
    assert datum.pick((10, -5, 100), (1, 0, 0), 25) is None  # runs along Top, never reaches the others
    # looking down at 45 degrees from in front: whichever plane the ray meets first
    d = np.array([0, 1, -1]) / np.sqrt(2)
    assert datum.pick((5, -40, 20), d, 25)[0] == "Top"  # meets z = 0 at y = -20, before y = 0
    assert datum.pick((5, -20, 40), d, 25)[0] == "Front"  # meets y = 0 at z = 20, before z = 0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.view.camera.look_at_plane((0, 0, 0), (1, -1, 1), (1, 1, 0), 150)  # isometric
    return w


def _px(w, p):
    v = w.view
    return tuple(v.camera.project(np.array([p], float), v.width(), v.height())[0])


def _key(w, key, text=""):
    QApplication.sendEvent(w.view, QKeyEvent(QKeyEvent.KeyPress, key, Qt.NoModifier, text))


# seen from the front-right-top, each of these is on the side of the other planes facing the camera
ON = {"Top": (12, -12, 0), "Front": (12, 0, 12), "Right": (0, -12, 12)}


def test_hover_and_click_each_plane(qapp, monkeypatch):
    w = _window(monkeypatch)
    for name, p in ON.items():
        assert w.view.pick_plane(*_px(w, p)) == name
        w.on_hover(*_px(w, p))
        assert w.view.hover_plane == name
    w.on_click(*_px(w, ON["Front"]), Qt.LeftButton, Qt.NoModifier)
    assert w.view.sel_plane == "Front" and w.sel_tag.text() == "Front plane"
    assert [b.text() for b in w.quick.buttons][:1] == ["✏️ Sketch"]
    w.on_click(*_px(w, ON["Front"]), Qt.LeftButton, Qt.NoModifier)  # again: dropped
    assert w.view.sel_plane is None
    w.on_click(*_px(w, ON["Top"]), Qt.LeftButton, Qt.NoModifier)
    w.cancel()  # Esc
    assert w.view.sel_plane is None and w.sel_tag.text() == ""


def test_sketch_on_a_picked_plane(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.on_click(*_px(w, ON["Right"]), Qt.LeftButton, Qt.NoModifier)
    w.quick.buttons[0].click()
    assert w.mode == "sketch" and w.view.sel_plane is None
    pl = w.sketch.plane
    assert pl == standard_plane("YZ")
    assert w.view.pick_plane(*_px(w, ON["Right"])) is None  # planes are out of the way while sketching


def test_model_faces_and_regions_win_over_planes(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(5, 5, 40, 40)
    w.doc.features += [Feature("sketch", "S", sketch=sk)]
    w.refresh()
    w.on_click(*_px(w, (22, 22, 0)), Qt.LeftButton, Qt.NoModifier)
    assert w.view.sel_plane is None and sk.selected_regions()  # the region, not the Top plane
    w.doc.features.append(Feature("extrude", sketch_ref=0, value=8))
    w.doc.rebuild()
    w.refresh()
    w.cancel()
    w.on_click(*_px(w, (22, 22, 8)), Qt.LeftButton, Qt.NoModifier)  # the block's top face
    assert w.view.sel_plane is None and w.view.sel_faces


def test_p_toggles_the_planes(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.on_click(*_px(w, ON["Top"]), Qt.LeftButton, Qt.NoModifier)
    _key(w, Qt.Key_P, "p")
    assert not w.view.show_planes and w.view.sel_plane is None
    assert w.view.pick_plane(*_px(w, ON["Top"])) is None
    assert "hidden" in w.statusBar().currentMessage()
    _key(w, Qt.Key_P, "p")
    assert w.view.show_planes and w.view.pick_plane(*_px(w, ON["Top"])) == "Top"


def test_p_typed_into_a_size_is_not_a_shortcut(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.doc.variables["pitch"] = 4.0
    w.new_sketch()
    w.set_tool("line")
    w.on_click(*_px(w, (0, 0, 0)), Qt.LeftButton, Qt.NoModifier)
    _key(w, Qt.Key_P, "p")  # the first letter of a variable, typed into the length box
    assert w.view.show_planes
    assert w.fields.boxes["L"].text().startswith("p")
