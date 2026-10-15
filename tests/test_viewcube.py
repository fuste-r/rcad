import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication

from rcad import viewcube
from rcad.camera import Camera
from rcad.viewport import Viewport

FACES = ["Top", "Bottom", "Front", "Back", "Left", "Right"]


def camera_facing(name):
    cam = Camera()
    normal, right = viewcube.view_for(name)
    cam.look_at_plane(cam.target, normal, right)
    return cam


def test_six_labelled_faces():
    assert sorted(viewcube.FACES) == sorted(FACES)


def test_default_view_shows_three_faces():
    vis = viewcube.visible_faces(viewcube.rotation(Camera()))
    assert len(vis) == 3 and "Top" in vis and "Front" in vis


@pytest.mark.parametrize("name", FACES)
def test_each_face_view_is_head_on_and_pickable(name):
    cam = camera_facing(name)
    rot = viewcube.rotation(cam)
    assert viewcube.visible_faces(rot) == [name]
    r = (0, 0, viewcube.SIZE)
    cx, cy = viewcube.to_screen(rot, viewcube.FACES[name][0], r)
    assert viewcube.pick(rot, cx, cy, r) == name
    assert np.allclose(cam.back, viewcube.FACES[name][0])


def test_directions_match_the_labels():
    r = (0, 0, viewcube.SIZE)
    centre = viewcube.to_screen(np.eye(3), (0, 0, 0), r)

    def offset(name, axis):
        x, y = viewcube.to_screen(viewcube.rotation(camera_facing(name)), axis, r)
        return x - centre[0], y - centre[1]

    assert offset("Front", (1, 0, 0))[0] > 0  # right side of the model is on the right
    assert offset("Front", (0, 0, 1))[1] < 0  # up is up
    assert offset("Back", (1, 0, 0))[0] < 0  # seen from behind, +X is on the left
    assert offset("Right", (0, 1, 0))[0] > 0  # looking at the right face, +Y points right
    assert offset("Left", (0, 1, 0))[0] < 0
    assert offset("Top", (0, 1, 0))[1] < 0  # from above, +Y is up the screen
    for side in ("Front", "Back", "Left", "Right"):
        assert offset(side, (0, 0, 1))[1] < 0


@pytest.mark.parametrize("name", viewcube.CORNERS)
def test_each_corner_selects_its_isometric_view(name):
    normal, right = viewcube.view_for(name)
    cam = Camera()
    cam.look_at_plane(cam.target, normal, right)
    rot = viewcube.rotation(cam)
    r = (0, 0, viewcube.SIZE)
    x, y = viewcube.to_screen(rot, np.asarray(viewcube.CORNERS[name]) * 1.025, r)

    assert viewcube.pick(rot, x, y, r) == name
    assert np.allclose(cam.back, np.asarray(viewcube.CORNERS[name]) / np.sqrt(3))
    assert np.dot(cam.up, (0, 0, 1)) > 0


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _event(kind, x, y, button=Qt.NoButton):
    return QMouseEvent(kind, QPointF(x, y), QPointF(x, y), button, button, Qt.NoModifier)


def move(vp, x, y):
    vp.mouseMoveEvent(_event(QEvent.MouseMove, x, y))


def click(vp, x, y):
    vp.mousePressEvent(_event(QEvent.MouseButtonPress, x, y, Qt.LeftButton))
    vp.mouseReleaseEvent(_event(QEvent.MouseButtonRelease, x, y, Qt.LeftButton))


def test_clicking_the_cube_snaps_the_view(qapp):
    vp = Viewport()
    vp.resize(800, 600)
    clicks = []
    vp.clicked.connect(lambda *a: clicks.append(a))

    rect = vp._cube_rect()
    rot = viewcube.rotation(vp.camera)
    x, y = viewcube.to_screen(rot, viewcube.FACES["Top"][0], rect)
    move(vp, x, y)
    assert vp.cube_hover == "Top"

    click(vp, x, y)
    assert np.allclose(vp.camera.back, (0, 0, 1))
    assert clicks == []  # the cube swallows the click

    click(vp, 50, 400)
    assert len(clicks) == 1

    move(vp, 50, 400)
    assert vp.cube_hover is None


@pytest.mark.parametrize("name", viewcube.CORNERS)
def test_clicking_a_corner_snaps_to_isometric_view(qapp, name):
    vp = Viewport()
    vp.resize(800, 600)
    normal, right = viewcube.view_for(name)
    vp.camera.look_at_plane(vp.camera.target, normal, right)
    rot = viewcube.rotation(vp.camera)
    x, y = viewcube.to_screen(rot, np.asarray(viewcube.CORNERS[name]) * 1.025, vp._cube_rect())

    click(vp, x, y)

    assert np.allclose(vp.camera.back, normal)


def test_cube_follows_camera_orbit(qapp):
    vp = Viewport()
    before = viewcube.visible_faces(viewcube.rotation(vp.camera))
    vp.camera.orbit(180, 0)
    after = viewcube.visible_faces(viewcube.rotation(vp.camera))
    assert "Front" in before and "Back" in after and "Front" not in after
