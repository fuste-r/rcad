import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from rcad import app as appmod
from rcad.app import MainWindow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def px(w, p):
    v = w.view
    return tuple(v.camera.project(np.array([p], float), v.width(), v.height())[0])


def click(w, p, mods=Qt.NoModifier):
    x, y = px(w, p)
    w.on_click(x, y, Qt.LeftButton, mods)


def test_full_workflow(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)

    # sketch a rectangle on XY and dimension it
    w.plane_combo.setCurrentText("XY")
    w.new_sketch()
    assert w.mode == "sketch"
    w.set_tool("rect")
    click(w, (0, 0, 0))
    click(w, (40, 30, 0))
    sk = w.sketch
    assert len(sk.points) == 4 and len(sk.lines) == 4

    w.set_tool("select")
    click(w, (20, 0, 0))  # bottom edge
    assert w.view.sketch_sel == {("l", 0)}
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (50.0, True))
    w.add_constraint("distance")
    assert sk.constraints[-1].kind == "distance"
    xs = sorted({round(p[0], 4) for p in sk.points})
    assert xs[1] - xs[0] == pytest.approx(50.0)

    # a conflicting dimension is rejected and rolled back
    n = len(sk.constraints)
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (60.0, True))
    w.view.sketch_sel = {("l", 0)}
    w.add_constraint("distance")
    assert len(sk.constraints) == n

    # a circle with a diameter
    w.set_tool("circle")
    click(w, (25, 15, 0))
    click(w, (30, 15, 0))
    assert len(sk.circles) == 1
    w.set_tool("select")
    w.view.sketch_sel = {("c", 0)}
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (8.0, True))
    w.add_constraint("diameter")
    assert sk.circles[0][1] == pytest.approx(4.0)

    w.finish_sketch()
    assert w.mode == "model"

    # extrude
    monkeypatch.setattr(appmod.ExtrudeDialog, "exec", lambda self: True)
    w.extrude()
    assert not w.doc.errors
    vol = 50 * 30 * 10 - np.pi * 16 * 10
    assert w.doc.part.volume == pytest.approx(vol, rel=1e-3)

    # push/pull the top face
    w.view.camera.look_at_plane((25, 15, 0), (0.3, -0.6, 0.74), (1, 0, 0), 150)
    top = [i for i, f in enumerate(w.doc.mesh.faces) if f.normal is not None and f.normal[2] > 0.99]
    assert top
    w.view.sel_faces = {top[0]}
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (5.0, True))
    w.pushpull()
    assert not w.doc.errors
    assert w.doc.part.volume == pytest.approx(vol + (50 * 30 - np.pi * 16) * 5, rel=1e-3)

    # fillet / chamfer a picked outer edge
    edge = max(range(len(w.doc.mesh.edges)), key=lambda i: w.doc.mesh.edges[i].mid[2] + w.doc.mesh.edges[i].mid[0])
    w.view.sel_edges = {edge}
    before = w.doc.part.volume
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (1.0, True))
    w.fillet()
    assert not w.doc.errors and w.doc.part.volume < before

    # picking in model mode selects faces
    w.view.camera.look_at_plane((25, 15, 0), (0.3, -0.6, 0.74), (1, 0, 0), 150)
    f = w.view.pick_face(*px(w, (25, 15, 15)))
    assert f is not None

    # save / reload
    path = str(tmp_path / "proj.rcad")
    w.path = path
    w.file_save()
    w2 = MainWindow()
    w2._load(appmod.Document.load(path), path)
    assert not w2.doc.errors
    assert w2.doc.part.volume == pytest.approx(w.doc.part.volume, rel=1e-9)

    # deleting the sketch removes its extrude
    w.doc.delete(0)
    assert [f.kind for f in w.doc.features][:1] != ["extrude"]
