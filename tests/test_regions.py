import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad import app as appmod
from rcad.kernel import Document, Feature
from rcad.sketch import Sketch, standard_plane


def two_rects():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 10)
    sk.add_rect(30, 0, 50, 10)
    return sk


def test_region_lookup_with_holes_and_islands():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 100, 100)
    sk.add_circle(sk.add_point(50, 50), 30)  # hole
    sk.add_rect(45, 45, 55, 55)  # island inside the hole
    assert len(sk.regions()) == 2
    ring, hole, island = sk.region_at(5, 5), sk.region_at(50, 50 + 20), sk.region_at(50, 50)
    assert ring is not None and hole is None and island is not None and ring != island


def test_extrude_only_selected_region():
    sk = two_rects()
    doc = Document([Feature("sketch", "S", sketch=sk),
                    Feature("extrude", sketch_ref=0, value=5, profiles=[[10, 5]])])
    doc.rebuild()
    assert not doc.errors
    assert doc.part.volume == pytest.approx(20 * 10 * 5)

    doc.features[1].profiles = None
    doc.rebuild()
    assert doc.part.volume == pytest.approx(2 * 20 * 10 * 5)

    doc.features[1].profiles = [[100, 100]]  # nothing there
    doc.rebuild()
    assert 1 in doc.errors


def test_selection_toggle_and_persistence(tmp_path):
    sk = two_rects()
    sk.toggle_region(10, 5, False)
    sk.toggle_region(40, 5, True)
    assert len(sk.selected_regions()) == 2
    sk.toggle_region(12, 4, True)  # second click in A deselects it
    assert len(sk.selected_regions()) == 1
    sk.toggle_region(100, 100, False)  # empty space clears
    assert sk.selection == []
    sk.toggle_region(10, 5, False)
    doc = Document([Feature("sketch", "S", sketch=sk)])
    path = str(tmp_path / "r.rcad")
    doc.save(path)
    assert Document.load(path).features[0].sketch.selected_regions() == sk.selected_regions()
    tris = sk.region_triangles(next(iter(sk.selected_regions())))
    assert tris.shape[1:] == (3, 3) and len(tris) >= 2


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _px(w, p):
    v = w.view
    return tuple(v.camera.project(np.array([p], float), v.width(), v.height())[0])


def _click(w, p, mods=Qt.NoModifier):
    x, y = _px(w, p)
    w.on_click(x, y, Qt.LeftButton, mods)


def test_ui_highlight_and_extrude_selected(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(appmod.ExtrudeDialog, "exec", lambda self: True)
    w = appmod.MainWindow()
    w.view.resize(1000, 700)
    w.plane_combo.setCurrentText("XY")
    w.new_sketch()
    w.view.camera.dist = 120
    w.set_tool("rect")
    _click(w, (-30, 0, 0))
    _click(w, (-10, 10, 0))
    assert w.tool == "select" and w.tool_actions["select"].isChecked()  # drawing ends the tool
    w.set_tool("rect")
    _click(w, (10, 0, 0))
    _click(w, (30, 10, 0))
    assert w.tool == "select"
    assert w.view.region_fill == []

    x, y = _px(w, (0, 5, 0))
    w.on_hover(*_px(w, (-20, 5, 0)))
    assert len(w.view.region_fill) == 1  # hover highlight

    _click(w, (-20, 5, 0))
    assert len(w.sketch.selected_regions()) == 1
    assert any(color[0] == 1.0 for _, color in w.view.region_fill)  # selected highlight

    w.extrude_from_sketch()
    assert not w.doc.errors
    assert w.doc.features[1].profiles is not None and len(w.doc.features[1].profiles) == 1
    assert w.doc.part.volume == pytest.approx(20 * 10 * 10)


def test_model_extrude_asks_for_region_when_ambiguous(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = appmod.MainWindow()
    w.doc.features.append(Feature("sketch", "S", sketch=two_rects()))
    w.refresh()
    w.extrude()
    assert w.mode == "sketch" and w.tool == "select"  # sent back to pick regions
    w.set_tool("line")
    w.cancel()  # Esc with nothing in progress returns to Select
    assert w.tool == "select"


def test_select_regions_in_model_mode(qapp, monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(appmod.ExtrudeDialog, "exec", lambda self: True)
    w = appmod.MainWindow()
    w.view.resize(1000, 700)
    w.doc.features.append(Feature("sketch", "S", sketch=two_rects()))
    w.refresh()
    w.view.camera.look_at_plane((25, 5, 0), (0, 0, 1), (1, 0, 0), 120)
    w.on_hover(*_px(w, (10, 5, 0)))
    assert w.hover_region is not None and len(w.view.region_fill) == 1
    _click(w, (10, 5, 0))
    assert w.doc.features[0].sketch.selected_regions() and w.region_sketch == 0
    assert any(c[0] == 1.0 for _, c in w.view.region_fill)
    _click(w, (40, 5, 0), Qt.ControlModifier)
    assert len(w.doc.features[0].sketch.selected_regions()) == 2
    _click(w, (40, 5, 0), Qt.ControlModifier)
    assert len(w.doc.features[0].sketch.selected_regions()) == 1
    w.extrude()  # selection made outside sketch mode drives the extrude
    assert w.mode == "model" and not w.doc.errors
    assert w.doc.features[1].profiles is not None and len(w.doc.features[1].profiles) == 1
    assert w.doc.part.volume == pytest.approx(20 * 10 * 10)

    # a region behind the new body is not pickable; hiding the layer removes it from picking
    w.view.camera.look_at_plane((25, 5, 0), (0, 0, 1), (1, 0, 0), 120)
    assert w._pick_region(*_px(w, (10, 5, 0))) is None
    w.cancel()
    assert w.doc.features[0].sketch.selection == [] and w.view.region_fill == []


def test_coplanar_region_wins_over_face(qapp):
    w = appmod.MainWindow()
    w.view.resize(1000, 700)
    w.doc.features += [Feature("sketch", "S", sketch=two_rects()), Feature("extrude", sketch_ref=0, value=-5)]
    w.doc.rebuild()
    w.refresh()
    w.view.camera.look_at_plane((25, 5, 0), (0, 0, 1), (1, 0, 0), 120)
    _click(w, (10, 5, 0))
    assert w.doc.features[0].sketch.selected_regions() and not w.view.sel_faces
