import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math

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
    assert len(sk.regions()) == 3  # every enclosed area can be picked, the circle's inside too
    ring, disc, island = sk.region_at(5, 5), sk.region_at(50, 50 + 20), sk.region_at(50, 50)
    assert None not in (ring, disc, island) and len({ring, disc, island}) == 3
    assert sk.region_at(-5, -5) is None
    # with nothing picked the circle is still a hole and the island solid again
    assert sorted(sk.filled_regions()) == sorted([ring, island])


def _extrude(sk, profiles, d=5):
    doc = Document([Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=d, profiles=profiles)])
    doc.rebuild()
    assert not doc.errors, doc.errors
    return doc.part


def test_overlapping_rects_split_into_three_regions():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 10)
    sk.add_rect(10, 5, 30, 20)
    assert len(sk.regions()) == 3
    a, both, b = sk.region_at(2, 2), sk.region_at(15, 7), sk.region_at(25, 15)
    assert None not in (a, both, b) and len({a, both, b}) == 3
    assert _extrude(sk, [[15, 7]]).volume == pytest.approx(10 * 5 * 5)  # just the overlap
    assert _extrude(sk, None).volume == pytest.approx((200 + 300 - 50) * 5)  # nothing picked: the union


def test_line_across_a_rect_divides_it():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 10)
    sk.add_line(sk.add_point(12, -5), sk.add_point(12, 15))  # free ends stick out
    assert len(sk.regions()) == 2
    assert _extrude(sk, [[3, 3]]).volume == pytest.approx(12 * 10 * 5)
    # both halves picked merge into one seamless block
    part = _extrude(sk, [[3, 3], [15, 3]])
    assert part.volume == pytest.approx(20 * 10 * 5) and len(part.faces()) == 6


def test_line_ending_on_a_corner_does_not_break_the_rect():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 10)
    sk.add_line(0, 2)  # diagonal between two of the rectangle's own corners
    assert len(sk.regions()) == 2
    assert _extrude(sk, None).volume == pytest.approx(20 * 10 * 5)


def test_open_line_inside_a_region_is_ignored():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 10)
    sk.add_line(sk.add_point(5, 5), sk.add_point(15, 5))
    sk.add_line(sk.add_point(30, 0), sk.add_point(40, 0))  # nowhere near anything
    assert len(sk.regions()) == 1
    part = _extrude(sk, [[2, 2]])
    assert part.volume == pytest.approx(200 * 5) and len(part.faces()) == 6


def test_circle_crossing_a_rect_edge():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 10)
    sk.add_circle(sk.add_point(20, 5), 4)
    assert len(sk.regions()) == 3
    half_disc = math.pi * 16 / 2
    assert _extrude(sk, [[19, 5]]).volume == pytest.approx(half_disc * 5, rel=1e-3)
    assert _extrude(sk, [[22, 5]]).volume == pytest.approx(half_disc * 5, rel=1e-3)
    assert _extrude(sk, [[5, 5]]).volume == pytest.approx((200 - half_disc) * 5, rel=1e-3)


def test_crossing_lines_enclose_a_triangle():
    sk = Sketch(standard_plane("XY"))
    for a, b in [((-5, 0), (25, 0)), ((-1, -2), (11, 22)), ((21, -2), (9, 22))]:
        sk.add_line(sk.add_point(*a), sk.add_point(*b))
    assert len(sk.regions()) == 1
    assert sk.region_at(10, 5) is not None and sk.region_at(10, 30) is None
    assert sk.filled_regions() == [0]


def test_regions_follow_geometry_edits():
    sk = two_rects()
    assert len(sk.regions()) == 2
    sk.points[1][0] = 35  # stretch the first rectangle into the second
    sk.points[2][0] = 35
    assert len(sk.regions()) == 3


def test_revolve_uses_picked_region():
    sk = Sketch(standard_plane("XZ"))
    sk.add_rect(10, 0, 20, 10)
    sk.add_line(sk.add_point(15, -5), sk.add_point(15, 15))
    doc = Document([Feature("sketch", "S", sketch=sk),
                    Feature("revolve", sketch_ref=0, value=360, axis="v", profiles=[[12, 5]])])
    doc.rebuild()
    assert not doc.errors
    assert doc.part.volume == pytest.approx(math.pi * (15 ** 2 - 10 ** 2) * 10, rel=1e-3)


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
    w.tool_bar.finish()
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


# -- picking and dragging regions in the model

from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest


def _cross_window(monkeypatch):
    """Model view of a sketch with a rectangle and a line dividing it, from above at an angle."""
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = appmod.MainWindow()
    w.view.resize(1000, 700)
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 40, 20)
    sk.add_line(sk.add_point(20, -5), sk.add_point(20, 25))
    w.doc.features.append(Feature("sketch", "S", sketch=sk))
    w.refresh()
    w.view.camera.look_at_plane((20, 10, 0), (0.0, -0.5, 0.87), (1, 0, 0), 150)
    return w, sk


def _qp(w, p):
    x, y = _px(w, p)
    return QPoint(int(round(x)), int(round(y)))


def _drag_up(w, start, height, end_offset=None):
    """Left drag from world point start straight up the screen by `height` mm along +Z."""
    v = w.view
    QTest.mousePress(v, Qt.LeftButton, Qt.NoModifier, _qp(w, start))
    for f in (0.3, 0.7, 1.0):
        QTest.mouseMove(v, _qp(w, np.add(start, (0, 0, height * f))))
    QTest.mouseRelease(v, Qt.LeftButton, Qt.NoModifier, _qp(w, np.add(start, (0, 0, height))))


def test_clicking_regions_shows_the_arrow_and_adds_without_ctrl(qapp, monkeypatch):
    w, sk = _cross_window(monkeypatch)
    assert w.view.region_arrow is None
    _click(w, (10, 10, 0))
    assert len(sk.selected_regions()) == 1 and w.view.region_arrow is not None
    assert np.allclose(w.view.region_arrow[1], (0, 0, 1))
    _click(w, (30, 10, 0))  # no Ctrl: joins the picked region
    assert len(sk.selected_regions()) == 2
    _click(w, (30, 10, 0))  # clicked again: dropped
    assert len(sk.selected_regions()) == 1
    _click(w, (100, 100, 0))  # empty space clears, and the arrow goes
    assert sk.selection == [] and w.view.region_arrow is None


def test_drag_a_region_extrudes_it(qapp, monkeypatch):
    w, sk = _cross_window(monkeypatch)
    _drag_up(w, (10, 10, 0), 8)  # nothing picked beforehand: just this half
    f = w.doc.features[-1]
    assert f.kind == "extrude" and f.auto and len(f.profiles) == 1 and not w.doc.errors
    assert f.value == pytest.approx(8, rel=0.05)
    assert w.doc.part.volume == pytest.approx(20 * 20 * f.value)
    assert sk.selection == [] and w.view.region_arrow is None

    # the value box left behind retypes the new extrude
    w.view.value_box.start_typing("1")
    w.view.value_box.accepted.emit(12.0)
    assert f.value == pytest.approx(12) and w.doc.part.volume == pytest.approx(20 * 20 * 12)


def test_dragging_the_arrow_extrudes_every_picked_region_as_one_block(qapp, monkeypatch):
    w, sk = _cross_window(monkeypatch)
    _click(w, (10, 10, 0))
    _click(w, (30, 10, 0))
    base, _ = w.view._arrow_ends()
    tip = np.asarray(base) + (0, 0, 0.5 * w.view.ARROW_PX / w.view._px_per_world(base))
    assert w.view.on_region_arrow(*_px(w, tip))
    _drag_up(w, tuple(tip), 6)
    f = w.doc.features[-1]
    assert f.kind == "extrude" and len(f.profiles) == 2 and not w.doc.errors
    assert w.doc.part.volume == pytest.approx(40 * 20 * f.value)
    assert len(w.doc.part.faces()) == 6  # one block, no seam where the line was


def test_dragging_another_region_takes_the_picked_ones_along(qapp, monkeypatch):
    w, sk = _cross_window(monkeypatch)
    _click(w, (10, 10, 0))
    _drag_up(w, (30, 10, 0), 5)
    f = w.doc.features[-1]
    assert f.kind == "extrude" and len(f.profiles) == 2


def test_a_tiny_drag_makes_nothing(qapp, monkeypatch):
    w, sk = _cross_window(monkeypatch)
    n = len(w.doc.features)
    w.view._extrude_state = None
    w.on_region_extrude(0.001, {"sketch": 0, "profiles": [[10, 10]]})
    assert len(w.doc.features) == n
