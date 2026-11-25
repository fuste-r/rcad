import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad import app as appmod
from rcad.app import MainWindow
from rcad.tool_bar import ScrubField
from rcad.units import INCH


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _rect_sketch(monkeypatch, size=(40, 30)):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.sketch.add_rect(0, 0, *size)
    w.sketch.solve()
    w.finish_sketch()
    return w


def test_extrude_opens_bar_and_previews_live(qapp, monkeypatch):
    w = _rect_sketch(monkeypatch)
    depth = len(w._undo)
    w.extrude()
    bar = w.tool_bar
    assert bar.active
    assert bar.values() == {"distance": 10.0}  # no body yet: no operation choice
    f = w.doc.features[-1]
    assert f.kind == "extrude" and w.doc.part.volume == pytest.approx(40 * 30 * 10)
    bar.fields["distance"].setText("2.5cm")
    assert f.value == pytest.approx(25.0) and w.doc.part.volume == pytest.approx(40 * 30 * 25)
    bar.fields["distance"].setText("2 +")  # invalid text: model keeps the last good preview
    assert f.value == pytest.approx(25.0) and not bar.tick.isEnabled()
    bar.fields["distance"].setText("12")
    bar.fields["distance"].returnPressed.emit()  # Enter finishes
    assert not bar.active and w.doc.features[-1] is f and f.value == pytest.approx(12.0)
    assert len(w._undo) == depth + 1  # one undo step for the whole session
    w.undo()
    assert all(g.kind != "extrude" for g in w.doc.features)


def test_cancel_removes_new_feature(qapp, monkeypatch):
    w = _rect_sketch(monkeypatch)
    n, depth = len(w.doc.features), len(w._undo)
    w.extrude()
    w.tool_bar.fields["distance"].setText("40")
    w.tool_bar.cross.click()
    assert not w.tool_bar.active
    assert len(w.doc.features) == n and len(w._undo) == depth and w.doc.part is None


def test_esc_cancels_and_restores_edited_feature(qapp, monkeypatch):
    w = _rect_sketch(monkeypatch)
    w.extrude()
    w.tool_bar.finish()
    f = w.doc.features[-1]
    w.edit_feature(w.features.item(w.features.count() - 1))
    assert w.tool_bar.values()["distance"] == pytest.approx(10.0)
    w.tool_bar.fields["distance"].setText("30")
    assert w.doc.part.volume == pytest.approx(40 * 30 * 30)
    w.cancel()  # Esc
    assert not w.tool_bar.active and f.value == pytest.approx(10.0)
    assert w.doc.part.volume == pytest.approx(40 * 30 * 10)


def test_operation_word_cycles_and_previews(qapp, monkeypatch):
    w = _rect_sketch(monkeypatch)
    w.extrude()
    w.tool_bar.finish()
    # a second extrude of the same sketch, now with a body to add to or cut from
    w.extrude()
    bar = w.tool_bar
    f = w.doc.features[-1]
    button = bar.choices["mode"][0]
    assert button.text() == "Auto" and f.auto
    button.click()
    assert button.text() == "Add" and not f.auto and not f.cut
    button.click()
    assert button.text() == "Cut" and f.cut and not f.auto
    bar.fields["distance"].setText("5")
    assert w.doc.part.volume == pytest.approx(40 * 30 * 5, rel=1e-6)  # cut the top half off
    button.click()
    assert button.text() == "Auto"


def test_failing_settings_block_done_and_are_dropped_when_left(qapp, monkeypatch):
    w = _rect_sketch(monkeypatch)
    w.extrude()
    bar = w.tool_bar
    bar.fields["distance"].setText("0")
    i = len(w.doc.features) - 1
    assert i in w.doc.errors
    assert bar.error_label.text() and not bar.tick.isEnabled()
    bar.finish()
    assert bar.active  # still open: nothing to keep yet
    bar.fields["distance"].setText("3")
    assert not w.doc.errors and bar.tick.isEnabled() and not bar.error_label.isVisible()
    bar.fields["distance"].setText("0")
    w.new_sketch()  # starting something else drops the failing feature
    assert not bar.active and all(g.kind != "extrude" for g in w.doc.features)


def test_other_actions_keep_the_open_feature(qapp, monkeypatch):
    w = _rect_sketch(monkeypatch)
    w.extrude()
    w.tool_bar.fields["distance"].setText("20")
    w.new_sketch()  # starting a sketch keeps the previewed extrude
    assert not w.tool_bar.active
    assert any(g.kind == "extrude" and g.value == pytest.approx(20.0) for g in w.doc.features)


def test_extrude_starts_at_last_distance(qapp, monkeypatch):
    w = _rect_sketch(monkeypatch)
    w.extrude()
    w.tool_bar.fields["distance"].setText("7")
    w.tool_bar.finish()
    w.extrude()
    assert w.tool_bar.values()["distance"] == pytest.approx(7.0)
    w.tool_bar.cancel()


def test_bar_uses_display_units(qapp, monkeypatch):
    w = _rect_sketch(monkeypatch)
    w.unit_combo.setCurrentText("in")
    w.extrude()
    field = w.tool_bar.fields["distance"]
    assert field.text() == "0.3937 in" and w.doc.features[-1].value == 10.0
    field.setText("1")
    assert w.doc.features[-1].value == pytest.approx(25.4)
    w.tool_bar.finish()


def test_bar_sits_at_bottom_centre(qapp, monkeypatch):
    w = _rect_sketch(monkeypatch)
    w.extrude()
    bar, v = w.tool_bar, w.view
    assert abs((bar.x() + bar.width() / 2) - v.width() / 2) <= 1
    assert bar.y() + bar.height() <= v.height() and bar.y() > v.height() / 2
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent
    old = v.size()
    v.resize(800, 500)
    # hidden widgets get their resize event later (and a GL repaint headless fails): hand it to the bar
    bar.eventFilter(v, QResizeEvent(QSize(800, 500), old))
    assert abs((bar.x() + bar.width() / 2) - 400) <= 1 and bar.y() + bar.height() <= 500


def test_scrub_field_steps_and_limits(qapp):
    f = ScrubField(360.0, "angle", lo=1.0, hi=360.0)
    assert f.scrub_to(360.0, -20) == pytest.approx(350.0)  # 0.5° per pixel
    assert f.value() == pytest.approx(350.0)
    assert f.scrub_to(350.0, 100) == pytest.approx(360.0)  # clamped
    g = ScrubField(10.0, "length", INCH)
    assert g.scrub_to(25.4, 10, Qt.ControlModifier) == pytest.approx(25.4 + 10 * 0.05 * 25.4)


# -- single-value tools: push/pull, fillet, chamfer, sketch dimensions
def _box(monkeypatch):
    w = _rect_sketch(monkeypatch)
    w.extrude()
    w.tool_bar.finish()
    return w


def _top_face(w):
    return next(i for i, f in enumerate(w.doc.mesh.faces) if f.normal is not None and f.normal[2] > 0.9)


def test_pushpull_in_bar(qapp, monkeypatch):
    w = _box(monkeypatch)
    n, depth = len(w.doc.features), len(w._undo)
    w.view.sel_faces = {_top_face(w)}
    w.pushpull()
    bar = w.tool_bar
    assert bar.active and bar.values() == {"value": 5.0}
    assert w.doc.part.volume == pytest.approx(40 * 30 * 15)  # previewing at 5 mm
    bar.fields["value"].setText("0")
    assert "can't be 0" in bar.error_label.text() and not bar.tick.isEnabled()
    bar.fields["value"].setText("-4")
    assert w.doc.part.volume == pytest.approx(40 * 30 * 6)
    bar.cancel()
    assert len(w.doc.features) == n and len(w._undo) == depth
    assert w.doc.part.volume == pytest.approx(40 * 30 * 10)


def test_fillet_and_chamfer_in_bar(qapp, monkeypatch):
    w = _box(monkeypatch)
    edge = max(range(len(w.doc.mesh.edges)), key=lambda i: w.doc.mesh.edges[i].mid[2] + w.doc.mesh.edges[i].mid[0])
    w.view.sel_edges = {edge}
    vol = w.doc.part.volume
    w.fillet()
    assert w.tool_bar.values() == {"value": 1.0}
    w.tool_bar.fields["value"].setText("3")
    f = w.doc.features[-1]
    assert f.kind == "fillet" and f.value == pytest.approx(3.0) and w.doc.part.volume < vol
    w.tool_bar.finish()
    assert f.result_edges  # recorded on ✓ so the fillet can be dragged later
    # editing it from the history opens the bar with its radius; ✕ restores it
    w.edit_feature(w.features.item(w.features.count() - 1))
    assert w.tool_bar.values() == {"value": pytest.approx(3.0)}
    w.tool_bar.fields["value"].setText("5")
    assert f.value == pytest.approx(5.0)
    w.tool_bar.cancel()
    assert f.value == pytest.approx(3.0)


def _dim_sketch(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.sketch.add_rect(0, 0, 40, 30)
    w.sketch.solve()
    return w


def _width(w):
    xs = sorted({round(p[0], 6) for p in w.sketch.points})
    return xs[-1] - xs[0]


def test_dimension_in_bar_solves_live_and_cancels(qapp, monkeypatch):
    w = _dim_sketch(monkeypatch)
    sk = w.sketch
    n, depth = len(sk.constraints), len(w._sketch_undo.get(w.sketch_index, []))
    w.view.sketch_sel = {("l", 0)}
    w.add_constraint("distance")
    bar = w.tool_bar
    assert bar.values() == {"value": pytest.approx(40.0)}  # starts at the current length
    assert len(sk.constraints) == n + 1
    bar.fields["value"].setText("55")
    assert _width(w) == pytest.approx(55.0)  # the sketch re-solves as you type
    bar.cancel()
    assert len(sk.constraints) == n and _width(w) == pytest.approx(40.0)
    assert len(w._sketch_undo.get(w.sketch_index, [])) == depth


def test_dimension_kept_and_edited(qapp, monkeypatch):
    w = _dim_sketch(monkeypatch)
    sk = w.sketch
    w.view.sketch_sel = {("l", 0)}
    w.add_constraint("distance")
    w.tool_bar.fields["value"].setText("50")
    w.tool_bar.finish()
    assert _width(w) == pytest.approx(50.0) and not w.tool_bar.active
    c_row = len(sk.constraints) - 1
    w.edit_constraint(w.constraints.item(c_row))  # used to crash (undefined name)
    assert w.tool_bar.values() == {"value": pytest.approx(50.0)}
    w.tool_bar.fields["value"].setText("2in")
    w.tool_bar.finish()
    assert _width(w) == pytest.approx(50.8)
    w.undo_sketch()
    assert _width(w) == pytest.approx(50.0)
    w.constraints.setCurrentRow(c_row)
    w.delete_constraint()  # also used to crash
    assert len(sk.constraints) == c_row


def test_sketch_click_keeps_open_dimension(qapp, monkeypatch):
    w = _dim_sketch(monkeypatch)
    w.view.sketch_sel = {("l", 0)}
    w.add_constraint("distance")
    w.tool_bar.fields["value"].setText("45")
    w.set_tool("line")
    x, y = w.view.camera.project(np.array([(100, 100, 0)], float), 1000, 700)[0][:2]
    w.on_click(x, y, Qt.LeftButton, Qt.NoModifier)  # drawing starts: the dimension is kept
    assert not w.tool_bar.active and _width(w) == pytest.approx(45.0)
    assert any(c.kind == "distance" and c.value == pytest.approx(45.0) for c in w.sketch.constraints)
