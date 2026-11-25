import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad import app as appmod
from rcad.app import MainWindow
from rcad.units import INCH, MM
from rcad.value_box import parse_length


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# -- parsing
@pytest.mark.parametrize("text, unit, expected", [
    ("12", MM, 12.0),
    ("  7.5 ", MM, 7.5),
    ("2", INCH, 50.8),
    ("2in", MM, 50.8),
    ('1"', MM, 25.4),
    ("5cm", MM, 50.0),
    ("0.1 m", MM, 100.0),
    ("1ft", MM, 304.8),
    ("2in + 5mm", MM, 55.8),
    ("40/3 + 2", MM, 40 / 3 + 2),
    ("40/3", INCH, 40 / 3 * 25.4),
    ("-4", MM, -4.0),
    ("(1+2)*3", MM, 9.0),
    ("1e1", MM, 10.0),
])
def test_parse_length(text, unit, expected):
    assert parse_length(text, unit) == pytest.approx(expected)


def test_parse_length_variables_are_millimetres():
    assert parse_length("wall * 2", MM, {"wall": 3.0}) == pytest.approx(6.0)
    assert parse_length("wall", INCH, {"wall": 3.0}) == pytest.approx(3.0)
    assert parse_length("wall + 1in", MM, {"wall": 3.0}) == pytest.approx(28.4)
    # a variable whose name starts like a unit is still a variable
    assert parse_length("min_wall", MM, {"min_wall": 4.0}) == pytest.approx(4.0)


@pytest.mark.parametrize("text", ["", "   ", "abc", "2 +", "1/0", "2in in", "__import__('os')", "1 m m"])
def test_parse_length_rejects(text):
    with pytest.raises(ValueError):
        parse_length(text, MM)


# -- viewport: typing while dragging
def _box_model(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.set_tool("rect")
    for p in [(0, 0, 0), (40, 30, 0)]:
        x, y = w.view.camera.project(np.array([p], float), 1000, 700)[0][:2]
        w.on_click(x, y, Qt.LeftButton, Qt.NoModifier)
    w.finish_sketch()
    monkeypatch.setattr(appmod.ExtrudeDialog, "exec", lambda self: True)
    w.extrude()
    assert not w.doc.errors
    return w


def _start_extrude_drag(w):
    """Put the viewport in the state of a face drag in progress (top face, +Z)."""
    v = w.view
    top = next(i for i, f in enumerate(w.doc.mesh.faces)
               if f.normal is not None and f.normal[2] > 0.9)
    info = w.doc.mesh.faces[top]
    v._extrude_state = {"face": top, "center": np.array(info.center, float),
                        "normal": np.array(info.normal, float), "start_screen": (0, 0),
                        "outline": []}
    v._extrude_dist = 3.0
    v._dragged = True
    return info


def _key(widget, text):
    widget.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, 0, Qt.NoModifier, text))


def test_type_during_extrude_drag_commits_typed_value(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    v = w.view
    _start_extrude_drag(w)
    vol0 = w.doc.part.volume
    _key(v, "1")
    assert v.value_box.typing and v.value_box.text() == "1"
    v.value_box.setText("12 + 0.5")
    v.value_box._on_text(v.value_box.text())
    assert v._extrude_dist == pytest.approx(12.5)  # live preview follows the text
    # releasing the mouse while a typed value is pending does not commit it
    v._press = None
    v.mouseReleaseEvent(type("E", (), {"position": lambda self: None})())
    assert w.doc.features[-1].kind != "pushpull"
    v.value_box.returnPressed.emit()
    f = w.doc.features[-1]
    assert f.kind == "pushpull" and f.value == pytest.approx(12.5)
    assert w.doc.part.volume == pytest.approx(vol0 + 40 * 30 * 12.5, rel=1e-6)
    # the box stays on the model for the new feature
    assert v._value_post["value"] == pytest.approx(12.5)
    assert w._value_feature is f


def test_value_box_after_drag_edits_feature(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    v = w.view
    _start_extrude_drag(w)
    v._finish_extrude()  # plain release at 3 mm
    f = w.doc.features[-1]
    assert f.kind == "pushpull" and f.value == pytest.approx(3.0)
    assert w._value_feature is f
    _key(v, "8")
    v.value_box.returnPressed.emit()
    assert f.value == pytest.approx(8.0)
    assert not w.doc.errors
    # undo returns to the 3 mm push/pull
    w.undo()
    assert w.doc.features[-1].value == pytest.approx(3.0)
    # the edited feature object is gone after undo: the next edit dismisses the box
    _key(v, "9")
    v.value_box.returnPressed.emit()
    assert v._value_post is None and not v.value_box.isVisible()


def test_value_box_esc_cancels_extrude(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    v = w.view
    n = len(w.doc.features)
    _start_extrude_drag(w)
    _key(v, "5")
    v.value_box.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert v._extrude_state is None
    assert len(w.doc.features) == n
    assert not v.value_box.isVisible()


def test_value_box_invalid_text_is_not_accepted(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    v = w.view
    n = len(w.doc.features)
    _start_extrude_drag(w)
    _key(v, "x")
    v.value_box.returnPressed.emit()
    assert len(w.doc.features) == n  # nothing committed
    assert v._extrude_state is not None  # still waiting for a valid value


def test_value_box_uses_display_units(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    v = w.view
    v.unit = INCH
    _start_extrude_drag(w)
    _key(v, "1")
    v.value_box.returnPressed.emit()
    assert w.doc.features[-1].value == pytest.approx(25.4)
    v.set_post_value(25.4)
    v._sync_value_box()
    assert v.value_box.text() == "1 in"


def test_fillet_typed_and_cancelled(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    v = w.view
    n = len(w.doc.features)
    undo_depth = len(w._undo)
    mids = [tuple(map(float, w.doc.mesh.edges[0].mid))]
    v._fillet_state = {"mids": mids, "outward": None, "start_screen": (0, 0), "radius": 1.0}
    v._dragged = True
    v.fillet_preview.emit(1.0, mids)  # what the drag would have emitted
    assert w.doc.features[-1].kind == "fillet"
    _key(v, "2")
    assert w.doc.features[-1].value == pytest.approx(2.0)  # typed value previews live
    v.value_box.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert len(w.doc.features) == n and w._fillet_live_idx is None
    assert len(w._undo) == undo_depth


def test_fillet_typed_then_enter_commits(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    v = w.view
    mids = [tuple(map(float, w.doc.mesh.edges[0].mid))]
    v._fillet_state = {"mids": mids, "outward": None, "start_screen": (0, 0), "radius": 1.0}
    v._dragged = True
    v.fillet_preview.emit(1.0, mids)
    _key(v, "2")
    v.value_box.setText("2.5")
    v.value_box._on_text("2.5")
    v.value_box.returnPressed.emit()
    f = w.doc.features[-1]
    assert f.kind == "fillet" and f.value == pytest.approx(2.5) and not w.doc.errors
    assert w._value_feature is f
    # retyping a radius too large for the part is refused and reverted
    _key(v, "9")
    v.value_box.setText("999")
    v.value_box._on_text("999")
    v.value_box.returnPressed.emit()
    assert f.value == pytest.approx(2.5) and not w.doc.errors
    assert v._value_post["value"] == pytest.approx(2.5)
