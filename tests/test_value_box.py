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


# -- angles
@pytest.mark.parametrize("text, expected", [("90", 90.0), ("45°", 45.0), ("30 deg", 30.0),
                                            ("180/4", 45.0), ("-15", -15.0)])
def test_parse_angle(text, expected):
    from rcad.value_box import parse_angle
    assert parse_angle(text) == pytest.approx(expected)


# -- scrubbing
def test_scrub_steps_follow_units_and_modifiers(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    box = w.view.value_box
    box.setText("10")
    box.begin_scrub(0)
    assert box.scrub_to(25) == pytest.approx(12.5)        # 0.1 mm per pixel
    assert box.scrub_to(25, Qt.ShiftModifier) == pytest.approx(10.25)
    assert box.scrub_to(-3, Qt.ControlModifier) == pytest.approx(7.0)
    w.view.unit = INCH
    box.setText("1")
    box.begin_scrub(0)
    assert box.scrub_to(10) == pytest.approx(25.4 + 10 * 0.005 * 25.4)  # 0.005 in per pixel


def test_tab_keeps_focus_in_a_lone_box(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    assert w.view.value_box.focusNextPrevChild(True) is True
    _start_extrude_drag(w)
    assert w.view.focusNextPrevChild(True) is False  # Tab doesn't leave the view mid-drag


def test_scrub_after_drag_edits_live_with_one_undo_step(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    v = w.view
    _start_extrude_drag(w)
    v._finish_extrude()  # 3 mm push/pull
    f = w.doc.features[-1]
    depth = len(w._undo)
    box = v.value_box
    box.setText(box.format(3.0))
    box.begin_scrub(0)
    box.scrub_to(20)
    assert f.value == pytest.approx(5.0)  # the model follows the scrub
    box.scrub_to(40)
    assert f.value == pytest.approx(7.0)
    box.scrub_finished.emit(7.0)
    assert f.value == pytest.approx(7.0) and not w.doc.errors
    assert len(w._undo) == depth + 1
    w.undo()
    assert w.doc.features[-1].value == pytest.approx(3.0)


def test_scrub_to_failing_fillet_reverts(qapp, monkeypatch):
    w = _box_model(monkeypatch)
    v = w.view
    mids = [tuple(map(float, w.doc.mesh.edges[0].mid))]
    v._fillet_state = {"mids": mids, "outward": None, "start_screen": (0, 0), "radius": 2.0}
    v._dragged = True
    v.fillet_preview.emit(2.0, mids)
    v._finish_fillet()
    f = w.doc.features[-1]
    assert f.kind == "fillet" and w._value_feature is f
    depth = len(w._undo)
    box = v.value_box
    box.setText(box.format(2.0))
    box.begin_scrub(0)
    box.scrub_to(9000)  # 902 mm: far too big
    box.scrub_finished.emit(902.0)
    assert f.value == pytest.approx(2.0) and not w.doc.errors
    assert len(w._undo) == depth


# -- sizes while drawing in a sketch
def _sketch(monkeypatch, tool):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.set_tool(tool)
    return w


def _click_at(w, p):
    x, y = w.view.camera.project(np.array([p], float), 1000, 700)[0][:2]
    w.on_click(x, y, Qt.LeftButton, Qt.NoModifier)
    return x, y


def _type(box, text):
    box.setText(text)
    box._on_text(text)


def _tab(widget, back=False):
    key = Qt.Key_Backtab if back else Qt.Key_Tab
    mods = Qt.ShiftModifier if back else Qt.NoModifier
    QApplication.sendEvent(widget, QKeyEvent(QKeyEvent.KeyPress, key, mods, "\t"))


def _points(w):
    return {tuple(round(c, 6) for c in p) for p in w.sketch.points}


def test_rect_width_tab_height_enter(qapp, monkeypatch):
    w = _sketch(monkeypatch, "rect")
    _click_at(w, (0, 0, 0))
    f = w.fields
    assert f.active and f.keys == ["W", "H"] and f.active_index == 0
    _key(w.view, "4")
    _type(f.boxes["W"], "40")
    _tab(w.view)  # from the view, or from the focused box: both move to H
    assert f.active_index == 1
    _type(f.boxes["H"], "25")
    _tab(f.boxes["H"], back=True)
    assert f.active_index == 0
    f.boxes["W"].returnPressed.emit()
    assert {(0, 0), (40, 0), (40, 25), (0, 25)} <= _points(w)
    assert w.tool == "select"
    assert f.active and f.post  # the boxes stay on the new rectangle


def test_rect_typed_width_mouse_height(qapp, monkeypatch):
    w = _sketch(monkeypatch, "rect")
    _click_at(w, (0, 0, 0))
    _key(w.view, "4")
    _type(w.fields.boxes["W"], "40")
    x, y = w.view.camera.project(np.array([(-10, -15, 0)], float), 1000, 700)[0][:2]
    w.on_hover(x, y)
    assert w.fields.boxes["H"].text() == "15 mm"  # the untyped size follows the mouse
    assert w.fields.boxes["W"].text() == "40"
    _click_at(w, (-10, -15, 0))
    assert {(0, 0), (-40, 0), (-40, -15), (0, -15)} <= _points(w)  # typed width keeps the mouse's side


def test_line_length_and_angle_then_next_segment(qapp, monkeypatch):
    w = _sketch(monkeypatch, "line")
    _click_at(w, (0, 0, 0))
    _key(w.view, "3")
    _type(w.fields.boxes["L"], "30")
    _tab(w.view)
    _type(w.fields.boxes["A"], "90")
    w.fields.boxes["A"].returnPressed.emit()
    assert (0, 30) in _points(w)
    assert len(w.sketch.lines) == 1
    # the chain goes on from the new point with nothing typed
    assert w.pending is not None and w.fields.active
    assert w.fields.locked("L") is None and w.fields.locked("A") is None
    _key(w.view, "1")
    _type(w.fields.boxes["L"], "10in/25.4")  # 10 mm, any expression works
    w.view.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Return, Qt.NoModifier, "\r"))
    assert (10, 30) in _points(w)  # angle 0 with no mouse move


def test_circle_diameter(qapp, monkeypatch):
    w = _sketch(monkeypatch, "circle")
    _click_at(w, (5, 5, 0))
    assert w.fields.keys == ["D"]
    _key(w.view, "1")
    _type(w.fields.boxes["D"], "12")
    w.fields.boxes["D"].returnPressed.emit()
    assert len(w.sketch.circles) == 1 and w.sketch.circles[0][1] == pytest.approx(6.0)


def test_drawing_fields_invalid_and_escape(qapp, monkeypatch):
    w = _sketch(monkeypatch, "rect")
    _click_at(w, (0, 0, 0))
    _key(w.view, "x")
    w.fields.accept()  # an unparseable size blocks placing the shape
    assert w.pending is not None and not w.sketch.lines
    w.fields.boxes["W"].keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Escape, Qt.NoModifier))
    assert w.pending is None and not w.fields.active


# -- boxes left on a shape after it is drawn
def _draw_rect(w, size=(40, 25)):
    _click_at(w, (0, 0, 0))
    _key(w.view, "1")
    _type(w.fields.boxes["W"], str(size[0]))
    _tab(w.view)
    _type(w.fields.boxes["H"], str(size[1]))
    w.fields.boxes["H"].returnPressed.emit()


def test_rect_boxes_stay_and_resize_it(qapp, monkeypatch):
    w = _sketch(monkeypatch, "rect")
    _draw_rect(w)
    f = w.fields
    assert f.post and f.anchor is not None
    assert f.boxes["W"].text() == "40 mm" and f.boxes["H"].text() == "25 mm"
    _tab(w.view)  # to H
    _key(w.view, "1")
    _type(f.boxes["H"], "10")
    assert {(0, 0), (40, 0), (40, 10), (0, 10)} <= _points(w)  # follows the typing
    f.boxes["H"].returnPressed.emit()
    assert f.post and f.boxes["H"].text() == "10 mm"
    w.undo_sketch()  # one undo step back to 25
    assert {(0, 0), (40, 0), (40, 25), (0, 25)} <= _points(w)
    assert not f.active


def test_rect_box_scrub_resizes_live(qapp, monkeypatch):
    w = _sketch(monkeypatch, "rect")
    _draw_rect(w)
    box = w.fields.boxes["W"]
    box.begin_scrub(0)
    box.scrub_to(50)  # +5 mm
    assert (45, 25) in _points(w)
    box.scrub_finished.emit(45.0)
    assert (45, 25) in _points(w)


def test_circle_box_stays_and_resizes(qapp, monkeypatch):
    w = _sketch(monkeypatch, "circle")
    _click_at(w, (5, 5, 0))
    _key(w.view, "1")
    _type(w.fields.boxes["D"], "12")
    w.fields.boxes["D"].returnPressed.emit()
    assert w.fields.post and w.fields.boxes["D"].text() == "12 mm"
    _key(w.view, "2")
    _type(w.fields.boxes["D"], "20")
    w.fields.boxes["D"].returnPressed.emit()
    assert w.sketch.circles[0][1] == pytest.approx(10.0)


def test_line_chain_esc_keeps_last_segment_boxes(qapp, monkeypatch):
    w = _sketch(monkeypatch, "line")
    _click_at(w, (0, 0, 0))
    _click_at(w, (20, 0, 0))
    w.cancel()  # Esc ends the chain
    f = w.fields
    assert f.post and f.boxes["L"].text() == "20 mm" and f.boxes["A"].text() == "0°"
    _key(w.view, "3")
    _type(f.boxes["L"], "35")
    f.boxes["L"].returnPressed.emit()
    assert (35, 0) in _points(w) and (20, 0) not in _points(w)
    w.cancel()  # Esc again leaves the line tool and puts the boxes away
    assert not f.active and w.tool == "select"


def test_line_closing_on_existing_point_has_no_boxes(qapp, monkeypatch):
    w = _sketch(monkeypatch, "line")
    for p in [(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 0, 0)]:
        _click_at(w, p)
    assert w.pending is None and not w.fields.active


def test_click_elsewhere_puts_drawn_boxes_away(qapp, monkeypatch):
    from PySide6.QtCore import QPoint
    from PySide6.QtTest import QTest
    w = _sketch(monkeypatch, "rect")
    _draw_rect(w)
    assert w.fields.post
    QTest.mouseClick(w.view, Qt.LeftButton, Qt.NoModifier, QPoint(5, 5))
    assert not w.fields.active
    assert {(0, 0), (40, 0), (40, 25), (0, 25)} <= _points(w)


# -- number fields in dialogs and toolbars
def test_number_field_parses_units_maths_and_variables(qapp):
    from rcad.value_box import NumberField
    f = NumberField(10.0, "length", INCH, {"wall": 3.0})
    assert f.text() == "0.3937 in" and f.value() == 10.0  # untouched: exact
    f.setText("2")
    assert f.value() == pytest.approx(50.8)
    f.setText("1in + 5mm")
    assert f.value() == pytest.approx(30.4)
    f.setText("wall * 2")
    assert f.value() == pytest.approx(6.0)
    f.setText("2 +")
    assert f.value() is None and not f.is_valid() and "border" in f.styleSheet()
    assert f.last_valid == pytest.approx(6.0)


def test_number_field_range_and_ok_button(qapp):
    from PySide6.QtWidgets import QDialogButtonBox
    from rcad.value_box import NumberField, bind_ok
    f = NumberField(360.0, "angle", lo=1.0, hi=360.0)
    assert f.text() == "360°"
    box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    bind_ok(box, [f])
    ok = box.button(QDialogButtonBox.Ok)
    assert ok.isEnabled()
    f.setText("400")
    assert f.value() is None and "At most 360°" in f.error() and not ok.isEnabled()
    f.setText("360/4")
    assert f.value() == pytest.approx(90.0) and ok.isEnabled()


def test_ask_length_accepts_expressions(qapp, monkeypatch):
    w = MainWindow()
    w.doc.variables = {"wall": 2.0}

    def exec_(self):
        self.field.setText("wall * 3 + 1cm")
        return True
    monkeypatch.setattr(appmod.NumberDialog, "exec", exec_)
    assert w.ask_length("t", "l", 5.0) == (pytest.approx(16.0), True)
    monkeypatch.setattr(appmod.NumberDialog, "exec", lambda self: False)
    assert w.ask_length("t", "l", 5.0) == (5.0, False)


def test_variables_dialog_values_take_expressions(qapp):
    d = appmod.VariablesDialog(None, {"wall": 2.0})
    d.name_edit.setText("depth")
    d.value_edit.setText("wall * 5 + 1in")
    d._add()
    assert d.variables()["depth"] == pytest.approx(35.4)
    d.name_edit.setText("bad name")
    d.value_edit.setText("3")
    d._add()
    assert "bad name" not in d.variables()


def test_snap_field_follows_units_and_takes_expressions(qapp):
    w = MainWindow()
    w.unit_combo.setCurrentText("in")
    assert w.snap.text() == "0.05 in"
    w.snap.setText("1/8")
    assert w.snap.value() == pytest.approx(3.175)
    w.snap.setText("1/")  # mid-edit: keep snapping to the last good value
    w.new_sketch()
    x, y = w.view.camera.project(np.array([(3.0, 3.3, 0)], float), w.view.width(), w.view.height())[0][:2]
    _pi, uv = w._snap(x, y)
    assert uv[0] == pytest.approx(3.175) and uv[1] == pytest.approx(3.175)
