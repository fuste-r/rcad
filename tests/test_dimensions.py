import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QEnterEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow
from rcad.sketch import Constraint


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(monkeypatch):
    """A sketch with a 40 x 30 rectangle, its width dimensioned (40), and a circle of diameter 10."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    sk = w.sketch
    sk.add_rect(0, 0, 40, 30)
    sk.constraints.append(Constraint("fixed", [0]))
    sk.constraints.append(Constraint("distance", [0, 1], 40.0))  # constraint 5: the bottom line
    sk.add_circle(sk.add_point(70, 15), 5)
    sk.constraints.append(Constraint("diameter", [0], 10.0))  # constraint 6
    assert sk.solve()
    w.refresh_constraints()
    w.view._sync_dim_labels()
    return w


def _label(w, cid):
    return next(l for l in w.view._dim_labels if l.cid == cid and not l.isHidden())


def _press(lbl, button=Qt.LeftButton):
    p = QPointF(3, 3)
    lbl.mousePressEvent(QMouseEvent(QMouseEvent.MouseButtonPress, p, lbl.mapToGlobal(p), button, button,
                                    Qt.NoModifier))


def test_labels_name_their_dimensions(qapp, monkeypatch):
    w = _window(monkeypatch)
    assert _label(w, 5).text() == "40 mm" and _label(w, 6).text() == "Ø 10 mm"


def test_hover_highlights_the_label_and_what_it_measures(qapp, monkeypatch):
    w = _window(monkeypatch)
    lbl = _label(w, 5)
    lbl.enterEvent(QEnterEvent(QPointF(2, 2), QPointF(2, 2), QPointF(2, 2)))
    assert w.view.dim_hover == 5 and lbl._look == "hover"
    assert w.view.dim_items(5) == {("p", 0), ("p", 1), ("l", 0)}  # the two corners and the line between
    lbl.leaveEvent(None)
    assert w.view.dim_hover is None and lbl._look == "normal"
    assert w.view.dim_items(6) == {("c", 0)}


def test_click_changes_the_dimension(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    _press(_label(w, 5))
    assert w.tool_bar.active and w.view.dim_active == 5 and _label(w, 5)._look == "active"
    w.tool_bar.fields["value"].setText("55")  # the sketch follows as it is typed
    assert sk.points[1][0] == pytest.approx(55)
    w.tool_bar.finish()
    assert not w.tool_bar.active and w.view.dim_active is None
    w.view._sync_dim_labels()  # (as a repaint would)
    assert sk.constraints[5].value == pytest.approx(55) and _label(w, 5).text() == "55 mm"
    w.undo()  # one step puts it back
    assert w.sketch.constraints[5].value == pytest.approx(40)
    assert w.sketch.points[1][0] == pytest.approx(40)


def test_escape_puts_it_back(qapp, monkeypatch):
    w = _window(monkeypatch)
    _press(_label(w, 6))
    w.tool_bar.fields["value"].setText("22")
    assert w.sketch.circles[0][1] == pytest.approx(11)
    w.cancel()
    assert not w.tool_bar.active and w.view.dim_active is None
    assert w.sketch.constraints[6].value == pytest.approx(10) and w.sketch.circles[0][1] == pytest.approx(5)


def test_a_value_the_sketch_cannot_take_is_refused(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    sk.constraints.append(Constraint("distance", [0, 2], 50.0))  # the diagonal: 40 x 30 -> 50
    assert sk.solve()
    w.refresh_constraints()
    w.view._sync_dim_labels()
    _press(_label(w, 7))
    w.tool_bar.fields["value"].setText("20")  # shorter than the fixed 40 mm width
    assert w.tool_bar.finish() is False or w.tool_bar.active  # the bar stays open with the error
    w.cancel()
    assert sk.constraints[7].value == pytest.approx(50) and sk.solve()


def test_right_click_deletes_the_dimension(qapp, monkeypatch):
    w = _window(monkeypatch)
    n = len(w.sketch.constraints)
    _press(_label(w, 6), Qt.RightButton)
    menu = w.dimension_menu
    assert [a.text() for a in menu.actions()] == ["✎ Change", "🗑 Delete dimension"]
    menu.actions()[1].trigger()
    assert len(w.sketch.constraints) == n - 1 and all(c.kind != "diameter" for c in w.sketch.constraints)
    w.view._sync_dim_labels()  # (as a repaint would)
    assert not any(l.cid == 6 and not l.isHidden() for l in w.view._dim_labels)
    w.undo()
    assert len(w.sketch.constraints) == n


def test_clicking_a_dimension_while_drawing_drops_the_half_drawn_shape(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.set_tool("line")
    w.pending = (None, (100.0, 100.0))
    _press(_label(w, 5))
    assert w.pending is None and w.tool_bar.active
    w.tool_bar.finish()
