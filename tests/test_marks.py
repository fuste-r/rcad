import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QEnterEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow
from rcad.sketch import Constraint


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(monkeypatch):
    """A sketch with a 40 x 30 rectangle (constraints 0-3: H, H, V, V)."""
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.sketch.add_rect(0, 0, 40, 30)
    assert w.sketch.solve()
    w.refresh_constraints()
    _sync(w)
    return w


def _sync(w):
    w.view._sync_dim_labels()  # (as a repaint would)


def _marks(w, cid=None):
    return [l for l in w.view._mark_labels if not l.isHidden() and (cid is None or l.cid == cid)]


def _press(lbl, button=Qt.LeftButton):
    p = QPointF(2, 2)
    lbl.mousePressEvent(QMouseEvent(QMouseEvent.MouseButtonPress, p, lbl.mapToGlobal(p), button, button,
                                    Qt.NoModifier))


def test_each_constraint_gets_a_mark_beside_its_geometry(qapp, monkeypatch):
    w = _window(monkeypatch)
    marks = _marks(w)
    assert sorted((m.cid, m.text()) for m in marks) == [(0, "H"), (1, "H"), (2, "V"), (3, "V")]
    sk = w.sketch
    sk.constraints.append(Constraint("equal", [0, 1]))
    assert sk.solve()
    _sync(w)
    assert [m.text() for m in _marks(w, 4)] == ["=", "="]  # one beside each of the two lines
    # the bottom line's marks sit side by side, not on top of each other
    bottom = sorted((m for m in _marks(w) if m.cid in (0, 4) and m.y() > 300), key=lambda m: m.x())
    assert len(bottom) == 2 and bottom[0].x() + bottom[0].width() <= bottom[1].x()


def test_hover_and_select_a_mark_then_delete_it(qapp, monkeypatch):
    w = _window(monkeypatch)
    mark = _marks(w, 2)[0]
    mark.enterEvent(QEnterEvent(QPointF(1, 1), QPointF(1, 1), QPointF(1, 1)))
    assert w.view.dim_hover == 2 and mark.look == "hover"
    assert w.view.dim_items(2) == {("l", 1)}
    mark.leaveEvent(None)
    _press(mark)
    assert w.view.sel_constraint == 2 and mark.look == "active"
    assert "Delete removes it" in w.statusBar().currentMessage()
    QApplication.sendEvent(w.view, QKeyEvent(QKeyEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
    assert [c.kind for c in w.sketch.constraints] == ["horizontal", "horizontal", "vertical"]
    assert w.view.sel_constraint is None
    w.undo()
    assert len(w.sketch.constraints) == 4


def test_right_click_a_mark_offers_delete_only(qapp, monkeypatch):
    w = _window(monkeypatch)
    _press(_marks(w, 0)[0], Qt.RightButton)
    actions = w.dimension_menu.actions()
    assert [a.text() for a in actions] == ["🗑 Delete constraint"]
    actions[0].trigger()
    assert len(w.sketch.constraints) == 3


def test_marks_let_drawing_clicks_through_and_can_be_hidden(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.set_tool("line")
    _sync(w)
    assert all(m.testAttribute(Qt.WA_TransparentForMouseEvents) for m in _marks(w))
    w.set_tool("select")
    _sync(w)
    assert not any(m.testAttribute(Qt.WA_TransparentForMouseEvents) for m in _marks(w))
    w.marks_action.trigger()
    _sync(w)
    assert _marks(w) == [] and not w.view.show_marks
    w.marks_action.trigger()
    _sync(w)
    assert len(_marks(w)) == 4


def test_a_click_on_the_sketch_or_esc_drops_the_selected_mark(qapp, monkeypatch):
    w = _window(monkeypatch)
    _press(_marks(w, 1)[0])
    assert w.view.sel_constraint == 1
    w.cancel()
    assert w.view.sel_constraint is None
    _press(_marks(w, 1)[0])
    _press(_marks(w, 1)[0])  # again: dropped
    assert w.view.sel_constraint is None


# -- strict constraints

def test_a_constraint_that_would_collapse_a_line_is_refused(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.view.sketch_sel = {("l", 0)}  # the bottom line, already horizontal
    w.add_constraint("vertical")
    sk = w.sketch
    assert [c.kind for c in sk.constraints].count("vertical") == 2  # only the rectangle's own
    assert "shrink a line to nothing" in w.statusBar().currentMessage()
    assert sk.points[1] == pytest.approx([40, 0])


def test_a_contradiction_names_what_it_clashes_with(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    sk.constraints.append(Constraint("distance", [0, 1], 40.0))
    assert sk.solve()
    n = len(sk.constraints)
    w._add_strict([Constraint("distance", [0, 1], 55.0)], "distance")  # the same two corners
    assert len(sk.constraints) == n
    msg = w.statusBar().currentMessage()
    assert msg.startswith("Distance refused: it contradicts or repeats") and "Distance 40 mm" in msg
    assert 4 in w.view.conflicts and w.view.conflicts <= set(range(n))
    _sync(w)
    assert next(l for l in w.view._dim_labels if l.cid == 4).look == "conflict"
    assert sk.solve() and sk.points[1] == pytest.approx([40, 0])


def test_an_unreachable_value_says_why(qapp, monkeypatch):
    w = _window(monkeypatch)
    sk = w.sketch
    sk.constraints += [Constraint("distance", [0, 1], 40.0), Constraint("distance", [1, 2], 30.0)]
    assert sk.solve()
    n = len(sk.constraints)
    w._add_strict([Constraint("distance", [0, 2], 10.0)], "distance")  # a 40 x 30 rectangle's diagonal
    assert len(sk.constraints) == n and sk.points[2] == pytest.approx([40, 30])
    assert w.statusBar().currentMessage().startswith("Distance refused:")
    assert "DOF" not in w.statusBar().currentMessage()


def test_a_repeated_constraint_is_refused(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.view.sketch_sel = {("l", 0), ("l", 2)}  # top and bottom: already both horizontal
    w.add_constraint("parallel")
    assert all(c.kind != "parallel" for c in w.sketch.constraints)
    assert "refused" in w.statusBar().currentMessage()
