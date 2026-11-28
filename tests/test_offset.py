import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.app import MainWindow
from rcad.offset import OffsetError, chains, offset
from rcad.sketch import Constraint, Sketch, standard_plane

RECT = [("l", i) for i in range(4)]


def _apply(sk, items, d, flip=False):
    """Offset the way the tool does: the holding constraints go in one by one, refused ones dropped."""
    res = offset(sk, items, d, flip)
    for c in res.keep:
        sk.constraints.append(c)
        if not sk.solve():
            sk.constraints.pop()
    assert sk.solve(), sk.status
    return res


def rect(fixed=True):
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 40, 20)
    if fixed:
        sk.constraints += [Constraint("fixed", [0]), Constraint("distance", [0, 1], 40.0),
                           Constraint("distance", [1, 2], 20.0)]
    assert sk.solve()
    return sk


def _box(sk, items):
    pts = [sk.points[p] for k, i in items if k == "l" for p in sk.lines[i]]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return pytest.approx((min(xs), min(ys), max(xs), max(ys)))


def test_spacing_keeps_its_side_whichever_way_the_line_was_drawn():
    for flip in (False, True):
        sk = Sketch(standard_plane("XY"))
        a, b = sk.add_point(0, 0), sk.add_point(40, 0)
        line = sk.add_line(*((b, a) if flip else (a, b)))
        p = sk.add_point(10, 3)
        sk.constraints += [Constraint("fixed", [a]), Constraint("fixed", [b]), Constraint("spacing", [p, line], 5.0)]
        assert sk.solve() and sk.points[p] == pytest.approx([10, 5])  # above, as it was


def test_chains_follow_the_outline():
    sk = rect()
    (ch,), circles = chains(sk, RECT)
    assert ch.closed and len(ch.segs) == 4 and circles == []
    for s, t in zip(ch.segs, ch.segs[1:] + ch.segs[:1]):
        assert s.end == t.start


def test_rectangle_outward_and_inward():
    sk = rect()
    res = _apply(sk, RECT, 3)
    assert _box(sk, res.items) == (-3, -3, 43, 23)
    assert len(sk.regions()) == 2
    sk = rect()
    res = _apply(sk, RECT, 3, flip=True)
    assert _box(sk, res.items) == (3, 3, 37, 17)


def test_open_chain_to_either_side_meets_at_its_corner():
    for flip, corner in ((False, (18, 2)), (True, (22, -2))):
        sk = Sketch(standard_plane("XY"))
        a, b, c = sk.add_point(0, 0), sk.add_point(20, 0), sk.add_point(20, 15)
        sk.add_line(a, b)
        sk.add_line(b, c)
        sk.constraints += [Constraint("fixed", [p]) for p in (a, b, c)]
        sk.solve()
        res = _apply(sk, [("l", 0), ("l", 1)], 2, flip)
        (l1, l2) = [sk.lines[i] for _, i in res.items]
        shared = set(l1) & set(l2)
        assert len(shared) == 1 and sk.points[shared.pop()] == pytest.approx(list(corner))


def test_slot_outward_grows_its_arcs():
    sk = Sketch(standard_plane("XY"))
    p = [sk.add_point(*q) for q in [(0, 0), (30, 0), (30, 10), (0, 10), (30, 5), (0, 5)]]
    sk.add_line(p[0], p[1])
    sk.add_line(p[2], p[3])
    sk.add_arc(p[4], p[1], p[2])
    sk.add_arc(p[5], p[3], p[0])
    sk.constraints += [Constraint("tangent", [0, 0]), Constraint("tangent", [0, 1]), Constraint("tangent", [1, 0]),
                       Constraint("tangent", [1, 1]), Constraint("radius", [0], 5.0), Constraint("fixed", [p[0]]),
                       Constraint("horizontal", [0]), Constraint("distance", [p[0], p[1]], 30.0)]
    assert sk.solve()
    res = _apply(sk, [("l", 0), ("l", 1), ("a", 0), ("a", 1)], 2)
    radii = sorted(round(sk.arc_geometry(i)[1], 6) for k, i in res.items if k == "a")
    assert radii == [7, 7]
    ring = sk.profile_faces(None)
    assert len(ring) == 1 and ring[0].area == pytest.approx((30 * 14 + math.pi * 49) - (30 * 10 + math.pi * 25))


def test_one_distance_moves_the_whole_offset_and_it_follows_the_original():
    sk = rect()
    res = _apply(sk, RECT, 3)
    group = {c.group for c in sk.constraints if c.kind == "spacing"}
    assert len(group) == 1 and None not in group
    for c in sk.constraints:
        if c.group:
            c.value = 6 if c.value is not None else c.value
    assert sk.solve() and _box(sk, res.items) == (-6, -6, 46, 26)
    width = next(c for c in sk.constraints if c.kind == "distance" and c.value == 40)
    width.value = 60  # widen the original: the offset follows, still 6 away
    assert sk.solve() and _box(sk, res.items) == (-6, -6, 66, 26)


def test_circles_stay_concentric_and_too_large_is_refused():
    sk = Sketch(standard_plane("XY"))
    sk.add_circle(sk.add_point(0, 0), 5)
    res = _apply(sk, [("c", 0)], 2)
    c2 = sk.circles[res.items[0][1]]
    assert c2[0] == sk.circles[0][0] and c2[1] == pytest.approx(7)
    with pytest.raises(OffsetError, match="too large"):
        offset(sk, [("c", 0)], 6, flip=True)
    sk = rect()
    with pytest.raises(OffsetError, match="too large"):
        offset(sk, RECT, 12, flip=True)  # a 40 x 20 rectangle can't shrink by 12 on each side


def test_a_failed_offset_leaves_the_sketch_alone_in_the_tool(qapp, monkeypatch):
    w = _window(monkeypatch)
    before = w.sketch.to_dict()
    w.view.sketch_sel = set(RECT)
    w.offset_selection()
    w.tool_bar.fields["distance"].setText("12")
    w.tool_bar._set_choice("side", "in")
    w.tool_bar._changed()
    assert w.tool_bar.error_label.text().startswith("Can't offset: the distance is too large")
    assert w.tool_bar.finish() is False or w.tool_bar.active  # ✓ refuses it; the bar stays open
    w.cancel()
    assert w.sketch.to_dict()["lines"] == before["lines"]


# -- the tool

@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _window(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    s = rect()
    sk = w.sketch
    sk.points, sk.lines, sk.constraints = s.points, s.lines, s.constraints
    sk.solve()
    return w


def test_offset_in_the_settings_bar(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.on_double_click(*w.view.camera.project(np.array([(20, 0, 0)], float), 1000, 700)[0][:2], Qt.NoModifier)
    assert w.view.sketch_sel == set(RECT)
    w._update_regions()
    assert "⧉ Offset" in [a.text() for a in w.quick.more_menu.actions()]
    w.offset_selection()
    assert w.tool_bar.active
    w.tool_bar.fields["distance"].setText("5")  # previews as it is typed
    assert len(w.sketch.lines) == 8
    w.tool_bar.finish()
    sk = w.sketch
    assert _box(sk, [("l", i) for i in range(4, 8)]) == (-5, -5, 45, 25)
    w.view._sync_dim_labels()
    offset_labels = [l for l in w.view._dim_labels if not l.isHidden() and sk.constraints[l.cid].group]
    assert len(offset_labels) == 1 and offset_labels[0].text() == "5 mm"  # one label for all four
    w.edit_dimension(offset_labels[0].cid)
    w.tool_bar.fields["value"].setText("8")
    w.tool_bar.finish()
    assert _box(sk, [("l", i) for i in range(4, 8)]) == (-8, -8, 48, 28)
    w.undo()
    w.undo()
    assert len(w.sketch.lines) == 4  # one undo step each


def test_escape_takes_the_offset_away(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.view.sketch_sel = set(RECT)
    w.offset_selection()
    w.tool_bar.fields["distance"].setText("4")
    assert len(w.sketch.lines) == 8
    w.cancel()
    assert len(w.sketch.lines) == 4 and not w.tool_bar.active
