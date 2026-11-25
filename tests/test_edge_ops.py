"""Fillet and chamfer regressions: circular and mixed edges, seams and smooth joins, separate
bodies, bad sizes, edits to earlier features, undo/redo, and save/load.

Where a result is checked by volume, it is compared with the same operation picked directly
on a freshly built part of the edited size.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad.kernel import Document, Feature
from rcad.sketch import Sketch, standard_plane


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _doc(build, height=10.0):
    sk = Sketch(standard_plane("XY"))
    build(sk)
    doc = Document([Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=height)])
    doc.rebuild(0)
    assert not doc.errors, doc.errors
    return doc


def _cylinder(r=10.0, centre=(0.0, 0.0)):
    def build(sk):
        sk.add_circle(sk.add_point(*centre), r)
    return build


def _box(w=40.0, d=30.0):
    def build(sk):
        sk.add_rect(0, 0, w, d)
    return build


def _holed(hole=(20.0, 15.0), r=5.0):
    def build(sk):
        sk.add_rect(0, 0, 40, 30)
        sk.add_circle(sk.add_point(*hole), r)
    return build


def _add(doc, kind, value, mids):
    doc.features.append(Feature(kind, value=value, edges=[tuple(map(float, m)) for m in mids]))
    doc.rebuild()


def _edit(doc, i):
    doc.features[i].version += 1
    doc.rebuild()


def _edit_sketch(doc, fn):
    sk = doc.features[0].sketch
    fn(sk)
    sk.version += 1
    doc.rebuild()


def _circle_mid(doc, z, near=None):
    """Mid of a circular mesh edge at height z (the one nearest `near`, if given)."""
    rims = [e.mid for e in doc.mesh.edges if len(e.polyline) > 2 and abs(e.mid[2] - z) < 1e-6]
    assert rims
    return rims[0] if near is None else min(rims, key=lambda m: np.linalg.norm(m[:2] - np.asarray(near)))


def _fresh(build, height, kind, value, mids):
    doc = _doc(build, height)
    _add(doc, kind, value, mids)
    assert not doc.errors, doc.errors
    return doc.part.volume


# -- circular edges
@pytest.mark.parametrize("kind", ["fillet", "chamfer"])
def test_circular_rim_follows_height_radius_and_centre(kind):
    doc = _doc(_cylinder(), 20)
    _add(doc, kind, 2.0, [_circle_mid(doc, 20)])
    assert not doc.errors, doc.errors
    doc.features[1].value = 30
    _edit(doc, 1)
    _edit_sketch(doc, lambda sk: sk.circles.__setitem__(0, (sk.circles[0][0], 15.0)))
    _edit_sketch(doc, lambda sk: sk.points.__setitem__(0, [5.0, 5.0]))  # moved by less than the radius
    assert not doc.errors, doc.errors
    sig = doc.features[2].ref_sigs[0]
    assert np.allclose(sig["center"], (5, 5, 30)) and sig["radius"] == pytest.approx(15)
    fresh = _doc(_cylinder(15, (5, 5)), 30)
    assert doc.part.volume == pytest.approx(_fresh(_cylinder(15, (5, 5)), 30, kind, 2.0,
                                                   [_circle_mid(fresh, 30)]), rel=1e-9)


def test_hole_edge_follows_a_moved_hole_but_not_to_another_hole():
    doc = _doc(_holed())
    _add(doc, "chamfer", 1.0, [_circle_mid(doc, 10)])
    _edit_sketch(doc, lambda sk: sk.points.__setitem__(4, [23.0, 15.0]))  # hole moved 3 mm (< radius)
    assert not doc.errors, doc.errors
    assert np.allclose(doc.features[2].ref_sigs[0]["center"], (23, 15, 10))
    _edit_sketch(doc, lambda sk: sk.points.__setitem__(4, [33.0, 15.0]))  # moved 10 mm: further than its size
    assert 2 in doc.errors and "can't be found" in doc.errors[2]

    # two holes; the chamfered one is removed: the other is never used instead
    def two(sk):
        sk.add_rect(0, 0, 60, 30)
        sk.add_circle(sk.add_point(15, 15), 5)
        sk.add_circle(sk.add_point(45, 15), 5)
    doc = _doc(two)
    _add(doc, "fillet", 1.0, [_circle_mid(doc, 10, near=(15, 15))])
    assert not doc.errors, doc.errors
    _edit_sketch(doc, lambda sk: sk.circles.pop(0))
    assert 2 in doc.errors and "can't be found" in doc.errors[2]


def test_mixed_circular_and_straight_edges_follow_together_and_round_trip(tmp_path):
    doc = _doc(_holed())
    _add(doc, "chamfer", 1.0, [_circle_mid(doc, 10), (20, 0, 10)])
    assert not doc.errors, doc.errors
    doc.features[1].value = 15
    _edit(doc, 1)
    assert not doc.errors, doc.errors
    f = doc.features[2]
    assert f.ref_sigs[0]["type"] == "CIRCLE" and f.ref_sigs[1]["type"] == "LINE"
    assert np.allclose(f.ref_sigs[0]["center"], (20, 15, 15)) and np.allclose(f.edges[1], (20, 0, 15))
    fresh = _doc(_holed(), 15)
    assert doc.part.volume == pytest.approx(
        _fresh(_holed(), 15, "chamfer", 1.0, [_circle_mid(fresh, 15), (20, 0, 15)]), rel=1e-9)
    path = tmp_path / "mixed.rcad"
    doc.save(str(path))
    loaded = Document.load(str(path))
    assert not loaded.errors and loaded.part.volume == pytest.approx(doc.part.volume, rel=1e-9)
    assert loaded.features[2].ref_sigs == f.ref_sigs


# -- edges with no corner
def test_a_holes_seam_is_refused_alone_and_skipped_beside_a_corner():
    doc = _doc(_holed())
    seam = next(e.mid for e in doc.mesh.edges if len(e.polyline) == 2
                and abs(np.linalg.norm(e.mid[:2] - (20, 15)) - 5) < 1e-6)
    _add(doc, "fillet", 1.0, [seam])
    assert 2 in doc.errors and "seam" in doc.errors[2]
    doc.features[2].edges = [tuple(seam), (20.0, 0.0, 10.0)]
    doc.features[2].ref_sigs = []
    _edit(doc, 2)
    assert not doc.errors, doc.errors
    assert doc.part.volume == pytest.approx(_fresh(_holed(), 10, "fillet", 1.0, [(20, 0, 10)]), rel=1e-9)


@pytest.mark.parametrize("kind", ["fillet", "chamfer"])
def test_the_smooth_edge_beside_a_fillet_is_refused_clearly(kind):
    doc = _doc(_box())
    _add(doc, "fillet", 2.0, [(20, 0, 10)])
    _add(doc, kind, 1.0, [(20, 2, 10)])  # where the rounded face meets the top: no corner
    assert 3 in doc.errors and "smooth join" in doc.errors[3]


# -- sizes
@pytest.mark.parametrize("kind, value, message", [
    ("fillet", 0.0, "fillet radius must be greater than zero"),
    ("fillet", -1.0, "fillet radius must be greater than zero"),
    ("chamfer", 0.0, "chamfer length must be greater than zero"),
    ("fillet", 15.0, "fillet radius is too large"),  # the box is 10 tall
    ("chamfer", 100.0, "chamfer length is too large"),
])
def test_bad_sizes_are_reported_and_the_part_before_is_kept(kind, value, message):
    doc = _doc(_box())
    _add(doc, kind, value, [(20, 0, 10)])
    assert 2 in doc.errors and doc.errors[2].startswith(message)
    assert "max_fillet" not in doc.errors[2]  # not build123d's API advice
    assert doc.part.volume == pytest.approx(40 * 30 * 10)


# -- separate bodies
def _two_blocks(sk, shift=0.0):
    sk.add_rect(0, 0, 20, 20)
    sk.add_rect(40 + shift, 0, 60 + shift, 20)


def test_one_feature_over_two_bodies_keeps_both_through_edits():
    doc = _doc(_two_blocks)
    _add(doc, "fillet", 2.0, [(10, 0, 10), (50, 0, 10)])
    assert not doc.errors and len(doc.part.solids()) == 2
    doc.features[1].value = 12
    _edit(doc, 1)
    assert not doc.errors and len(doc.part.solids()) == 2
    assert doc.part.volume == pytest.approx(
        _fresh(_two_blocks, 12, "fillet", 2.0, [(10, 0, 12), (50, 0, 12)]), rel=1e-9)


def test_each_body_keeps_its_own_operation_when_one_moves():
    doc = _doc(_two_blocks)
    _add(doc, "fillet", 2.0, [(10, 0, 10)])
    _add(doc, "chamfer", 2.0, [(50, 0, 10)])

    def shift(sk):
        for p in sk.points[4:]:
            p[0] += 10
    _edit_sketch(doc, shift)
    assert not doc.errors, doc.errors
    assert np.allclose(doc.features[2].edges[0], (10, 0, 10)) and np.allclose(doc.features[3].edges[0], (60, 0, 10))
    fresh = _doc(lambda sk: _two_blocks(sk, 10))
    _add(fresh, "fillet", 2.0, [(10, 0, 10)])
    _add(fresh, "chamfer", 2.0, [(60, 0, 10)])
    assert len(doc.part.solids()) == 2 and doc.part.volume == pytest.approx(fresh.part.volume, rel=1e-9)


# -- edits to earlier features
def test_fillet_radius_edits_carry_the_adjacent_chamfer_and_recover_from_a_failure():
    doc = _doc(_box())
    _add(doc, "fillet", 2.0, [(20, 0, 10)])
    _add(doc, "chamfer", 1.0, [(40, 15, 10)])  # the top right edge, which meets the fillet
    assert not doc.errors, doc.errors
    doc.features[2].value = 4.0
    _edit(doc, 2)
    assert not doc.errors and np.allclose(doc.features[3].edges[0], (40, 17, 10))  # shortened by the fillet
    doc.features[2].value = 0.5  # OpenCascade can't chamfer 1 mm next to a 0.5 mm fillet
    _edit(doc, 2)
    assert 3 in doc.errors and "too large" in doc.errors[3]
    doc.features[2].value = 2.0
    _edit(doc, 2)
    assert not doc.errors, doc.errors
    fresh = _doc(_box())
    _add(fresh, "fillet", 2.0, [(20, 0, 10)])
    _add(fresh, "chamfer", 1.0, [(40, 16, 10)])
    assert doc.part.volume == pytest.approx(fresh.part.volume, rel=1e-9)


def test_a_lost_reference_comes_back_when_the_edit_is_reverted():
    doc = _doc(_two_blocks)
    _add(doc, "fillet", 2.0, [(10, 0, 10)])
    before = doc.part.volume
    saved = [list(p) for p in doc.features[0].sketch.points], list(doc.features[0].sketch.lines), \
        list(doc.features[0].sketch.constraints)

    def drop_left(sk):
        keep = Sketch(standard_plane("XY"))
        keep.add_rect(40, 0, 60, 20)
        sk.points, sk.lines, sk.constraints = keep.points, keep.lines, keep.constraints
    _edit_sketch(doc, drop_left)
    assert 2 in doc.errors

    def restore(sk):
        sk.points, sk.lines, sk.constraints = saved
    _edit_sketch(doc, restore)
    assert not doc.errors and doc.part.volume == pytest.approx(before, rel=1e-9)


def test_a_cut_inserted_before_a_fillet_keeps_it():
    doc = _doc(_box())
    _add(doc, "fillet", 2.0, [(20, 0, 10)])
    hole = Sketch(standard_plane("XY"))
    hole.add_circle(hole.add_point(20, 15), 3)
    doc.features[2:2] = [Feature("sketch", "H", sketch=hole), Feature("extrude", sketch_ref=2, value=10, cut=True)]
    doc.rebuild(0)
    assert not doc.errors, doc.errors
    fresh = _doc(_box())
    fresh.features += [Feature("sketch", "H", sketch=hole), Feature("extrude", sketch_ref=2, value=10, cut=True)]
    fresh.rebuild(0)
    _add(fresh, "fillet", 2.0, [(20, 0, 10)])
    assert doc.part.volume == pytest.approx(fresh.part.volume, rel=1e-9)


# -- undo / redo in the app
def test_undo_and_redo_restore_edge_operations_and_their_references(qapp, monkeypatch):
    from rcad.app import MainWindow
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w = MainWindow()
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 40, 30)
    w.doc.features = [Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=10)]
    w.doc.rebuild(0)
    assert w.commit(Feature("fillet", value=2.0, edges=[(20.0, 0.0, 10.0)]))
    assert w.commit(Feature("chamfer", value=1.0, edges=[(40.0, 30.0, 5.0)]))
    v10 = w.doc.part.volume
    w._push_undo()  # an edit to the extrude height
    w.doc.features[1].value = 20
    w.doc.features[1].version += 1
    w.rebuild()
    assert not w.doc.errors and np.allclose(w.doc.features[2].edges[0], (20, 0, 20))
    v20 = w.doc.part.volume

    w.undo()
    assert not w.doc.errors and w.doc.part.volume == pytest.approx(v10, rel=1e-9)
    assert np.allclose(w.doc.features[2].edges[0], (20, 0, 10))
    w.redo()
    assert not w.doc.errors and w.doc.part.volume == pytest.approx(v20, rel=1e-9)
    assert np.allclose(w.doc.features[3].ref_sigs[0]["mid"], (40, 30, 10))
    # the rounded edges can still be found (to drag or delete) after redo
    assert all(w._find_fillet_by_edge(m) == 2 for m in w.doc.features[2].result_edges)

    w.undo()
    w.undo()  # the chamfer
    assert [f.kind for f in w.doc.features] == ["sketch", "extrude", "fillet"]
    w.undo()  # the fillet
    assert w.doc.part.volume == pytest.approx(40 * 30 * 10)
    w.redo()
    w.redo()
    assert [f.kind for f in w.doc.features][-2:] == ["fillet", "chamfer"] and not w.doc.errors
    assert w.doc.part.volume == pytest.approx(v10, rel=1e-9)
