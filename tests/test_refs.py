"""Faces and edges picked by later features follow upstream edits, and a reference that
can't be found again (or matches two things equally) is reported instead of rebinding."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from build123d import Box

from rcad import refs
from rcad.kernel import Document, Feature
from rcad.sketch import Sketch, standard_plane


def _doc(rects=((0, 0, 40, 30),), height=10.0):
    sk = Sketch(standard_plane("XY"))
    for r in rects:
        sk.add_rect(*r)
    doc = Document()
    doc.features = [Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=height)]
    doc.rebuild(0)
    assert not doc.errors
    return doc


def _set_height(doc, h):
    doc.features[1].value = h
    doc.features[1].version += 1
    doc.rebuild()


def _set_width(doc, w):
    sk = doc.features[0].sketch
    for p in sk.points:
        if p[0] > 1:
            p[0] = float(w)
    sk.version += 1
    doc.rebuild()


def _pushpull_top(doc, value, center=(20, 15, 10)):
    doc.features.append(Feature("pushpull", value=value, face=(tuple(map(float, center)), (0.0, 0.0, 1.0))))
    doc.rebuild()


def _edge(doc, kind, value, mid):
    doc.features.append(Feature(kind, value=value, edges=[tuple(map(float, mid))]))
    doc.rebuild()


def test_pushpull_follows_extrude_height_and_sketch_width():
    doc = _doc()
    _pushpull_top(doc, 5)
    assert not doc.errors and doc.part.volume == pytest.approx(40 * 30 * 15)
    assert doc.features[2].ref_sigs  # a signature is stored on the first rebuild
    _set_height(doc, 20)
    assert not doc.errors and doc.part.volume == pytest.approx(40 * 30 * 25)
    assert doc.features[2].face[0][2] == pytest.approx(20)  # the reference followed the face
    _set_width(doc, 60)
    assert not doc.errors and doc.part.volume == pytest.approx(60 * 30 * 25)
    assert doc.features[2].face[0][0] == pytest.approx(30)


def _fresh_volume(width, height, picks):
    """Volume of the same fillets/chamfers picked directly on a box of this size."""
    doc = _doc(((0, 0, width, 30),), height)
    for kind, value, mid in picks:
        _edge(doc, kind, value, mid)
    assert not doc.errors, doc.errors
    return doc.part.volume


def test_fillets_and_chamfers_follow_height_and_width_changes():
    doc = _doc()
    _edge(doc, "fillet", 2.0, (20, 0, 10))  # top front edge
    _edge(doc, "chamfer", 3.0, (40, 30, 5))  # back right vertical edge
    assert not doc.errors
    _set_height(doc, 20)
    _set_width(doc, 60)
    assert not doc.errors, doc.errors
    fillet, chamfer = doc.features[2], doc.features[3]
    assert np.allclose(fillet.edges[0], (30, 0, 20)) and np.allclose(chamfer.edges[0], (60, 30, 10))
    expected = _fresh_volume(60, 20, [("fillet", 2.0, (30, 0, 20)), ("chamfer", 3.0, (60, 30, 10))])
    assert doc.part.volume == pytest.approx(expected, rel=1e-9)
    # the fillet's sampled path follows too, so its rounded edges can be found and dragged again
    assert np.allclose(np.mean(fillet.edge_refs[0], axis=0), (30, 0, 20))


def test_a_look_alike_elsewhere_is_not_used():
    # two blocks; push/pull and fillet the left one, then remove it from the sketch
    doc = _doc(((0, 0, 20, 20), (40, 0, 60, 20)))
    _pushpull_top(doc, 5, center=(10, 10, 10))
    assert not doc.errors
    sk = doc.features[0].sketch
    keep = Sketch(standard_plane("XY"))
    keep.add_rect(40, 0, 60, 20)
    sk.points, sk.lines, sk.constraints = keep.points, keep.lines, keep.constraints
    sk.version += 1
    doc.rebuild()
    assert 2 in doc.errors and "face can't be found" in doc.errors[2]  # not the right block's top
    doc2 = _doc(((0, 0, 20, 20), (40, 0, 60, 20)))
    _edge(doc2, "fillet", 2.0, (10, 0, 10))
    assert not doc2.errors
    sk = doc2.features[0].sketch
    sk.points, sk.lines, sk.constraints = keep.points, keep.lines, keep.constraints
    sk.version += 1
    doc2.rebuild()
    assert 2 in doc2.errors and "edge can't be found" in doc2.errors[2]


def test_two_equally_good_candidates_are_reported_as_ambiguous():
    topo = refs.Topology(Box(40, 30, 10))  # centred: top front/back edges at y = -15 / +15, z = 5
    sig = {"kind": "edge", "type": "LINE", "mid": [0, 0, 5], "dir": [1, 0, 0], "length": 40.0}
    with pytest.raises(refs.RefError, match="ambiguous"):
        refs.resolve_edge(topo, sig)


def test_a_face_that_is_gone_is_reported():
    topo = refs.Topology(Box(40, 30, 10))
    sig = {"kind": "face", "center": [0, 0, 5], "normal": [0.7071, 0.7071, 0]}
    with pytest.raises(refs.RefError, match="can't be found"):
        refs.resolve_face(topo, sig)


def test_exact_matches_still_resolve_directly():
    topo = refs.Topology(Box(40, 30, 10))
    top = refs.resolve_face(topo, {"kind": "face", "center": [0, 0, 5], "normal": [0, 0, 1]})
    assert top.center().Z == pytest.approx(5)
    e = refs.resolve_edge(topo, {"kind": "edge", "mid": [0, -15, 5]})
    assert (e @ 0.5).Y == pytest.approx(-15) and (e @ 0.5).Z == pytest.approx(5)


def test_failure_names_which_of_several_edges(tmp_path):
    doc = _doc(((0, 0, 20, 20), (40, 0, 60, 20)))
    doc.features.append(Feature("fillet", value=1.0, edges=[(50.0, 0.0, 10.0), (10.0, 0.0, 10.0)]))
    doc.rebuild()
    assert not doc.errors
    sk = doc.features[0].sketch
    keep = Sketch(standard_plane("XY"))
    keep.add_rect(40, 0, 60, 20)
    sk.points, sk.lines, sk.constraints = keep.points, keep.lines, keep.constraints
    sk.version += 1
    doc.rebuild()
    assert "edge 2 of 2" in doc.errors[2]


def test_signatures_survive_save_and_load_and_old_files_gain_them(tmp_path):
    doc = _doc()
    _edge(doc, "fillet", 2.0, (20, 0, 10))
    path = tmp_path / "m.rcad"
    doc.save(str(path))
    loaded = Document.load(str(path))
    assert loaded.features[2].ref_sigs == doc.features[2].ref_sigs
    # a file from before signatures: no ref_sigs; it resolves as before and gains them
    data = json.loads(path.read_text())
    for f in data["features"]:
        f.pop("ref_sigs", None)
    path.write_text(json.dumps(data))
    assert all("ref_sigs" not in f for f in json.loads(path.read_text())["features"])
    old = Document.load(str(path))  # loading rebuilds: the old references resolve and gain signatures
    assert not old.errors and old.features[2].ref_sigs
    _set_height(old, 20)  # and then follows edits like a new one
    assert not old.errors and np.allclose(old.features[2].edges[0], (20, 0, 20))
