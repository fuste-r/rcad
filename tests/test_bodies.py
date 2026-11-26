"""Move and Mirror for whole bodies: the kernel features, how they find their bodies again after
upstream edits, and the quick buttons / settings bar that create and edit them."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad import refs
from rcad.kernel import Document, Feature
from rcad.sketch import Sketch, standard_plane


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)


def _two_blocks(height=10.0):
    """Blocks 0..20 and 40..60 in x, 20 deep."""
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 20, 20)
    sk.add_rect(40, 0, 60, 20)
    doc = Document([Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=height)])
    doc.rebuild(0)
    return doc


def _solid(doc, x):
    return next(s for s in doc.part.solids() if s.bounding_box().min.X <= x <= s.bounding_box().max.X)


def _centres(doc):
    return sorted(tuple(round(v, 6) for v in s.center()) for s in doc.part.solids())


def _height(doc, h):
    doc.features[1].value = h
    doc.features[1].version += 1
    doc.rebuild()


# -- kernel
def test_move_translates_and_rotates_one_body_about_its_centre():
    doc = _two_blocks()
    doc.features.append(Feature("move", offset=(0, 30, 5), angle=45, rot_axis="z",
                                ref_sigs=[refs.body_signature(_solid(doc, 50))]))
    doc.rebuild()
    assert not doc.errors, doc.errors
    assert _centres(doc) == [(10, 10, 5), (50, 40, 10)]
    moved = _solid(doc, 50)
    bb = moved.bounding_box()
    assert bb.max.X - bb.min.X == pytest.approx(20 * np.sqrt(2))  # turned 45° about its own centre
    assert doc.part.volume == pytest.approx(2 * 20 * 20 * 10)


def test_move_with_copy_keeps_the_original():
    doc = _two_blocks()
    doc.features.append(Feature("move", offset=(0, 0, 20), copy=True,
                                ref_sigs=[refs.body_signature(_solid(doc, 10))]))
    doc.rebuild()
    assert not doc.errors and len(doc.part.solids()) == 3
    assert _centres(doc) == [(10, 10, 5), (10, 10, 25), (50, 10, 5)]


@pytest.mark.parametrize("plane, at, centre", [
    ("YZ", 0.0, (-10, 10, 5)), ("YZ", 30.0, (50, 10, 5)), ("XZ", -5.0, (10, -20, 5)), ("XY", 20.0, (10, 10, 35)),
])
def test_mirror_reflects_across_an_offset_world_plane(plane, at, centre):
    doc = _two_blocks()
    doc.features.append(Feature("mirror", plane=plane, value=at, ref_sigs=[refs.body_signature(_solid(doc, 10))]))
    doc.rebuild()
    assert not doc.errors, doc.errors
    assert tuple(centre) in _centres(doc)
    assert (10, 10, 5) not in _centres(doc)  # without copy the original goes


def test_a_mirrored_copy_that_touches_joins_the_original():
    doc = _two_blocks()
    doc.features.append(Feature("mirror", plane="YZ", value=0.0, copy=True,
                                ref_sigs=[refs.body_signature(_solid(doc, 10))]))
    doc.rebuild()
    assert not doc.errors and len(doc.part.solids()) == 2  # the left pair is now one body
    assert doc.part.volume == pytest.approx(3 * 20 * 20 * 10)


def test_moving_several_bodies_and_overlapping_another_joins_them():
    doc = _two_blocks()
    sigs = [refs.body_signature(s) for s in doc.part.solids()]
    doc.features.append(Feature("move", offset=(100, 0, 0), ref_sigs=sigs))
    doc.rebuild()
    assert not doc.errors and _centres(doc) == [(110, 10, 5), (150, 10, 5)]
    doc.features[2].ref_sigs = [refs.body_signature(_solid(_two_blocks(), 10))]
    doc.features[2].offset = (30, 0, 0)  # the left block onto the right one, overlapping by half
    doc.features[2].version += 1
    doc.rebuild()
    assert not doc.errors and len(doc.part.solids()) == 1
    assert doc.part.volume == pytest.approx(30 * 20 * 10)


def test_bodies_are_followed_through_upstream_edits_and_round_trip(tmp_path):
    doc = _two_blocks()
    doc.features.append(Feature("move", offset=(0, 40, 0), ref_sigs=[refs.body_signature(_solid(doc, 50))]))
    doc.features.append(Feature("mirror", plane="XY", value=0.0, copy=True,
                                ref_sigs=[refs.body_signature(_solid(doc, 10))]))
    doc.rebuild()
    assert not doc.errors, doc.errors
    _height(doc, 15)  # every body grows: still the same bodies
    assert not doc.errors, doc.errors
    assert _centres(doc) == [(10, 10, 0), (50, 50, 7.5)]  # the left pair joined at z = 0
    sk = doc.features[0].sketch
    for p in sk.points[4:]:
        p[0] += 5  # the right block slides a little
    sk.version += 1
    doc.rebuild()
    assert not doc.errors and (55, 50, 7.5) in _centres(doc)
    path = tmp_path / "bodies.rcad"
    doc.save(str(path))
    loaded = Document.load(str(path))
    assert not loaded.errors and _centres(loaded) == _centres(doc)
    assert [f.label() for f in loaded.features] == [f.label() for f in doc.features]
    assert loaded.features[3].plane == "XY" and loaded.features[3].copy


def test_a_body_that_is_gone_is_reported_not_swapped():
    doc = _two_blocks()
    doc.features.append(Feature("move", offset=(0, 30, 0), ref_sigs=[refs.body_signature(_solid(doc, 10))]))
    doc.rebuild()
    keep = Sketch(standard_plane("XY"))
    keep.add_rect(40, 0, 60, 20)
    sk = doc.features[0].sketch
    sk.points, sk.lines, sk.constraints = keep.points, keep.lines, keep.constraints
    sk.version += 1
    doc.rebuild()
    assert 2 in doc.errors and "body can't be found" in doc.errors[2]


def test_two_equally_likely_bodies_are_ambiguous():
    sig = {"kind": "body", "center": [30, 10, 5], "volume": 4000}  # halfway between two equal bodies
    from build123d import Box, Pos
    solids = [Pos(25, 10, 5) * Box(20, 20, 10), Pos(35, 10, 5) * Box(20, 20, 10)]
    with pytest.raises(refs.RefError, match="ambiguous"):
        refs.resolve_body(solids, sig, 60.0)


def test_old_files_without_move_fields_still_load(tmp_path):
    doc = _two_blocks()
    data = doc.to_dict()
    for f in data["features"]:
        for k in ("offset", "angle", "rot_axis", "plane", "copy"):
            f.pop(k)
    f = Feature.from_dict(data["features"][1])
    assert f.offset == (0.0, 0.0, 0.0) and f.plane == "YZ" and not f.copy


# -- app
def _window():
    from rcad.app import MainWindow
    w = MainWindow()
    w.view.resize(1000, 700)
    w.doc = _two_blocks()
    w.refresh()
    w.view.camera.look_at_plane((30, 10, 5), (0.3, -0.6, 0.74), (1, 0, 0), 160)
    return w


def _select_body(w, x):
    mesh = w.doc.mesh
    body = next(b for f, b in zip(mesh.faces, mesh.face_body) if (f.center[0] < 30) == (x < 30))
    w.view.sel_faces = w._body_faces(body)
    w._update_regions()


def test_a_selected_body_offers_move_and_mirror(qapp, quiet):
    w = _window()
    _select_body(w, 50)
    assert w.quick.labels() == ["✥ Move", "⇋ Mirror"]
    w.view.sel_faces = {next(iter(w.view.sel_faces))}  # one face of it: face tools instead
    w._update_regions()
    assert w.quick.labels() == ["✏️ Sketch", "↕️ Push/Pull"]


def test_move_button_opens_the_settings_bar_and_previews(qapp, quiet):
    w = _window()
    _select_body(w, 50)
    next(b for b in w.quick.buttons if b.text() == "✥ Move").click()
    bar = w.tool_bar
    assert bar.active and set(bar.fields) == {"x", "y", "z", "angle"} and set(bar.choices) == {"rot_axis", "copy"}
    assert not w.view.sel_faces  # the selection's faces renumber as the body moves
    bar.fields["y"].setText("30")
    assert (50, 40, 5) in _centres(w.doc)  # live preview
    bar.choices["copy"][0].click()  # Move -> Copy
    assert len(w.doc.part.solids()) == 3
    bar.finish()
    f = w.doc.features[-1]
    assert f.kind == "move" and f.offset == (0, 30, 0) and f.copy
    w.undo()
    assert w.doc.features[-1].kind == "extrude" and _centres(w.doc) == [(10, 10, 5), (50, 10, 5)]


def test_mirror_defaults_to_copy_and_cancel_puts_the_model_back(qapp, quiet):
    w = _window()
    _select_body(w, 10)
    w.mirror_bodies()
    bar = w.tool_bar
    assert bar.active and bar.choices["copy"][1].current == "copy"
    assert len(w.doc.part.solids()) == 2 and w.doc.part.volume == pytest.approx(3 * 4000)  # joined at x = 0
    bar.choices["plane"][0].click()  # YZ -> XZ: the copy lands at y -20..0 and joins along y = 0
    assert (10, 0, 5) in _centres(w.doc)
    bar.cancel()
    assert w.doc.features[-1].kind == "extrude" and _centres(w.doc) == [(10, 10, 5), (50, 10, 5)]


def test_editing_a_move_reopens_its_values(qapp, quiet):
    w = _window()
    _select_body(w, 50)
    w.move_bodies()
    w.tool_bar.fields["x"].setText("5")
    w.tool_bar.fields["angle"].setText("90")
    w.tool_bar.finish()
    i = len(w.doc.features) - 1
    w.edit_feature(w.features.item(i))
    bar = w.tool_bar
    assert bar.active and bar.fields["x"].value() == pytest.approx(5) and bar.fields["angle"].value() == 90
    bar.fields["x"].setText("10")
    bar.cancel()  # back to 5
    assert w.doc.features[i].offset == (5, 0, 0) and (55, 10, 5) in _centres(w.doc)


def test_move_without_a_body_selected_explains(qapp, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: shown.append(a[2]))
    w = _window()
    w.move_bodies()
    assert not w.tool_bar.active and shown and "double-click" in shown[0]
