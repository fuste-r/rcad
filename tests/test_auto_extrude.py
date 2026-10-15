import pytest

from rcad.kernel import Document, Feature
from rcad.sketch import Sketch, SketchPlane, standard_plane


def base_doc():
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, 40, 30)
    return Document([Feature("sketch", "S1", sketch=sk), Feature("extrude", sketch_ref=0, value=10, auto=True)])


def top_sketch(doc):
    sk = Sketch(SketchPlane((0, 0, 10), (1, 0, 0), (0, 0, 1)))
    sk.add_rect(10, 10, 20, 20)
    doc.features.append(Feature("sketch", "S2", sketch=sk))
    return len(doc.features) - 1


def test_auto_adds_outward_and_cuts_inward():
    doc = base_doc()
    s = top_sketch(doc)
    doc.features.append(Feature("extrude", sketch_ref=s, value=5, auto=True))
    doc.rebuild()
    assert not doc.errors
    assert doc.part.volume == pytest.approx(40 * 30 * 10 + 100 * 5)
    assert doc.features[-1].resolved_cut is False

    doc.features[-1].value = -4
    doc.rebuild()
    assert doc.part.volume == pytest.approx(40 * 30 * 10 - 100 * 4)
    assert doc.features[-1].resolved_cut is True
    assert doc.features[-1].label().startswith("Cut auto")


def test_first_extrude_is_always_an_add():
    doc = base_doc()
    doc.rebuild()
    assert doc.features[1].resolved_cut is False
    assert doc.part.volume == pytest.approx(40 * 30 * 10)


def test_explicit_modes_ignore_direction():
    doc = base_doc()
    s = top_sketch(doc)
    doc.features.append(Feature("extrude", sketch_ref=s, value=-4, auto=False, cut=False))  # add inside: no change
    doc.rebuild()
    assert doc.part.volume == pytest.approx(40 * 30 * 10)
    doc.features[-1].cut = True
    doc.rebuild()
    assert doc.part.volume == pytest.approx(40 * 30 * 10 - 100 * 4)


def test_auto_survives_save_and_load(tmp_path):
    doc = base_doc()
    s = top_sketch(doc)
    doc.features.append(Feature("extrude", sketch_ref=s, value=-4, auto=True))
    doc.rebuild()
    path = str(tmp_path / "a.rcad")
    doc.save(path)
    again = Document.load(path)
    assert again.features[-1].auto and again.features[-1].resolved_cut is True
    assert again.part.volume == pytest.approx(doc.part.volume)
