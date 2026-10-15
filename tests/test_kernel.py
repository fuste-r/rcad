import numpy as np
import pytest

from rcad.kernel import Document, Feature
from rcad.sketch import Constraint, Sketch, standard_plane


def rect_sketch(w=40, h=30):
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, w, h)
    return sk


def test_constraints_solve():
    sk = rect_sketch(37, 22)
    sk.constraints.append(Constraint("fixed", [0]))
    sk.constraints.append(Constraint("distance", [0, 1], 50.0))
    sk.constraints.append(Constraint("distance", [1, 2], 20.0))
    assert sk.solve()
    assert sk.dof == 0
    xs = sorted({round(p[0], 6) for p in sk.points})
    ys = sorted({round(p[1], 6) for p in sk.points})
    assert xs == [0, 50] and ys == [0, 20]


def test_inconsistent_reports_constraint():
    sk = rect_sketch()
    sk.constraints.append(Constraint("distance", [0, 1], 10.0))
    sk.constraints.append(Constraint("distance", [0, 1], 20.0))
    before = [list(p) for p in sk.points]
    assert not sk.solve()
    assert "inconsistent" in sk.status
    assert sk.points == before


def test_extrude_with_hole_and_features():
    sk = rect_sketch(40, 30)
    c = sk.add_point(20, 15)
    sk.add_circle(c, 5)
    sk.constraints.append(Constraint("diameter", [0], 12.0))
    assert sk.solve()
    doc = Document([Feature("sketch", "S1", sketch=sk), Feature("extrude", sketch_ref=0, value=10)])
    doc.rebuild()
    assert not doc.errors
    expected = 40 * 30 * 10 - np.pi * 6 ** 2 * 10
    assert doc.part.volume == pytest.approx(expected, rel=1e-3)
    assert doc.mesh.faces and doc.mesh.edges


def test_pushpull_fillet_chamfer_and_roundtrip(tmp_path):
    doc = Document([Feature("sketch", "S1", sketch=rect_sketch()), Feature("extrude", sketch_ref=0, value=10)])
    doc.rebuild()
    top = [f for f in doc.mesh.faces if f.normal is not None and f.normal[2] > 0.99][0]
    doc.features.append(Feature("pushpull", value=5, face=(tuple(top.center), tuple(top.normal))))
    doc.rebuild()
    assert not doc.errors
    assert doc.part.volume == pytest.approx(40 * 30 * 15, rel=1e-6)

    doc.features.append(Feature("pushpull", value=-3, face=(tuple(top.center + [0, 0, 5]), (0, 0, 1))))
    doc.rebuild()
    assert not doc.errors, doc.errors
    assert doc.part.volume == pytest.approx(40 * 30 * 12, rel=1e-6)

    edge = doc.mesh.edges[0]
    doc.features.append(Feature("fillet", value=1.0, edges=[tuple(edge.mid)]))
    doc.rebuild()
    assert not doc.errors, doc.errors
    assert doc.part.volume < 40 * 30 * 12
    edge = doc.mesh.edges[3]
    doc.features.append(Feature("chamfer", value=1.0, edges=[tuple(edge.mid)]))
    doc.rebuild()
    assert not doc.errors, doc.errors

    path = str(tmp_path / "p.rcad")
    doc.save(path)
    again = Document.load(path)
    assert not again.errors
    assert again.part.volume == pytest.approx(doc.part.volume, rel=1e-9)
    assert [f.kind for f in again.features] == [f.kind for f in doc.features]


def test_bad_feature_reports_error_and_keeps_earlier():
    doc = Document([Feature("sketch", "S1", sketch=rect_sketch()), Feature("extrude", sketch_ref=0, value=10),
                    Feature("fillet", value=1.0, edges=[(99, 99, 99)])])
    doc.rebuild()
    assert 2 in doc.errors
    assert doc.part is not None


def test_sketch_on_face_plane():
    doc = Document([Feature("sketch", "S1", sketch=rect_sketch()), Feature("extrude", sketch_ref=0, value=10)])
    doc.rebuild()
    top = [f for f in doc.mesh.faces if f.normal is not None and f.normal[2] > 0.99][0]
    from rcad.sketch import SketchPlane
    pl = SketchPlane(tuple(top.center), (1, 0, 0), tuple(top.normal))
    sk = Sketch(pl)
    sk.add_rect(-5, -5, 5, 5)
    doc.features += [Feature("sketch", "S2", sketch=sk), Feature("extrude", sketch_ref=2, value=-4, cut=True)]
    doc.rebuild()
    assert not doc.errors, doc.errors
    assert doc.part.volume == pytest.approx(40 * 30 * 10 - 100 * 4, rel=1e-6)


def test_layers_suppress_and_persist(tmp_path):
    doc = Document([Feature("sketch", "S1", sketch=rect_sketch()), Feature("extrude", sketch_ref=0, value=10)])
    doc.features.append(Feature("fillet", value=1.0, edges=[(0, 0, 5)]))
    doc.rebuild()
    assert not doc.errors
    filleted = doc.part.volume
    doc.features[2].visible = False
    doc.rebuild()
    assert doc.part.volume == pytest.approx(40 * 30 * 10, rel=1e-6) and filleted < doc.part.volume
    doc.features[1].visible = False
    doc.rebuild()
    assert doc.part is None
    doc.features[0].visible = False  # hiding a sketch layer must not affect geometry
    doc.features[1].visible = True
    doc.rebuild()
    assert doc.part.volume == pytest.approx(40 * 30 * 10, rel=1e-6)
    path = str(tmp_path / "l.rcad")
    doc.save(path)
    assert [f.visible for f in Document.load(path).features] == [False, True, False]
