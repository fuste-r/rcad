import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from rcad import app as appmod
from rcad.app import MainWindow
from rcad.kernel import Feature
from rcad.sketch import Sketch, SketchPlane, standard_plane


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def px(w, p):
    v = w.view
    return tuple(v.camera.project(np.array([p], float), v.width(), v.height())[0])


def click(w, p, mods=Qt.NoModifier):
    x, y = px(w, p)
    w.on_click(x, y, Qt.LeftButton, mods)


def test_full_workflow(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = MainWindow()
    w.view.resize(1000, 700)

    # sketch a rectangle on XY and dimension it
    w.plane_combo.setCurrentText("XY")
    w.new_sketch()
    assert w.mode == "sketch"
    w.set_tool("rect")
    click(w, (0, 0, 0))
    click(w, (40, 30, 0))
    sk = w.sketch
    assert len(sk.points) == 4 and len(sk.lines) == 4

    w.set_tool("select")
    click(w, (20, 0, 0))  # bottom edge
    assert w.view.sketch_sel == {("l", 0)}
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (50.0, True))
    w.add_constraint("distance")
    assert sk.constraints[-1].kind == "distance"
    xs = sorted({round(p[0], 4) for p in sk.points})
    assert xs[1] - xs[0] == pytest.approx(50.0)

    # a conflicting dimension is rejected and rolled back
    n = len(sk.constraints)
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (60.0, True))
    w.view.sketch_sel = {("l", 0)}
    w.add_constraint("distance")
    assert len(sk.constraints) == n

    # a circle with a diameter
    w.set_tool("circle")
    click(w, (25, 15, 0))
    click(w, (30, 15, 0))
    assert len(sk.circles) == 1
    w.set_tool("select")
    w.view.sketch_sel = {("c", 0)}
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (8.0, True))
    w.add_constraint("diameter")
    assert sk.circles[0][1] == pytest.approx(4.0)

    w.finish_sketch()
    assert w.mode == "model"

    # extrude
    monkeypatch.setattr(appmod.ExtrudeDialog, "exec", lambda self: True)
    w.extrude()
    assert not w.doc.errors
    vol = 50 * 30 * 10 - np.pi * 16 * 10
    assert w.doc.part.volume == pytest.approx(vol, rel=1e-3)

    # push/pull the top face
    w.view.camera.look_at_plane((25, 15, 0), (0.3, -0.6, 0.74), (1, 0, 0), 150)
    top = [i for i, f in enumerate(w.doc.mesh.faces) if f.normal is not None and f.normal[2] > 0.99]
    assert top
    w.view.sel_faces = {top[0]}
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (5.0, True))
    w.pushpull()
    assert not w.doc.errors
    assert w.doc.part.volume == pytest.approx(vol + (50 * 30 - np.pi * 16) * 5, rel=1e-3)

    # fillet / chamfer a picked outer edge
    edge = max(range(len(w.doc.mesh.edges)), key=lambda i: w.doc.mesh.edges[i].mid[2] + w.doc.mesh.edges[i].mid[0])
    w.view.sel_edges = {edge}
    before = w.doc.part.volume
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (1.0, True))
    w.fillet()
    assert not w.doc.errors and w.doc.part.volume < before

    # picking in model mode selects faces
    w.view.camera.look_at_plane((25, 15, 0), (0.3, -0.6, 0.74), (1, 0, 0), 150)
    f = w.view.pick_face(*px(w, (25, 15, 15)))
    assert f is not None

    # save / reload
    path = str(tmp_path / "proj.rcad")
    w.path = path
    w.file_save()
    w2 = MainWindow()
    w2._load(appmod.Document.load(path), path)
    assert not w2.doc.errors
    assert w2.doc.part.volume == pytest.approx(w.doc.part.volume, rel=1e-9)

    # deleting the sketch removes its extrude
    w.doc.delete(0)
    assert [f.kind for f in w.doc.features][:1] != ["extrude"]


def test_expression_constraint_create_undo_redo_and_roundtrip(qapp, monkeypatch, tmp_path):
    w = MainWindow()
    sketch = Sketch(standard_plane("XY"), points=[[0, 0], [10, 0]])
    w.doc.features = [Feature("sketch", "S1", sketch=sketch)]
    w.doc.variables = {"a": 12.5}
    w.enter_sketch(0)
    w.view.sketch_sel = {("p", 0), ("p", 1)}

    class AcceptedExpressionDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return True

        def expression(self):
            return "2*a"

    monkeypatch.setattr(appmod, "ExpressionDialog", AcceptedExpressionDialog)
    w.add_constraint("expression")
    assert len(sketch.constraints) == 1
    assert sketch.constraints[0].expression == "2*a"
    assert np.linalg.norm(np.array(sketch.points[1]) - np.array(sketch.points[0])) == pytest.approx(25)

    assert w.undo_sketch()
    assert sketch.constraints == []
    assert w.redo_sketch()
    assert sketch.constraints[0].expression == "2*a"
    assert np.linalg.norm(np.array(sketch.points[1]) - np.array(sketch.points[0])) == pytest.approx(25)

    w.finish_sketch()
    w.doc.rebuild(0)
    assert sketch.constraints[0].expression == "2*a"

    path = tmp_path / "expression.rcad"
    w.doc.save(str(path))
    loaded = appmod.Document.load(str(path))
    loaded_sketch = loaded.features[0].sketch
    loaded_sketch.variables = loaded.variables
    assert loaded_sketch.constraints[0].expression == "2*a"
    assert loaded_sketch.solve()
    assert np.linalg.norm(np.array(loaded_sketch.points[1]) - np.array(loaded_sketch.points[0])) == pytest.approx(25)


def test_revolve_workflow_and_edit(qapp, monkeypatch):
    w = MainWindow()
    w.plane_combo.setCurrentText("XZ")
    w.new_sketch()
    w.sketch.add_rect(10, 0, 20, 30)
    w.sketch.solve()
    w.finish_sketch()
    class AcceptedRevolveDialog:
        angle_value = 360.0

        def __init__(self, *args, **kwargs):
            self.angle = type("Angle", (), {"value": lambda _: self.angle_value})()
            self.axis_name = "u"
            self.cut = False

        def exec(self):
            return True

    monkeypatch.setattr(appmod, "RevolveDialog", AcceptedRevolveDialog)

    w.revolve()
    feature = w.doc.features[-1]
    expected = np.pi * (20**2 - 10**2) * 30
    assert feature.kind == "revolve"
    assert feature.axis == "u"
    assert not w.doc.errors, w.doc.errors
    assert w.doc.part.volume == pytest.approx(expected, rel=1e-6)

    feature_item = w.features.item(w.features.count() - 1)
    AcceptedRevolveDialog.angle_value = 180.0
    w.edit_feature(feature_item)
    assert feature.value == pytest.approx(180.0)
    assert w.doc.part.volume == pytest.approx(expected / 2, rel=1e-6)


def test_sweep_and_loft_actions(qapp, monkeypatch):
    profile = Sketch(standard_plane("XY"))
    profile.add_rect(0, 0, 10, 4)
    path = Sketch(standard_plane("XZ"))
    path.points = [[0, 0], [0, 20]]
    path.add_line(0, 1)
    lower = Sketch(standard_plane("XY"))
    lower.add_rect(-5, -5, 5, 5)
    upper = Sketch(SketchPlane((0, 0, 10), (1, 0, 0), (0, 0, 1)))
    upper.add_rect(-10, -10, 10, 10)

    class AcceptedSweepDialog:
        path_ref = 1
        cut = False

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return True

    class AcceptedLoftDialog:
        section_refs = [0, 1]
        cut = False

        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return True

    w = MainWindow()
    w.doc.features = [Feature("sketch", "Profile", sketch=profile),
                      Feature("sketch", "Path", sketch=path),
                      Feature("sketch", "Lower", sketch=lower),
                      Feature("sketch", "Upper", sketch=upper)]
    w.doc.rebuild()
    w.refresh()
    w.features.setCurrentRow(0)
    monkeypatch.setattr(appmod, "SweepDialog", AcceptedSweepDialog)
    w.sweep()
    assert w.doc.features[-1].kind == "sweep"
    assert not w.doc.errors, w.doc.errors
    assert w.doc.part.volume == pytest.approx(800.0, rel=1e-6)

    w.doc.features = [w.doc.features[2], w.doc.features[3]]
    w.doc.rebuild(0)
    w.refresh()
    monkeypatch.setattr(appmod, "LoftDialog", AcceptedLoftDialog)
    w.loft()
    assert w.doc.features[-1].kind == "loft"
    assert not w.doc.errors, w.doc.errors
    assert w.doc.part.volume == pytest.approx(10 / 3 * (100 + 200 + 400), rel=1e-6)


def test_drag_fillet_preview_commit(qapp, monkeypatch):
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.set_tool("rect")
    click(w, (0, 0, 0))
    click(w, (40, 30, 0))
    w.finish_sketch()
    monkeypatch.setattr(appmod.ExtrudeDialog, "exec", lambda self: True)
    w.extrude()
    assert not w.doc.errors
    vol0 = w.doc.part.volume

    # a picked edge midpoint to fillet
    edge = w.doc.mesh.edges[0]
    mids = [tuple(edge.mid)]

    # start a live fillet preview (viewport would emit this on drag)
    w.on_fillet_preview(2.0, mids)
    assert w.doc.features[-1].kind == "fillet"
    assert w.doc.part is not None and w.doc.part.volume < vol0
    assert not w.doc.errors

    # dragging to a slightly larger radius updates the same temp feature
    w.on_fillet_preview(2.5, mids)
    assert w.doc.features[-1].kind == "fillet"
    assert w.doc.features[-1].value == pytest.approx(2.5)

    # release commits the fillet at the final radius
    w.on_fillet_commit(2.5, mids)
    assert w._fillet_live_idx is None
    assert w.doc.features[-1].kind == "fillet"
    assert w.doc.features[-1].value == pytest.approx(2.5)
    assert not w.doc.errors
    assert w.doc.part.volume < vol0

    # cancel path: preview then release with ~0 radius rolls the model back
    edge = w.doc.mesh.edges[0]
    mids = [tuple(edge.mid)]
    vol_before_cancel = w.doc.part.volume
    w.on_fillet_preview(1.0, mids)
    assert w.doc.part.volume != vol_before_cancel
    w.on_fillet_commit(0.0, mids)
    assert w._fillet_live_idx is None
    assert not w.doc.errors
    assert w.doc.part.volume == pytest.approx(vol_before_cancel)

    # failing fillet (radius too large) on release also rolls back
    w.on_fillet_preview(999.0, mids)
    assert w.doc.errors  # the too-large fillet failed
    w.on_fillet_commit(999.0, mids)
    assert w._fillet_live_idx is None
    assert not w.doc.errors
    assert w.doc.part.volume == pytest.approx(vol_before_cancel)


def test_drag_fillet_edit_existing(qapp, monkeypatch):
    """Dragging an existing fillet's rounded edge edits its radius in place."""
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.set_tool("rect")
    click(w, (0, 0, 0))
    click(w, (40, 30, 0))
    w.finish_sketch()
    monkeypatch.setattr(appmod.ExtrudeDialog, "exec", lambda self: True)
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    w.extrude()
    assert not w.doc.errors

    edge = w.doc.mesh.edges[0]
    mids = [tuple(edge.mid)]
    vol0 = w.doc.part.volume

    # create a fillet via drag (new)
    w.on_fillet_preview(2.0, mids)
    w.on_fillet_commit(2.0, mids)
    fillet = w.doc.features[-1]
    assert fillet.kind == "fillet"
    assert fillet.value == pytest.approx(2.0)
    assert not w.doc.errors
    assert fillet.result_edges  # the rounded-edge mids are recorded
    vol_fillet = w.doc.part.volume
    assert vol_fillet < vol0

    # select the rounded edge's mid (at radius 2.0) and drag it -> edits the same fillet
    rmid = [tuple(fillet.result_edges[0])]
    w.on_fillet_preview(3.0, rmid)
    assert w._fillet_is_mod  # recognized as an edit, not a new fillet
    assert w.doc.features[-1] is fillet  # same feature, edited in place
    assert fillet.value == pytest.approx(3.0)
    w.on_fillet_commit(3.0, rmid)
    assert w._fillet_is_mod is False
    assert fillet.value == pytest.approx(3.0)
    assert not w.doc.errors
    assert w.doc.part.volume != vol_fillet  # the radius change moved the shape
    # refresh the rounded-edge mid (it moved with the radius)
    rmid = [tuple(fillet.result_edges[0])]

    # failing radius on release reverts to the pre-drag value
    w.on_fillet_preview(999.0, rmid)
    assert w.doc.errors  # the too-large fillet failed
    w.on_fillet_commit(999.0, rmid)
    assert not w.doc.errors
    assert fillet.value == pytest.approx(3.0)  # reverted to pre-drag radius
    rmid = [tuple(fillet.result_edges[0])]

    # dragging to ~0 removes the fillet
    vol_before_remove = w.doc.part.volume
    w.on_fillet_preview(3.0, rmid)
    assert w._fillet_is_mod
    w.on_fillet_commit(0.0, rmid)
    assert fillet not in w.doc.features  # the fillet is gone
    assert not w.doc.errors
    assert w.doc.part is not None  # the extrude is still live
    assert w.doc.part.volume == pytest.approx(vol0)


def test_fillet_result_edges_stay_current_across_fillets(qapp, monkeypatch, tmp_path):
    """Both fillets' result_edges must match the current mesh after the second,
    including after save/reload — earlier fillets used to keep stale mids."""
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.set_tool("rect")
    click(w, (0, 0, 0))
    click(w, (40, 30, 0))
    w.finish_sketch()
    monkeypatch.setattr(appmod.ExtrudeDialog, "exec", lambda self: True)
    w.extrude()
    assert not w.doc.errors

    mids0 = [tuple(w.doc.mesh.edges[0].mid)]
    e1 = next(e for e in w.doc.mesh.edges
              if float(np.linalg.norm(np.array(e.mid) - np.array(mids0[0]))) > 1.0)
    mids1 = [tuple(e1.mid)]

    # first fillet
    w.on_fillet_preview(2.0, mids0)
    w.on_fillet_commit(2.0, mids0)
    f0 = w.doc.features[-1]
    assert f0.kind == "fillet" and not w.doc.errors

    # second fillet on a different edge (look it up by its original mid)
    w.on_fillet_preview(2.0, mids1)
    w.on_fillet_commit(2.0, mids1)
    f1 = w.doc.features[-1]
    assert f1.kind == "fillet" and f1 is not f0 and not w.doc.errors

    curve = [tuple(map(float, e.mid)) for e in w.doc.mesh.edges if len(e.polyline) > 2]
    assert len(curve) == 4

    def covers(f):
        return all(any(all(abs(a - b) < 1e-6 for a, b in zip(m, c)) for c in curve)
                   for m in f.result_edges)

    assert len(f0.result_edges) == 2 and covers(f0)
    assert len(f1.result_edges) == 2 and covers(f1)
    assert not (set(map(tuple, f0.result_edges)) & set(map(tuple, f1.result_edges)))

    # reload: the stored sets must still match the rebuilt mesh
    path = str(tmp_path / "proj.rcad")
    w.path = path
    w.file_save()
    w2 = MainWindow()
    w2._load(appmod.Document.load(path), path)
    assert not w2.doc.errors
    curve2 = [tuple(map(float, e.mid)) for e in w2.doc.mesh.edges if len(e.polyline) > 2]
    assert len(curve2) == 4
    for f in (f for f in w2.doc.features if f.kind == "fillet"):
        assert len(f.result_edges) == 2
        assert all(any(all(abs(a - b) < 1e-6 for a, b in zip(m, c)) for c in curve2)
                   for m in f.result_edges)


def test_layer_checkbox_toggles(qapp, monkeypatch):
    w = MainWindow()
    w.view.resize(1000, 700)
    w.new_sketch()
    w.set_tool("rect")
    click(w, (0, 0, 0))
    click(w, (20, 10, 0))
    w.finish_sketch()
    monkeypatch.setattr(appmod.ExtrudeDialog, "exec", lambda self: True)
    w.extrude()
    vol = w.doc.part.volume
    assert w.view.overlay_sketches == [w.doc.features[0].sketch]
    w.features.item(0).setCheckState(Qt.Unchecked)
    assert not w.doc.features[0].visible and w.view.overlay_sketches == []
    assert w.doc.part.volume == pytest.approx(vol)
    w.features.item(1).setCheckState(Qt.Unchecked)
    assert w.doc.part is None
    w.features.item(1).setCheckState(Qt.Checked)
    assert w.doc.part.volume == pytest.approx(vol)


def test_symmetry_preserves_shared_vertices(qapp):
    w = MainWindow()
    w.new_sketch()
    sk = w.sketch
    axis_a = sk.add_point(3, -2)
    axis_b = sk.add_point(3, 3)
    p0 = sk.add_point(0, 1)
    p1 = sk.add_point(1, 1)
    p2 = sk.add_point(1, 2)
    axis = sk.add_line(axis_a, axis_b)
    first = sk.add_line(p0, p1)
    second = sk.add_line(p1, p2)

    w.view.sketch_sel = {("l", axis)}
    w.add_constraint("symmetry")
    w.view.sketch_sel = {("l", first), ("l", second)}
    w.add_constraint("symmetry")

    mirrored_first = sk.lines[-2]
    mirrored_second = sk.lines[-1]
    assert set(mirrored_first) & set(mirrored_second)
    mirrored_points = {point for line in (mirrored_first, mirrored_second) for point in line}
    assert sorted((sk.points[p][0], sk.points[p][1]) for p in mirrored_points) == [
        pytest.approx((5, 1)),
        pytest.approx((5, 2)),
        pytest.approx((6, 1)),
    ]
