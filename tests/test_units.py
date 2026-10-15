import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QInputDialog, QMessageBox

from rcad import app as appmod
from rcad.kernel import Document, Feature
from rcad.sketch import Constraint, Sketch, standard_plane
from rcad.units import INCH, MM


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_unit_conversion_and_format():
    assert INCH.to_mm(2) == pytest.approx(50.8)
    assert INCH.from_mm(25.4) == pytest.approx(1)
    assert INCH.fmt(25.4) == "1 in"
    assert MM.fmt(12.5) == "12.5 mm"
    assert INCH.fmt(10) == "0.3937 in"


def test_labels_follow_units():
    f = Feature("extrude", sketch_ref=0, value=25.4)
    assert "25.4 mm" in f.label() and "1 in" in f.label(INCH)
    assert "1 in" in Feature("fillet", value=25.4, edges=[(0, 0, 0)]).label(INCH)
    assert "= 2 in" in Constraint("distance", [0, 1], 50.8).label(INCH)


def test_units_saved_and_old_files_default_to_mm(tmp_path):
    doc = Document([Feature("sketch", "S", sketch=Sketch(standard_plane("XY")))], units="in")
    path = str(tmp_path / "u.rcad")
    doc.save(path)
    assert Document.load(path).units == "in"

    data = json.load(open(path))
    del data["units"]
    json.dump(data, open(path, "w"))
    assert Document.load(path).units == "mm"

    data["units"] = "furlongs"
    json.dump(data, open(path, "w"))
    with pytest.raises(ValueError):
        Document.load(path)


def test_ui_inches(qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    w = appmod.MainWindow()
    w.view.resize(1000, 700)
    assert w.unit is MM and w.view.unit_mm == 1.0

    w.unit_combo.setCurrentText("in")
    assert w.doc.units == "in" and w.dirty
    assert w.view.unit_mm == pytest.approx(25.4)
    assert w.snap.suffix() == " in" and w.snap.value() == pytest.approx(0.05)

    # lengths typed in inches are stored as millimetres
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (2.0, True))
    assert w.ask_length("t", "l", 10.0) == (pytest.approx(50.8), True)

    # an untouched value keeps its exact millimetre value
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (round(10 / 25.4, 4), True))
    assert w.ask_length("t", "l", 10.0)[0] == 10.0

    d = appmod.ExtrudeDialog(w, 10.0, unit=INCH)
    assert d.distance_mm == 10.0
    d.dist.setValue(1.0)
    assert d.distance_mm == pytest.approx(25.4)

    # dimension a sketch line in inches
    w.plane_combo.setCurrentText("XY")
    w.new_sketch()
    w.sketch.add_rect(0, 0, 30, 20)
    w.view.sketch_sel = {("l", 0)}
    monkeypatch.setattr(QInputDialog, "getDouble", lambda *a, **k: (2.0, True))
    w.add_constraint("distance")
    c = w.sketch.constraints[-1]
    assert c.value == pytest.approx(50.8)
    assert "2 in" in w.constraints.item(w.constraints.count() - 1).text()
    w.finish_sketch()

    path = str(tmp_path / "in.rcad")
    w.path = path
    w.file_save()
    w2 = appmod.MainWindow()
    w2._load(Document.load(path), path)
    assert w2.unit_combo.currentText() == "in" and w2.view.unit_mm == pytest.approx(25.4)
