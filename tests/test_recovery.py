import json
import os
import stat
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from rcad import kernel, recovery
from rcad.app import MainWindow
from rcad.kernel import Document, Feature
from rcad.sketch import Sketch, standard_plane


def _doc(width=40.0):
    sk = Sketch(standard_plane("XY"))
    sk.add_rect(0, 0, width, 20)
    return Document([Feature("sketch", "S", sketch=sk), Feature("extrude", sketch_ref=0, value=5)])


@pytest.fixture(autouse=True)
def clean_recovery():
    """Each test starts (and ends) with an empty recovery folder."""
    for f in recovery.recovery_dir().glob("*"):
        f.unlink()
    yield
    for f in recovery.recovery_dir().glob("*"):
        f.unlink()


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _orphan(file):
    """Make a recovery copy look like it was left by an rCad that has since crashed."""
    payload = json.loads(file.read_text())
    payload["pid"] = _dead_pid()
    file.write_text(json.dumps(payload))


# -- saving

def test_save_keeps_the_previous_version_as_bak(tmp_path):
    path = str(tmp_path / "p.rcad")
    _doc(40).save(path)
    first = open(path).read()
    _doc(55).save(path)
    assert open(path + ".bak").read() == first
    assert Document.load(path).features[0].sketch.points[1][0] == pytest.approx(55)
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []


def test_a_failed_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    path = str(tmp_path / "p.rcad")
    _doc(40).save(path)
    before = open(path).read()

    def crash(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(kernel.os, "replace", crash)
    with pytest.raises(OSError, match="disk full"):
        _doc(55).save(path)
    assert open(path).read() == before
    assert [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")] == []  # no litter


def test_a_document_that_cannot_be_written_touches_nothing(tmp_path, monkeypatch):
    path = str(tmp_path / "p.rcad")
    _doc(40).save(path)
    before = open(path).read()
    doc = _doc(55)
    monkeypatch.setattr(doc, "to_dict", lambda: {"bad": object()})
    with pytest.raises(TypeError):
        doc.save(path)
    assert open(path).read() == before and not os.path.exists(path + ".bak")


def test_save_keeps_file_permissions(tmp_path):
    path = str(tmp_path / "p.rcad")
    _doc().save(path)
    os.chmod(path, 0o640)
    _doc(50).save(path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o640
    fresh = str(tmp_path / "new.rcad")
    _doc().save(fresh)
    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(os.stat(fresh).st_mode) == 0o666 & ~umask


# -- recovery copies

def test_a_running_rcads_copy_is_not_offered():
    saver = recovery.Autosaver()
    saver.write(_doc(), None)
    assert recovery.find_recoveries() == []  # this process is alive
    _orphan(saver.file)
    found = recovery.find_recoveries()
    assert len(found) == 1 and found[0].project is None and found[0].name == "an untitled project"
    assert found[0].document().features[0].sketch.points[1][0] == pytest.approx(40)


def test_copies_older_than_their_project_and_broken_ones_are_dropped(tmp_path):
    path = str(tmp_path / "p.rcad")
    saver = recovery.Autosaver()
    saver.write(_doc(), path)
    _orphan(saver.file)
    _doc().save(path)  # saved after the copy was taken
    payload = json.loads(saver.file.read_text())
    os.utime(path, (payload["saved_at"] + 5, payload["saved_at"] + 5))
    broken = recovery.recovery_dir() / f"junk{recovery.SUFFIX}"
    broken.write_text("{not json")
    assert recovery.find_recoveries() == []
    assert not saver.file.exists() and not broken.exists()


def test_renaming_the_project_moves_the_copy(tmp_path):
    saver = recovery.Autosaver()
    first = saver.write(_doc(), None)
    second = saver.write(_doc(), str(tmp_path / "named.rcad"))
    assert not first.exists() and second.exists() and second.name.startswith("named-")


# -- the window

def _window(monkeypatch):
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    return MainWindow()


def test_autosave_only_with_unsaved_changes_and_cleared_by_save(qapp, monkeypatch, tmp_path):
    w = _window(monkeypatch)
    assert w.autosave_timer.isActive() and w.autosave_timer.interval() == 60_000
    assert w.autosave() is False and w.autosaver.file is None  # nothing changed yet
    w.doc = _doc()
    w.mark_dirty()
    assert w.autosave() is True and w.autosaver.file.exists()
    copy = w.autosaver.file
    w.path = str(tmp_path / "p.rcad")
    w.file_save()
    assert not copy.exists() and w.autosaver.file is None and os.path.exists(w.path)
    assert w.autosave() is False


def test_closing_on_purpose_leaves_nothing_to_recover(qapp, monkeypatch):
    w = _window(monkeypatch)
    w.doc = _doc()
    w.mark_dirty()
    w.autosave()
    copy = w.autosaver.file
    w.closeEvent(QCloseEvent())  # "discard unsaved changes?" -> yes
    assert not copy.exists() and not w.autosave_timer.isActive()


def test_restore_after_a_crash(qapp, monkeypatch, tmp_path):
    path = str(tmp_path / "p.rcad")
    _doc(40).save(path)
    crashed = _window(monkeypatch)
    crashed._load(Document.load(path), path)
    crashed.doc.features[0].sketch.points[1][0] = 70  # unsaved work
    crashed.mark_dirty()
    crashed.autosave()
    _orphan(crashed.autosaver.file)  # ... and then rCad died
    crashed.autosaver.file = None

    w = _window(monkeypatch)
    monkeypatch.setattr(w, "_ask_recovery", lambda rec: "restore")
    assert w.offer_recovery() == "restore"
    assert w.path == os.path.abspath(path) and w.dirty
    assert w.doc.features[0].sketch.points[1][0] == pytest.approx(70)
    assert w.autosaver.file is not None and w.autosaver.file.exists()  # kept until it is saved
    w.file_save()
    assert recovery.find_recoveries() == [] and Document.load(path).features[0].sketch.points[1][0] == 70


def test_discard_or_later(qapp, monkeypatch):
    saver = recovery.Autosaver()
    saver.write(_doc(), None)
    _orphan(saver.file)
    w = _window(monkeypatch)
    monkeypatch.setattr(w, "_ask_recovery", lambda rec: "later")
    assert w.offer_recovery() == "later" and saver.file.exists()
    monkeypatch.setattr(w, "_ask_recovery", lambda rec: "discard")
    assert w.offer_recovery() == "discard" and not saver.file.exists()
    assert w.offer_recovery() is None  # nothing left
