"""Project files: new, open, save (crash-safe), export, autosave and recovery."""
from __future__ import annotations

import time

from PySide6.QtWidgets import QFileDialog, QMessageBox

from .kernel import Document
from .recovery import find_recoveries


FILE_FILTER = "rCad project (*.rcad)"


class FilesMixin:
    """Project files: new, open, save (crash-safe), export, autosave and recovery. Mixed into MainWindow."""

    # ---- files
    AUTOSAVE_MS = 60_000  # how often unsaved changes are copied for recovery after a crash

    def autosave(self) -> bool:
        """Copy unsaved changes to the recovery folder (never to the project file itself)."""
        if not self.dirty:
            return False
        try:
            self.autosaver.write(self.doc, self.path)
            return True
        except Exception as e:  # a full disk or a document that won't serialise: keep working
            self.statusBar().showMessage(f"Autosave failed: {e}", 8000)
            return False

    def offer_recovery(self) -> str | None:
        """At start-up: offer to restore work left unsaved by an rCad that closed without saving
        (the newest copy; any others are offered next time). Returns what was done."""
        recs = find_recoveries()
        if not recs:
            return None
        rec = recs[0]
        answer = self._ask_recovery(rec)
        if answer == "restore":
            try:
                doc = rec.document()
            except (ValueError, KeyError, TypeError) as e:
                QMessageBox.warning(self, "Recovery failed", f"The recovery copy can't be read: {e}")
                rec.discard()
                return "failed"
            self._load(doc, rec.project)
            self.autosaver.adopt(rec)
            self.mark_dirty(True)
            self.statusBar().showMessage(f"Restored unsaved work on {rec.name}: save to keep it.", 10000)
        elif answer == "discard":
            rec.discard()
        return answer

    def _ask_recovery(self, rec) -> str:
        """'restore', 'discard' or 'later'."""
        box = QMessageBox(self)
        box.setWindowTitle("Recover unsaved work")
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(rec.saved_at))
        box.setText(f"rCad closed without saving changes to {rec.name}.\n"
                    f"A copy from {when} was kept. Restore it?")
        restore = box.addButton("Restore", QMessageBox.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.DestructiveRole)
        box.addButton("Not now", QMessageBox.RejectRole)
        box.setDefaultButton(restore)
        box.exec()
        clicked = box.clickedButton()
        return "restore" if clicked is restore else "discard" if clicked is discard else "later"

    def _confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved changes", "Discard unsaved changes?")
        return r == QMessageBox.Yes

    def _load(self, doc, path):
        self._cancel_session()
        self.autosaver.clear()  # the document it copied is being replaced
        self.doc, self.path = doc, path
        self.sketch_index = -1
        self._undo.clear()
        self._redo.clear()
        self._clear_sketch_history()
        self.set_mode("model")
        self.mark_dirty(False)
        self._apply_units()
        self.view.frame_all()
        self._refresh_fillet_result_edges()

    def file_new(self):
        if self._confirm_discard():
            self._load(Document(), None)

    def file_open(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", FILE_FILTER)
        if path:
            try:
                self._load(Document.load(path), path)
            except (OSError, ValueError, KeyError) as e:
                QMessageBox.warning(self, "Open failed", str(e))

    def file_save(self):
        if self.path is None:
            return self.file_save_as()
        try:
            self.doc.save(self.path)
            self.mark_dirty(False)
            self.autosaver.clear()
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))

    def file_save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self.path or "model.rcad", FILE_FILTER)
        if path:
            self.path = path if path.endswith(".rcad") else path + ".rcad"
            self.file_save()

    def file_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export", "model.step", "STEP (*.step *.stp);;STL (*.stl)")
        if path:
            try:
                self.doc.export(path)
            except Exception as e:  # kernel or I/O error
                QMessageBox.warning(self, "Export failed", str(e))

    def closeEvent(self, e):
        if not self._confirm_discard():
            e.ignore()
            return
        self.autosave_timer.stop()
        self.autosaver.clear()  # closed on purpose: nothing to recover
        e.accept()
