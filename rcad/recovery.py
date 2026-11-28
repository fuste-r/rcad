"""Autosaved recovery copies of unsaved work, and finding them again after a crash.

Recovery copies live in rCad's data folder, never beside or in place of the project. Each one
names its project (or none, for an untitled one), when it was written, and the process that
wrote it, so a copy belonging to another rCad that is still running is never offered."""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QStandardPaths

from .kernel import Document, atomic_write

SUFFIX = ".rcad-recovery"


def recovery_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.GenericDataLocation) or os.path.expanduser("~")
    d = Path(base) / "rcad" / "recovery"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Recovery:
    file: Path
    project: str | None  # the project file it belongs to, or None if it was never saved
    saved_at: float
    data: dict  # the document

    @property
    def name(self) -> str:
        return os.path.basename(self.project) if self.project else "an untitled project"

    def document(self) -> Document:
        return Document.from_dict(self.data)

    def discard(self):
        try:
            self.file.unlink()
        except OSError:
            pass


class Autosaver:
    """Writes the recovery copy for one window's document and removes it when it is not needed."""

    def __init__(self):
        self.file: Path | None = None
        self._untitled = uuid.uuid4().hex[:12]

    def _file_for(self, project: str | None) -> Path:
        if project:
            key = hashlib.sha1(os.path.abspath(project).encode()).hexdigest()[:12]
            stem = Path(project).stem
        else:
            key, stem = self._untitled, "untitled"
        return recovery_dir() / f"{stem}-{key}{SUFFIX}"

    def write(self, doc: Document, project: str | None) -> Path:
        target = self._file_for(project)
        if self.file is not None and self.file != target:
            self.clear()  # the project was renamed or replaced: drop the old copy
        payload = {"rcad_recovery": 1, "project": os.path.abspath(project) if project else None,
                   "saved_at": time.time(), "pid": os.getpid(), "document": doc.to_dict()}
        atomic_write(str(target), json.dumps(payload))
        self.file = target
        return target

    def adopt(self, rec: Recovery):
        """Carry on with a restored recovery copy (it is rewritten under this process at the next autosave)."""
        self.file = rec.file

    def clear(self):
        if self.file is not None:
            try:
                self.file.unlink()
            except OSError:
                pass
        self.file = None


def _alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def find_recoveries() -> list[Recovery]:
    """Recovery copies left by an rCad that is no longer running, newest first. Copies older than
    their project file (it was saved again since) and unreadable ones are removed."""
    out = []
    for f in recovery_dir().glob(f"*{SUFFIX}"):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
            project, saved_at, pid = payload.get("project"), float(payload["saved_at"]), int(payload["pid"])
            data = payload["document"]
        except (OSError, ValueError, KeyError, TypeError):
            f.unlink(missing_ok=True)
            continue
        if _alive(pid):
            continue  # another rCad (or this one) is still working on it
        if project and os.path.exists(project) and os.path.getmtime(project) >= saved_at:
            f.unlink(missing_ok=True)  # the project was saved after this copy: nothing to recover
            continue
        out.append(Recovery(f, project, saved_at, data))
    return sorted(out, key=lambda r: r.saved_at, reverse=True)
