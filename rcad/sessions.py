"""The tool settings bar: a tool's values are edited there with the model previewing each change; ✓
keeps it, ✕ puts everything back."""
from __future__ import annotations

from .kernel import Feature
from .tool_bar import Number


class SessionsMixin:
    """The tool settings bar: a tool's values are edited there with the model previewing each change; ✓ keeps it, ✕ puts everything back. Mixed into MainWindow."""

    def _value_session(self, f: Feature, title: str, label: str, new: bool, lo: float, hi: float,
                       nonzero: bool = False):
        """One length on a feature (push/pull distance, fillet radius, chamfer size)."""
        items = [Number("value", label, f.value, lo=lo, hi=hi)]

        def apply(f, v):
            f.value = v["value"]

        def validate(v):
            return f"{label} can't be 0" if nonzero and abs(v["value"]) < 1e-9 else None

        def done(f):
            if f.kind == "fillet":
                f.result_edges = self._fillet_result_edges(f)

        self._begin_session(title, items, f, new, apply, done, validate)

    def _begin_session(self, title, items, f: Feature, new: bool, apply, done=None, validate=None):
        """Open the settings bar on a feature: new ones are added to the history right away and
        every change rebuilds the model; ✓ keeps it, ✕ puts the history back. One undo step."""
        self._settle_session()
        self._push_undo()
        snapshot = {k: getattr(f, k) for k in ("value", "cut", "auto", "axis", "offset", "angle", "rot_axis",
                                               "plane", "copy")}
        if new:
            self.doc.features.append(f)

        def index():
            return next((i for i, x in enumerate(self.doc.features) if x is f), None)

        def preview(values):
            if validate is not None:
                err = validate(values)
                if err:
                    return err
            apply(f, values)
            f.version += 1
            self.rebuild()
            i = index()
            return "The feature is gone" if i is None else self.doc.errors.get(i)

        def check():
            i = index()
            return "The feature is gone" if i is None else self.doc.errors.get(i)

        def restore():
            if new:
                self.doc.features[:] = [x for x in self.doc.features if x is not f]
            else:
                for k, v in snapshot.items():
                    setattr(f, k, v)
                f.version += 1
            if self._undo:
                self._undo.pop()  # the snapshot taken when the tool opened
            self.rebuild()

        self._open_session(title, items, preview, check, restore, (lambda: done(f)) if done else None)

    def _open_session(self, title, items, preview, check, restore, done=None):
        """preview(values) -> error or None; check() -> error or None; restore() undoes everything."""
        self._session = {"preview": preview, "check": check, "restore": restore, "done": done, "error": None}
        self.tool_bar.begin(title, items, self._session_changed, self._finish_session, self._cancel_session)
        self._session_changed(self.tool_bar.values())

    def _session_changed(self, values: dict | None):
        s = self._session
        if s is None or values is None:
            return
        s["error"] = s["preview"](values)
        self.tool_bar.set_error(s["error"])

    def _finish_session(self) -> bool:
        """✓ / Enter: keep the result as previewed. Returns False (bar stays open) if it fails."""
        s = self._session
        if s is None:
            return True
        err = s["error"] or s["check"]()
        if err:
            self.tool_bar.set_error(err)
            return False
        self._session = None
        self.tool_bar.end()
        if s["done"]:
            s["done"]()
        self.view.setFocus()
        return True

    def _cancel_session(self):
        """✕ / Esc: put everything back as it was before the tool opened."""
        s, self._session = self._session, None
        self.tool_bar.end()
        if s is None:
            return
        s["restore"]()
        self.view.setFocus()

    def _settle_session(self):
        """Something else is starting: keep the open tool's result, or drop it if it fails."""
        if self._session is not None and not self._finish_session():
            self._cancel_session()
