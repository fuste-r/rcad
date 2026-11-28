"""Modelling: extrude, revolve, sweep and loft from sketches (and by dragging regions), push/pull,
fillet and chamfer, moving and mirroring bodies."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from . import datum, refs
from .dialogs import LoftDialog, SweepDialog
from .kernel import Feature
from .tool_bar import Choice, Number


class ModelToolsMixin:
    """Modelling: extrude, revolve, sweep and loft from sketches (and by dragging regions), push/pull, fillet and chamfer, moving and mirroring bodies. Mixed into MainWindow."""

    def _sketch_on_plane(self, name: str):
        """Start a sketch on a reference plane (Top, Front or Right)."""
        self.view.sel_plane = None
        self.plane_combo.setCurrentText(datum.PLANES[name][0])
        self.new_sketch()

    def _on_planes_toggled(self, shown: bool):
        self.statusBar().showMessage("Reference planes shown (P hides them)" if shown
                                     else "Reference planes hidden (P shows them)", 4000)
        self._update_regions()

    def _sketch_on_selected_face(self):
        self.plane_combo.setCurrentText("Selected face")
        self.new_sketch()

    def _region_drag_target(self, x, y, on_arrow: bool):
        """What a left drag from (x, y) extrudes: the picked regions when it starts on one of them or
        on their arrow, else the region under the pointer (added to the picked ones of its sketch)."""
        feats = self.doc.features
        if self.mode != "model" or self._session is not None:
            return None
        if on_arrow:
            i = self.region_sketch
            if not (0 <= i < len(feats) and feats[i].kind == "sketch"):
                return None
            points = list(feats[i].sketch.selection)
        else:
            if self.view.sel_faces and self.view.pick_face(x, y) in self.view.sel_faces:
                return None  # a picked face drags as push/pull
            hit = self._pick_region(x, y)
            if hit is None:
                return None
            i, r, uv = hit
            sk = feats[i].sketch
            if self._picking_regions_of(i) or (i == self.region_sketch and r in sk.selected_regions()):
                points = list(sk.selection) + ([] if r in sk.selected_regions() else [tuple(uv)])
            else:
                points = [tuple(uv)]
        sk = feats[i].sketch
        by_region = {}
        for p in points:
            r = sk.region_at(*p)
            if r is not None:
                by_region.setdefault(r, [float(p[0]), float(p[1])])
        if not by_region:
            return None
        profiles = list(by_region.values())
        pl = sk.plane
        segments, corners = [], []
        for face in sk.profile_faces(profiles):
            for e in face.edges():
                n = 2 if e.geom_type.name == "LINE" else 48
                pts = [pl.to3d(q.X, q.Y) for q in (e.position_at(t) for t in np.linspace(0, 1, n))]
                segments += list(zip(pts[:-1], pts[1:]))
            corners += [pl.to3d(q.X, q.Y) for q in face.vertices()]
        tris = np.concatenate([sk.region_triangles(r) for r in by_region]).reshape(-1, 3)
        return {"sketch": i, "profiles": profiles, "center": tris.mean(axis=0),
                "normal": np.array(pl.normal, float), "segments": segments,
                "corners": corners}

    def on_region_extrude(self, dist: float, target: dict):
        """A sketch region was dragged: extrude the regions it carried by the dragged distance."""
        self._value_feature = None
        if abs(dist) < 0.01:
            self.view.dismiss_value_box()
            return
        self._settle_session()
        i = target["sketch"]
        feats = self.doc.features
        if not (0 <= i < len(feats) and feats[i].kind == "sketch"):
            self.view.dismiss_value_box()
            return
        feature = Feature("extrude", sketch_ref=i, value=float(dist), auto=True,
                          profiles=[list(p) for p in target["profiles"]])
        if self.commit(feature):
            self._value_feature = feature
            self._last_extrude = float(dist)
            feats[i].sketch.selection = []  # the regions are inside the new body now
            self._update_regions()
        else:
            self.view.dismiss_value_box()

    def on_wheel_adjust(self, x, y, delta, mods):
        edge = self.view.hover_edge
        if edge is None and self.view.sel_edges:
            edge = sorted(self.view.sel_edges)[0]
        feat_i = self._feature_for_edge(edge)
        if feat_i is None:
            row = self.features.currentRow()
            if 0 <= row < len(self.doc.features) and self.doc.features[row].kind in ("fillet", "chamfer"):
                feat_i = row
            else:
                candidates = [i for i, f in enumerate(self.doc.features) if f.kind in ("fillet", "chamfer")]
                feat_i = candidates[-1] if candidates else None
        if feat_i is None:
            return
        self._push_undo()
        f = self.doc.features[feat_i]
        step = 0.5 if not (mods & Qt.ShiftModifier) else 0.1
        f.value = max(1e-4, f.value + (delta / 120.0) * step)
        f.version += 1
        self.rebuild()
        self.statusBar().showMessage(f"{f.kind.title()} {self.unit.fmt(f.value)}", 2000)

    def _sketch_for_extrude(self) -> int | None:
        feats = self.doc.features
        if 0 <= self.region_sketch < len(feats) and feats[self.region_sketch].kind == "sketch" \
                and feats[self.region_sketch].sketch.selected_regions():
            return self.region_sketch
        row = self.features.currentRow()
        if 0 <= row < len(feats) and feats[row].kind == "sketch":
            return row
        idx = [i for i, f in enumerate(feats) if f.kind == "sketch"]
        return idx[-1] if idx else None

    def extrude(self):
        i = self._sketch_for_extrude()
        if i is None:
            QMessageBox.information(self, "⇈", "Create a sketch with a closed profile first.")
            return
        sk = self.doc.features[i].sketch
        if len(sk.profile_faces()) > 1 and not sk.selected_regions():
            self.enter_sketch(i)
            self.statusBar().showMessage("Click the regions to extrude (Ctrl+click for several), then press Extrude. "
                                         "With nothing highlighted, every region is extruded.", 15000)
            return
        self._extrude_sketch(i)

    def extrude_from_sketch(self):
        i = self.sketch_index
        self.finish_sketch()
        self._extrude_sketch(i)

    def _extrude_sketch(self, i: int):
        sk = self.doc.features[i].sketch
        if not sk.regions():
            QMessageBox.information(self, "⇈", "The sketch has no closed profile.")
            return
        chosen = sorted(sk.selected_regions())
        profiles = [list(next(p for p in sk.selection if sk.region_at(*p) == r)) for r in chosen] or None
        title = f"Extrude {len(profiles)} region(s)" if profiles else "Extrude"
        show_mode = self.doc.part is not None
        self._extrude_session(Feature("extrude", sketch_ref=i, value=self._last_extrude, auto=True,
                                      profiles=profiles), title, new=True, show_mode=show_mode)

    # ---- tool settings bar sessions
    EXTRUDE_MODES = [("auto", "Auto"), ("add", "Add"), ("cut", "Cut")]
    REVOLVE_AXES = [("u", "Sketch U axis"), ("v", "Sketch V axis")]
    ADD_CUT = [("add", "Add"), ("cut", "Cut")]

    def _extrude_session(self, f: Feature, title: str, new: bool, show_mode: bool = True):
        items = [Number("distance", "Distance", f.value, lo=-1e5, hi=1e5)]
        if show_mode:
            items.append(Choice("mode", "Operation", self.EXTRUDE_MODES,
                                "auto" if f.auto else "cut" if f.cut else "add"))

        def apply(f, v):
            f.value = v["distance"]
            mode = v.get("mode", "auto")
            f.auto, f.cut = mode == "auto", mode == "cut"

        def done(f):
            self._last_extrude = f.value

        self._begin_session(title, items, f, new, apply, done)

    def _revolve_session(self, f: Feature, title: str, new: bool):
        items = [Number("angle", "Angle", f.value, "angle", lo=1.0, hi=360.0),
                 Choice("axis", "Axis", self.REVOLVE_AXES, f.axis),
                 Choice("mode", "Operation", self.ADD_CUT, "cut" if f.cut else "add")]

        def apply(f, v):
            f.value, f.axis, f.cut = v["angle"], v["axis"], v["mode"] == "cut"

        self._begin_session(title, items, f, new, apply)

    MOVE_COPY = [("move", "Move"), ("copy", "Copy")]
    ROT_AXES = [("x", "X axis"), ("y", "Y axis"), ("z", "Z axis")]
    MIRROR_PLANES = [("YZ", "YZ plane"), ("XZ", "XZ plane"), ("XY", "XY plane")]

    def _body_feature(self, kind: str) -> Feature | None:
        """A new move/mirror feature on the selected bodies, or None (with a hint) if none are."""
        bodies = self._selected_bodies()
        if not bodies:
            QMessageBox.information(self, kind.title(), "Select a body first: double-click it (Ctrl adds more), "
                                                        "or Ctrl+A for all of them.")
            return None
        solids = self.doc.part.solids()
        sigs = [refs.body_signature(solids[b]) for b in bodies if b < len(solids)]
        f = Feature(kind, ref_sigs=sigs, copy=kind == "mirror")
        self._clear_model_selection()  # the faces renumber as the bodies move
        self._update_regions()
        return f

    def move_bodies(self):
        f = self._body_feature("move")
        if f is not None:
            self._move_session(f, "Move", new=True)

    def mirror_bodies(self):
        f = self._body_feature("mirror")
        if f is not None:
            self._mirror_session(f, "Mirror", new=True)

    def _move_session(self, f: Feature, title: str, new: bool):
        dx, dy, dz = f.offset
        items = [Number("x", "X", dx, lo=-1e5, hi=1e5), Number("y", "Y", dy, lo=-1e5, hi=1e5),
                 Number("z", "Z", dz, lo=-1e5, hi=1e5),
                 Number("angle", "Rotate", f.angle, "angle", lo=-360.0, hi=360.0),
                 Choice("rot_axis", "About", self.ROT_AXES, f.rot_axis),
                 Choice("copy", "Keep", self.MOVE_COPY, "copy" if f.copy else "move")]

        def apply(f, v):
            f.offset = (v["x"], v["y"], v["z"])
            f.angle, f.rot_axis, f.copy = v["angle"], v["rot_axis"], v["copy"] == "copy"

        self._begin_session(title, items, f, new, apply)

    def _mirror_session(self, f: Feature, title: str, new: bool):
        items = [Choice("plane", "Plane", self.MIRROR_PLANES, f.plane),
                 Number("offset", "At", f.value, lo=-1e5, hi=1e5),
                 Choice("copy", "Keep", self.MOVE_COPY, "copy" if f.copy else "move")]

        def apply(f, v):
            f.plane, f.value, f.copy = v["plane"], v["offset"], v["copy"] == "copy"

        self._begin_session(title, items, f, new, apply)

    def revolve(self):
        i = self._sketch_for_extrude()
        if i is None:
            QMessageBox.information(self, "Revolve", "Create a sketch with a closed profile first.")
            return
        sketch = self.doc.features[i].sketch
        if len(sketch.profile_faces()) > 1 and not sketch.selected_regions():
            self.enter_sketch(i)
            self.statusBar().showMessage("Click the regions to revolve (Ctrl+click for several), then finish the sketch.",
                                         12000)
            return
        chosen = sorted(sketch.selected_regions())
        profiles = [list(next(p for p in sketch.selection if sketch.region_at(*p) == r)) for r in chosen] or None
        self._revolve_session(Feature("revolve", sketch_ref=i, value=360.0, axis="v", profiles=profiles),
                              "Revolve", new=True)

    def sweep(self):
        profile_ref = self._sketch_for_extrude()
        if profile_ref is None:
            QMessageBox.information(self, "Sweep", "Create a sketch with one closed profile first.")
            return
        profile = self.doc.features[profile_ref].sketch
        picked = [list(p) for p in profile.selection] or None
        if len(profile.profile_faces(picked)) != 1:
            QMessageBox.information(self, "Sweep", "Sweep currently requires exactly one closed profile region.")
            return
        paths = [(i, f"{i}: {f.name}") for i, f in enumerate(self.doc.features)
                 if f.kind == "sketch" and i != profile_ref and f.sketch.lines]
        if not paths:
            QMessageBox.information(self, "Sweep", "Create another sketch with connected path lines first.")
            return
        dialog = SweepDialog(self, paths)
        if dialog.exec():
            self.commit(Feature("sweep", sketch_ref=profile_ref, path_ref=dialog.path_ref, cut=dialog.cut,
                                profiles=picked))

    def loft(self):
        sections = [(i, f"{i}: {f.name}") for i, f in enumerate(self.doc.features)
                    if f.kind == "sketch" and len(f.sketch.profile_faces()) == 1]
        if len(sections) < 2:
            QMessageBox.information(self, "Loft", "Create at least two sketches with one closed profile each.")
            return
        dialog = LoftDialog(self, sections)
        if dialog.exec():
            refs = dialog.section_refs
            if len(refs) < 2:
                QMessageBox.information(self, "Loft", "Select at least two section sketches.")
                return
            self.commit(Feature("loft", section_refs=refs, cut=dialog.cut))

    def pushpull(self):
        if len(self.view.sel_faces) != 1:
            QMessageBox.information(self, "↕️", "Select exactly one planar face.")
            return
        info = self.doc.mesh.faces[next(iter(self.view.sel_faces))]
        if info.normal is None:
            QMessageBox.information(self, "↕️", "Only planar faces are supported.")
            return
        f = Feature("pushpull", value=5.0, face=(tuple(map(float, info.center)), tuple(map(float, info.normal))))
        self._value_session(f, "Push/pull", "Distance (+ out)", new=True, lo=-1e5, hi=1e5, nonzero=True)

    def on_extrude_commit(self, dist: float, center: tuple, normal: tuple):
        """Drag-to-extrude: commit a pushpull feature from the drag distance."""
        self._value_feature = None
        if abs(dist) < 0.01:
            self.view.dismiss_value_box()
            return
        feature = Feature("pushpull", value=dist, face=(center, normal))
        if self.commit(feature):
            self._value_feature = feature
        else:
            self.view.dismiss_value_box()

    def on_value_edit(self, kind: str, value: float):
        """A size typed into the value box left on the model after a drag."""
        f = self._value_feature
        expected = {"extrude": ("pushpull", "extrude"), "fillet": ("fillet",), "chamfer": ("chamfer",)}.get(kind, ())
        if f is None or f.kind not in expected or not any(x is f for x in self.doc.features):
            self._value_feature = None  # gone (undo, delete, reload): nothing to edit
            self.view.dismiss_value_box()
            return
        scrub, self._scrub_prev = self._scrub_prev, None
        if scrub is not None and scrub[0] is not f:
            scrub = None
        prev = scrub[1] if scrub is not None else f.value
        if (kind in ("fillet", "chamfer") and value <= 1e-4) or abs(value) < 0.01:
            self.statusBar().showMessage("Value too small", 5000)
            if scrub is not None:  # put back the size from before the scrub
                f.value = prev
                f.version += 1
                self._undo.pop()
                self.rebuild()
            self.view.set_post_value(f.value)
            return
        if scrub is None:  # a scrub already saved the undo snapshot
            self._push_undo()
        f.value = float(value)
        f.version += 1
        self.rebuild()
        idx = next(i for i, x in enumerate(self.doc.features) if x is f)
        if idx in self.doc.errors:
            msg = self.doc.errors[idx]
            f.value = prev
            f.version += 1
            self._undo.pop()
            self.rebuild()
            self.view.set_post_value(prev)
            QMessageBox.warning(self, "Operation failed", msg)
        if f.kind == "fillet":
            f.result_edges = self._fillet_result_edges(f)

    def on_value_scrub(self, kind: str, value: float):
        """Live size while the value box left after a drag is scrubbed; on_value_edit finishes it."""
        f = self._value_feature
        expected = {"extrude": ("pushpull", "extrude"), "fillet": ("fillet",), "chamfer": ("chamfer",)}.get(kind, ())
        if f is None or f.kind not in expected or not any(x is f for x in self.doc.features):
            return
        if (kind in ("fillet", "chamfer") and value <= 1e-4) or abs(value) < 0.01:
            return
        if self._scrub_prev is None or self._scrub_prev[0] is not f:
            self._scrub_prev = (f, f.value)
            self._push_undo()
        f.value = float(value)
        f.version += 1
        self.rebuild()

    def on_fillet_cancel(self):
        """Esc during a fillet drag: put the history back as it was before the drag."""
        idx = self._fillet_live_idx
        if idx is None:
            return
        f = self.doc.features[idx]
        was_mod = self._fillet_is_mod
        if was_mod:
            f.value = float(self._fillet_mod_prev)
            f.kind = self._fillet_mod_prev_kind or "fillet"
            f.version += 1
        else:
            self.doc.features.pop(idx)
        if self._undo:
            self._undo.pop()  # the snapshot pushed when the live preview started
        self._fillet_live_idx = None
        self._fillet_preview_r = None
        self._fillet_is_mod = False
        self._fillet_mod_prev = None
        self._fillet_mod_prev_kind = None
        self._fillet_preview_kind = None
        self.rebuild()
        if was_mod:
            f.result_edges = self._fillet_result_edges(f)

    def _fillet_edge_dirs(self, mids: list) -> list:
        """Aligned unit directions of picked edges; curved edges have no direction."""
        mesh = self.doc.mesh
        if mesh is None:
            return []
        dirs = []
        for m in mids:
            m_arr = np.asarray(m, float)
            best_e, best_d = None, 1e-4  # edge mid within ~0.1mm is the picked edge
            for e in mesh.edges:
                d = float(np.linalg.norm(e.mid - m_arr))
                if d < best_d:
                    best_e, best_d = e, d
            if best_e is None:
                dirs.append(None)
                continue
            if len(best_e.polyline) > 2:
                dirs.append(None)
                continue
            p = np.asarray(best_e.polyline, float)
            v = p[-1] - p[0]
            n = float(np.linalg.norm(v))
            dirs.append(tuple((v / n).tolist()) if n > 1e-9 else None)
        return dirs

    def _fillet_edge_refs(self, mids: list) -> list:
        """Sampled path for each picked mesh edge, preserving selection order."""
        mesh = self.doc.mesh
        if mesh is None:
            return []
        refs = []
        for mid in mids:
            point = np.asarray(mid, float)
            best = min(mesh.edges, key=lambda edge: float(np.linalg.norm(edge.mid - point)), default=None)
            if best is None or float(np.linalg.norm(best.mid - point)) >= 1e-4:
                refs.append(None)
            else:
                refs.append([tuple(map(float, p)) for p in best.polyline])
        return refs

    @staticmethod
    def _point_path_distance(point: np.ndarray, path: list) -> float:
        points = np.asarray(path, float)
        if len(points) == 1:
            return float(np.linalg.norm(point - points[0]))
        starts, directions = points[:-1], points[1:] - points[:-1]
        lengths_sq = np.einsum("ij,ij->i", directions, directions)
        fractions = np.divide(np.einsum("ij,ij->i", point - starts, directions), lengths_sq,
                              out=np.zeros_like(lengths_sq), where=lengths_sq > 1e-18)
        closest = starts + np.clip(fractions, 0, 1)[:, None] * directions
        return float(np.min(np.linalg.norm(closest - point, axis=1)))

    def _fillet_result_edges(self, f: Feature) -> list:
        """Mids of the curve edges this fillet created in the current mesh.

        A fillet rounds the picked edge into arc geometry: the midpoints of
        those arcs sit a constant distance from the picked edge's original
        line (r(√2-1) for a 90° edge, growing toward r on acute angles).
        Matching each curve mid against every picked edge's line therefore
        separates one fillet's arcs from another's, so a drag on a rounded
        arc maps back to its own fillet feature.
        """
        if f.kind != "fillet" or self.doc.mesh is None:
            return []
        mids = [tuple(map(float, e.mid)) for e in self.doc.mesh.edges
                if len(e.polyline) > 2]
        if f.edge_refs:
            fillets = [(i, feature) for i, feature in enumerate(self.doc.features)
                       if feature.kind == "fillet" and feature.edge_refs]
            target_index = next((i for i, feature in fillets if feature is f), None)
            result = []
            for mid in mids:
                point = np.asarray(mid, float)
                owner, best_score = None, float("inf")
                for index, feature in fillets:
                    paths = [path for path in feature.edge_refs if path]
                    if not paths or feature.value <= 0:
                        continue
                    distance = min(self._point_path_distance(point, path) for path in paths)
                    score = distance / feature.value
                    if distance <= feature.value + 1e-3 and score < best_score:
                        owner, best_score = index, score
                if owner == target_index:
                    result.append(mid)
            return result
        if not f.edge_dirs:  # legacy feature (saved before edge_dirs existed)
            return mids
        res = []
        for m in mids:
            m_arr = np.asarray(m, float)
            best = 1e30
            for mid_e, dir_e in zip(f.edges, f.edge_dirs):
                mid_arr = np.asarray(mid_e, float)
                dir_arr = np.asarray(dir_e, float)
                t = float(np.dot(m_arr - mid_arr, dir_arr))
                d = float(np.linalg.norm((m_arr - mid_arr) - t * dir_arr))
                if d < best:
                    best = d
            if best <= f.value + 1e-3:
                res.append(m)
        return res

    def _refresh_fillet_result_edges(self):
        """Refresh every fillet's ``result_edges`` from the current mesh.

        Fillets are found by matching a grabbed edge mid against ``result_edges``.
        Only the edited feature used to be refreshed, so after a second fillet
        (or any rebuild) earlier fillets kept mids that no longer exist — and
        a saved model kept them forever.  Re-derive for all fillets on every
        rebuild and after load.
        """
        for f in self.doc.features:
            if f.kind == "fillet":
                f.result_edges = self._fillet_result_edges(f)

    def _find_fillet_by_edge(self, mid: tuple) -> int | None:
        """Index of the fillet feature whose rounded edges contain `mid`, else None."""
        m = np.asarray(mid, float)
        best, best_d = None, 1e-4  # require an exact-ish match (within ~0.1mm)
        for i, f in enumerate(self.doc.features):
            if f.kind != "fillet" or not f.result_edges:
                continue
            for re in f.result_edges:
                d = float(np.linalg.norm(np.asarray(re, float) - m))
                if d < best_d:
                    best, best_d = i, d
        return best

    def on_fillet_commit(self, radius: float, mids: list, kind: str = "fillet"):
        """Finalize an edge drag: commit a new fillet/chamfer, or edit an existing fillet.

        ``kind`` is "fillet" when the drag pushed toward the body and "chamfer" when it pulled
        away; dragging an existing fillet the other way turns it into a chamfer.

        - New fillet: the live temp feature is finalized in place; if the fillet
          is still failing at release the temp feature is rolled back so the model
          returns to the pre-drag state.
        - Edit (existing fillet): the feature's radius is changed in place; if the
          new radius makes the fillet fail the radius is reverted to its pre-drag
          value; dragging to ~0 removes the fillet.
        """
        mids = [tuple(map(float, m)) for m in mids]
        self._value_feature = None
        f = self._fillet_commit_feature(radius, mids, kind)
        # The value box stays on the model only if the feature exists at the dragged size.
        if (f is not None and any(x is f for x in self.doc.features) and abs(f.value - radius) < 1e-9
                and f.kind == kind):
            self._value_feature = f
        else:
            self.view.dismiss_value_box()

    def _fillet_commit_feature(self, radius: float, mids: list, kind: str = "fillet"):
        """on_fillet_commit's work; returns the fillet/chamfer feature it made or edited, if any."""
        if self._fillet_live_idx is not None:
            idx = self._fillet_live_idx
            f = self.doc.features[idx]
            if self._fillet_is_mod:
                if radius > 1e-4:
                    f.kind = kind
                    f.value = float(radius)
                    f.version += 1
                    self.rebuild()
                    if idx in self.doc.errors:
                        # New size invalid: back to the pre-drag fillet.
                        f.value = float(self._fillet_mod_prev or f.value)
                        f.kind = self._fillet_mod_prev_kind or "fillet"
                        f.version += 1
                        self.rebuild()
                        f.result_edges = self._fillet_result_edges(f)
                        QMessageBox.warning(self, f"Edit {kind}", "Reverted (it failed at the new size).")
                    else:
                        f.result_edges = self._fillet_result_edges(f)
                else:
                    # Dragged to ~0: remove the fillet.
                    self.doc.features.pop(idx)
                    self.rebuild()
            else:
                if radius > 1e-4:
                    f.kind = kind
                    f.value = float(radius)
                    f.version += 1
                    self.rebuild()
                    if idx in self.doc.errors:
                        self.doc.features.pop(idx)
                        self.rebuild()
                    else:
                        f.result_edges = self._fillet_result_edges(f)
                else:
                    self.doc.features.pop(idx)
                    self.rebuild()
            self._fillet_live_idx = None
            self._fillet_preview_r = None
            self._fillet_is_mod = False
            self._fillet_mod_prev = None
            self._fillet_mod_prev_kind = None
            self._fillet_preview_kind = None
            return f
        if radius > 1e-4:
            feature = Feature(kind, value=float(radius), edges=mids,
                              edge_dirs=self._fillet_edge_dirs(mids), edge_refs=self._fillet_edge_refs(mids))
            if self.commit(feature):
                feature.result_edges = self._fillet_result_edges(feature)
                return feature
        return None

    def on_fillet_preview(self, radius: float, mids: list, kind: str = "fillet"):
        """Live fillet/chamfer: keep the feature in the history while dragging.

        ``kind`` follows the drag direction (push: fillet, pull: chamfer) and may change
        mid-drag; the live feature switches with it.

        The first move decides whether the drag edits an existing fillet (the
        selected edge's mid matches a fillet feature's result_edges) or creates
        a new one.  Either way the radius is changed in place and the model is
        rebuilt (throttled to >= 0.05mm).  Editing also refreshes the fillet's
        result_edges so the next drag can find it again.
        """
        mids = [tuple(map(float, m)) for m in mids]
        if self._fillet_live_idx is None:
            self._push_undo()  # remember the pre-drag history
            mod_idx = self._find_fillet_by_edge(mids[0])
            if mod_idx is not None:
                self._fillet_live_idx = mod_idx
                self._fillet_is_mod = True
                self._fillet_mod_prev = self.doc.features[mod_idx].value
                self._fillet_mod_prev_kind = self.doc.features[mod_idx].kind
            else:
                self.doc.features.append(Feature(kind, value=float(radius), edges=mids,
                                                 edge_dirs=self._fillet_edge_dirs(mids),
                                                 edge_refs=self._fillet_edge_refs(mids)))
                self._fillet_live_idx = len(self.doc.features) - 1
            self._fillet_preview_r = None
        elif (self._fillet_preview_r is not None and abs(radius - self._fillet_preview_r) < 0.05
              and kind == self._fillet_preview_kind):
            return  # not worth a rebuild this frame
        f = self.doc.features[self._fillet_live_idx]
        f.kind = kind
        f.value = float(radius)
        f.version += 1
        sel_edges = set(self.view.sel_edges)
        sel_faces = set(self.view.sel_faces)
        self.rebuild()  # runs the kernel + refresh (clears selection); re-apply it
        self.view.sel_edges = sel_edges
        self.view.sel_faces = sel_faces
        if self._fillet_is_mod:
            f.result_edges = self._fillet_result_edges(f)
        self._fillet_preview_r = radius
        self._fillet_preview_kind = kind

    def _edge_feature(self, kind):
        if not self.view.sel_edges:
            QMessageBox.information(self, kind.title(), "Select one or more edges first (Ctrl+click for several).")
            return
        mids = [tuple(map(float, self.doc.mesh.edges[i].mid)) for i in sorted(self.view.sel_edges)]
        dirs = self._fillet_edge_dirs(mids) if kind == "fillet" else []
        refs = self._fillet_edge_refs(mids) if kind == "fillet" else []
        f = Feature(kind, value=1.0, edges=mids, edge_dirs=dirs, edge_refs=refs)
        self._value_session(f, kind.title(), "Radius" if kind == "fillet" else "Size", new=True, lo=1e-4, hi=1e4)

    def fillet(self):
        self._edge_feature("fillet")

    def chamfer(self):
        self._edge_feature("chamfer")
