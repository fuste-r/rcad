"""What is picked: clicks, hover, double-click and box selection, sketch regions, the quick buttons,
right-click, and the selection tag."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtWidgets import QMenu

from . import datum
from .camera import ray_plane


class SelectionMixin:
    """What is picked: clicks, hover, double-click and box selection, sketch regions, the quick buttons, right-click, and the selection tag. Mixed into MainWindow."""

    def _update_regions(self):
        """Refresh the fill for selected and hovered sketch regions (hover_region = (feature index, region))."""
        fills = []
        arrow = None
        for i, f in enumerate(self.doc.features):
            if f.kind != "sketch":
                continue
            shown = i == self.sketch_index if self.mode == "sketch" else f.visible
            if not shown:
                continue
            chosen = f.sketch.selected_regions()
            chosen_fills = [f.sketch.region_triangles(r) for r in sorted(chosen)]
            fills += [(t, (1.0, 0.55, 0.15, 0.45)) for t in chosen_fills]
            if i == self.region_sketch and chosen_fills:
                self._region_anchor = chosen_fills[0].reshape(-1, 3).mean(axis=0)
                if self.mode == "model":
                    arrow = (np.concatenate(chosen_fills).reshape(-1, 3).mean(axis=0), np.array(f.sketch.plane.normal, float))
            if self.hover_region is not None and self.hover_region[0] == i and self.hover_region[1] not in chosen:
                fills.append((f.sketch.region_triangles(self.hover_region[1]), (0.95, 0.9, 0.3, 0.35)))
        self.view.region_fill = fills
        self.view.region_arrow = arrow
        self._refresh_quick()
        self._refresh_tag()
        self.view.update()

    def _quick_spec(self):
        """(key, world anchor, likeliest tools, more tools, delete tool) for the selection, or None."""
        if self._session is not None or self.tool_bar.active:
            return None
        v = self.view
        if self.mode == "sketch":
            if self.tool != "select" or not v.sketch_sel or self.sketch is None:
                return None
            sk = self.sketch
            sel = sorted(v.sketch_sel)
            pts = [i for k, i in sel if k == "p"]
            lines = [i for k, i in sel if k == "l"]
            circles = [i for k, i in sel if k == "c"]
            arcs = [i for k, i in sel if k == "a"]
            try:
                uvs = [sk.points[i] for i in pts]
                uvs += [np.mean([sk.points[a], sk.points[b]], axis=0) for a, b in (sk.lines[i] for i in lines)]
                uvs += [sk.points[sk.circles[i][0]] for i in circles]
                uvs += [sk.arc_polyline(i, 3)[1] for i in arcs]
            except IndexError:
                return None
            if not uvs:
                return None
            anchor = np.asarray(sk.plane.to3d(*np.mean(uvs, axis=0)), float)
            tools = [(self.CONSTRAINT_LABELS[k], lambda k=k: self.add_constraint(k), f"Add a {k} constraint")
                     for k in self._sketch_quick_kinds(pts, lines, circles, arcs)]
            more = tools[2:]
            if lines or circles or arcs:
                curves = [(k, i) for k, i in sel if k != "p"]
                on = not all(sk.is_construction(k, i) for k, i in curves)
                more.append(("┄ Construction" if on else "━ Normal geometry", self.toggle_construction,
                             "Reference geometry: dashed, for snapping and constraints, never part of a region"
                             if on else "Make the selected reference geometry normal again (Q)"))
            if lines or circles or arcs:
                more.append(("⧉ Offset", self.offset_selection,
                             "A parallel copy at a distance that stays at that distance"))
            delete = ("🗑", self.delete_selection, "Delete the selected sketch items")
            return ("sketch", tuple(sel)), anchor, tools[:2], more, delete
        if v.sel_plane is not None:
            name = v.sel_plane
            tools = [("✏️ Sketch", lambda name=name: self._sketch_on_plane(name), f"Start a sketch on the {name} plane")]
            _, u, up = datum.axes(name)  # beside the corner across from the plane's name
            return ("plane", name), (u - up) * v._datum_half * 0.8, tools, [], None
        mesh = self.doc.mesh
        bodies = self._selected_bodies()
        if bodies:
            faces = sorted(set().union(*(self._body_faces(b) for b in bodies)))
            anchor = np.max([mesh.faces[f].center for f in faces], axis=0)  # beside the body's far corner
            tools = [("✥ Move", self.move_bodies, "Move or rotate the selected bodies (or a copy)"),
                     ("⇋ Mirror", self.mirror_bodies, "Mirror the selected bodies across a plane")]
            return ("bodies", tuple(bodies)), anchor, tools, [], None
        if v.sel_edges and not v.sel_faces and mesh is not None:
            edges = sorted(e for e in v.sel_edges if 0 <= e < len(mesh.edges))
            if not edges:
                return None
            owners = {self._edge_owner(e) for e in edges}
            delete = None
            if len(owners) == 1 and None not in owners:
                i = owners.pop()
                delete = ("🗑", lambda i=i: self._delete_feature_at(i),
                          f"Delete the {self.doc.features[i].kind} this edge belongs to")
            tools = [("◐ Fillet", self.fillet, "Round the selected edges"),
                     ("◺ Chamfer", self.chamfer, "Bevel the selected edges")]
            return ("edges", tuple(edges)), np.asarray(mesh.edges[edges[0]].mid, float), tools, [], delete
        if len(v.sel_faces) == 1 and not v.sel_edges and mesh is not None:
            face = next(iter(v.sel_faces))
            if not (0 <= face < len(mesh.faces)) or mesh.faces[face].normal is None:
                return None
            tools = [("✏️ Sketch", self._sketch_on_selected_face, "Start a sketch on this face"),
                     ("↕️ Push/Pull", self.pushpull, "Move this face along its normal")]
            return ("face", face), np.asarray(mesh.faces[face].center, float), tools, [], None
        feats = self.doc.features
        if (not v.sel_faces and not v.sel_edges and 0 <= self.region_sketch < len(feats)
                and feats[self.region_sketch].kind == "sketch"
                and feats[self.region_sketch].sketch.selected_regions() and self._region_anchor is not None):
            chosen = tuple(sorted(feats[self.region_sketch].sketch.selected_regions()))
            tools = [("⇈ Extrude", self.extrude, "Extrude the highlighted regions"),
                     ("⟳ Revolve", self.revolve, "Revolve the highlighted regions")]
            more = [("↝ Sweep", self.sweep, "Sweep this profile along another sketch"),
                    ("⌁ Loft", self.loft, "Loft between sketch sections")]
            return ("region", self.region_sketch, chosen), self._region_anchor, tools, more, None
        return None

    def _refresh_quick(self):
        spec = self._quick_spec()
        if spec is None:
            self.quick.clear()
            return
        key, anchor, primary, more, delete = spec
        self.quick.set_tools(key, anchor, primary, more, delete)
        self._place_quick()

    def _place_quick(self):
        """Keep the quick buttons beside the selection; hidden while the mouse drags."""
        q, v = self.quick, self.view
        if q.anchor is None or not q.buttons:
            q.hide()
            return
        if v._press is not None and v._dragged:
            q.hide()
            return
        s = v.camera.project(np.asarray(q.anchor, float).reshape(1, 3), v.width(), v.height())[0]
        if np.isnan(s[0]) or np.isnan(s[1]):
            q.hide()
            return
        q.place(float(s[0]), float(s[1]))

    def _edge_owner(self, edge: int) -> int | None:
        """Index of the fillet (by its rounded edges) or fillet/chamfer (by its picked edges) an edge belongs to."""
        i = self._find_fillet_by_edge(tuple(map(float, self.doc.mesh.edges[edge].mid)))
        return i if i is not None else self._feature_for_edge(edge)

    def _picking_regions_of(self, i: int) -> bool:
        """Only regions of sketch i are picked: a plain click on another of its regions adds it."""
        v, feats = self.view, self.doc.features
        geometry = v.sketch_sel if self.mode == "sketch" else (v.sel_faces or v.sel_edges)
        return (not geometry and i == self.region_sketch and 0 <= i < len(feats)
                and feats[i].kind == "sketch" and bool(feats[i].sketch.selected_regions()))

    def _pick_region(self, x, y):
        """Visible-sketch region under the cursor and not hidden behind a model face: (feature, region, (u, v))."""
        cam, w, h = self.view.camera, self.view.width(), self.view.height()
        o, d = cam.ray(x, y, w, h)
        face_t = self.view.pick_face_hit(x, y)[1]
        best = None
        for i, f in enumerate(self.doc.features):
            if f.kind != "sketch" or not f.visible:
                continue
            pl = f.sketch.plane
            p = ray_plane(o, d, pl.origin, np.array(pl.normal))
            if p is None:
                continue
            t = float(np.dot(p - o, d))
            uv = pl.to_uv(p)
            r = f.sketch.region_at(*uv)
            if r is not None and (face_t is None or t <= face_t + 1e-4 * cam.dist) and (best is None or t < best[0]):
                best = (t, i, r, uv)
        return None if best is None else best[1:]

    # ---- model mode
    def on_click(self, x, y, button, mods):
        if button == Qt.RightButton:
            self.step_back(x, y)
            return
        if button != Qt.LeftButton:
            return
        if self.mode == "sketch":
            self._sketch_click(x, y, mods)
            return
        v = self.view
        multi = bool(mods & Qt.ControlModifier)
        edge = v.pick_edge(x, y)
        hit = None if edge is not None else self._pick_region(x, y)
        face = None if edge is not None or hit is not None else v.pick_face(x, y)
        adding = hit is not None and self._picking_regions_of(hit[0])  # more regions for the same extrude
        plane = None if edge is not None or hit is not None or face is not None else v.pick_plane(x, y)
        if plane is not None:  # a plane is picked on its own; clicking it again drops it
            again = plane == v.sel_plane
            self._clear_model_selection()
            v.sel_plane = None if again else plane
            self._update_regions()
            return
        if not multi:
            v.sel_plane = None
        if not multi:
            v.sel_faces.clear()
            v.sel_edges.clear()
            for j, f in enumerate(self.doc.features):
                if f.kind == "sketch" and (hit is None or j != hit[0]):
                    f.sketch.selection = []
        if edge is not None:
            v.sel_edges.symmetric_difference_update({edge})
        elif hit is not None:
            i, _, uv = hit
            self.doc.features[i].sketch.toggle_region(uv[0], uv[1], multi or adding)
            self.region_sketch = i
            self.mark_dirty()
        elif face is not None:
            v.sel_faces.symmetric_difference_update({face})
        self._update_regions()

    # -- selection shortcuts
    def _clear_model_selection(self):
        self.view.sel_faces.clear()
        self.view.sel_edges.clear()
        self.view.sel_plane = None
        for f in self.doc.features:
            if f.kind == "sketch":
                f.sketch.selection = []

    def _body_faces(self, body: int) -> set[int]:
        mesh = self.doc.mesh
        return {i for i, b in enumerate(mesh.face_body) if b == body}

    def _selected_bodies(self) -> list[int] | None:
        """The bodies whose every face is selected, when the selection is exactly those; else None."""
        v, mesh = self.view, self.doc.mesh
        if mesh is None or not v.sel_faces or v.sel_edges or self.mode != "model":
            return None
        faces = {f for f in v.sel_faces if 0 <= f < len(mesh.faces)}
        bodies = sorted({mesh.face_body[f] for f in faces})
        if not bodies or faces != set().union(*(self._body_faces(b) for b in bodies)):
            return None
        return bodies

    def on_double_click(self, x, y, mods):
        """Double-click: the whole body under the pointer, or in a sketch the connected chain of
        lines (or the circle). Drawing tools treat it as two clicks. Ctrl adds to the selection."""
        v, add = self.view, bool(mods & Qt.ControlModifier)
        if self.mode == "sketch":
            hit = v.pick_sketch(x, y) if self.tool == "select" else None
            if hit is None:
                self.on_click(x, y, Qt.LeftButton, mods)
                return
            chain = self._sketch_chain(hit)
            if not add:
                v.sketch_sel.clear()
            v.sketch_sel |= chain
            self._update_regions()
            return
        mesh = self.doc.mesh
        body = None
        if mesh is not None:
            edge = v.pick_edge(x, y)
            if edge is not None:
                body = mesh.edge_body[edge]
                v.sel_edges.discard(edge)  # the first click picked the edge: the body replaces it
            else:
                face = v.pick_face(x, y)
                body = None if face is None else mesh.face_body[face]
        if body is None:
            self.on_click(x, y, Qt.LeftButton, mods)
            return
        if not add:
            self._clear_model_selection()
        v.sel_faces |= self._body_faces(body)
        self.mark_dirty()
        self._update_regions()

    def select_all(self):
        """Ctrl+A: every body in the model, or everything in the sketch."""
        self._settle_session()
        v = self.view
        if self.mode == "sketch":
            if self.tool != "select":
                self.set_tool("select")
            sk = self.sketch
            v.sketch_sel = ({("p", i) for i in range(len(sk.points))} | {("l", i) for i in range(len(sk.lines))}
                            | {("c", i) for i in range(len(sk.circles))} | {("a", i) for i in range(len(sk.arcs))})
        else:
            self._clear_model_selection()
            if self.doc.mesh is not None:
                v.sel_faces |= set(range(len(self.doc.mesh.faces)))
            self.mark_dirty()
        self._update_regions()

    def on_box_select(self, x0, y0, x1, y1, mods):
        """A selection box: everything entirely inside it. In the model that is the visible edges
        (added to the selection: Ctrl draws the box); in a sketch, points, lines and circles
        (Ctrl adds, otherwise they replace the selection)."""
        v = self.view
        w, h = v.width(), v.height()

        def inside(world) -> bool:
            q = v.camera.project(np.asarray(world, float).reshape(-1, 3), w, h)
            return bool(not np.isnan(q).any() and (q[:, 0] >= x0).all() and (q[:, 0] <= x1).all()
                        and (q[:, 1] >= y0).all() and (q[:, 1] <= y1).all())

        if self.mode == "sketch":
            sk = self.sketch
            pl = sk.plane
            to3 = lambda uv: pl.to3d(*uv)
            found = {("p", i) for i, p in enumerate(sk.points) if inside([to3(p)])}
            found |= {("l", i) for i, (a, b) in enumerate(sk.lines) if inside([to3(sk.points[a]), to3(sk.points[b])])}
            ring = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            for i, (c, r) in enumerate(sk.circles):
                u, vv = sk.points[c]
                if inside([to3((u + r * np.cos(t), vv + r * np.sin(t))) for t in ring]):
                    found.add(("c", i))
            found |= {("a", i) for i in range(len(sk.arcs)) if inside([to3(q) for q in sk.arc_polyline(i, 24)])}
            if not mods & Qt.ControlModifier:
                v.sketch_sel.clear()
            v.sketch_sel |= found
            self._update_regions()
            return
        mesh = self.doc.mesh
        if mesh is None:
            return
        for i, e in enumerate(mesh.edges):
            if inside(e.polyline) and self._edge_visible(e.mid):
                v.sel_edges.add(i)
        self._update_regions()

    def _edge_visible(self, mid) -> bool:
        """True unless a model face hides this point of an edge from the camera."""
        v = self.view
        s = v.camera.project(np.asarray(mid, float).reshape(1, 3), v.width(), v.height())[0]
        o, d = v.camera.ray(float(s[0]), float(s[1]), v.width(), v.height())
        t_hit = v.pick_face_hit(float(s[0]), float(s[1]))[1]
        return t_hit is None or t_hit >= float(np.dot(np.asarray(mid, float) - o, d)) - 1e-3 * v.camera.dist

    def _selection_text(self) -> str:
        """A short name for what is selected ("Flat face", "3 edges", "Body 2", "2 lines, 1 circle"), or ""."""
        v = self.view

        def count(n, one, many=None):
            return f"{n} {one if n == 1 else many or one + 's'}"

        if self.mode == "sketch":
            kinds = [k for k, _ in v.sketch_sel]
            names = [("p", "point"), ("l", "line"), ("c", "circle"), ("a", "arc")]
            if len(kinds) == 1:
                return dict(names)[kinds[0]].capitalize()
            return ", ".join(count(kinds.count(k), name) for k, name in names if kinds.count(k))
        if v.sel_plane is not None:
            return f"{v.sel_plane} plane"
        mesh = self.doc.mesh
        parts = []
        if v.sel_faces and mesh is not None:
            faces = {f for f in v.sel_faces if 0 <= f < len(mesh.faces)}
            bodies = self._selected_bodies()
            if bodies:
                return f"Body {bodies[0] + 1}" if len(bodies) == 1 else f"{len(bodies)} bodies"
            if len(faces) == 1:
                parts.append("Flat face" if mesh.faces[next(iter(faces))].normal is not None else "Curved face")
            else:
                parts.append(count(len(faces), "face"))
        if v.sel_edges and mesh is not None:
            edges = [e for e in v.sel_edges if 0 <= e < len(mesh.edges)]
            owner = self._edge_owner(edges[0]) if len(edges) == 1 else None
            if owner is not None:
                parts.append(f"{self.doc.features[owner].kind.title()} edge")
            elif len(edges) == 1 and not parts:
                parts.append("Edge")
            else:
                parts.append(count(len(edges), "edge"))
        if parts:
            text = ", ".join(parts)
            return text[0].upper() + text[1:]
        feats = self.doc.features
        if 0 <= self.region_sketch < len(feats) and feats[self.region_sketch].kind == "sketch":
            n = len(feats[self.region_sketch].sketch.selected_regions())
            if n:
                return f"{'Region' if n == 1 else f'{n} regions'} of {feats[self.region_sketch].name}"
        return ""

    def _refresh_tag(self):
        text = self._selection_text()
        self.sel_tag.setText(text)
        self.sel_tag.adjustSize()
        self._place_tag()

    def _place_tag(self):
        t = self.sel_tag
        if not t.text():
            t.hide()
            return
        t.move(max((self.view.width() - t.width()) // 2, 4), 10)
        t.show()
        t.raise_()

    # -- right-click
    def step_back(self, x, y) -> str:
        """Right-click (without dragging): undo the last step of what is in progress, or offer tools.

        In order: an open settings-bar tool is cancelled; in a sketch a half-drawn shape (or a
        mirror pick) is dropped, then the drawing tool is left. With something selected, or
        something under the pointer (which is then selected), its tools open at the pointer.
        Otherwise right-click leaves the sketch. Returns what it did, for tests."""
        if self._session is not None:
            self._cancel_session()
            return "cancel tool"
        if self.mode == "sketch":
            if self.pending is not None or self._mirror_axis is not None:
                self.cancel()
                return "drop step"
            if self.tool != "select":
                self.set_tool("select")
                return "leave tool"
        self._select_under(x, y)
        if self._open_selection_menu(x, y):
            return "menu"
        if self.mode == "sketch":
            self.finish_sketch()
            return "exit sketch"
        return "nothing"

    def _select_under(self, x, y):
        """Select what is under the pointer, unless it is already part of the selection."""
        v = self.view
        if self.mode == "sketch":
            hit = v.pick_sketch(x, y)
            if hit is not None and hit not in v.sketch_sel:
                v.sketch_sel.clear()
                v.sketch_sel.add(hit)
                self._update_regions()
            return
        edge = v.pick_edge(x, y)
        if edge is not None:
            if edge not in v.sel_edges:
                self.on_click(x, y, Qt.LeftButton, Qt.NoModifier)
            return
        hit = self._pick_region(x, y)
        if hit is not None:
            i, r, _ = hit
            if r not in self.doc.features[i].sketch.selected_regions():
                self.on_click(x, y, Qt.LeftButton, Qt.NoModifier)
            return
        face = v.pick_face(x, y)
        if face is not None and face not in v.sel_faces:
            self.on_click(x, y, Qt.LeftButton, Qt.NoModifier)
            return
        plane = v.pick_plane(x, y) if face is None else None
        if plane is not None and plane != v.sel_plane:
            self.on_click(x, y, Qt.LeftButton, Qt.NoModifier)

    def _open_selection_menu(self, x, y) -> bool:
        """Pop up the selection's tools (the quick buttons' list) at view point (x, y)."""
        spec = self._quick_spec()
        if spec is None:
            return False
        _, _, primary, more, delete = spec
        menu = QMenu(self.view)
        for group in (primary, more, [delete] if delete is not None else []):
            if group and menu.actions():
                menu.addSeparator()
            for label, slot, tip in group:
                if label == "🗑":
                    label = "🗑 Delete"
                a = menu.addAction(label)
                a.setToolTip(tip)
                a.triggered.connect(lambda _=False, s=slot: s())
        self.selection_menu = menu
        menu.popup(self.view.mapToGlobal(QPoint(int(x), int(y))))
        return True

    def _feature_for_edge(self, edge):
        if edge is None or self.doc.mesh is None or not (0 <= edge < len(self.doc.mesh.edges)):
            return None
        mid = np.asarray(self.doc.mesh.edges[edge].mid, float)
        for i, f in enumerate(self.doc.features):
            if f.kind not in ("fillet", "chamfer"):
                continue
            for m in f.edges:
                if np.linalg.norm(np.asarray(m, float) - mid) < 1e-3:
                    return i
        return None

    def on_hover(self, x, y):
        if self.mode == "sketch":
            if self.tool != "select":
                self._update_preview(x, y)
                return
            uv = self.view.sketch_uv(x, y)
            near = self.view.pick_sketch(x, y) is not None
            r = None if uv is None or near else self.sketch.region_at(*uv)
            region = None if r is None else (self.sketch_index, r)
            if region != self.hover_region:
                self.hover_region = region
                self._update_regions()
            return
        edge = self.view.pick_edge(x, y)
        hit = None if edge is not None else self._pick_region(x, y)
        region = None if hit is None else hit[:2]
        face = None if edge is not None or hit is not None else self.view.pick_face(x, y)
        plane = None if edge is not None or hit is not None or face is not None else self.view.pick_plane(x, y)
        if plane != self.view.hover_plane:
            self.view.hover_plane = plane
            self.view.update()
        if (edge, face) != (self.view.hover_edge, self.view.hover_face) or region != self.hover_region:
            self.view.hover_edge, self.view.hover_face = edge, face
            self.hover_region = region
            self._update_regions()
