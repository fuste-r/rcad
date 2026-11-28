"""Drawing in a sketch: the tools, snapping, typed sizes, arcs, dragging geometry, construction
geometry and deleting sketch items."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from .sketch import Constraint, end_direction, tangent_arc
from .offset import chains, offset, OffsetError
from .tool_bar import Choice, Number
from .trim import piece_at, trim
from .snapping import find_snap, grid_step, Snap, START
from .viewport import Viewport


def _corner_kept(cons, start_idx, snap, n0):
    """A rectangle's corners are new points even when a click landed on an existing point:
    tie those corners to the points they were drawn on."""
    out = list(cons)
    if start_idx is not None:
        out.append(Constraint("coincident", [start_idx, n0]))
    if snap is not None and snap.point is not None:
        out.append(Constraint("coincident", [snap.point, n0 + 2]))
    return out


class SketchToolsMixin:
    """Drawing in a sketch: the tools, snapping, typed sizes, arcs, dragging geometry, construction geometry and deleting sketch items. Mixed into MainWindow."""

    # -- sketch drag
    def on_sketch_mouse(self, phase, x, y):
        """Left-button drag on sketch geometry (points, lines, circles).

        phase is "press" (track the geometry under the cursor), "move" (apply the
        cursor delta directly so the view updates live), or "release" (solve and
        either keep the move or revert it if the solve fails)."""
        if self.tool == "trim" and self.sketch is not None:
            self._trim_stroke_event(phase, x, y)
            return
        if phase == "press":
            if self.tool == "select" and self.sketch is not None:
                self._sketch_press_uv = self.view.sketch_uv(x, y)
                self._sketch_press_hit = self.view.pick_sketch(x, y)
                if self._sketch_press_hit is None:
                    self.view.start_box(x, y)  # a drag from empty space selects what it boxes
            else:
                self._sketch_press_uv = None
                self._sketch_press_hit = None
            self._sketch_drag_active = False
            self._sketch_drag_orig = None
        elif phase == "move":
            if not self._sketch_drag_active:
                if self._sketch_press_hit is None or self._sketch_press_uv is None:
                    return
                kind, idx = self._sketch_press_hit
                self._sketch_drag_active = True
                self._sketch_drag_kind = kind
                self._sketch_drag_idx = idx
                self._sketch_drag_orig = self._collect_drag_points(kind, idx)
                self._push_sketch_undo()  # snapshot so a failed solve can revert
            cur_uv = self.view.sketch_uv(x, y)
            if cur_uv is None or self._sketch_press_uv is None:
                return
            du = cur_uv[0] - self._sketch_press_uv[0]
            dv = cur_uv[1] - self._sketch_press_uv[1]
            du, dv = self._clamp_drag_delta(self._sketch_drag_kind, self._sketch_drag_idx, du, dv)
            sk = self.sketch
            orig = self._sketch_drag_orig
            if orig is not None:
                for pi, (u, v) in orig.items():
                    sk.points[pi] = [u + du, v + dv]
            # Show the silhouette only over the range this element is allowed to
            # sweep to (a fixed-angle line moves only along its legal direction).
            self._update_drag_silhouette(sk, du, dv)
            if self._sketch_press_hit is not None:
                self.view.sketch_sel = {self._sketch_press_hit}  # highlight while dragging
            self.view.update()
        elif phase == "release":
            if self._sketch_drag_active:
                self._finish_sketch_drag()
            self._sketch_drag_active = False
            self._sketch_press_uv = None
            self._sketch_press_hit = None
            self._sketch_drag_orig = None
            self._sketch_drag_kind = None
            self._sketch_drag_idx = None
            self.view.preview = []

    def _line_angle_fixed(self, idx):
        """Does the dragged line have a constraint fixing its angle?

        A horizontal/vertical line, or a line constrained parallel/perpendicular
        to another, can only translate perpendicular to its own direction.
        """
        for c in self.sketch.constraints:
            if c.kind in ("horizontal", "vertical") and c.refs == [idx]:
                return True
            if c.kind in ("parallel", "perpendicular") and idx in c.refs:
                return True
        return False

    def _clamp_drag_delta(self, kind, idx, du, dv):
        """Project (du, dv) onto the subspace the constraints allow.

        For a line with a fixed angle the legal motion is the direction
        perpendicular to the line; the perpendicular component of the cursor
        delta is dropped so the line never rotates off its angle. Returns the
        input unchanged when no restriction applies.
        """
        if kind != "l" or not self._line_angle_fixed(idx):
            return du, dv
        sk = self.sketch
        a, b = sk.lines[idx]
        pa = sk.points[a]
        pb = sk.points[b]
        dx, dy = pb[0] - pa[0], pb[1] - pa[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return du, dv
        dx, dy = dx / L, dy / L          # unit direction of the line
        px, py = -dy, dx                # unit perpendicular = allowed translation
        s = du * px + dv * py           # scalar component along the allowed direction
        return s * px, s * py

    def _update_drag_silhouette(self, sk, du, dv):
        """Set the viewport preview to the range the dragged line is allowed to
        sweep over: for a fixed-angle line, the strip between its start and
        current positions along the allowed (perpendicular) direction."""
        kind, idx = self._sketch_drag_kind, self._sketch_drag_idx
        if kind != "l" or not self._line_angle_fixed(idx):
            self.view.preview = []
            return
        orig = self._sketch_drag_orig
        if orig is None:
            self.view.preview = []
            return
        a, b = sk.lines[idx]
        p0a = orig[a]
        p0b = orig[b]
        p1a = sk.points[a]
        p1b = sk.points[b]
        if all(math.hypot(x1 - x0, y1 - y0) < 1e-9
               for (x0, y0), (x1, y1) in ((p0a, p1a), (p0b, p1b))):
            self.view.preview = []   # no sweep yet; keep it clean
            return
        pl = sk.plane
        # Closed strip boundary (first point repeated so the LINE_STRIP loops):
        # start-A -> current-A -> current-B -> start-B -> start-A
        quad = [pl.to3d(*p0a), pl.to3d(*p1a), pl.to3d(*p1b),
               pl.to3d(*p0b), pl.to3d(*p0a)]
        self.view.preview = [np.array(quad, dtype=float)]

    def _collect_drag_points(self, kind, idx):
        """Indices of the sketch points that move with the dragged element."""
        sk = self.sketch
        if kind == "p":
            return {idx: list(sk.points[idx])}
        if kind == "l":
            a, b = sk.lines[idx]
            return {a: list(sk.points[a]), b: list(sk.points[b])}
        if kind == "c":
            center, _ = sk.circles[idx]
            return {center: list(sk.points[center])}
        if kind == "a":  # the whole arc moves
            return {i: list(sk.points[i]) for i in sk.arcs[idx]}
        return {}

    def _finish_sketch_drag(self):
        """Solve after a drag; revert to the pre-drag positions if the solve fails."""
        sk = self.sketch
        orig = self._sketch_drag_orig
        if not sk.solve():
            if orig is not None:
                for pi, (u, v) in orig.items():
                    sk.points[pi] = [u, v]
            sk.solve()  # restores the known-good state
            self.statusBar().showMessage("Drag rejected: sketch constraints unsatisfied.", 6000)
        else:
            sk.version += 1
            hit = self._sketch_press_hit
            if hit is not None:
                kind, idx = hit
                label = {"p": f"point {idx}", "l": f"line {idx}", "c": f"circle {idx}", "a": f"arc {idx}"}[kind]
                self.statusBar().showMessage(f"Dragged {label}.", 2000)
        self.refresh_constraints()
        self._update_regions()
        self.mark_dirty()

    def _snap(self, x, y, mods=None) -> Snap | None:
        """Where a drawing click at view (x, y) lands, or None if the ray misses the plane.
        Ctrl places it freely: on the grid only, with no snapping to geometry and no constraints."""
        uv = self.view.sketch_uv(x, y)
        if uv is None:
            return None
        nxt = self.view.sketch_uv(x + 1, y)
        mm_per_px = math.dist(uv, nxt) if nxt is not None else 0.1
        g = self.snap.value()
        if g is None:
            g = self.snap.last_valid or 0.0  # keep snapping while the text is being edited
        if mods is None:
            mods = QApplication.keyboardModifiers()
        start = self.pending if self.tool == "line" else None
        last = self.pending is not None and (self.tool == "circle" or (self.tool == "arc" and self._arc_start))
        tangent = None
        if start is not None and start[0] is not None:  # a line leaving an arc's end: snap along its tangent
            came = end_direction(self.sketch, start[0], prefer=self._chain_item)
            if came is not None and came[0][0] == "a":
                tangent = (came[0][1], came[1])
        return find_snap(self.sketch, uv, self.SNAP_PX * mm_per_px, grid_step(g, mm_per_px), start=start,
                         smart=not mods & Qt.ControlModifier, curves=not last, tangent=tangent)

    SNAP_PX = 10  # how close (screen pixels) the pointer must be for a snap

    def _show_snap(self, snap: Snap | None):
        """Mark where the next point lands and draw its alignment guides; name the snap in the status bar."""
        self._snap_text = "" if snap is None or snap.kind in ("grid", "free") else snap.describe(self.sketch)
        if not self._snap_text:
            self.view.snap_mark = None
            return
        sk, pl = self.sketch, self.sketch.plane
        start_uv = self.pending[1] if self.pending is not None else None
        self.view.snap_mark = {"kind": snap.kind, "at": np.array(pl.to3d(*snap.uv), float),
                               "guides": [(np.array(pl.to3d(*a), float), np.array(pl.to3d(*b), float))
                                          for a, b in snap.guides(sk, start_uv)]}
        if self.pending is None:  # (while a shape is drawn, the coordinates line names it)
            self.statusBar().showMessage(f"Snap: {self._snap_text}", 2000)

    def _auto_constrain(self, cons: list[Constraint]):
        """Add the constraints a snap implies, one at a time, dropping any the sketch can't take."""
        sk = self.sketch
        sk.variables = self.doc.variables
        for c in cons:
            sk.constraints.append(c)
            if not sk.solve():
                sk.constraints.pop()

    def _solve_sketch(self):
        self.sketch.variables = self.doc.variables
        ok = self.sketch.solve()
        if not ok:
            self.statusBar().showMessage(f"Solver: {self.sketch.status}", 6000)
        self.refresh_constraints()
        self.mark_dirty()
        self._update_regions()
        return ok

    def _sketch_click(self, x, y, mods):
        self.view.sel_constraint = None
        self._push_sketch_undo()
        sk = self.sketch
        if self.tool == "trim":
            uv = self.view.sketch_uv(x, y)
            if uv is None or not self._trim_at([uv], self._trim_tol(x, y)):
                self._sketch_undo.get(self.sketch_index, [None]).pop()  # nothing changed: no undo step
            return
        if self.tool == "select":
            hit = self.view.pick_sketch(x, y)
            sel = self.view.sketch_sel
            multi = bool(mods & Qt.ControlModifier)
            # In mirror mode (self._mirror_axis is set) we always add hits to the selection,
            # without requiring Ctrl and without replacing the current selection.
            if self._mirror_axis is not None:
                if hit is not None:
                    sel.add(hit)
                # background clicks are ignored in mirror mode
                self._update_regions()
                return
            if hit is None:
                uv = self.view.sketch_uv(x, y)
                if not multi:
                    sel.clear()
                if uv is not None:
                    sk.toggle_region(uv[0], uv[1], multi or self._picking_regions_of(self.sketch_index))
                    self.region_sketch = self.sketch_index
                    self.mark_dirty()
            elif multi:
                sel.symmetric_difference_update({hit})
            else:
                sel.clear()
                sel.add(hit)
            self._update_regions()
            return
        snap = self._snap(x, y, mods)
        if snap is None:
            return
        self._draw_at(snap.point, snap.uv, snap)

    # ---- sizes typed while drawing
    FIELD_SPECS = {
        "line": [("L", "L", "length"), ("A", "∠", "angle")],
        "rect": [("W", "W", "length"), ("H", "H", "length")],
        "circle": [("D", "Ø", "length")],
        "arc": [("R", "R", "length"), ("S", "∠", "angle")],  # radius, then the angle it sweeps
        "tarc": [("R", "R", "length"), ("S", "∠", "angle")],
    }

    def _apply_locks(self, p0, uv):
        """Move the cursor point so the shape from p0 has the sizes typed into the fields."""
        f = self.fields
        du, dv = uv[0] - p0[0], uv[1] - p0[1]
        if self.tool == "line":
            length, angle = f.locked("L"), f.locked("A")
            if length is None and angle is None:
                return uv
            length = abs(length) if length is not None else math.hypot(du, dv)
            a = math.radians(angle) if angle is not None else (math.atan2(dv, du) if (du or dv) else 0.0)
            return p0[0] + length * math.cos(a), p0[1] + length * math.sin(a)
        if self.tool == "rect":
            w, h = f.locked("W"), f.locked("H")
            if w is not None:
                du = w if w < 0 else math.copysign(w, du)
            if h is not None:
                dv = h if h < 0 else math.copysign(h, dv)
            return p0[0] + du, p0[1] + dv
        if self.tool == "circle":
            d = f.locked("D")
            if d is None:
                return uv
            n = math.hypot(du, dv)
            ux, uy = (du / n, dv / n) if n > 1e-12 else (1.0, 0.0)
            return p0[0] + abs(d) / 2 * ux, p0[1] + abs(d) / 2 * uy
        if self.tool == "arc":
            if self._arc_start is None:  # placing the start: the radius
                r = f.locked("R")
                if r is None:
                    return uv
                n = math.hypot(du, dv)
                ux, uy = (du / n, dv / n) if n > 1e-12 else (1.0, 0.0)
                return p0[0] + abs(r) * ux, p0[1] + abs(r) * uy
            return self._arc_end(p0, uv)
        if self.tool == "tarc":
            hit = self._tangent_arc(p0, uv)
            return hit[1] if hit is not None else uv
        return uv

    def _tangent_arc(self, start, uv):
        """The tangent arc from the pending point toward uv, with any typed radius and angle:
        (center, end, signed radius, sweep degrees), or None (straight on, or nothing to follow)."""
        if self._tarc_from is None:
            return None
        f = self.fields
        return tangent_arc(start, self._tarc_from[1], uv, f.locked("R") if f.active else None,
                           f.locked("S") if f.active else None)

    def _arc_end(self, center, uv):
        """Where the arc's end goes: on its radius, at the typed sweep angle or toward the pointer."""
        su, sv = self._arc_start[1]
        r = math.dist(center, (su, sv))
        sweep = self.fields.locked("S") if self.fields.active else None
        if sweep is None:
            a = math.atan2(uv[1] - center[1], uv[0] - center[0])
        else:
            a = math.atan2(sv - center[1], su - center[0]) + math.radians(sweep)
        return center[0] + r * math.cos(a), center[1] + r * math.sin(a)

    def _track_arc_sweep(self, center, uv):
        """Follow the pointer around the centre, so the arc goes the way it was dragged."""
        a = math.degrees(math.atan2(uv[1] - center[1], uv[0] - center[0]))
        step = (a - self._arc_angle + 180) % 360 - 180
        self._arc_sweep = max(-359.9, min(359.9, self._arc_sweep + step))
        self._arc_angle = a

    def _arc_sweep_now(self) -> float:
        typed = self.fields.locked("S") if self.fields.active else None
        return typed if typed is not None else self._arc_sweep

    def _field_values(self, p0, uv) -> dict:
        du, dv = uv[0] - p0[0], uv[1] - p0[1]
        if self.tool == "line":
            return {"L": math.hypot(du, dv), "A": math.degrees(math.atan2(dv, du)) if (du or dv) else 0.0}
        if self.tool == "rect":
            return {"W": abs(du), "H": abs(dv)}
        if self.tool == "circle":
            return {"D": 2 * math.hypot(du, dv)}
        if self.tool == "tarc":
            hit = self._tangent_arc(p0, uv)
            return {"R": abs(hit[2]), "S": hit[3]} if hit is not None else {"R": 0.0, "S": 0.0}
        if self.tool == "arc":
            if self._arc_start is None:
                return {"R": math.hypot(du, dv), "S": 0.0}
            return {"R": math.dist(p0, self._arc_start[1]), "S": self._arc_sweep_now()}
        return {}

    # -- boxes left on a shape after it is drawn
    def _show_drawn(self, drawn: dict):
        """Keep the size boxes on the shape just drawn so it can be resized by typing or scrubbing.

        drawn: {"tool": "rect", "pts": [4 point indices]} | {"tool": "circle", "circle": index}
               | {"tool": "line", "a": start point, "b": end point}
        """
        self._drawn = dict(drawn, undo=False)
        self.fields.start(self.FIELD_SPECS[drawn["tool"]], post=True)
        self._refresh_drawn_fields()

    def _drawn_ok(self) -> bool:
        d = self._drawn
        if d is None or not self.fields.post or self.mode != "sketch":
            return False
        sk = self.sketch
        if d["tool"] == "rect":
            return all(i < len(sk.points) for i in d["pts"])
        if d["tool"] == "circle":
            return d["circle"] < len(sk.circles)
        return max(d["a"], d["b"]) < len(sk.points)

    def _drawn_geometry(self):
        """(start uv, end uv) for rect (opposite corners) / line, or (center uv, radius) for circle."""
        d, sk = self._drawn, self.sketch
        if d["tool"] == "rect":
            return tuple(sk.points[d["pts"][0]]), tuple(sk.points[d["pts"][2]])
        if d["tool"] == "circle":
            c, r = sk.circles[d["circle"]]
            return tuple(sk.points[c]), r
        return tuple(sk.points[d["a"]]), tuple(sk.points[d["b"]])

    def _refresh_drawn_fields(self):
        d = self._drawn
        p0, end = self._drawn_geometry()
        if d["tool"] == "circle":
            self.fields.set_values({"D": 2 * end})
            anchor = (p0[0] + end, p0[1])
        else:
            tool, self.tool = self.tool, d["tool"]  # _field_values reads the tool
            self.fields.set_values(self._field_values(p0, end))
            self.tool = tool
            anchor = end
        self.fields.anchor = self.sketch.plane.to3d(*anchor)
        self.view.update()

    def _apply_drawn(self):
        """Resize the shape just drawn to the sizes typed or scrubbed into its boxes."""
        if not self._drawn_ok():
            return
        d, sk, f = self._drawn, self.sketch, self.fields
        p0, end = self._drawn_geometry()
        if d["tool"] == "rect":
            du, dv = end[0] - p0[0], end[1] - p0[1]
            w, h = f.locked("W"), f.locked("H")
            if w is not None:
                du = w if w < 0 else math.copysign(w, du)
            if h is not None:
                dv = h if h < 0 else math.copysign(h, dv)
            if abs(du) < 1e-9 or abs(dv) < 1e-9:
                return
            u1, v1 = p0[0] + du, p0[1] + dv
            new = {d["pts"][1]: [u1, p0[1]], d["pts"][2]: [u1, v1], d["pts"][3]: [p0[0], v1]}
        elif d["tool"] == "circle":
            dia = f.locked("D")
            if dia is None or abs(dia) < 1e-9:
                return
            new = None
        else:
            length, angle = f.locked("L"), f.locked("A")
            du, dv = end[0] - p0[0], end[1] - p0[1]
            length = abs(length) if length is not None else math.hypot(du, dv)
            a = math.radians(angle) if angle is not None else math.atan2(dv, du)
            if length < 1e-9:
                return
            new = {d["b"]: [p0[0] + length * math.cos(a), p0[1] + length * math.sin(a)]}
        if not d["undo"]:  # one undo step per editing session
            self._push_sketch_undo()
            d["undo"] = True
        if new is None:
            c, _r = sk.circles[d["circle"]]
            sk.circles[d["circle"]] = (c, abs(dia) / 2)
        else:
            for i, uv in new.items():
                sk.points[i] = uv
        sk.version += 1
        self._solve_sketch()
        self._refresh_drawn_fields()

    def _show_last_segment(self):
        """At the end of a line chain, keep the last segment's boxes if its end point is free."""
        seg, self._last_segment = self._last_segment, None
        if seg is None or self.mode != "sketch":
            return
        a, b = seg
        if b >= len(self.sketch.points) or sum(b in ln for ln in self.sketch.lines) != 1:
            return  # closing onto other geometry: resizing would drag that along
        self._show_drawn({"tool": "line", "a": a, "b": b})

    def _start_fields(self):
        self._drawn = None
        specs = self.FIELD_SPECS.get(self.tool)
        if specs and self.pending is not None:
            self.fields.start(specs)
            self._draw_mouse = None
        else:
            self.fields.stop()

    def _on_fields_changed(self):
        if self.fields.post:
            self._apply_drawn()  # the shape follows typing and scrubbing
            return
        if self._draw_mouse is not None:
            self._update_preview(*self._draw_mouse[2:])
        elif self.pending is not None:
            self._show_draw_preview(self.pending[1], None, None)

    def _on_fields_accepted(self):
        """Enter in a drawing value box: place the shape at the typed sizes."""
        if self.fields.post:
            self._apply_drawn()
            if self._drawn is not None:
                self._drawn["undo"] = False  # the next edit is a new undo step
            self.fields.end_typing()
            if self._drawn_ok():
                self._refresh_drawn_fields()
            self.view.setFocus()
            return
        if self.mode != "sketch" or self.pending is None:
            return
        pi, uv = (self._draw_mouse[:2] if self._draw_mouse is not None else (None, self.pending[1]))
        self._push_sketch_undo()
        self._draw_at(pi, uv, self._draw_snap if self._draw_mouse is not None else None)
        self.view.setFocus()

    def _draw_at(self, pi, uv, snap: Snap | None = None):
        """Place the next point of the active drawing tool at uv (pi: existing point there). The snap
        that put it there, if any, adds its constraints to the points it creates."""
        sk = self.sketch
        if self.pending is not None and self.fields.active:
            fixed = self._apply_locks(self.pending[1], uv)
            if math.dist(fixed, uv) > 1e-9:
                pi, uv, snap = None, fixed, None  # typed sizes win over the snap
        start_snap = self._pending_snap
        if self.pending is None:
            self._pending_snap = snap if pi is None else None
        self.view.snap_mark = None
        if self.tool == "line":
            if self.pending is None:
                self.pending = (pi, uv)
                self._last_segment = None
                self._start_fields()
                return
            a_idx, a_uv = self.pending
            if a_idx is None and pi is None and math.dist(a_uv, uv) < 1e-9:
                return
            a = a_idx if a_idx is not None else sk.add_point(*a_uv)
            b = pi if pi is not None else sk.add_point(*uv)
            if a != b:
                li = sk.add_line(a, b)
                sk.version += 1
                self._last_segment = (a, b)
                self._chain_item = ("l", li)
                cons = start_snap.constraints(a) if start_snap is not None and a_idx is None else []
                if snap is not None and snap.kind == "tangent":
                    cons.append(Constraint("tangent", [li, snap.tangent]))
                if snap is not None and pi is None:
                    level = [ax for ax, ref in snap.align if ref == START]
                    cons += [Constraint("horizontal" if ax == "h" else "vertical", [li]) for ax in level]
                    snap.align = [x for x in snap.align if x[1] != START]
                    cons += snap.constraints(b)
                self._auto_constrain(cons)
            self._pending_snap = None
            self.pending = (b, tuple(sk.points[b])) if pi is None else None
            self._start_fields()  # the next segment starts with nothing typed
            self._solve_sketch()
            if self.pending is None:  # the chain ended on an existing point
                self._show_last_segment()
        elif self.tool == "tarc":
            if self.pending is None:  # where it starts: the end of a line or arc
                came = end_direction(sk, pi) if pi is not None else None
                if came is None:
                    self.statusBar().showMessage("Start a tangent arc on the end of a line or arc.", 6000)
                    return
                self._tarc_from = came
                self._chain_item = came[0]
                self.pending = (pi, tuple(sk.points[pi]))
                self._start_fields()
                return
            self._finish_tangent_arc(pi, uv, snap)
        elif self.tool == "rect":
            if self.pending is None:
                self.pending = (pi, uv)
                self._start_fields()
                return
            (u0, v0), (u1, v1) = self.pending[1], uv
            if abs(u1 - u0) > 1e-9 and abs(v1 - v0) > 1e-9:
                n0 = len(sk.points)
                sk.add_rect(u0, v0, u1, v1)
                sk.version += 1
                cons = start_snap.constraints(n0) if start_snap is not None else []
                cons += snap.constraints(n0 + 2) if snap is not None else []
                self._auto_constrain(_corner_kept(cons, self.pending[0], snap, n0))
                self._solve_sketch()
                n = len(sk.points)
                self.set_tool("select")  # so the new region can be highlighted right away
                self._show_drawn({"tool": "rect", "pts": list(range(n - 4, n))})
            else:
                self.set_tool("select")
        elif self.tool == "circle":
            if self.pending is None:
                self.pending = (pi, uv)
                self._start_fields()
                return
            c_idx, c_uv = self.pending
            r = math.dist(c_uv, uv)
            if r > 1e-9:
                c = c_idx if c_idx is not None else sk.add_point(*c_uv)
                ci = sk.add_circle(c, r)
                sk.version += 1
                cons = start_snap.constraints(c) if start_snap is not None and c_idx is None else []
                if pi is not None:
                    cons.append(Constraint("point_on_circle", [pi, ci]))  # through the point it was drawn to
                self._auto_constrain(cons)
                self._solve_sketch()
                self.set_tool("select")
                self._show_drawn({"tool": "circle", "circle": ci})
            else:
                self.set_tool("select")
        elif self.tool == "arc":
            if self.pending is None:  # the centre
                self.pending = (pi, uv)
                self._start_fields()
                return
            c_idx, c_uv = self.pending
            if self._arc_start is None:  # the start, which sets the radius
                if math.dist(c_uv, uv) < 1e-9:
                    return
                self._arc_start = (pi, tuple(uv), snap)
                self._arc_sweep = 0.0
                self._arc_angle = math.degrees(math.atan2(uv[1] - c_uv[1], uv[0] - c_uv[0]))
                self.fields.set_values(self._field_values(c_uv, uv))
                return
            self._finish_arc(c_idx, c_uv, pi, uv, start_snap)

    def _finish_tangent_arc(self, pi, uv, snap):
        """Add the tangent arc from the pending point to uv, tangent to what it leaves; the chain then
        goes on from its end."""
        sk = self.sketch
        s, s_uv = self.pending
        hit = self._tangent_arc(s_uv, uv)
        if hit is None or hit[3] < 0.5 or hit[3] > 359.5:
            return  # straight on, or a whole circle: keep drawing
        center, end, r, sweep = hit
        c = sk.add_point(*center)
        lands = pi is not None and pi not in (s, c) and math.dist(sk.points[pi], end) < 1e-6 * max(abs(r), 1)
        e = pi if lands else sk.add_point(*end)
        ai = sk.add_arc(c, s, e) if r > 0 else sk.add_arc(c, e, s)  # stored counter-clockwise
        sk.version += 1
        kind, j = self._tarc_from[0]
        cons = [Constraint("tangent", [j, ai]) if kind == "l" else Constraint("tangent_arcs", [j, ai])]
        if snap is not None and not lands and math.dist(snap.uv, end) < 1e-9:
            cons += snap.constraints(e)  # the end went where the pointer snapped
        if lands:  # closing onto the end of another line or arc it meets smoothly: tangent there too
            arrive = end_direction(sk, e, prefer=("a", ai))[1]
            for kind2, j2 in [("l", i) for i, ab in enumerate(sk.lines) if e in ab] + \
                    [("a", i) for i, (_, s2, e2) in enumerate(sk.arcs) if e in (s2, e2) and i != ai]:
                other = end_direction(sk, e, prefer=(kind2, j2))
                if other is None or other[0] != (kind2, j2):
                    continue
                ou, ov = other[1]
                if abs(arrive[0] * ov - arrive[1] * ou) < 1e-6:  # parallel at the joint: smooth
                    cons.append(Constraint("tangent", [j2, ai]) if kind2 == "l"
                                else Constraint("tangent_arcs", [j2, ai]))
        self._auto_constrain(cons)
        self._solve_sketch()
        self._chain_item = ("a", ai)
        self.statusBar().showMessage(f"Tangent arc R {self.unit.fmt(abs(r))}, {sweep:.4g}°", 4000)
        if lands:  # closed onto existing geometry: the chain ends
            self.pending = None
            self._tarc_from = None
            self._start_fields()
            return
        self._tarc_from = end_direction(sk, e, prefer=("a", ai))
        self.pending = (e, tuple(sk.points[e]))
        self._start_fields()

    def _finish_arc(self, c_idx, c_uv, pi, uv, centre_snap):
        """Third click of the arc tool: add the arc from its start to the end on its radius.
        The sweep tracked while the pointer moved decides the side; its direction, the end."""
        sk = self.sketch
        s_idx, s_uv, s_snap = self._arc_start
        sweep = self._arc_sweep_now()
        if self.fields.locked("S") is None:
            a_ptr = math.degrees(math.atan2(uv[1] - c_uv[1], uv[0] - c_uv[0]))
            a0 = math.degrees(math.atan2(s_uv[1] - c_uv[1], s_uv[0] - c_uv[0]))
            ccw = (a_ptr - a0) % 360
            sweep = ccw if sweep >= 0 else ccw - 360
        if abs(sweep) < 0.5 or abs(sweep) > 359.5:
            return  # nothing yet, or a whole circle: keep drawing
        r = math.dist(c_uv, s_uv)
        a = math.atan2(s_uv[1] - c_uv[1], s_uv[0] - c_uv[0]) + math.radians(sweep)
        e_uv = (c_uv[0] + r * math.cos(a), c_uv[1] + r * math.sin(a))
        c = c_idx if c_idx is not None else sk.add_point(*c_uv)
        s = s_idx if s_idx is not None else sk.add_point(*s_uv)
        reuse = pi is not None and pi not in (c, s) and abs(math.dist(sk.points[pi], c_uv) - r) < 1e-6 * max(r, 1)
        e = pi if reuse else sk.add_point(*e_uv)
        if sweep > 0:
            sk.add_arc(c, s, e)
        else:
            sk.add_arc(c, e, s)
        sk.version += 1
        cons = centre_snap.constraints(c) if centre_snap is not None and c_idx is None else []
        cons += s_snap.constraints(s) if s_snap is not None and s_idx is None else []
        self._auto_constrain(cons)
        self._solve_sketch()
        self.set_tool("select")
        self.statusBar().showMessage(f"Arc R {self.unit.fmt(r)}, {abs(sweep):.4g}°", 4000)

    def _update_preview(self, x, y):
        if self.tool == "trim":
            if not self._trim_stroke:
                uv = self.view.sketch_uv(x, y)
                piece = None if uv is None else piece_at(self.sketch, uv, self._trim_tol(x, y))
                self._show_trim([piece] if piece else [])
            return
        snap = self._snap(x, y)
        self._show_snap(snap)
        if snap is None or self.pending is None:
            self.view.preview = []
            self.view.update()
            return
        self._draw_mouse = (snap.point, snap.uv, x, y)
        self._draw_snap = snap
        if self.tool == "arc" and self._arc_start is not None:
            self._track_arc_sweep(self.pending[1], snap.uv)
        self._show_draw_preview(snap.uv, x, y)

    def _show_draw_preview(self, uv, x, y):
        """Rubber-band the shape from the pending point to uv (after typed sizes) and show its sizes."""
        pl = self.sketch.plane
        p0 = self.pending[1]
        if self.fields.active:
            uv = self._apply_locks(p0, uv)
            self.fields.set_values(self._field_values(p0, uv))
            if x is None:  # no hover yet: put the boxes by the pending point
                x, y = self.view.camera.project(np.array([pl.to3d(*p0)], float),
                                                self.view.width(), self.view.height())[0][:2]
            self.fields.place(x, y)
        self.view.preview = []
        if self.tool == "line":
            self.view.preview = [np.array([pl.to3d(*p0), pl.to3d(*uv)])]
        elif self.tool == "rect":
            c = [(p0[0], p0[1]), (uv[0], p0[1]), (uv[0], uv[1]), (p0[0], uv[1]), (p0[0], p0[1])]
            self.view.preview = [np.array([pl.to3d(*p) for p in c])]
        elif self.tool == "circle":
            self.view.preview = [np.array(Viewport._circle_world(pl, p0, math.dist(p0, uv)))]
        elif self.tool == "tarc":
            hit = self._tangent_arc(p0, uv)
            if hit is None:  # straight on: show the direction it would leave in
                t = self._tarc_from[1] if self._tarc_from else (1.0, 0.0)
                self.view.preview = [np.array([pl.to3d(*p0), pl.to3d(p0[0] + t[0], p0[1] + t[1])])]
            else:
                (cu, cv), _, r, sweep = hit
                a0 = math.atan2(p0[1] - cv, p0[0] - cu)
                turn = math.radians(sweep) * (1 if r > 0 else -1)
                self.view.preview = [np.array([pl.to3d(cu + abs(r) * math.cos(a0 + turn * k),
                                                       cv + abs(r) * math.sin(a0 + turn * k))
                                               for k in np.linspace(0, 1, 64)])]
        elif self.tool == "arc":
            if self._arc_start is None:  # the radius, and the circle the arc will lie on
                self.view.preview = [np.array([pl.to3d(*p0), pl.to3d(*uv)]),
                                     np.array(Viewport._circle_world(pl, p0, math.dist(p0, uv)))]
            else:
                su, sv = self._arc_start[1]
                r = math.dist(p0, (su, sv))
                a0, sweep = math.atan2(sv - p0[1], su - p0[0]), math.radians(self._arc_sweep_now())
                arc = [pl.to3d(p0[0] + r * math.cos(a0 + sweep * t), p0[1] + r * math.sin(a0 + sweep * t))
                       for t in np.linspace(0, 1, 64)]
                self.view.preview = [np.array([pl.to3d(*p0), pl.to3d(su, sv)]), np.array(arc),
                                     np.array([pl.to3d(*p0), arc[-1]])]
        self.view.update()
        u = self.unit
        snapped = f"   ·   snap: {self._snap_text}" if self._snap_text else ""
        self.statusBar().showMessage(f"u={u.from_mm(uv[0]):.{u.decimals}f}  v={u.from_mm(uv[1]):.{u.decimals}f} {u.name}"
                                     + snapped)

    # -- trim
    TRIM_PX = 8  # how close (screen pixels) the pointer must be to a curve to trim it

    def _trim_tol(self, x, y) -> float:
        a, b = self.view.sketch_uv(x, y), self.view.sketch_uv(x + 1, y)
        return self.TRIM_PX * (math.dist(a, b) if a is not None and b is not None else 0.1)

    def _show_trim(self, pieces):
        pl = self.sketch.plane
        self.view.trim_preview = [np.array([pl.to3d(*q) for q in p.polyline]) for p in pieces]
        self.view.update()

    def _trim_at(self, uvs, tol) -> bool:
        """Trim the piece under each point of uvs in turn (a click is one point, a drag many).
        If the sketch can't be solved afterwards, nothing is trimmed. Returns whether anything was."""
        sk = self.sketch
        snapshot = sk.to_dict()
        dropped, done = [], 0
        for uv in uvs:
            piece = piece_at(sk, uv, tol)
            if piece is None:
                continue
            result = trim(sk, piece)
            self._auto_constrain(result.keep)
            dropped += result.dropped
            done += 1
        self._show_trim([])
        if not done:
            if len(uvs) == 1:
                self.statusBar().showMessage("Nothing to trim here: point at a line, circle or arc.", 4000)
            return False
        if not sk.solve():
            self._restore_sketch(snapshot, self.sketch_index)
            self.statusBar().showMessage("Trim undone: the sketch can't be solved without that piece.", 8000)
            return False
        self.view.sketch_sel.clear()
        msg = f"Trimmed {done} piece{'s' if done > 1 else ''}."
        if dropped:
            names = sorted({self.CONSTRAINT_LABELS.get(k, k.replace('_', ' ').capitalize()) for k in dropped})
            msg += " Removed constraints that no longer applied: " + ", ".join(names) + "."
        self._solve_sketch()
        self.statusBar().showMessage(msg, 8000)
        return True

    def _trim_stroke_event(self, phase, x, y):
        """Drag across pieces: each one the stroke passes over is trimmed (shown red as it goes)."""
        uv = self.view.sketch_uv(x, y)
        if phase == "press":
            self._trim_stroke = [] if uv is None else [uv]
            self._trim_stroke_tol = self._trim_tol(x, y)
            return
        if uv is None or not self._trim_stroke:
            return
        tol = self._trim_stroke_tol
        if phase == "move":
            last = self._trim_stroke[-1]
            steps = max(1, int(math.dist(last, uv) / (tol / 2)))
            self._trim_stroke += [(last[0] + (uv[0] - last[0]) * k / steps, last[1] + (uv[1] - last[1]) * k / steps)
                                  for k in range(1, steps + 1)]
            self._show_trim(self._stroke_pieces(self._trim_stroke, tol))
            return
        stroke, self._trim_stroke = self._trim_stroke, []
        if len(stroke) > 1:  # (a click without dragging is trimmed by _sketch_click)
            # the pieces as they were when crossed, each trimmed at its own middle
            middles = [p.polyline[len(p.polyline) // 2] for p in self._stroke_pieces(stroke, tol)]
            self._push_sketch_undo()
            if not self._trim_at(middles, tol):
                self._sketch_undo.get(self.sketch_index, [None]).pop()

    def _stroke_pieces(self, stroke, tol):
        """The different pieces a stroke passes over, in the order it meets them."""
        seen, pieces = set(), []
        for q in stroke:
            p = piece_at(self.sketch, q, tol)
            if p is None:
                continue
            key = (p.kind, p.index, p.lo.uv if p.lo else None, p.hi.uv if p.hi else None)
            if key not in seen:
                seen.add(key)
                pieces.append(p)
        return pieces

    # -- offset
    def offset_selection(self):
        """A parallel copy of the selected lines, arcs and circles, at a distance set in the settings
        bar (typed or scrubbed, previewed live). It stays at that distance as the sketch changes."""
        if self.mode != "sketch":
            return
        self._settle_session()
        sk, index = self.sketch, self.sketch_index
        picked = [(k, i) for k, i in self.view.sketch_sel if k in ("l", "c", "a")]
        if not picked:
            self.statusBar().showMessage("Select lines, arcs or circles to offset (double-click picks a "
                                         "whole outline).", 6000)
            return
        found, circles = chains(sk, picked)
        closed = any(ch.closed for ch in found) or (circles and not found)
        sides = [("out", "Outside"), ("in", "Inside")] if closed else [("left", "Left"), ("right", "Right")]
        self._push_sketch_undo()
        snapshot = sk.to_dict()

        def put_back():
            self._sketch_restoring = True
            try:
                self._restore_sketch(snapshot, index)
            finally:
                self._sketch_restoring = False

        last = {"distance": self._last_offset}

        def preview(values):
            put_back()
            last["distance"] = values["distance"]
            try:
                result = offset(sk, picked, values["distance"], flip=values["side"] in ("in", "right"))
            except OffsetError as e:
                sk.solve()
                self._update_regions()
                return f"Can't offset: {e}."
            self._auto_constrain(result.keep)
            sk.version += 1
            if not self._solve_sketch():
                return f"Can't solve the sketch ({sk.status})"
            self.view.sketch_sel = set(result.items)
            return None

        def restore():
            put_back()
            stack = self._sketch_undo.get(index)
            if stack:
                stack.pop()  # the snapshot taken when the offset started
            sk.version += 1
            self._solve_sketch()

        def done():
            self._last_offset = last["distance"]
            self.view.sketch_sel.clear()
            self._update_regions()

        items = [Number("distance", "Distance", self._last_offset, lo=1e-6, hi=1e5),
                 Choice("side", "Side", sides, sides[0][0])]
        self._open_session("Offset", items, preview, lambda: None, restore, done)

    def toggle_construction(self):
        """Q: selected lines, circles and arcs become reference geometry, or normal again if they all are."""
        if self.mode != "sketch":
            return
        sk = self.sketch
        curves = [(k, i) for k, i in self.view.sketch_sel if k in ("l", "c", "a")]
        if not curves:
            self.statusBar().showMessage("Select lines, circles or arcs to make them construction geometry.", 6000)
            return
        self._push_sketch_undo()
        on = not all(sk.is_construction(k, i) for k, i in curves)
        sk.set_construction(curves, on)
        sk.version += 1
        self.statusBar().showMessage(f"{len(curves)} item(s) {'made construction' if on else 'made normal'}.", 4000)
        self.mark_dirty()
        self._update_regions()
        self.view.update()

    def delete_selection(self):
        if self.mode != "sketch":
            return
        if self.view.sel_constraint is not None and not self.view.sketch_sel:
            self.delete_dimension(self.view.sel_constraint)
            return
        self._push_sketch_undo()
        sk = self.sketch
        for kind in ("l", "c", "a", "p"):  # deleting points last keeps the other indices stable
            for idx in sorted((i for k, i in self.view.sketch_sel if k == kind), reverse=True):
                sk.delete(kind, idx)
        sk.version += 1
        self.view.sketch_sel.clear()
        self._solve_sketch()

    def _sketch_chain(self, hit) -> set:
        """The lines and arcs joined end to end (through shared points) with a sketch line, arc or
        point; a circle alone."""
        kind, i = hit
        if kind == "c":
            return {hit}
        sk = self.sketch
        ends = {("l", j): set(ab) for j, ab in enumerate(sk.lines)}
        ends.update({("a", j): {a, b} for j, (_, a, b) in enumerate(sk.arcs)})
        points = {i} if kind == "p" else set(ends[hit])
        chain = set() if kind == "p" else {hit}
        grew = True
        while grew:
            grew = False
            for item, pts in ends.items():
                if item not in chain and pts & points:
                    chain.add(item)
                    points |= pts
                    grew = True
        return chain or {hit}
