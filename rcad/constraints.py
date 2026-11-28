"""Sketch constraints: adding them (refusing ones that clash), dimensions edited on the sketch,
constraint marks, and mirroring."""
from __future__ import annotations

import math

from PySide6.QtCore import QPoint, QTimer
from PySide6.QtWidgets import QMenu

from .dialogs import ExpressionDialog
from .sketch import Constraint, CONSTRAINT_KINDS, reflect_point_across_line
from .tool_bar import Number


class ConstraintsMixin:
    """Sketch constraints: adding them (refusing ones that clash), dimensions edited on the sketch, constraint marks, and mirroring. Mixed into MainWindow."""

    # ---- quick buttons beside the selection
    CONSTRAINT_LABELS = {
        "horizontal": "Horizontal", "vertical": "Vertical", "coincident": "Coincident", "parallel": "Parallel",
        "perpendicular": "Perpendicular", "equal": "Equal", "distance": "Distance", "spacing": "Spacing",
        "diameter": "Diameter", "expression": "Expression", "fixed": "Fixed", "point_on_line": "Point on line",
        "midpoint": "Midpoint", "point_on_circle": "Point on circle", "point_on_arc": "Point on arc",
        "radius": "Radius", "tangent": "Tangent", "concentric": "Concentric",
    }

    def _sketch_quick_kinds(self, pts, lines, circles, arcs=()) -> list[str]:
        """Constraints that fit a sketch selection, likeliest first (same rules as add_constraint)."""
        if arcs:
            if len(arcs) == 1 and not pts and not lines and not circles:
                return ["radius"]
            if len(arcs) == 1 and len(lines) == 1 and not pts and not circles:
                return ["tangent"]
            if len(arcs) == 2 and not pts and not lines and not circles:
                return ["tangent", "concentric"]
            if len(arcs) + len(circles) == 2 and not pts and not lines:
                return ["concentric"]
            if len(pts) == 1 and len(arcs) == 1 and not lines and not circles:
                return ["point_on_arc"]
            return []
        if len(circles) == 2 and not pts and not lines:
            return ["concentric"]
        if lines and not pts and not circles:
            if len(lines) == 1:
                return ["distance", "horizontal", "vertical"]
            if len(lines) == 2:
                return ["parallel", "perpendicular", "equal", "spacing"]
        if pts and not lines and not circles:
            if len(pts) == 2:
                return ["coincident", "distance", "horizontal", "vertical", "expression", "fixed"]
            return ["fixed"]
        if len(pts) == 1 and len(lines) == 1 and not circles:
            return ["point_on_line"]
        if len(circles) == 1 and not pts and not lines:
            return ["diameter"]
        return []

    # ---- constraints
    def add_constraint(self, kind):
        self._push_sketch_undo()
        sk = self.sketch
        sel = sorted(self.view.sketch_sel)
        pts = [i for k, i in sel if k == "p"]
        lines = [i for k, i in sel if k == "l"]
        circles = [i for k, i in sel if k == "c"]
        arcs = [i for k, i in sel if k == "a"]
        new: list[Constraint] = []
        value = None
        if arcs and kind not in ("radius", "tangent", "concentric", "point_on_arc", "symmetry", "fixed"):
            self.statusBar().showMessage(f"'{kind}' does not apply to arcs.", 6000)
            return
        try:
            if kind == "radius":
                if len(arcs) == 1 and not pts and not lines and not circles:
                    cur = sk.arc_geometry(arcs[0])[1]
                    return self._constraint_session("Radius", Constraint("radius", arcs, cur), new=True)
            elif kind == "tangent":
                if len(lines) == 1 and len(arcs) == 1 and not pts and not circles:
                    if not set(sk.lines[lines[0]]) & set(sk.arcs[arcs[0]][1:]):
                        self.statusBar().showMessage("The line must end where the arc ends.", 6000)
                        return
                    new = [Constraint("tangent", [lines[0], arcs[0]])]
                elif len(arcs) == 2 and not pts and not lines and not circles:
                    if not set(sk.arcs[arcs[0]][1:]) & set(sk.arcs[arcs[1]][1:]):
                        self.statusBar().showMessage("The arcs must share an end.", 6000)
                        return
                    new = [Constraint("tangent_arcs", arcs)]
            elif kind == "concentric":
                centres = [sk.circles[i][0] for i in circles] + [sk.arcs[i][0] for i in arcs]
                if len(centres) == 2 and not pts and not lines and centres[0] != centres[1]:
                    new = [Constraint("coincident", centres)]
            elif kind == "point_on_arc":
                if len(pts) == 1 and len(arcs) == 1 and not lines and not circles:
                    new = [Constraint("point_on_arc", [pts[0], arcs[0]])]
            elif kind in ("horizontal", "vertical"):
                if len(lines) == 1 and not pts and not circles:
                    new = [Constraint(kind, lines)]
                elif len(pts) == 2 and not lines and not circles:
                    new = [Constraint("p" + kind, pts)]
            elif kind == "coincident" and len(pts) == 2 and not lines and not circles:
                new = [Constraint(kind, pts)]
            elif kind in ("parallel", "perpendicular", "equal") and len(lines) == 2 and not pts and not circles:
                new = [Constraint(kind, lines)]
            elif kind == "spacing" and len(lines) == 2 and not pts and not circles:
                a = sk.points[sk.lines[lines[0]][0]]
                b = sk.points[sk.lines[lines[1]][0]]
                c = sk.points[sk.lines[lines[1]][1]]
                ax, ay = a
                (x1, y1), (x2, y2) = b, c
                cur = abs((x2 - x1) * (y1 - ay) - (x1 - ax) * (y2 - y1)) / max(math.hypot(x2 - x1, y2 - y1), 1e-12)
                return self._constraint_session(
                    "Spacing", Constraint("spacing", [sk.lines[lines[0]][0], lines[1]], cur), new=True)
            elif kind == "expression" and len(pts) == 2 and not lines and not circles:
                dialog = ExpressionDialog(self, "(e.g. 2*a + 3)π")
                if not dialog.exec():
                    return
                expr = dialog.expression()
                if not expr:
                    return
                new = [Constraint("expression", [pts[0], pts[1]], expression=expr)]
                self.statusBar().showMessage(f"Expression: {expr}", 6000)
            elif kind == "distance":
                ends = pts if len(pts) == 2 and not lines else list(sk.lines[lines[0]]) if len(lines) == 1 and not pts else None
                if ends:
                    cur = math.dist(sk.points[ends[0]], sk.points[ends[1]])
                    return self._constraint_session("Distance", Constraint(kind, ends, cur), new=True)
            elif kind == "diameter" and len(circles) == 1 and not pts and not lines:
                cur = 2 * sk.circles[circles[0]][1]
                return self._constraint_session("Diameter", Constraint(kind, circles, cur), new=True)
            elif kind == "fixed" and pts and not lines and not circles:
                new = [Constraint(kind, [p]) for p in pts]
            elif kind == "point_on_line" and len(pts) == 1 and len(lines) == 1 and not circles:
                new = [Constraint(kind, [pts[0], lines[0]])]
            elif kind == "symmetry":
                # Two-step mirror: set mirror line
                if self._mirror_axis is None:
                    # Expect exactly one line in selection
                    lines = [i for k, i in sel if k == "l"]
                    if len(lines) == 1 and not pts and not circles:
                        line_idx = lines[0]
                        line_pts = sk.lines[line_idx]
                        a_idx, b_idx = line_pts
                        a = sk.points[a_idx]
                        b = sk.points[b_idx]
                        self._mirror_axis = (a_idx, b_idx, line_idx)
                        self._mirroring = True
                        self.statusBar().showMessage(
                            "Select points/lines to mirror across this line, then press Symmetry again.",
                            6000,
                        )
                        # Keep the selection (so the user can add more items)
                        return
                    else:
                        self.statusBar().showMessage(
                            "To set a mirror axis, select exactly one line first.", 6000
                        )
                        return
                else:
                    # We have a mirror axis; mirror the selected items
                    if not pts and not lines and not circles and not arcs:
                        self.statusBar().showMessage(
                            "Select points, lines, circles or arcs to mirror (or press Escape to cancel).", 6000
                        )
                        return
                    self._push_sketch_undo()
                    sk = self.sketch
                    a_idx, b_idx, mirror_line_idx = self._mirror_axis
                    # Exclude the mirror line and its endpoints from the items to mirror
                    lines = [i for i in lines if i != mirror_line_idx]
                    pts = [i for i in pts if i not in (a_idx, b_idx)]
                    circles = [i for i in circles if not (sk.circles[i][0] in (a_idx, b_idx))]
                    a = sk.points[a_idx]
                    b = sk.points[b_idx]
                    new_constraints = []
                    # Fix only the mirror line endpoints so the axis stays in place
                    new_constraints.append(Constraint("fixed", [a_idx]))
                    new_constraints.append(Constraint("fixed", [b_idx]))

                    mirrored_points = {}

                    def mirrored_point(point_idx):
                        if point_idx not in mirrored_points:
                            reflected = reflect_point_across_line(sk.points[point_idx], a, b)
                            mirrored_points[point_idx] = sk.add_point(reflected[0], reflected[1])
                            new_constraints.append(
                                Constraint("symmetry", [point_idx, mirrored_points[point_idx], mirror_line_idx])
                            )
                        return mirrored_points[point_idx]

                    for p in pts:
                        mirrored_point(p)

                    for l_idx in lines:
                        (p1, p2) = sk.lines[l_idx]
                        new_p1_idx = mirrored_point(p1)
                        new_p2_idx = mirrored_point(p2)
                        sk.add_line(new_p1_idx, new_p2_idx)

                    for ci in circles:
                        center_idx, r = sk.circles[ci]
                        new_center_idx = mirrored_point(center_idx)
                        sk.circles.append((new_center_idx, r))

                    for ai in arcs:  # a mirror image runs the other way round: swap its ends
                        c0, s0, e0 = sk.arcs[ai]
                        sk.add_arc(mirrored_point(c0), mirrored_point(e0), mirrored_point(s0))

                    if new_constraints:
                        sk.constraints += new_constraints
                        sk.version += 1
                        self._mirror_axis = None
                        self._mirroring = False
                        self.view.sketch_sel.clear()
                        self._solve_sketch()
                        self.statusBar().showMessage(
                            f"Mirrored {len(pts) + len(lines) + len(circles) + len(arcs)} item(s).", 6000
                        )
                    else:
                        self.statusBar().showMessage(
                            "No items selected to mirror.", 6000
                        )
                    return
                # Fallback to old behavior (two points + one line)
                new = [Constraint("symmetry", [pts[0], pts[1], lines[0]])]
        except IndexError:
            new = []
        if not new:
            needs = {"concentric": "two circles or arcs"}.get(kind) or CONSTRAINT_KINDS[kind][0]
            self.statusBar().showMessage(f"Select the right items for '{kind}' ({needs}).", 6000)
            return
        self._add_strict(new, kind)

    def _add_strict(self, new: list[Constraint], kind: str):
        """Add constraints only if the sketch can take them as it is: one that contradicts or repeats
        others is refused with their names (shown red on the sketch), and so is one that would shrink
        a line, circle or arc to nothing."""
        sk = self.sketch
        n0, before = len(sk.constraints), sk.collapsed()
        snapshot = [list(p) for p in sk.points], list(sk.circles)
        sk.constraints += new
        sk.variables = self.doc.variables
        ok = sk.solve()
        squashed = sk.collapsed() - before if ok else set()
        if ok and not squashed:
            sk.version += 1
            self.view.conflicts = set()
            self._solve_sketch()
            return True
        clash = [i for i in sk.failed if i < n0] if not ok else []
        del sk.constraints[n0:]
        sk.points, sk.circles = snapshot
        if not ok and not clash:  # the solver named only the new one: find what it clashes with
            clash = sk.clashing(new)
        sk.solve()
        label = self.CONSTRAINT_LABELS.get(kind, kind.replace("_", " ").capitalize())
        if squashed:
            names = {"l": "line", "c": "circle", "a": "arc"}
            what = ", ".join(sorted({names[k] for k, _ in squashed}))
            msg = f"{label} refused: it would shrink a {what} to nothing."
        elif clash:
            msg = (f"{label} refused: it contradicts or repeats "
                   + ", ".join(self._constraint_name(sk.constraints[i]) for i in clash) + " (shown in red).")
            self._flash_conflicts(clash)
        else:
            msg = f"{label} refused: no shape fits it together with the sketch's other constraints."
        self.statusBar().showMessage(msg, 8000)
        self.refresh_constraints()
        self._update_regions()
        return False

    def edit_constraint(self, item):
        self.edit_dimension(self.constraints.row(item))

    def edit_dimension(self, i: int):
        """Change constraint i's value (a click on its label, or a double-click in the list)."""
        if self.mode != "sketch" or not 0 <= i < len(self.sketch.constraints):
            return
        self._settle_session()
        if self.pending is not None:
            self.set_tool(self.tool)  # drop a half-drawn shape
        self.constraints.setCurrentRow(i)
        sk = self.sketch
        c = sk.constraints[i]
        if c.expression is not None:
            dialog = ExpressionDialog(self, c.expression)
            if dialog.exec():
                expr = dialog.expression()
                if expr:
                    self._push_sketch_undo()
                    old = c.expression
                    c.expression = expr
                    sk.version += 1
                    if not self._solve_sketch():
                        c.expression = old
                        self.sketch.solve()
                        self.refresh_constraints()
            return
        if c.value is None:
            return
        self._push_sketch_undo()
        self._constraint_session(self.CONSTRAINT_LABELS.get(c.kind, c.kind.title()), c, new=False)

    def delete_dimension(self, i: int):
        """Remove constraint i (a dimension or a mark)."""
        if self.mode != "sketch" or not 0 <= i < len(self.sketch.constraints):
            return
        self.view.sel_constraint = None
        self.view.conflicts = set()
        self._settle_session()
        self._push_sketch_undo()
        group = self.sketch.constraints[i].group
        if group:  # one of an offset's distances: the offset stops following, all of it together
            self.sketch.constraints = [c for c in self.sketch.constraints if c.group != group]
        else:
            self.sketch.remove_constraint(i)
        self.sketch.version += 1
        self.view.set_dim_hover(None)
        self._solve_sketch()

    def select_constraint(self, i: int):
        """A click on a constraint mark: select it (again: drop it), so Delete removes it."""
        if self.mode != "sketch" or not 0 <= i < len(self.sketch.constraints):
            return
        v = self.view
        v.sel_constraint = None if v.sel_constraint == i else i
        v.sketch_sel.clear()
        if v.sel_constraint is not None:
            self.constraints.setCurrentRow(i)
            c = self.sketch.constraints[i]
            self.statusBar().showMessage(f"{self._constraint_name(c)} selected: Delete removes it.", 4000)
        v._restyle_dims()
        self._update_regions()

    def _constraint_name(self, c: Constraint) -> str:
        name = self.CONSTRAINT_LABELS.get(c.kind, c.kind.replace("_", " ").capitalize())
        return f"{name} {self.unit.fmt(c.value)}" if c.value is not None else name

    def _toggle_marks(self, on: bool):
        self.view.show_marks = bool(on)
        self.view.update()

    def _flash_conflicts(self, cids):
        """Show the constraints a refused one clashed with in red for a few seconds."""
        self.view.conflicts = set(cids)
        self.view._restyle_dims()
        self.view.update()
        token = object()
        self._conflict_token = token

        def clear():
            if getattr(self, "_conflict_token", None) is token:
                self.view.conflicts = set()
                self.view._restyle_dims()
                self.view.update()
        QTimer.singleShot(4000, clear)

    def _dimension_menu(self, i: int, x: float, y: float):
        """Right-click on a dimension's label: change it or delete it."""
        if self.mode != "sketch" or not 0 <= i < len(self.sketch.constraints):
            return
        c = self.sketch.constraints[i]
        menu = QMenu(self.view)
        sized = c.value is not None or c.expression is not None
        if sized:
            menu.addAction("✎ Change", lambda: self.edit_dimension(i))
        menu.addAction("🗑 Delete dimension" if sized else "🗑 Delete constraint", lambda: self.delete_dimension(i))
        self.dimension_menu = menu
        menu.popup(self.view.mapToGlobal(QPoint(int(x), int(y))))

    def delete_constraint(self):
        i = self.constraints.currentRow()
        if i >= 0:
            self._push_sketch_undo()
            self.sketch.remove_constraint(i)
            self.sketch.version += 1
            self._solve_sketch()

    def _constraint_session(self, title: str, c: Constraint, new: bool):
        """A sketch dimension in the bar: the sketch re-solves as the value changes.
        The caller has pushed the sketch undo snapshot; new constraints are added here."""
        sk, index = self.sketch, self.sketch_index
        snapshot = sk.to_dict()
        if new:
            sk.constraints.append(c)
        self.view.set_dim_active(next((i for i, x in enumerate(sk.constraints) if x is c), None))

        def preview(values):
            c.value = values["value"]
            for other in sk.constraints:  # an offset's distances change together
                if c.group and other.group == c.group and other.value is not None:
                    other.value = c.value
            sk.version += 1
            return None if self._solve_sketch() else f"Can't solve the sketch ({sk.status})"

        def restore():
            self.view.set_dim_active(None)
            self._sketch_restoring = True
            try:
                self._restore_sketch(snapshot, index)
            finally:
                self._sketch_restoring = False
            stack = self._sketch_undo.get(index)
            if stack:
                stack.pop()  # the snapshot taken when the dimension was started
            sk.version += 1
            self._solve_sketch()

        def done():
            self.view.set_dim_active(None)

        items = [Number("value", "Value", c.value, lo=1e-6, hi=1e6)]
        self._open_session(title, items, preview, lambda: None, restore, done)
