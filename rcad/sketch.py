"""2D sketches on a plane, solved with SolveSpace's libslvs."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from build123d import Face, Plane, Shape, Solid, Vector, Wire
from py_slvs import slvs

from .units import MM, Unit

BASE_GROUP, SKETCH_GROUP = 1, 2

# kind -> (what refs point at, needs a dimension value)
CONSTRAINT_KINDS = {
    "coincident": ("pp", False),
    "horizontal": ("l", False),
    "vertical": ("l", False),
    "phorizontal": ("pp", False),
    "pvertical": ("pp", False),
    "distance": ("pp", True),
    "parallel": ("ll", False),
    "perpendicular": ("ll", False),
    "equal": ("ll", False),
    "diameter": ("c", True),
    "fixed": ("p", False),
    "point_on_line": ("pl", False),
}


@dataclass
class SketchPlane:
    origin: tuple[float, float, float]
    xdir: tuple[float, float, float]
    normal: tuple[float, float, float]

    @property
    def ydir(self):
        n, x = self.normal, self.xdir
        return (n[1] * x[2] - n[2] * x[1], n[2] * x[0] - n[0] * x[2], n[0] * x[1] - n[1] * x[0])

    def to3d(self, u: float, v: float):
        y = self.ydir
        return tuple(self.origin[i] + u * self.xdir[i] + v * y[i] for i in range(3))

    def to_uv(self, p):
        d = [p[i] - self.origin[i] for i in range(3)]
        y = self.ydir
        return (sum(d[i] * self.xdir[i] for i in range(3)), sum(d[i] * y[i] for i in range(3)))

    def b123d(self) -> Plane:
        return Plane(origin=self.origin, x_dir=self.xdir, z_dir=self.normal)


def standard_plane(name: str) -> SketchPlane:
    return {
        "XY": SketchPlane((0, 0, 0), (1, 0, 0), (0, 0, 1)),
        "XZ": SketchPlane((0, 0, 0), (1, 0, 0), (0, -1, 0)),
        "YZ": SketchPlane((0, 0, 0), (0, 1, 0), (1, 0, 0)),
    }[name]


@dataclass
class Constraint:
    kind: str
    refs: list[int]
    value: float | None = None

    def label(self, unit: Unit = MM) -> str:
        v = f" = {unit.fmt(self.value)}" if self.value is not None else ""
        return f"{self.kind} {self.refs}{v}"


@dataclass
class Sketch:
    plane: SketchPlane
    points: list[list[float]] = field(default_factory=list)
    lines: list[tuple[int, int]] = field(default_factory=list)
    circles: list[tuple[int, float]] = field(default_factory=list)  # (center point, radius)
    constraints: list[Constraint] = field(default_factory=list)
    selection: list[tuple[float, float]] = field(default_factory=list)  # points inside the highlighted regions
    status: str = "unsolved"
    dof: int = -1

    # -- editing
    def add_point(self, u: float, v: float) -> int:
        self.points.append([u, v])
        return len(self.points) - 1

    def add_line(self, a: int, b: int) -> int:
        self.lines.append((a, b))
        return len(self.lines) - 1

    def add_circle(self, center: int, r: float) -> int:
        self.circles.append((center, r))
        return len(self.circles) - 1

    def add_rect(self, u0, v0, u1, v1):
        p = [self.add_point(u0, v0), self.add_point(u1, v0), self.add_point(u1, v1), self.add_point(u0, v1)]
        ls = [self.add_line(p[i], p[(i + 1) % 4]) for i in range(4)]
        self.constraints += [Constraint("horizontal", [ls[0]]), Constraint("horizontal", [ls[2]]),
                             Constraint("vertical", [ls[1]]), Constraint("vertical", [ls[3]])]

    def remove_constraint(self, i: int):
        del self.constraints[i]

    def delete(self, kind: str, idx: int):
        """Delete a point ('p'), line ('l') or circle ('c') with everything that depends on it."""
        if kind == "p":
            for i in sorted((i for i, (a, b) in enumerate(self.lines) if idx in (a, b)), reverse=True):
                self.delete("l", i)
            for i in sorted((i for i, (c, _) in enumerate(self.circles) if c == idx), reverse=True):
                self.delete("c", i)
            del self.points[idx]
            self.lines = [(a - (a > idx), b - (b > idx)) for a, b in self.lines]
            self.circles = [(c - (c > idx), r) for c, r in self.circles]
        elif kind == "l":
            del self.lines[idx]
        elif kind == "c":
            del self.circles[idx]
        kept = []
        for c in self.constraints:
            types = CONSTRAINT_KINDS[c.kind][0]
            if any(t == kind and r == idx for t, r in zip(types, c.refs)):
                continue
            c.refs = [r - (t == kind and r > idx) for t, r in zip(types, c.refs)]
            kept.append(c)
        self.constraints = kept

    # -- solving
    def solve(self) -> bool:
        """Solve in place; on failure positions are left untouched and status explains why."""
        s = slvs.System()
        g0, g = BASE_GROUP, SKETCH_GROUP
        o = s.addPoint3dV(0, 0, 0, g0)
        wp = s.addWorkplane(o, s.addNormal3dV(1, 0, 0, 0, g0), g0)
        pts = [s.addPoint2dV(wp, u, v, g) for u, v in self.points]
        lines = [s.addLineSegment(pts[a], pts[b], g) for a, b in self.lines]
        n2 = s.addNormal2d(wp, g)
        radii = [s.addDistanceV(r, g) for _, r in self.circles]
        circles = [s.addCircle(pts[c], n2, radii[i], g) for i, (c, _) in enumerate(self.circles)]
        handle_to_constraint = {}
        for ci, c in enumerate(self.constraints):
            r = c.refs
            k = c.kind
            if k == "coincident":
                h = s.addPointsCoincident(pts[r[0]], pts[r[1]], wp, g)
            elif k == "horizontal":
                h = s.addLineHorizontal(lines[r[0]], wp, g)
            elif k == "vertical":
                h = s.addLineVertical(lines[r[0]], wp, g)
            elif k == "phorizontal":
                h = s.addPointsHorizontal(pts[r[0]], pts[r[1]], wp, g)
            elif k == "pvertical":
                h = s.addPointsVertical(pts[r[0]], pts[r[1]], wp, g)
            elif k == "distance":
                h = s.addPointsDistance(c.value, pts[r[0]], pts[r[1]], wp, g)
            elif k == "parallel":
                h = s.addParallel(lines[r[0]], lines[r[1]], wp, g)
            elif k == "perpendicular":
                h = s.addPerpendicular(lines[r[0]], lines[r[1]], wp, g)
            elif k == "equal":
                h = s.addEqualLength(lines[r[0]], lines[r[1]], wp, g)
            elif k == "diameter":
                h = s.addDiameter(c.value, circles[r[0]], g)
            elif k == "fixed":
                h = s.addWhereDragged(pts[r[0]], wp, g)
            elif k == "point_on_line":
                h = s.addPointOnLine(pts[r[0]], lines[r[1]], wp, g)
            else:
                raise ValueError(f"unknown constraint {k}")
            handle_to_constraint[h] = ci

        res = s.solve(g, True)
        self.dof = s.Dof
        if res != 0:
            bad = sorted({handle_to_constraint[h] for h in s.Failed if h in handle_to_constraint})
            why = {1: "inconsistent", 2: "did not converge", 3: "too many unknowns"}.get(res, f"error {res}")
            self.status = f"{why}" + (f" (constraints {bad})" if bad else "")
            return False
        for i, h in enumerate(pts):
            self.points[i] = [s.getParam(s.getEntityParam(h, k)).val for k in range(2)]
        self.circles = [(c, s.getParam(s.getEntityParam(radii[i], 0)).val) for i, (c, _) in enumerate(self.circles)]
        self.status = "fully constrained" if self.dof == 0 else f"{self.dof} DOF"
        return True

    # -- profiles
    def loops(self) -> list[tuple[str, object]]:
        """Closed profiles: ('poly', [(u,v)...]) or ('circle', (cu, cv, r))."""
        out: list[tuple[str, object]] = [("circle", (*self.points[c], r)) for c, r in self.circles if r > 1e-9]
        at: dict[int, list[int]] = {}
        for li, (a, b) in enumerate(self.lines):
            at.setdefault(a, []).append(li)
            at.setdefault(b, []).append(li)
        used: set[int] = set()
        for start in range(len(self.lines)):
            if start in used:
                continue
            a, b = self.lines[start]
            first, cur = a, b
            ok = True
            seq = [start]
            verts = [a, b]
            while cur != first:
                nxt = [x for x in at[cur] if x != seq[-1]]
                if len(at[cur]) != 2 or len(nxt) != 1 or nxt[0] in seq:
                    ok = False
                    break
                seq.append(nxt[0])
                p, q = self.lines[nxt[0]]
                cur = q if p == cur else p
                verts.append(cur)
            if ok and len(seq) >= 3:
                used.update(seq)
                out.append(("poly", [tuple(self.points[v]) for v in verts[:-1]]))
        return out

    def regions(self) -> list[tuple[tuple, list[tuple]]]:
        """Fillable areas as (outer loop, hole loops): even-depth loops with their direct children as holes."""
        loops = self.loops()
        areas = [_area(l) for l in loops]
        containers = [[j for j, o in enumerate(loops) if j != i and areas[j] > areas[i] and _inside(l, o)]
                      for i, l in enumerate(loops)]
        parent = [min(c, key=lambda j: areas[j]) if c else None for c in containers]
        return [(loops[i], [loops[j] for j in range(len(loops)) if parent[j] == i])
                for i in range(len(loops)) if len(containers[i]) % 2 == 0]

    def region_at(self, u: float, v: float) -> int | None:
        """Index into regions() of the filled area under (u, v); None over empty space or holes."""
        regs = self.regions()
        best, best_area = None, math.inf
        for i, (outer, holes) in enumerate(regs):
            if _contains(outer, (u, v)) and not any(_contains(h, (u, v)) for h in holes):
                a = _area(outer)
                if a < best_area:
                    best, best_area = i, a
        return best

    def selected_regions(self) -> set[int]:
        return {i for p in self.selection if (i := self.region_at(*p)) is not None}

    def toggle_region(self, u: float, v: float, additive: bool) -> None:
        idx = self.region_at(u, v)
        if idx is None:
            if not additive:
                self.selection = []
            return
        if idx in self.selected_regions():
            self.selection = [p for p in self.selection if self.region_at(*p) != idx]
        else:
            self.selection = (self.selection if additive else []) + [(u, v)]

    def _region_face(self, region) -> Face:
        outer, holes = region
        return Face(_loop_wire(outer), [_loop_wire(h) for h in holes])

    def region_triangles(self, index: int) -> np.ndarray:
        """World-space triangles (N,3,3) filling region `index`, for highlighting."""
        face = self.plane.b123d().from_local_coords(self._region_face(self.regions()[index]))
        verts, tris = face.tessellate(0.05)
        v = np.array([[p.X, p.Y, p.Z] for p in verts], dtype=float)
        return v[np.array(tris, dtype=int)].reshape(-1, 3, 3)

    def solid(self, distance: float, profiles: list | None = None) -> Shape | None:
        """Extrude filled regions along the plane normal; `profiles` (points in the sketch) limits which ones."""
        regs = self.regions()
        if profiles is not None:
            keep = {i for p in profiles if (i := self.region_at(*p)) is not None}
            regs = [r for i, r in enumerate(regs) if i in keep]
        if not regs or abs(distance) < 1e-9:
            return None
        pl = self.plane.b123d()
        n = self.plane.normal
        d = Vector(float(n[0] * distance), float(n[1] * distance), float(n[2] * distance))
        result = None
        for region in regs:
            solid = Solid.extrude(pl.from_local_coords(self._region_face(region)), d)
            result = solid if result is None else result.fuse(solid)
        return result

    # -- persistence
    def to_dict(self) -> dict:
        return {"plane": asdict(self.plane), "points": self.points, "lines": [list(l) for l in self.lines],
                "circles": [list(c) for c in self.circles],
                "constraints": [asdict(c) for c in self.constraints], "selection": [list(p) for p in self.selection]}

    @staticmethod
    def from_dict(d: dict) -> "Sketch":
        p = d["plane"]
        sk = Sketch(SketchPlane(tuple(p["origin"]), tuple(p["xdir"]), tuple(p["normal"])))
        sk.points = [list(p) for p in d["points"]]
        sk.lines = [tuple(l) for l in d["lines"]]
        sk.circles = [tuple(c) for c in d["circles"]]
        sk.constraints = [Constraint(**c) for c in d["constraints"]]
        sk.selection = [tuple(p) for p in d.get("selection", [])]
        return sk


def _area(loop) -> float:
    if loop[0] == "circle":
        return math.pi * loop[1][2] ** 2
    p = loop[1]
    return abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1] for i in range(len(p)))) / 2


def _probe(loop):
    return (loop[1][0] + loop[1][2], loop[1][1]) if loop[0] == "circle" else loop[1][0]


def _inside(loop, outer) -> bool:
    return _contains(outer, _probe(loop))


def _contains(outer, pt) -> bool:
    x, y = pt
    if outer[0] == "circle":
        u, v, r = outer[1]
        return math.hypot(x - u, y - v) < r
    p, inside = outer[1], False
    for i in range(len(p)):
        (x1, y1), (x2, y2) = p[i], p[(i + 1) % len(p)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _loop_wire(loop) -> Wire:
    if loop[0] == "circle":
        u, v, r = loop[1]
        return Wire.make_circle(r, Plane(origin=(u, v, 0)))
    return Wire.make_polygon([Vector(float(u), float(v), 0.0) for u, v in loop[1]])
