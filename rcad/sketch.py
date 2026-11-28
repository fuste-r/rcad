"""2D sketches on a plane, solved with SolveSpace's libslvs."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from build123d import Compound, Edge, Face, Plane, Shape, Solid, Vector
from build123d.topology.shape_core import List_TopoDS_Shape
from OCP.BRepAlgoAPI import BRepAlgoAPI_Splitter
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.gp import gp_Pnt
from OCP.TopAbs import TopAbs_IN, TopAbs_INTERNAL
from py_slvs import slvs

from .units import MM, Unit
from .expression_eval import _evaluate_expression

BASE_GROUP, SKETCH_GROUP = 1, 2

# kind -> (what refs point at, needs a dimension value)
CONSTRAINT_KINDS = {
    "coincident": ("pp", False),
    "horizontal": ("l", False),
    "vertical": ("l", False),
    "phorizontal": ("pp", False),
    "pvertical": ("pp", False),
    "distance": ("pp", True),
    "spacing": ("pl", True),
    "parallel": ("ll", False),
    "perpendicular": ("ll", False),
    "equal": ("ll", False),
    "diameter": ("c", True),
    "fixed": ("p", False),
    "point_on_line": ("pl", False),
    "midpoint": ("pl", False),
    "point_on_circle": ("pc", False),
    "point_on_arc": ("pa", False),
    "radius": ("a", True),
    "tangent": ("la", False),  # a line and an arc that share an end point
    "tangent_arcs": ("aa", False),  # two arcs that share an end point
    "expression": ("pp", True),
    "symmetry": ("ppl", False),
}

def reflect_point_across_line(p, a, b):
    """Reflect point p across the line from a to b (2D coordinates).

    p, a, b are (x, y) tuples/sequences.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    den = dx * dx + dy * dy
    if den == 0:
        return list(p)
    t = ((px - ax) * dx + (py - ay) * dy) / den
    foot_x = ax + t * dx
    foot_y = ay + t * dy
    return [2 * foot_x - px, 2 * foot_y - py]


def _mirror_points_across_line(points, a, b):
    """Return reflected coordinates for a list of points across line a-b."""
    return [reflect_point_across_line(p, a, b) for p in points]



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
    expression: str | None = None  # expression using variables, when not None

    def label(self, unit: Unit = MM) -> str:
        v = f" = {unit.fmt(self.value)}" if self.value is not None else ""
        return f"{self.kind} {self.refs}{v}"


@dataclass
class Sketch:
    plane: SketchPlane
    points: list[list[float]] = field(default_factory=list)
    lines: list[tuple[int, int]] = field(default_factory=list)
    circles: list[tuple[int, float]] = field(default_factory=list)  # (center point, radius)
    arcs: list[tuple[int, int, int]] = field(default_factory=list)  # (center, start, end), counter-clockwise
    construction: set = field(default_factory=set)  # ("l" | "c" | "a", index): reference geometry, no regions
    constraints: list[Constraint] = field(default_factory=list)
    selection: list[tuple[float, float]] = field(default_factory=list)  # points inside the highlighted regions
    variables: dict[str, float] = field(default_factory=dict)  # expression variables
    status: str = "unsolved"
    dof: int = -1
    version: int = 0  # incremented on edit to enable shape caching
    failed: list = field(default_factory=list, repr=False, compare=False)  # constraints the last failed solve named
    _region_cache: tuple | None = field(default=None, init=False, repr=False, compare=False)

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

    def add_arc(self, center: int, start: int, end: int) -> int:
        """An arc around `center` running counter-clockwise from `start` to `end` (radius: center to start)."""
        self.arcs.append((center, start, end))
        return len(self.arcs) - 1

    def arc_geometry(self, i: int):
        """(center uv, radius, start angle, sweep) of arc i, angles in radians, sweep in (0, 2 pi]."""
        c, s, e = self.arcs[i]
        cu, cv = self.points[c]
        a0 = math.atan2(self.points[s][1] - cv, self.points[s][0] - cu)
        a1 = math.atan2(self.points[e][1] - cv, self.points[e][0] - cu)
        sweep = (a1 - a0) % (2 * math.pi)
        return (cu, cv), math.dist(self.points[c], self.points[s]), a0, sweep or 2 * math.pi

    def arc_polyline(self, i: int, n: int = 48) -> list[tuple[float, float]]:
        """Points along arc i in sketch coordinates, from its start to its end."""
        (cu, cv), r, a0, sweep = self.arc_geometry(i)
        return [(cu + r * math.cos(a0 + sweep * t), cv + r * math.sin(a0 + sweep * t))
                for t in np.linspace(0, 1, n)]

    def _constraint_points(self, c: Constraint) -> set:
        """The points constraint c holds, directly or through its lines, circles and arcs."""
        out = set()
        for t, r in zip(CONSTRAINT_KINDS[c.kind][0], c.refs):
            if t == "p":
                out.add(r)
            elif t == "l":
                out |= set(self.lines[r])
            elif t == "c":
                out.add(self.circles[r][0])
            elif t == "a":
                out |= set(self.arcs[r])
        return out

    def clashing(self, new: list[Constraint]) -> list[int]:
        """Existing constraints that stop `new` from fitting: each one whose removal lets the sketch
        solve with `new` added. Only constraints sharing points with `new` are tried."""
        touched = set().union(*(self._constraint_points(c) for c in new))
        out = []
        for i, c in enumerate(self.constraints):
            if not self._constraint_points(c) & touched:
                continue
            trial = Sketch.from_dict(self.to_dict())
            trial.variables = self.variables
            trial.constraints = [x for j, x in enumerate(trial.constraints) if j != i] + \
                [Constraint(**asdict(x)) for x in new]
            try:
                if trial.solve():
                    out.append(i)
            except ValueError:
                pass
        return out

    def collapsed(self, tol: float = 1e-6) -> set:
        """Lines, circles and arcs shrunk to nothing, as ("l" | "c" | "a", index)."""
        out = {("l", i) for i, (a, b) in enumerate(self.lines) if math.dist(self.points[a], self.points[b]) < tol}
        out |= {("c", i) for i, (_, r) in enumerate(self.circles) if r < tol}
        out |= {("a", i) for i, (c, a, _) in enumerate(self.arcs) if math.dist(self.points[c], self.points[a]) < tol}
        return out

    def is_construction(self, kind: str, i: int) -> bool:
        return (kind, i) in self.construction

    def set_construction(self, items, on: bool):
        for k, i in items:
            if k in ("l", "c", "a"):
                (self.construction.add if on else self.construction.discard)((k, i))

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
            for i in sorted((i for i, arc in enumerate(self.arcs) if idx in arc), reverse=True):
                self.delete("a", i)
            del self.points[idx]
            self.lines = [(a - (a > idx), b - (b > idx)) for a, b in self.lines]
            self.circles = [(c - (c > idx), r) for c, r in self.circles]
            self.arcs = [tuple(p - (p > idx) for p in arc) for arc in self.arcs]
        elif kind == "l":
            del self.lines[idx]
        elif kind == "c":
            del self.circles[idx]
        elif kind == "a":
            del self.arcs[idx]
        if kind != "p":
            self.construction = {(k, i - (k == kind and i > idx)) for k, i in self.construction
                                 if (k, i) != (kind, idx)}
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
        arcs = [s.addArcOfCircle(wp, pts[c], pts[a], pts[b], g) for c, a, b in self.arcs]
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
            elif k == "spacing":
                h = s.addPointLineDistance(c.value, pts[r[0]], lines[r[1]], wp, g)
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
            elif k == "expression":
                if c.expression is None:
                    raise ValueError("expression constraint has no expression")
                val = _evaluate_expression(c.expression, self.variables)
                h = s.addPointsDistance(val, pts[r[0]], pts[r[1]], wp, g)
            elif k == "symmetry":
                p1, p2, axis = r
                p1_xy, p2_xy = self.points[p1], self.points[p2]
                axis_a, axis_b = self.lines[r[2]]
                if math.dist(p1_xy, reflect_point_across_line(p1_xy, self.points[axis_a], self.points[axis_b])) < 1e-9:
                    h = (
                        s.addPointsCoincident(pts[p1], pts[p2], wp, g),
                        s.addPointOnLine(pts[p1], lines[axis], wp, g),
                    )
                else:
                    pair_line = s.addLineSegment(pts[p1], pts[p2], g)
                    midpoint = s.addPoint2dV(wp, (p1_xy[0] + p2_xy[0]) / 2, (p1_xy[1] + p2_xy[1]) / 2, g)
                    h = (
                        s.addMidPoint(midpoint, pair_line, wp, g),
                        s.addPointOnLine(midpoint, lines[axis], wp, g),
                        s.addPerpendicular(pair_line, lines[axis], wp, g),
                    )
            elif k == "point_on_line":
                h = s.addPointOnLine(pts[r[0]], lines[r[1]], wp, g)
            elif k == "midpoint":
                h = s.addMidPoint(pts[r[0]], lines[r[1]], wp, g)
            elif k == "point_on_circle":
                h = s.addPointOnCircle(pts[r[0]], circles[r[1]], g)
            elif k == "point_on_arc":
                h = s.addPointOnCircle(pts[r[0]], arcs[r[1]], g)
            elif k == "radius":
                h = s.addDiameter(2 * c.value, arcs[r[0]], g)
            elif k == "tangent":
                shared = set(self.lines[r[0]]) & set(self.arcs[r[1]][1:])
                if not shared:
                    raise ValueError("a tangent line and arc must share an end point")
                h = s.addArcLineTangent(self.arcs[r[1]][2] in shared, arcs[r[1]], lines[r[0]], g)
            elif k == "tangent_arcs":
                (_, s1, e1), (_, s2, e2) = self.arcs[r[0]], self.arcs[r[1]]
                shared = {s1, e1} & {s2, e2}
                if not shared:
                    raise ValueError("tangent arcs must share an end point")
                p = shared.pop()
                h = s.addCurvesTangent(p == e1, p == e2, arcs[r[0]], arcs[r[1]], wp, g)
            else:
                raise ValueError(f"unknown constraint {k}")
            for handle in h if isinstance(h, tuple) else (h,):
                handle_to_constraint[handle] = ci

        res = s.solve(g, True)
        self.dof = s.Dof
        if res != 0:
            bad = sorted({handle_to_constraint[h] for h in s.Failed if h in handle_to_constraint})
            self.failed = bad
            why = {1: "inconsistent", 2: "did not converge", 3: "too many unknowns"}.get(res, f"error {res}")
            self.status = f"{why}" + (f" (constraints {bad})" if bad else "")
            return False
        for i, h in enumerate(pts):
            self.points[i] = [s.getParam(s.getEntityParam(h, k)).val for k in range(2)]
        self.circles = [(c, s.getParam(s.getEntityParam(radii[i], 0)).val) for i, (c, _) in enumerate(self.circles)]
        self.status = "fully constrained" if self.dof == 0 else f"{self.dof} DOF"
        return True

    # -- profiles
    def _curves(self) -> list:
        """The sketch's lines and circles as edges on the local plane (z = 0)."""
        edges = []
        for i, (a, b) in enumerate(self.lines):
            (u0, v0), (u1, v1) = self.points[a], self.points[b]
            if math.hypot(u1 - u0, v1 - v0) > 1e-9 and ("l", i) not in self.construction:
                edges.append(Edge.make_line((u0, v0, 0), (u1, v1, 0)))
        for i, (c, r) in enumerate(self.circles):
            if r > 1e-9 and ("c", i) not in self.construction:
                u, v = self.points[c]
                edges.append(Edge.make_circle(r, Plane(origin=(u, v, 0))))
        for i, (c, a, b) in enumerate(self.arcs):
            if ("a", i) in self.construction:
                continue
            (cu, cv), r, a0, sweep = self.arc_geometry(i)
            if r < 1e-9 or math.dist(self.points[a], self.points[b]) < 1e-9:
                continue
            m = (cu + r * math.cos(a0 + sweep / 2), cv + r * math.sin(a0 + sweep / 2), 0)
            # through the end points exactly, so the arc meets lines that end there
            edges.append(Edge.make_three_point_arc((*self.points[a], 0), m, (*self.points[b], 0)))
        return edges

    def regions(self) -> list["Region"]:
        """The smallest areas the sketch's curves enclose. Curves split each other where they cross, so
        overlapping shapes give one region per piece; open ends that enclose nothing are ignored."""
        key = (tuple(map(tuple, self.points)), tuple(self.lines), tuple(self.circles), tuple(self.arcs),
               frozenset(self.construction))
        if self._region_cache is None or self._region_cache[0] != key:
            self._region_cache = (key, _arrangement(self._curves()), {})
        return self._region_cache[1]

    def region_at(self, u: float, v: float) -> int | None:
        """Index into regions() of the region under (u, v); None over empty space."""
        for i, r in enumerate(self.regions()):
            if r.contains(u, v):
                return i
        return None

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

    def filled_regions(self) -> list[int]:
        """The regions used when none are picked: each separate outline flips between solid and hole,
        so a circle inside a rectangle is a hole, an island inside that is solid again, and shapes
        that overlap or are divided by lines are solid throughout."""
        regs = self.regions()
        cache = self._region_cache[2]
        if "filled" not in cache:
            groups = _outline_groups(regs)
            envelopes = [Face(_fuse([regs[i].face for i in g])[0].outer_wire()) for g in groups]
            cache["filled"] = [i for i, r in enumerate(regs)
                               if sum(_face_contains(env, *r.sample) for env in envelopes) % 2 == 1]
        return cache["filled"]

    def profile_faces(self, profiles: list | None = None) -> list[Face]:
        """Faces on the local plane for the regions under `profiles` (points in the sketch), or the
        filled regions when None. Neighbouring regions merge into one face."""
        regs = self.regions()
        if profiles is None:
            keep = self.filled_regions()
        else:
            keep = sorted({i for p in profiles if (i := self.region_at(*p)) is not None})
        return _fuse([regs[i].face for i in keep])

    def region_triangles(self, index: int) -> np.ndarray:
        """World-space triangles (N,3,3) filling region `index`, for highlighting."""
        local = self.regions()[index].triangles
        pl = self.plane
        o, x, y = np.array(pl.origin, float), np.array(pl.xdir, float), np.array(pl.ydir, float)
        return o + local[..., :1] * x + local[..., 1:2] * y

    def solid(self, distance: float, profiles: list | None = None) -> Shape | None:
        """Extrude filled regions along the plane normal; `profiles` (points in the sketch) limits which ones."""
        faces = self.profile_faces(profiles)
        if not faces or abs(distance) < 1e-9:
            return None
        pl = self.plane.b123d()
        n = self.plane.normal
        d = Vector(float(n[0] * distance), float(n[1] * distance), float(n[2] * distance))
        result = None
        for face in faces:
            solid = Solid.extrude(pl.from_local_coords(face), d)
            result = solid if result is None else result.fuse(solid)
        return result

    # -- persistence
    def to_dict(self) -> dict:
        return {"plane": asdict(self.plane), "points": [list(p) for p in self.points], "lines": [list(l) for l in self.lines],
                "circles": [list(c) for c in self.circles], "arcs": [list(a) for a in self.arcs],
                "construction": sorted([k, i] for k, i in self.construction),
                "constraints": [asdict(c) for c in self.constraints], "selection": [list(p) for p in self.selection],
                "version": self.version}

    @staticmethod
    def from_dict(d: dict) -> "Sketch":
        p = d["plane"]
        sk = Sketch(SketchPlane(tuple(p["origin"]), tuple(p["xdir"]), tuple(p["normal"])))
        sk.points = [list(p) for p in d["points"]]
        sk.lines = [tuple(l) for l in d["lines"]]
        sk.circles = [tuple(c) for c in d["circles"]]
        sk.arcs = [tuple(a) for a in d.get("arcs", [])]
        sk.construction = {(k, i) for k, i in d.get("construction", [])}
        sk.constraints = [Constraint(**c) for c in d["constraints"]]
        sk.selection = [tuple(p) for p in d.get("selection", [])]
        sk.version = d.get("version", 0)
        return sk



@dataclass(eq=False)
class Region:
    """One enclosed area of a sketch, on the local plane (u, v, 0)."""
    face: Face
    sample: tuple[float, float]  # a point well inside it
    triangles: np.ndarray  # (N, 3, 3) local triangles, for highlighting

    def contains(self, u: float, v: float) -> bool:
        return _face_contains(self.face, u, v)


def _face_contains(face: Face, u: float, v: float) -> bool:
    c = BRepClass_FaceClassifier(face.wrapped, gp_Pnt(u, v, 0), 1e-7)
    return c.State() == TopAbs_IN


def _arrangement(edges: list) -> list[Region]:
    """Split a big face by every curve and keep the pieces that do not reach its border."""
    if not edges:
        return []
    bb = Compound(edges).bounding_box()
    size = 4 * max(bb.size.X, bb.size.Y, 1.0)
    cx, cy = (bb.min.X + bb.max.X) / 2, (bb.min.Y + bb.max.Y) / 2
    big = Face.make_rect(size, size, Plane(origin=(cx, cy, 0)))
    args, tools = List_TopoDS_Shape(), List_TopoDS_Shape()
    args.Append(big.wrapped)
    for e in edges:
        tools.Append(e.wrapped)
    splitter = BRepAlgoAPI_Splitter()
    splitter.SetArguments(args)
    splitter.SetTools(tools)
    splitter.SetFuzzyValue(1e-7)
    splitter.Build()
    if not splitter.IsDone():
        return []
    border = size / 2 * 0.999
    regions = []
    for f in Compound(splitter.Shape()).faces():
        fb = f.bounding_box()
        if max(abs(fb.min.X - cx), abs(fb.max.X - cx), abs(fb.min.Y - cy), abs(fb.max.Y - cy)) > border:
            continue  # the outside
        # open curves poking into the region come back as internal edges: drop them
        holes = [w for w in f.inner_wires() if any(e.wrapped.Orientation() != TopAbs_INTERNAL for e in w.edges())]
        face = Face(f.outer_wire(), holes)
        if face.area < 1e-9:
            continue
        verts, tris = face.tessellate(0.05)
        v = np.array([[p.X, p.Y, p.Z] for p in verts], dtype=float)
        tri = v[np.array(tris, dtype=int)].reshape(-1, 3, 3)
        ab, ac = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        big_tri = tri[int(np.argmax(np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])))]
        cu, cv = big_tri[:, :2].mean(axis=0)
        regions.append(Region(face, (float(cu), float(cv)), tri))
    return regions


def _outline_groups(regs: list[Region]) -> list[list[int]]:
    """Regions whose outer boundaries share an edge, grouped: each group is one separate outline."""
    parent = list(range(len(regs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owner: dict = {}
    for i, r in enumerate(regs):
        for e in r.face.outer_wire().edges():
            j = owner.setdefault(hash(e), i)
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(regs)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _fuse(faces: list[Face]) -> list[Face]:
    """Merge faces that touch into single faces without seams."""
    if len(faces) <= 1:
        return list(faces)
    merged = faces[0].fuse(*faces[1:]).clean()
    return merged.faces()
