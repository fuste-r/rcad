"""Smart snapping while drawing in a sketch, and the constraints a snap implies."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .sketch import Constraint, Sketch

START = -1  # alignment reference: the drawing's pending start point (it may not exist yet)


@dataclass
class Snap:
    uv: tuple[float, float]
    kind: str = "free"  # point, midpoint, on_line, on_circle, align, grid, free
    point: int | None = None  # existing point the pointer is on (reused, so it is coincident)
    line: int | None = None  # line for midpoint / on_line
    circle: int | None = None  # circle for on_circle
    arc: int | None = None  # arc for on_arc
    tangent: int | None = None  # kind "tangent": the arc a line drawn from its end leaves tangentially
    align: list[tuple[str, int]] = field(default_factory=list)  # ("h" same v | "v" same u, point or START)

    def constraints(self, p: int, start: int | None = None) -> list[Constraint]:
        """Constraints that keep new point p where this snap put it. Alignment to the start point is
        left out (start=None): a line drawn level with its own start gets a horizontal/vertical line."""
        out = []
        if self.kind == "midpoint":
            out.append(Constraint("midpoint", [p, self.line]))
        elif self.line is not None:
            out.append(Constraint("point_on_line", [p, self.line]))
        elif self.circle is not None:
            out.append(Constraint("point_on_circle", [p, self.circle]))
        elif self.arc is not None:
            out.append(Constraint("point_on_arc", [p, self.arc]))
        for axis, ref in self.align:
            ref = start if ref == START else ref
            if ref is not None and ref != p:
                out.append(Constraint("phorizontal" if axis == "h" else "pvertical", [ref, p]))
        return out

    def describe(self, sk: Sketch) -> str:
        names = {"point": "point", "midpoint": "midpoint", "on_line": "on line", "on_circle": "on circle",
                 "on_arc": "on arc", "tangent": "tangent to the arc"}
        parts = [names[self.kind]] if self.kind in names else []
        for axis, ref in self.align:
            what = ("the start" if ref == START
                    else "a circle centre" if any(c == ref for c, _ in sk.circles)
                    else "an arc centre" if any(a[0] == ref for a in sk.arcs) else "a point")
            parts.append(f"{'horizontal' if axis == 'h' else 'vertical'} with {what}")
        return ", ".join(parts)

    def guides(self, sk: Sketch, start_uv=None) -> list[tuple[tuple, tuple]]:
        """Dashed alignment lines from each reference point to the snapped point (sketch coordinates)."""
        out = []
        if self.kind == "tangent" and start_uv is not None:
            out.append((tuple(start_uv), self.uv))
        for _, ref in self.align:
            ref_uv = start_uv if ref == START else tuple(sk.points[ref])
            if ref_uv is not None:
                out.append((tuple(ref_uv), self.uv))
        return out


def grid_step(base: float, mm_per_px: float, min_px: float = 5.0) -> float:
    """The snap grid step: base (the Snap field) times 1, 2, 5, 10, ... until steps are min_px apart."""
    if base <= 0:
        return 0.0
    for k in range(30):
        step = base * (1, 2, 5)[k % 3] * 10 ** (k // 3)
        if step / max(mm_per_px, 1e-12) >= min_px:
            return step
    return step


def _round(uv, g):
    return (round(uv[0] / g) * g, round(uv[1] / g) * g) if g > 0 else tuple(uv)


def find_snap(sk: Sketch, uv, tol: float, grid: float = 0.0, start=None, smart: bool = True,
              curves: bool = True, skip_points=(), tangent=None) -> Snap:
    """Where a point drawn at uv lands. tol: snap distance in sketch units; start: (index or None, uv) of
    the pending start point, used for alignment; curves: allow snapping onto lines and circles;
    tangent: (arc index, unit direction) when a line starts at an arc's end, to snap along its tangent."""
    uv = (float(uv[0]), float(uv[1]))
    if not smart:
        return Snap(_round(uv, grid), "grid" if grid > 0 else "free")
    skip = set(skip_points)

    # 1. an existing point
    best, best_d = None, tol
    for i, p in enumerate(sk.points):
        d = math.dist(p, uv)
        if i not in skip and d <= best_d:
            best, best_d = i, d
    if best is not None:
        return Snap(tuple(sk.points[best]), "point", point=best)

    # 2. a line's midpoint
    if curves:
        best, best_d = None, tol
        for i, (a, b) in enumerate(sk.lines):
            m = ((sk.points[a][0] + sk.points[b][0]) / 2, (sk.points[a][1] + sk.points[b][1]) / 2)
            d = math.dist(m, uv)
            if d <= best_d:
                best, best_d = (i, m), d
        if best is not None:
            return Snap(best[1], "midpoint", line=best[0])

    # 3. straight on from an arc's end, along its tangent
    if tangent is not None and start is not None:
        arc_i, (tu, tv) = tangent
        su, sv = start[1]
        du, dv = uv[0] - su, uv[1] - sv
        along, across = du * tu + dv * tv, du * tv - dv * tu
        if along > tol and abs(across) <= tol:
            if grid > 0:
                along = max(grid, round(along / grid) * grid)
            return Snap((su + tu * along, sv + tv * along), "tangent", tangent=arc_i)

    # alignment candidates: the nearest point (or the start) level with uv, per axis
    refs = [(i, tuple(p)) for i, p in enumerate(sk.points) if i not in skip]
    if start is not None:
        s_idx, s_uv = start
        refs = [(r, p) for r, p in refs if r != s_idx] + [(START, tuple(s_uv))]
    align = {}
    for axis, k in (("h", 1), ("v", 0)):
        cands = [(abs(p[k] - uv[k]), math.dist(p, uv), r, p) for r, p in refs if abs(p[k] - uv[k]) <= tol]
        if cands:
            _, _, r, p = min(cands, key=lambda c: (c[0] > 1e-9, c[1]))  # exact ties: nearest reference
            align[axis] = (r, p)

    # 4. a curve, alone or where an alignment crosses it
    curve = _nearest_curve(sk, uv, tol) if curves else None
    if curve is not None:
        kind, idx, foot = curve
        for axis, (r, p) in align.items():
            hit = _curve_cross(sk, kind, idx, axis, p, uv)
            if hit is not None and math.dist(hit, uv) <= tol * 1.5:
                return Snap(hit, "on_" + kind, **{kind: idx}, align=[(axis, r)])
        return Snap(foot, "on_" + kind, **{kind: idx})

    # 5. alignment: both axes pin the point; one axis pins one coordinate and the grid the other
    if align:
        u, v = _round(uv, grid)
        if "v" in align:
            u = align["v"][1][0]
        if "h" in align:
            v = align["h"][1][1]
        return Snap((u, v), "align", align=[(a, r) for a, (r, _) in sorted(align.items())])

    return Snap(_round(uv, grid), "grid" if grid > 0 else "free")


def _nearest_curve(sk: Sketch, uv, tol):
    best, best_d = None, tol
    for i, (a, b) in enumerate(sk.lines):
        foot, d = _foot_on_segment(sk.points[a], sk.points[b], uv)
        if d <= best_d:
            best, best_d = ("line", i, foot), d
    for i, (c, r) in enumerate(sk.circles):
        cu, cv = sk.points[c]
        d0 = math.dist((cu, cv), uv)
        if r > 1e-9 and d0 > 1e-12 and abs(d0 - r) <= best_d:
            best_d = abs(d0 - r)
            best = ("circle", i, (cu + (uv[0] - cu) * r / d0, cv + (uv[1] - cv) * r / d0))
    for i in range(len(sk.arcs)):
        (cu, cv), r, a0, sweep = sk.arc_geometry(i)
        d0 = math.dist((cu, cv), uv)
        if r < 1e-9 or d0 < 1e-12 or abs(d0 - r) > best_d:
            continue
        if (math.atan2(uv[1] - cv, uv[0] - cu) - a0) % (2 * math.pi) > sweep:
            continue  # beside the arc's gap
        best_d = abs(d0 - r)
        best = ("arc", i, (cu + (uv[0] - cu) * r / d0, cv + (uv[1] - cv) * r / d0))
    return best


def _foot_on_segment(a, b, p):
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    den = dx * dx + dy * dy
    t = 0.0 if den < 1e-18 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / den))
    foot = (ax + t * dx, ay + t * dy)
    return foot, math.dist(foot, p)


def _curve_cross(sk: Sketch, kind, idx, axis, ref, uv):
    """Where the curve meets the horizontal (axis 'h': v = ref v) or vertical ('v': u = ref u) line
    through ref, nearest uv; None if it does not."""
    k = 1 if axis == "h" else 0  # the fixed coordinate
    level = ref[k]
    if kind == "line":
        a, b = sk.points[sk.lines[idx][0]], sk.points[sk.lines[idx][1]]
        if abs(b[k] - a[k]) < 1e-12:
            return None
        t = (level - a[k]) / (b[k] - a[k])
        if not -1e-9 <= t <= 1 + 1e-9:
            return None
        return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    if kind != "circle":
        return None  # (arcs: plain on-arc snap)
    c, r = sk.circles[idx]
    center = sk.points[c]
    off = level - center[k]
    if abs(off) > r:
        return None
    other = math.sqrt(r * r - off * off)
    o = 1 - k
    hits = []
    for s in (other, -other):
        p = [0.0, 0.0]
        p[k], p[o] = level, center[o] + s
        hits.append(tuple(p))
    return min(hits, key=lambda h: math.dist(h, uv))
