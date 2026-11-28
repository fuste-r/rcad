"""Trim: remove the piece of a sketch line, circle or arc between the places where other curves
cross it (or out to its free end)."""
from __future__ import annotations

import math
from dataclasses import dataclass

from .sketch import CONSTRAINT_KINDS, Constraint, Sketch

EPS = 1e-7
TAU = 2 * math.pi


@dataclass
class Cut:
    param: float  # where along the target: 0..1 on a line, the angle from its start on a circle or arc
    uv: tuple[float, float]
    cutter: tuple[str, int]  # the curve that crosses there


@dataclass
class Piece:
    kind: str  # "l" | "c" | "a": the curve it belongs to
    index: int
    lo: Cut | None  # None: the curve's own start (a line's or arc's first end)
    hi: Cut | None  # None: the curve's own end
    polyline: list  # sketch points along the piece, for showing it


# -- the curves as geometry: ("seg", A, B) or ("circ", centre, r, start angle, sweep)

def _shape(sk: Sketch, kind: str, i: int):
    if kind == "l":
        a, b = sk.lines[i]
        return ("seg", tuple(sk.points[a]), tuple(sk.points[b]))
    if kind == "c":
        c, r = sk.circles[i]
        return ("circ", tuple(sk.points[c]), r, 0.0, TAU)
    centre, r, a0, sweep = sk.arc_geometry(i)
    return ("circ", centre, r, a0, sweep)


def _curves(sk: Sketch):
    yield from (("l", i) for i in range(len(sk.lines)))
    yield from (("c", i) for i in range(len(sk.circles)))
    yield from (("a", i) for i in range(len(sk.arcs)))


def _param(shape, p) -> float:
    if shape[0] == "seg":
        (ax, ay), (bx, by) = shape[1], shape[2]
        dx, dy = bx - ax, by - ay
        return ((p[0] - ax) * dx + (p[1] - ay) * dy) / max(dx * dx + dy * dy, 1e-300)
    (cx, cy), _, a0, _ = shape[1:]
    return (math.atan2(p[1] - cy, p[0] - cx) - a0) % TAU


def _on(shape, p, tol=1e-6) -> bool:
    """Is p on the curve (within its ends)?"""
    if shape[0] == "seg":
        t = _param(shape, p)
        (ax, ay), (bx, by) = shape[1], shape[2]
        q = (ax + t * (bx - ax), ay + t * (by - ay))
        length = math.dist(shape[1], shape[2])
        return -tol / max(length, 1e-12) <= t <= 1 + tol / max(length, 1e-12) and math.dist(p, q) <= tol
    c, r, _, sweep = shape[1:]
    if abs(math.dist(c, p) - r) > tol:
        return False
    off = _param(shape, p)
    return off <= sweep + tol / max(r, 1e-12) or off >= TAU - tol / max(r, 1e-12)


def _meet(s1, s2) -> list:
    """Points where two curves cross (within their ends)."""
    out = []
    if s1[0] == "seg" and s2[0] == "seg":
        (ax, ay), (bx, by) = s1[1], s1[2]
        (cx, cy), (dx, dy) = s2[1], s2[2]
        rx, ry, sx, sy = bx - ax, by - ay, dx - cx, dy - cy
        den = rx * sy - ry * sx
        if abs(den) > 1e-12:
            t = ((cx - ax) * sy - (cy - ay) * sx) / den
            out.append((ax + t * rx, ay + t * ry))
    elif s1[0] == "seg" or s2[0] == "seg":
        seg, circ = (s1, s2) if s1[0] == "seg" else (s2, s1)
        (ax, ay), (bx, by) = seg[1], seg[2]
        (cx, cy), r = circ[1], circ[2]
        dx, dy = bx - ax, by - ay
        fx, fy = ax - cx, ay - cy
        a, b, c = dx * dx + dy * dy, 2 * (fx * dx + fy * dy), fx * fx + fy * fy - r * r
        disc = b * b - 4 * a * c
        if a > 1e-300 and disc >= -1e-9 * max(a * r * r, 1e-300):
            root = math.sqrt(max(disc, 0.0))
            for t in {(-b - root) / (2 * a), (-b + root) / (2 * a)}:
                out.append((ax + t * dx, ay + t * dy))
    else:
        (x1, y1), r1 = s1[1], s1[2]
        (x2, y2), r2 = s2[1], s2[2]
        d = math.dist((x1, y1), (x2, y2))
        if 1e-12 < d <= r1 + r2 + 1e-9 and d >= abs(r1 - r2) - 1e-9:
            a = (r1 * r1 - r2 * r2 + d * d) / (2 * d)
            h = math.sqrt(max(r1 * r1 - a * a, 0.0))
            mx, my = x1 + a * (x2 - x1) / d, y1 + a * (y2 - y1) / d
            for s in {h, -h}:
                out.append((mx + s * (y2 - y1) / d, my - s * (x2 - x1) / d))
    return [p for p in out if _on(s1, p) and _on(s2, p)]


def _ends(shape):
    if shape[0] == "seg":
        return [shape[1], shape[2]]
    c, r, a0, sweep = shape[1:]
    if sweep >= TAU - 1e-12:
        return []
    return [(c[0] + r * math.cos(a), c[1] + r * math.sin(a)) for a in (a0, a0 + sweep)]


def cuts(sk: Sketch, kind: str, i: int) -> list[Cut]:
    """Where other curves cross or touch curve (kind, i), along it, not counting its own ends."""
    target = _shape(sk, kind, i)
    found: list[Cut] = []
    for other in _curves(sk):
        if other == (kind, i):
            continue
        shape = _shape(sk, *other)
        pts = _meet(target, shape) + [p for p in _ends(shape) if _on(target, p)]  # a curve ending on it
        for p in pts:
            t = _param(target, p)
            if any(math.dist(p, f.uv) < 1e-6 for f in found):
                continue
            found.append(Cut(t, p, other))
    span = 1.0 if target[0] == "seg" else target[4]
    full = kind == "c"
    tiny = 1e-9 if target[0] == "seg" else 1e-9 / max(target[2], 1e-12)
    keep = [f for f in found if full or tiny < f.param < span - tiny]
    return sorted(keep, key=lambda f: f.param)


def piece_at(sk: Sketch, uv, tol: float) -> Piece | None:
    """The piece of the curve nearest uv (within tol) between the cuts on either side of uv."""
    best = None
    for item in _curves(sk):
        shape = _shape(sk, *item)
        if shape[0] == "seg":
            t = min(1.0, max(0.0, _param(shape, uv)))
            (ax, ay), (bx, by) = shape[1], shape[2]
            d = math.dist(uv, (ax + t * (bx - ax), ay + t * (by - ay)))
        else:
            c, r, _, sweep = shape[1:]
            if _param(shape, uv) > sweep:
                continue
            d = abs(math.dist(c, uv) - r)
        if d <= tol and (best is None or d < best[0]):
            best = (d, item, shape)
    if best is None:
        return None
    _, (kind, i), shape = best
    t = _param(shape, uv)
    cs = cuts(sk, kind, i)
    if kind == "c":
        if len(cs) < 2:
            lo = hi = None  # nothing (or one point) cuts it: the whole circle
        else:
            after = [c for c in cs if c.param > t]
            hi = after[0] if after else cs[0]
            before = [c for c in cs if c.param <= t]
            lo = before[-1] if before else cs[-1]
    else:
        lo = next((c for c in reversed(cs) if c.param <= t), None)
        hi = next((c for c in cs if c.param > t), None)
    return Piece(kind, i, lo, hi, _polyline(shape, lo, hi))


def _polyline(shape, lo, hi, n=48):
    if shape[0] == "seg":
        (ax, ay), (bx, by) = shape[1], shape[2]
        t0, t1 = (lo.param if lo else 0.0), (hi.param if hi else 1.0)
        return [(ax + t * (bx - ax), ay + t * (by - ay)) for t in (t0, t1)]
    c, r, a0, sweep = shape[1:]
    o0 = lo.param if lo else 0.0
    o1 = hi.param if hi else sweep
    if o1 <= o0:
        o1 += TAU  # round past the circle's start
    return [(c[0] + r * math.cos(a0 + o0 + (o1 - o0) * k / (n - 1)),
             c[1] + r * math.sin(a0 + o0 + (o1 - o0) * k / (n - 1))) for k in range(n)]


# -- trimming

@dataclass
class TrimResult:
    keep: list[Constraint]  # constraints that hold the new ends in place; add them one by one
    dropped: list[str]  # kinds of constraints that no longer applied and were removed


def _cut_point(sk: Sketch, cut: Cut, keep: list) -> int:
    """A point at the cut: an existing one if one is there, else a new one held on the cutter."""
    for i, p in enumerate(sk.points):
        if math.dist(p, cut.uv) < 1e-6:
            return i
    p = sk.add_point(*cut.uv)
    kind, j = cut.cutter
    keep.append(Constraint({"l": "point_on_line", "c": "point_on_circle", "a": "point_on_arc"}[kind], [p, j]))
    return p


def _uses(sk: Sketch, p: int) -> bool:
    return (any(p in ab for ab in sk.lines) or any(p == c for c, _ in sk.circles)
            or any(p in arc for arc in sk.arcs))


def _retarget_tangents(sk: Sketch, kind: str, old: int, new: int, joint: int):
    """Tangent constraints on curve `old` at point `joint` now belong to curve `new`."""
    for c in sk.constraints:
        if c.kind == "tangent" and kind == "l" and c.refs[0] == old and joint in sk.arcs[c.refs[1]][1:]:
            c.refs = [new, c.refs[1]]
        elif c.kind == "tangent" and kind == "a" and c.refs[1] == old and joint in sk.lines[c.refs[0]]:
            c.refs = [c.refs[0], new]
        elif c.kind == "tangent_arcs" and kind == "a" and old in c.refs:
            other = c.refs[1] if c.refs[0] == old else c.refs[0]
            if joint in sk.arcs[other][1:]:
                c.refs = [new if r == old else r for r in c.refs]


def _drop_broken(sk: Sketch, items: set) -> set:
    """Remove constraints on the trimmed curves that no longer make sense: a midpoint or equal length
    of a line whose length changed, and tangents whose curves no longer share an end. A point held on
    a curve that it is now an end of is absorbed (returned by id: nothing was lost)."""
    kept, absorbed = [], set()
    for c in sk.constraints:
        refs = set(zip(_types(c), c.refs))
        if c.kind in ("midpoint", "equal") and refs & items:
            continue
        if (c.kind == "point_on_line" and c.refs[0] in sk.lines[c.refs[1]]) or \
                (c.kind == "point_on_arc" and c.refs[0] in sk.arcs[c.refs[1]][1:]):
            absorbed.add(id(c))  # the point is now the curve's own end
            continue
        if c.kind == "tangent" and not set(sk.lines[c.refs[0]]) & set(sk.arcs[c.refs[1]][1:]):
            continue
        if c.kind == "tangent_arcs" and not set(sk.arcs[c.refs[0]][1:]) & set(sk.arcs[c.refs[1]][1:]):
            continue
        kept.append(c)
    sk.constraints = kept
    return absorbed


def _types(c: Constraint) -> str:
    return CONSTRAINT_KINDS[c.kind][0]


def trim(sk: Sketch, piece: Piece) -> TrimResult:
    """Remove `piece` from its curve. Points left attached to nothing are removed too."""
    before = list(sk.constraints)
    keep: list[Constraint] = []
    converted: set = set()
    orphans: set = set()
    kind, i = piece.kind, piece.index
    lo, hi = piece.lo, piece.hi
    construction = (kind, i) in sk.construction

    if kind == "l":
        a, b = sk.lines[i]
        if lo is None and hi is None:
            sk.delete("l", i)
            orphans |= {a, b}
        elif lo is None:
            sk.lines[i] = (_cut_point(sk, hi, keep), b)
            orphans.add(a)
            converted |= _drop_broken(sk, {("l", i)})
        elif hi is None:
            sk.lines[i] = (a, _cut_point(sk, lo, keep))
            orphans.add(b)
            converted |= _drop_broken(sk, {("l", i)})
        else:
            p_lo, p_hi = _cut_point(sk, lo, keep), _cut_point(sk, hi, keep)
            sk.lines[i] = (a, p_lo)
            j = sk.add_line(p_hi, b)
            if construction:
                sk.construction.add(("l", j))
            _retarget_tangents(sk, "l", i, j, b)
            keep += [Constraint("point_on_line", [p_hi, i]), Constraint("point_on_line", [b, i])]  # in line
            converted |= _drop_broken(sk, {("l", i), ("l", j)})
    elif kind == "c":
        c, r = sk.circles[i]
        if lo is None or hi is None:
            sk.delete("c", i)
            orphans.add(c)
        else:
            s, e = _cut_point(sk, hi, keep), _cut_point(sk, lo, keep)  # what is left runs from hi round to lo
            j = sk.add_arc(c, s, e)
            if construction:
                sk.construction.add(("a", j))
            for con in list(sk.constraints):
                if con.kind == "diameter" and con.refs == [i]:
                    sk.constraints.append(Constraint("radius", [j], con.value / 2 if con.value else con.value))
                    converted.add(id(con))
                elif con.kind == "point_on_circle" and con.refs[1] == i:
                    sk.constraints.append(Constraint("point_on_arc", [con.refs[0], j]))
                    converted.add(id(con))
            sk.delete("c", i)
    else:
        c, s, e = sk.arcs[i]
        if lo is None and hi is None:
            sk.delete("a", i)
            orphans |= {c, s, e}
        elif lo is None:
            sk.arcs[i] = (c, _cut_point(sk, hi, keep), e)
            orphans.add(s)
            converted |= _drop_broken(sk, {("a", i)})
        elif hi is None:
            sk.arcs[i] = (c, s, _cut_point(sk, lo, keep))
            orphans.add(e)
            converted |= _drop_broken(sk, {("a", i)})
        else:
            p_lo, p_hi = _cut_point(sk, lo, keep), _cut_point(sk, hi, keep)
            sk.arcs[i] = (c, s, p_lo)
            j = sk.add_arc(c, p_hi, e)
            if construction:
                sk.construction.add(("a", j))
            _retarget_tangents(sk, "a", i, j, e)
            keep.append(Constraint("point_on_arc", [p_hi, i]))  # the same radius
            converted |= _drop_broken(sk, {("a", i), ("a", j)})

    # removing points renumbers the rest: let the new constraints be renumbered with the others
    sk.constraints += keep
    for p in sorted(orphans, reverse=True):
        if p < len(sk.points) and not _uses(sk, p):
            sk.delete("p", p)  # (its other constraints go with it)
    held = {id(c) for c in keep}
    keep = [c for c in sk.constraints if id(c) in held]
    sk.constraints = [c for c in sk.constraints if id(c) not in held]
    after = {id(c) for c in sk.constraints}
    dropped = [c.kind for c in before if id(c) not in after and id(c) not in converted]
    sk.version += 1
    return TrimResult(keep, dropped)
