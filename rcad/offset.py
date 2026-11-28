"""Offset: a parallel copy of sketch lines, arcs and circles at a distance, held there by constraints."""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field

from .sketch import Constraint, Sketch

TAU = 2 * math.pi


class OffsetError(ValueError):
    pass


@dataclass
class Seg:
    """One selected line or arc, in the order and direction its chain runs."""
    kind: str  # "l" | "a"
    index: int
    forward: bool  # line: from its first point to its second; arc: counter-clockwise (s to e)
    start: int  # point indices, in travel order
    end: int


@dataclass
class Chain:
    segs: list[Seg]
    closed: bool


@dataclass
class OffsetResult:
    keep: list[Constraint] = field(default_factory=list)  # add one by one; any the sketch refuses is dropped
    items: list = field(default_factory=list)  # the new ("l" | "a" | "c", index) curves


def chains(sk: Sketch, items) -> tuple[list[Chain], list[int]]:
    """Group selected lines and arcs into chains joined end to end (circles are returned apart).
    The order is fixed (never set order): an open chain runs from its earliest-drawn end (the lower
    point index), which is usually the way it was drawn, so "left" and "right" mean the same each time."""
    segs = sorted({(k, i) for k, i in items if k in ("l", "a")})
    circles = sorted(i for k, i in items if k == "c")

    def ends(item):
        k, i = item
        return tuple(sk.lines[i]) if k == "l" else (sk.arcs[i][1], sk.arcs[i][2])

    at: dict[int, list] = {}
    for item in segs:
        for p in ends(item):
            at.setdefault(p, []).append(item)
    out, used = [], set()
    # open chains start at a point only one selected curve reaches; then what is left is closed loops
    starts = sorted(p for p, its in at.items() if len(its) == 1)
    for first in starts + sorted(at):
        for item in at.get(first, []):
            if item in used:
                continue
            chain, p, cur = [], first, item
            while cur is not None and cur not in used:
                used.add(cur)
                a, b = ends(cur)
                forward = a == p
                nxt_p = b if forward else a
                chain.append(Seg(cur[0], cur[1], forward, p, nxt_p))
                p = nxt_p
                cur = next((x for x in at.get(p, []) if x not in used), None)
            out.append(Chain(chain, chain[0].start == chain[-1].end and len(chain) > 1))
    return out, circles


def _left(du, dv):
    n = math.hypot(du, dv)
    return (-dv / n, du / n)


def _shape(sk: Sketch, seg: Seg, d: float):
    """The offset of one segment: ("line", P', Q') or ("arc", centre point, radius, ccw) with its
    offset end points, in travel order. d > 0 is to the left of travel."""
    p, q = sk.points[seg.start], sk.points[seg.end]
    if seg.kind == "l":
        nu, nv = _left(q[0] - p[0], q[1] - p[1])
        return {"kind": "l", "a": (p[0] + nu * d, p[1] + nv * d), "b": (q[0] + nu * d, q[1] + nv * d),
                "dir": (q[0] - p[0], q[1] - p[1])}
    c = sk.arcs[seg.index][0]
    cu, cv = sk.points[c]
    r = math.dist((cu, cv), p)
    ccw = seg.forward
    r2 = r - d if ccw else r + d  # left of counter-clockwise travel is toward the centre
    if r2 <= 1e-9:
        raise OffsetError(f"the distance is too large for an arc of radius {r:g}")

    def on(pt):
        a = math.atan2(pt[1] - cv, pt[0] - cu)
        return (cu + r2 * math.cos(a), cv + r2 * math.sin(a))
    return {"kind": "a", "centre": c, "c": (cu, cv), "r": r2, "ccw": ccw, "a": on(p), "b": on(q)}


def _cross(s1, s2, near):
    """Where two offset curves (extended: whole lines and circles) meet, nearest `near`; or None."""
    pts = []
    if s1["kind"] == "l" and s2["kind"] == "l":
        (ax, ay), (rx, ry) = s1["a"], s1["dir"]
        (cx, cy), (sx, sy) = s2["a"], s2["dir"]
        den = rx * sy - ry * sx
        if abs(den) > 1e-12 * math.hypot(rx, ry) * math.hypot(sx, sy):
            t = ((cx - ax) * sy - (cy - ay) * sx) / den
            pts.append((ax + t * rx, ay + t * ry))
    elif s1["kind"] == "l" or s2["kind"] == "l":
        ln, ci = (s1, s2) if s1["kind"] == "l" else (s2, s1)
        (ax, ay), (dx, dy) = ln["a"], ln["dir"]
        (cx, cy), r = ci["c"], ci["r"]
        fx, fy = ax - cx, ay - cy
        a, b, c = dx * dx + dy * dy, 2 * (fx * dx + fy * dy), fx * fx + fy * fy - r * r
        disc = b * b - 4 * a * c
        if disc >= 0:
            for t in ((-b - math.sqrt(disc)) / (2 * a), (-b + math.sqrt(disc)) / (2 * a)):
                pts.append((ax + t * dx, ay + t * dy))
    else:
        (x1, y1), r1, (x2, y2), r2 = s1["c"], s1["r"], s2["c"], s2["r"]
        dd = math.dist((x1, y1), (x2, y2))
        if 1e-12 < dd <= r1 + r2 and dd >= abs(r1 - r2):
            a = (r1 * r1 - r2 * r2 + dd * dd) / (2 * dd)
            h = math.sqrt(max(r1 * r1 - a * a, 0.0))
            mx, my = x1 + a * (x2 - x1) / dd, y1 + a * (y2 - y1) / dd
            for sgn in (1, -1):
                pts.append((mx + sgn * h * (y2 - y1) / dd, my - sgn * h * (x2 - x1) / dd))
    return min(pts, key=lambda p: math.dist(p, near)) if pts else None


def offset(sk: Sketch, items, distance: float, flip: bool = False) -> OffsetResult:
    """Add the offset of the selected lines, arcs and circles to the sketch. distance > 0. Closed
    outlines go outward (inward when flip); open chains go to their left (right when flip).
    Raises OffsetError (the sketch untouched) when it can't be made."""
    if distance <= 0:
        raise OffsetError("the distance must be more than zero")
    found, circles = chains(sk, items)
    if not found and not circles:
        raise OffsetError("select lines, arcs or circles to offset")
    group = "offset-" + uuid.uuid4().hex[:8]
    res = OffsetResult()
    plans = []
    for ch in found:
        d = distance
        if ch.closed:
            # outward: the outline runs counter-clockwise when its signed area is positive; left is then inward
            area = _signed_area(sk, ch)
            d = -distance if area > 0 else distance
        if flip:
            d = -d
        plans.append((ch, d, [_shape(sk, s, d) for s in ch.segs]))
    for ch, d, shapes in plans:
        _build(sk, ch, d, shapes, group, distance, res)
    for ci in circles:
        c, r = sk.circles[ci]
        r2 = r - distance if flip else r + distance
        if r2 <= 1e-9:
            raise OffsetError(f"the distance is too large for a circle of radius {r:g}")
        res.items.append(("c", sk.add_circle(c, r2)))  # the same centre point: they stay concentric
    return res


def _signed_area(sk: Sketch, ch: Chain) -> float:
    pts = []
    for s in ch.segs:
        if s.kind == "l":
            pts.append(sk.points[s.start])
        else:
            (cu, cv), r, a0, sweep = sk.arc_geometry(s.index)
            ts = [k / 16 for k in range(16)]
            arc = [(cu + r * math.cos(a0 + sweep * t), cv + r * math.sin(a0 + sweep * t)) for t in ts]
            pts += arc if s.forward else [sk.points[s.start]] + arc[::-1][:-1]
    return sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
               for i in range(len(pts))) / 2


def _build(sk, ch, d, shapes, group, distance, res: OffsetResult):
    n = len(shapes)
    joints = []  # per joint between shape k and k+1: ("shared", point uv) or ("round", (end uv, start uv))
    pairs = range(n) if ch.closed else range(n - 1)
    for k in pairs:
        s1, s2 = shapes[k], shapes[(k + 1) % n]
        e, s = s1["b"], s2["a"]
        if math.dist(e, s) < 1e-9 * max(1.0, distance):
            joints.append(("smooth", e))
            continue
        near = ((e[0] + s[0]) / 2, (e[1] + s[1]) / 2)
        hit = _cross(s1, s2, near)
        if hit is not None and math.dist(hit, near) <= 4 * distance + math.dist(e, s):
            joints.append(("corner", hit))
        else:
            joints.append(("round", (e, s)))
    # points: one per joint (two for a round one), plus the open ends
    pt_start, pt_end = [None] * n, [None] * n
    for k, (how, where) in zip(pairs, joints):
        nxt = (k + 1) % n
        if how == "round":
            pt_end[k], pt_start[nxt] = sk.add_point(*where[0]), sk.add_point(*where[1])
        else:
            pt_end[k] = pt_start[nxt] = sk.add_point(*where)
    if not ch.closed:
        pt_start[0] = sk.add_point(*shapes[0]["a"])
        pt_end[-1] = sk.add_point(*shapes[-1]["b"])
    new = []
    for k, (seg, sh) in enumerate(zip(ch.segs, shapes)):
        a, b = pt_start[k], pt_end[k]
        if seg.kind == "l":
            o = sk.points[seg.start], sk.points[seg.end]
            na, nb = sk.points[a], sk.points[b]
            if (nb[0] - na[0]) * (o[1][0] - o[0][0]) + (nb[1] - na[1]) * (o[1][1] - o[0][1]) <= 0:
                raise OffsetError("the distance is too large: a line would turn back on itself")
            li = sk.add_line(a, b)
            res.items.append(("l", li))
            res.keep += [Constraint("parallel", [li, seg.index], group=group),
                         Constraint("spacing", [a, seg.index], distance, group=group)]
            new.append(("l", li))
        else:
            ai = sk.add_arc(sh["centre"], a, b) if sh["ccw"] else sk.add_arc(sh["centre"], b, a)
            res.items.append(("a", ai))
            new.append(("a", ai))
    # smooth joints stay smooth; round joins are arcs round the original corner, tangent to both sides
    for k, (how, where) in zip(pairs, joints):
        nxt = (k + 1) % n
        if how == "smooth":
            res.keep += _tangent(new[k], new[nxt])
        elif how == "round":
            corner = ch.segs[k].end
            e, s = pt_end[k], pt_start[nxt]
            cu, cv = sk.points[corner]
            a_e = math.atan2(sk.points[e][1] - cv, sk.points[e][0] - cu)
            a_s = math.atan2(sk.points[s][1] - cv, sk.points[s][0] - cu)
            # the short way round the corner, stored counter-clockwise
            ai = sk.add_arc(corner, e, s) if (a_s - a_e) % TAU <= math.pi else sk.add_arc(corner, s, e)
            res.items.append(("a", ai))
            res.keep.append(Constraint("radius", [ai], distance, group=group))
            res.keep += _tangent(new[k], ("a", ai)) + _tangent(("a", ai), new[nxt])


def _tangent(x, y):
    if x[0] == "l" and y[0] == "l":
        return []  # two lines meeting smoothly are simply in line
    if x[0] == "l":
        return [Constraint("tangent", [x[1], y[1]])]
    if y[0] == "l":
        return [Constraint("tangent", [y[1], x[1]])]
    return [Constraint("tangent_arcs", [x[1], y[1]])]
