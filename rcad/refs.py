"""Stable references to faces and edges across rebuilds.

A feature that works on a face (push/pull) or edges (fillet, chamfer) stores a
*signature* of each: its type, position, size, direction, and the normals of the
faces around it. On rebuild the reference is found again in the new part:

1. an exact match (same place, same type) is taken directly;
2. otherwise every candidate of the same kind is scored by how far it moved and
   how much its size and surroundings changed, and the best one is taken only if
   it is clearly better than the runner-up;
3. a candidate only counts if it is *local*: the old position, projected onto the
   candidate's plane (for an edge: onto each flat face beside it), lands on it.
   That holds when a face grows or moves along its normal (a height or width
   change), and fails for a look-alike face or edge somewhere else;
4. if nothing fits, or two candidates fit about equally well, the feature fails
   with a message saying so, instead of silently using a different face or edge.

After a successful match the caller stores the new signature, so references
follow the geometry through a series of edits.
"""
from __future__ import annotations

import math

import numpy as np
from build123d import GeomType, Shape, Vector

EXACT_TOL = 1e-3  # mm: the old position still holds a matching face/edge
MAX_SCORE = 1.5  # beyond this, the best candidate is too different to be the same one
MARGIN = 0.1  # the runner-up must score at least this much worse (plus a share of the best score)
MARGIN_SHARE = 0.25
PRESELECT = 6  # candidates that get the (slower) locality and surroundings checks


class RefError(ValueError):
    """A stored face or edge can't be found again unambiguously."""


def _vec(v) -> np.ndarray:
    return np.array([v.X, v.Y, v.Z], dtype=float)


def _on_plane_face(face, point: np.ndarray, tol: float) -> bool:
    """True if `point`, projected onto the plane of planar `face`, lies on the face."""
    c, n = _vec(face.center()), _vec(face.normal_at())
    p = point - float(np.dot(point - c, n)) * n
    try:
        return float(face.distance_to(Vector(*p))) <= tol
    except Exception:
        return False


class Topology:
    """A part's faces and edges, with edge -> adjacent faces built on demand."""

    def __init__(self, part: Shape):
        self.part = part
        self.faces = part.faces()
        self.edges = part.edges()
        self.scale = max(float(part.bounding_box().diagonal), 1e-6)
        self.tol = max(EXACT_TOL, 1e-5 * self.scale)  # "lands on the face" tolerance
        self._edge_faces = None

    def faces_of(self, edge) -> list:
        if self._edge_faces is None:
            self._edge_faces = {}
            for f in self.faces:
                for fe in f.edges():
                    self._edge_faces.setdefault(hash(fe), []).append((fe, f))
        out = []
        for fe, f in self._edge_faces.get(hash(edge), []):
            if fe.wrapped.IsSame(edge.wrapped) and not any(f.wrapped.IsSame(g.wrapped) for g in out):
                out.append(f)
        return out


# -- signatures
def _normal_at(face, point) -> list[float]:
    try:
        n = _vec(face.normal_at(point)) if face.geom_type != GeomType.PLANE else _vec(face.normal_at())
    except Exception:
        return [0.0, 0.0, 0.0]
    return n.tolist()


def _line_dir(edge) -> np.ndarray | None:
    if edge.geom_type != GeomType.LINE:
        return None
    d = _vec(edge @ 1) - _vec(edge @ 0)
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-12 else None


def face_signature(topo: Topology, face) -> dict:
    nbr = []
    for e in face.edges():
        mid = e @ 0.5
        for g in topo.faces_of(e):
            if not g.wrapped.IsSame(face.wrapped):
                nbr.append(_normal_at(g, mid))
    return {"kind": "face", "type": face.geom_type.name, "center": _vec(face.center()).tolist(),
            "normal": _vec(face.normal_at()).tolist(), "area": float(face.area), "nbr": nbr}


def edge_signature(topo: Topology, edge) -> dict:
    mid = edge @ 0.5
    d = _line_dir(edge)
    radius = None
    if edge.geom_type == GeomType.CIRCLE:
        try:
            radius = float(edge.radius)
        except Exception:
            radius = None
    return {"kind": "edge", "type": edge.geom_type.name, "mid": _vec(mid).tolist(),
            "dir": None if d is None else d.tolist(), "length": float(edge.length), "radius": radius,
            "faces": [_normal_at(f, mid) for f in topo.faces_of(edge)]}


def edge_path(edge) -> list[tuple]:
    """Sampled points along an edge (2 for a line, 32 otherwise), like the mesh polylines."""
    n = 2 if edge.geom_type == GeomType.LINE else 32
    return [tuple(map(float, _vec(edge @ (i / (n - 1))))) for i in range(n)]


# -- scoring
def _ratio(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return min(abs(math.log(max(a, 1e-9) / max(b, 1e-9))), 3.0)


def _normals_mismatch(ns0, ns1) -> float:
    """Share of normals in one list with no (nearly) equal normal in the other."""
    if not ns0 and not ns1:
        return 0.0
    a, b = np.asarray(ns0, float).reshape(-1, 3), np.asarray(ns1, float).reshape(-1, 3)
    matched = sum(1 for n in a if len(b) and float(np.max(b @ n)) > 0.999)
    return 1.0 - matched / max(len(a), len(b))


def _pick(scored: list[tuple[float, object]], what: str):
    """The clearly-best candidate, or a RefError saying why there isn't one."""
    if not scored:
        raise RefError(f"{what} can't be found after the change")
    scored.sort(key=lambda t: t[0])
    best = scored[0][0]
    if best > MAX_SCORE:
        raise RefError(f"{what} changed too much to be found again")
    if len(scored) > 1 and scored[1][0] - best < MARGIN + MARGIN_SHARE * best:
        close = sum(1 for s, _ in scored if s - best < MARGIN + MARGIN_SHARE * best)
        raise RefError(f"{what} is ambiguous after the change ({close} similar candidates)")
    return scored[0][1]


def resolve_face(topo: Topology, sig: dict, what: str = "its face"):
    """The planar face in `topo` that `sig` refers to; raises RefError."""
    c0, n0 = np.asarray(sig["center"], float), np.asarray(sig["normal"], float)
    n0 = n0 / max(float(np.linalg.norm(n0)), 1e-12)
    planes = [(f, _vec(f.center()), _vec(f.normal_at())) for f in topo.faces if f.geom_type == GeomType.PLANE]
    same_dir = [(f, c) for f, c, n in planes if float(np.dot(n, n0)) > 0.999]
    for f, c in same_dir:  # exact: still there
        if float(np.linalg.norm(c - c0)) < EXACT_TOL:
            return f
    area0 = sig.get("area")
    cheap = sorted(((float(np.linalg.norm(c - c0)) / topo.scale + 0.5 * _ratio(f.area, area0), f)
                    for f, c in same_dir), key=lambda t: t[0])[:PRESELECT]
    scored = []
    for s, f in cheap:
        if not _on_plane_face(f, c0, topo.tol):
            continue  # a look-alike elsewhere, not this face moved or resized
        if "nbr" in sig:
            s += 0.5 * _normals_mismatch(sig["nbr"], face_signature(topo, f)["nbr"])
        scored.append((s, f))
    return _pick(scored, what)


def resolve_edge(topo: Topology, sig: dict, what: str = "its edge"):
    """The edge in `topo` that `sig` refers to; raises RefError."""
    m0 = np.asarray(sig["mid"], float)
    etype = sig.get("type")
    d0 = None if sig.get("dir") is None else np.asarray(sig["dir"], float)
    cands = []
    for e in topo.edges:
        if etype is not None and e.geom_type.name != etype:
            continue
        mid = _vec(e @ 0.5)
        if float(np.linalg.norm(mid - m0)) < EXACT_TOL:
            return e  # exact: still there
        if d0 is not None:
            d = _line_dir(e)
            if d is None or abs(float(np.dot(d, d0))) < 0.999:
                continue  # a straight edge keeps its direction
        s = float(np.linalg.norm(mid - m0)) / topo.scale + 0.5 * _ratio(e.length, sig.get("length"))
        if sig.get("radius") is not None and e.geom_type == GeomType.CIRCLE:
            try:
                s += 0.5 * _ratio(float(e.radius), sig["radius"])
            except Exception:
                pass
        cands.append((s, e))
    cands.sort(key=lambda t: t[0])
    scored = []
    for s, e in cands[:PRESELECT]:
        faces = topo.faces_of(e)
        if not all(_on_plane_face(f, m0, topo.tol) for f in faces if f.geom_type == GeomType.PLANE):
            continue  # a look-alike elsewhere, not this edge moved or resized
        if sig.get("faces"):
            mid = e @ 0.5
            s += 0.5 * _normals_mismatch(sig["faces"], [_normal_at(f, mid) for f in faces])
        scored.append((s, e))
    return _pick(scored, what)
