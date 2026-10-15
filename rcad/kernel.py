"""Feature history, rebuild, tessellation with face/edge reference data, and project files."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from build123d import GeomType, Shape, Solid, Vector, export_step, export_stl

from .sketch import Sketch

FORMAT_VERSION = 1
KINDS = ("sketch", "extrude", "pushpull", "fillet", "chamfer")


@dataclass
class Feature:
    kind: str
    name: str = ""
    sketch: Sketch | None = None  # kind == "sketch"
    sketch_ref: int = -1  # kind == "extrude": index of the sketch feature
    value: float = 0.0  # extrude/pushpull distance, fillet radius, chamfer length
    cut: bool = False
    face: tuple | None = None  # pushpull: (center xyz, normal xyz) of the picked planar face
    edges: list = field(default_factory=list)  # fillet/chamfer: midpoints of picked edges

    def label(self) -> str:
        if self.kind == "sketch":
            return f"{self.name} [{self.sketch.status}]"
        if self.kind == "extrude":
            return f"{'Cut' if self.cut else 'Extrude'} ({self.value:g}) of #{self.sketch_ref}"
        if self.kind == "pushpull":
            return f"Push/Pull {self.value:+g}"
        return f"{self.kind.title()} {self.value:g} x{len(self.edges)}"

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "name": self.name, "sketch_ref": self.sketch_ref, "value": self.value,
             "cut": self.cut, "face": self.face, "edges": [list(e) for e in self.edges]}
        if self.sketch:
            d["sketch"] = self.sketch.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict) -> "Feature":
        if d["kind"] not in KINDS:
            raise ValueError(f"unknown feature kind {d['kind']!r}")
        return Feature(d["kind"], d.get("name", ""), Sketch.from_dict(d["sketch"]) if "sketch" in d else None,
                       d.get("sketch_ref", -1), d.get("value", 0.0), d.get("cut", False),
                       tuple(map(tuple, d["face"])) if d.get("face") else None,
                       [tuple(e) for e in d.get("edges", [])])


@dataclass
class FaceInfo:
    center: np.ndarray
    normal: np.ndarray | None  # None for non-planar faces


@dataclass
class EdgeInfo:
    mid: np.ndarray
    polyline: np.ndarray


@dataclass
class MeshData:
    verts: np.ndarray  # (N,3) float32
    tris: np.ndarray  # (M,3) uint32 into verts
    tri_face: np.ndarray  # (M,) index into faces per triangle
    faces: list[FaceInfo]
    edges: list[EdgeInfo]


def _vec(v) -> np.ndarray:
    return np.array([v.X, v.Y, v.Z], dtype=float)


def _main_solid(shape: Shape) -> Solid:
    solids = shape.solids()
    if not solids:
        raise ValueError("model has no solid")
    return max(solids, key=lambda s: s.volume)


def build_mesh(shape: Shape, tol: float = 0.05) -> MeshData:
    verts, tris, owner, faces = [], [], [], []
    base = 0
    for face in shape.faces():
        v, t = face.tessellate(tol)
        if not t:
            continue
        verts += [[p.X, p.Y, p.Z] for p in v]
        tris += [[a + base, b + base, c + base] for a, b, c in t]
        owner += [len(faces)] * len(t)
        base += len(v)
        planar = face.geom_type == GeomType.PLANE
        faces.append(FaceInfo(_vec(face.center()), _vec(face.normal_at()) if planar else None))
    edges = []
    for e in shape.edges():
        n = 2 if e.geom_type == GeomType.LINE else 32
        pts = np.array([_vec(e @ (i / (n - 1))) for i in range(n)])
        edges.append(EdgeInfo(_vec(e @ 0.5), pts))
    return MeshData(np.array(verts, np.float32).reshape(-1, 3), np.array(tris, np.uint32).reshape(-1, 3),
                    np.array(owner, np.int32), faces, edges)


def _find_planar_face(part: Shape, center, normal):
    c, n = np.asarray(center, float), np.asarray(normal, float)
    best, best_d = None, 1e9
    for f in part.faces():
        if f.geom_type != GeomType.PLANE:
            continue
        fn, fc = _vec(f.normal_at()), _vec(f.center())
        if np.dot(fn, n) < 0.999 or abs(np.dot(fc - c, n)) > 1e-4:
            continue
        d = np.linalg.norm(fc - c)
        if d < best_d:
            best, best_d = f, d
    return best


def _find_edge(part: Shape, mid, tol=1e-3):
    m = np.asarray(mid, float)
    best, best_d = None, tol
    for e in part.edges():
        d = np.linalg.norm(_vec(e @ 0.5) - m)
        if d < best_d:
            best, best_d = e, d
    return best


@dataclass
class Document:
    features: list[Feature] = field(default_factory=list)
    part: Shape | None = None
    mesh: MeshData | None = None
    errors: dict[int, str] = field(default_factory=dict)

    def delete(self, i: int) -> None:
        """Remove a feature; deleting a sketch also removes the extrudes that use it."""
        drop = {i}
        if self.features[i].kind == "sketch":
            drop |= {j for j, g in enumerate(self.features) if g.kind == "extrude" and g.sketch_ref == i}
        keep = [g for j, g in enumerate(self.features) if j not in drop]
        for g in keep:
            if g.kind == "extrude":
                g.sketch_ref -= sum(1 for d in drop if d < g.sketch_ref)
        self.features = keep

    def rebuild(self) -> None:
        """Replay the history; the first failing feature stops the replay and is reported in `errors`."""
        part, self.errors = None, {}
        for i, f in enumerate(self.features):
            try:
                part = self._apply(part, f)
            except Exception as e:  # kernel errors are reported per feature
                self.errors[i] = str(e) or type(e).__name__
                break
        self.part = part
        self.mesh = build_mesh(part) if part is not None else None

    def _apply(self, part: Shape | None, f: Feature) -> Shape | None:
        if f.kind == "sketch":
            return part
        if f.kind == "extrude":
            if not 0 <= f.sketch_ref < len(self.features) or self.features[f.sketch_ref].kind != "sketch":
                raise ValueError("extrude has no valid sketch")
            solid = self.features[f.sketch_ref].sketch.solid(f.value)
            if solid is None:
                raise ValueError("sketch has no closed profile or distance is zero")
            return self._combine(part, solid, f.cut)
        if f.kind == "pushpull":
            if part is None:
                raise ValueError("nothing to push/pull")
            face = _find_planar_face(part, *f.face)
            if face is None:
                raise ValueError("face no longer exists")
            n = np.asarray(f.face[1], float) * f.value
            solid = Solid.extrude(face, Vector(float(n[0]), float(n[1]), float(n[2])))
            return self._combine(part, solid, f.value < 0)
        if f.kind in ("fillet", "chamfer"):
            if part is None:
                raise ValueError("nothing to modify")
            edges = [_find_edge(part, m) for m in f.edges]
            if not edges or any(e is None for e in edges):
                raise ValueError("edge no longer exists")
            solid = _main_solid(part)
            return solid.fillet(f.value, edges) if f.kind == "fillet" else solid.chamfer(f.value, None, edges)
        raise ValueError(f"unknown feature {f.kind}")

    @staticmethod
    def _combine(part, solid, cut):
        if cut:
            if part is None:
                raise ValueError("cut needs an existing body")
            return part.cut(solid).clean()
        return solid if part is None else part.fuse(solid).clean()

    # -- files
    def to_dict(self) -> dict:
        return {"format": "rcad", "version": FORMAT_VERSION, "features": [f.to_dict() for f in self.features]}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @staticmethod
    def load(path: str) -> "Document":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("format") != "rcad" or d.get("version", 0) > FORMAT_VERSION:
            raise ValueError("not an rCad project (or saved by a newer version)")
        doc = Document([Feature.from_dict(f) for f in d["features"]])
        doc.rebuild()
        return doc

    def export(self, path: str) -> None:
        if self.part is None:
            raise ValueError("Nothing to export")
        if path.lower().endswith((".step", ".stp")):
            export_step(self.part, path)
        else:
            export_stl(self.part, path)
