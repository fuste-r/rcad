"""Feature history, rebuild, tessellation with face/edge reference data, and project files."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from build123d import Axis, GeomType, Shape, Solid, Vector, export_step, export_stl

from .sketch import Sketch
from .units import MM, UNITS, Unit

FORMAT_VERSION = 1
KINDS = ("sketch", "extrude", "revolve", "pushpull", "fillet", "chamfer")


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
    visible: bool = True  # layer toggle: sketches show/hide curves, other features are suppressed when off
    profiles: list | None = None  # extrude: points inside the sketch regions to use; None = all regions
    auto: bool = False  # extrude: cut when the extrusion mostly lies inside the body, otherwise add
    resolved_cut: bool | None = field(default=None, compare=False, repr=False)  # last auto decision
    version: int = 0  # incremented on edit to enable shape caching
    result_edges: list = field(default_factory=list, repr=False)  # fillet: mids of the rounded edges it created
    edge_dirs: list = field(default_factory=list, repr=False)  # fillet: unit direction of each picked edge (pre-fillet mesh)
    axis: str = "v"  # revolve: sketch-local axis, "u" or "v"

    def label(self, unit: Unit = MM) -> str:
        if self.kind == "sketch":
            return f"{self.name} [{self.sketch.status}]"
        if self.kind == "extrude":
            part = f", {len(self.profiles)} region(s)" if self.profiles is not None else ""
            cut = self.resolved_cut if self.auto and self.resolved_cut is not None else self.cut
            mode = " auto" if self.auto else ""
            return f"{'Cut' if cut else 'Extrude'}{mode} ({unit.fmt(self.value)}{part}) of #{self.sketch_ref}"
        if self.kind == "revolve":
            mode = "Cut" if self.cut else "Revolve"
            part = f", {len(self.profiles)} region(s)" if self.profiles is not None else ""
            return f"{mode} ({self.value:g} deg, {self.axis.upper()} axis{part}) of #{self.sketch_ref}"
        if self.kind == "pushpull":
            return f"Push/Pull {'+' if self.value >= 0 else '-'}{unit.fmt(abs(self.value))}"
        return f"{self.kind.title()} {unit.fmt(self.value)} x{len(self.edges)}"

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "name": self.name, "sketch_ref": self.sketch_ref, "value": self.value,
             "cut": self.cut, "face": self.face, "edges": [list(e) for e in self.edges], "visible": self.visible,
             "profiles": self.profiles,             "auto": self.auto, "result_edges": [list(e) for e in self.result_edges],
             "axis": self.axis,
             "edge_dirs": [list(d) for d in self.edge_dirs]}
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
                       [tuple(e) for e in d.get("edges", [])], d.get("visible", True), d.get("profiles"),
                       d.get("auto", False), result_edges=[tuple(e) for e in d.get("result_edges", [])],
                       edge_dirs=[tuple(d) for d in d.get("edge_dirs", [])], axis=d.get("axis", "v"))


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


def _mostly_inside(part: Shape, solid: Shape) -> bool:
    """True when more than half of `solid` overlaps `part`, i.e. the extrusion goes into the body."""
    try:
        res = part.intersect(solid)
        shapes = res if isinstance(res, list) else [res] if res else []
        return sum(s.volume for s in shapes) > 0.5 * solid.volume
    except Exception:  # degenerate boolean: treat as outside
        return False


def _find_edge(part: Shape, mid, tol=1e-3):
    m = np.asarray(mid, float)
    best, best_d = None, tol
    for e in part.edges():
        d = np.linalg.norm(_vec(e @ 0.5) - m)
        if d < best_d:
            best, best_d = e, d
    return best


def _is_seam(part: Shape, e) -> bool:
    """True when the edge is a seam: both sides are the same (or parallel) surface,
    e.g. the seam of an extruded cylinder — a fillet/chamfer cannot round it."""
    faces = [f for f in part.faces() if e in f.edges()]
    if len(faces) < 2:
        return True
    d = abs(float(np.dot(_vec(faces[0].normal_at()), _vec(faces[1].normal_at()))))
    return d > 1 - 1e-6


@dataclass
class Document:
    features: list[Feature] = field(default_factory=list)
    part: Shape | None = None
    mesh: MeshData | None = None
    errors: dict[int, str] = field(default_factory=dict)
    units: str = "mm"  # display/input units; geometry is always stored in mm
    variables: dict[str, float] = field(default_factory=dict)  # expression variables
    _shapes: list[Shape | None] = field(default_factory=list, repr=False)  # cached shape after each feature
    _dirty_from: int = 0  # index from which to recompute; 0 = full rebuild
    _last_versions: list[int] = field(default_factory=list, repr=False)  # feature version at last rebuild

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
        # invalidating the whole cache is safe: indices shifted
        self._shapes = []
        self._dirty_from = 0
        self._last_versions = []

    def invalidate_from(self, i: int) -> None:
        """Mark the cache invalid from feature i onward (their inputs changed)."""
        self._dirty_from = max(0, i)

    @staticmethod
    def _feature_version(f: Feature) -> int:
        """Combined version of a feature and its sketch (if present)."""
        v = f.version
        if f.sketch is not None and f.sketch.version > v:
            v = f.sketch.version
        return v

    def rebuild(self, dirty_from: int | None = None) -> None:
        """Replay the history; the first failing feature stops the replay and is reported in `errors`.

        `dirty_from` (default: auto-detected from feature/sketch versions, or the cached `_dirty_from`)
        is the first index to recompute; features before it reuse their cached shapes.
        Pass 0 for a full rebuild.
        """
        if dirty_from is None and self._shapes:
            # find the first feature whose version (or its sketch's version) changed since the last rebuild
            dirty_from = next((i for i, f in enumerate(self.features)
                               if i >= len(self._last_versions) or self._last_versions[i] != self._feature_version(f)),
                               self._dirty_from)
        if dirty_from is not None:
            self._dirty_from = dirty_from
        start = self._dirty_from
        part, self.errors = None, {}
        for i, f in enumerate(self.features):
            if i < start and i < len(self._shapes) and self._shapes[i] is not None:
                part = self._shapes[i]
                continue
            try:
                part = self._apply(part, f)
            except Exception as e:  # kernel errors are reported per feature
                self.errors[i] = str(e) or type(e).__name__
                break
            # cache this shape (even for sketches/hidden features, part is unchanged)
            self._shapes = self._shapes[:i] + [part]
        self._last_versions = [self._feature_version(f) for f in self.features]
        self.part = part
        self.mesh = build_mesh(part) if part is not None else None

    def _apply(self, part: Shape | None, f: Feature) -> Shape | None:
        if f.kind == "sketch" or not f.visible:
            return part
        if f.kind in ("extrude", "revolve"):
            if not 0 <= f.sketch_ref < len(self.features) or self.features[f.sketch_ref].kind != "sketch":
                raise ValueError(f"{f.kind} has no valid sketch")
            sketch = self.features[f.sketch_ref].sketch
            sketch.variables = self.variables
            if f.kind == "extrude":
                solid = sketch.solid(f.value, f.profiles)
                if solid is None:
                    raise ValueError("sketch has no matching closed region or distance is zero")
                cut = f.cut
                if f.auto:
                    cut = f.resolved_cut = part is not None and _mostly_inside(part, solid)
                return self._combine(part, solid, cut)
            if f.axis not in ("u", "v"):
                raise ValueError("revolve axis must be sketch U or V")
            if abs(f.value) < 1e-6 or abs(f.value) > 360:
                raise ValueError("revolve angle must be between 0 and 360 degrees")
            direction = sketch.plane.xdir if f.axis == "u" else sketch.plane.ydir
            axis = Axis(sketch.plane.origin, direction)
            regions = sketch.regions()
            if f.profiles is not None:
                selected = {i for p in f.profiles if (i := sketch.region_at(*p)) is not None}
                regions = [region for i, region in enumerate(regions) if i in selected]
            if not regions:
                raise ValueError("sketch has no matching closed region")
            solid = None
            for region in regions:
                face = sketch.plane.b123d().from_local_coords(sketch._region_face(region))
                revolved = Solid.revolve(face, f.value, axis)
                solid = revolved if solid is None else solid.fuse(revolved).clean()
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
            corners = [e for e in edges if not _is_seam(part, e)]
            if not corners:
                raise ValueError("picked edge is a seam (same surface on both sides) — cannot be rounded")
            solids = part.solids()
            modified: list[Solid] = []
            for s in solids:
                s_edges = [e for e in corners if e in s.edges()]
                if not s_edges:
                    modified.append(s)
                elif f.kind == "fillet":
                    modified.append(s.fillet(f.value, s_edges))
                else:
                    modified.append(s.chamfer(f.value, None, s_edges))
            # Multi-solid parts (e.g. a disconnected model) keep every modified
            # body; fuse into one Part rather than the old Solid(...) container.
            return modified[0] if len(modified) == 1 else modified[0].fuse(*modified[1:]).clean()
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
        return {"format": "rcad", "version": FORMAT_VERSION, "units": self.units,
                "features": [f.to_dict() for f in self.features],
                "variables": self.variables}

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @staticmethod
    def load(path: str) -> "Document":
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("format") != "rcad" or d.get("version", 0) > FORMAT_VERSION:
            raise ValueError("not an rCad project (or saved by a newer version)")
        doc = Document([Feature.from_dict(f) for f in d["features"]], units=d.get("units", "mm"))
        if doc.units not in UNITS:
            raise ValueError(f"unknown units {doc.units!r}")
        doc.variables = d.get("variables", {})
        if not isinstance(doc.variables, dict):
            doc.variables = {}
        for f in doc.features:
            if f.kind == "sketch" and f.sketch is not None:
                f.sketch.variables = doc.variables
        doc.rebuild()
        return doc

    def export(self, path: str) -> None:
        if self.part is None:
            raise ValueError("Nothing to export")
        if path.lower().endswith((".step", ".stp")):
            export_step(self.part, path)
        else:
            export_stl(self.part, path)
