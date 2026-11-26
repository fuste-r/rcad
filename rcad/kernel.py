"""Feature history, rebuild, tessellation with face/edge reference data, and project files."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from build123d import Axis, Edge, GeomType, Plane, Shape, Solid, Vector, Wire, export_step, export_stl, loft

from . import refs
from .sketch import Sketch
from .units import MM, UNITS, Unit

FORMAT_VERSION = 1
KINDS = ("sketch", "extrude", "revolve", "sweep", "loft", "pushpull", "fillet", "chamfer", "move", "mirror")


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
    edge_dirs: list = field(default_factory=list, repr=False)  # fillet: aligned unit directions; None for curved edges
    edge_refs: list = field(default_factory=list, repr=False)  # fillet: sampled picked-edge paths, one per edge
    axis: str = "v"  # revolve: sketch-local axis, "u" or "v"
    path_ref: int = -1  # sweep: sketch feature containing the path
    section_refs: list = field(default_factory=list)  # loft: ordered sketch section feature indices
    ref_sigs: list = field(default_factory=list, repr=False)  # pushpull: [face signature]; fillet/chamfer: one per edge
    # move/mirror: ref_sigs holds one body signature per body moved
    offset: tuple = (0.0, 0.0, 0.0)  # move: translation in mm
    angle: float = 0.0  # move: rotation in degrees about rot_axis through the body's centre
    rot_axis: str = "z"  # move: "x" | "y" | "z"
    plane: str = "YZ"  # mirror: "YZ" | "XZ" | "XY"; `value` is the plane's offset along its normal
    copy: bool = False  # move/mirror: keep the original body too

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
        if self.kind == "sweep":
            return f"{'Cut' if self.cut else 'Sweep'} #{self.sketch_ref} along #{self.path_ref}"
        if self.kind == "loft":
            return f"{'Cut' if self.cut else 'Loft'} x{len(self.section_refs)} sections"
        if self.kind == "pushpull":
            return f"Push/Pull {'+' if self.value >= 0 else '-'}{unit.fmt(abs(self.value))}"
        if self.kind == "move":
            parts = [f"{'Copy' if self.copy else 'Move'} {', '.join(unit.fmt(v) for v in self.offset)}"]
            if self.angle:
                parts.append(f"{self.angle:g} deg about {self.rot_axis.upper()}")
            return " ".join(parts) + f" x{len(self.ref_sigs)}"
        if self.kind == "mirror":
            keep = " (copy)" if self.copy else ""
            return f"Mirror {self.plane} at {unit.fmt(self.value)}{keep} x{len(self.ref_sigs)}"
        return f"{self.kind.title()} {unit.fmt(self.value)} x{len(self.edges)}"

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "name": self.name, "sketch_ref": self.sketch_ref, "value": self.value,
             "cut": self.cut, "face": self.face, "edges": [list(e) for e in self.edges], "visible": self.visible,
             "profiles": self.profiles,             "auto": self.auto, "result_edges": [list(e) for e in self.result_edges],
             "axis": self.axis,
             "path_ref": self.path_ref, "section_refs": self.section_refs,
             "edge_dirs": [list(d) if d is not None else None for d in self.edge_dirs],
             "edge_refs": [[list(point) for point in path] if path is not None else None
                           for path in self.edge_refs],
             "ref_sigs": self.ref_sigs, "offset": list(self.offset), "angle": self.angle,
             "rot_axis": self.rot_axis, "plane": self.plane, "copy": self.copy}
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
                       edge_dirs=[tuple(direction) if direction is not None else None
                                  for direction in d.get("edge_dirs", [])],
                       edge_refs=[[tuple(point) for point in path] if path is not None else None
                                  for path in d.get("edge_refs", [])], axis=d.get("axis", "v"),
                       path_ref=d.get("path_ref", -1), section_refs=list(d.get("section_refs", [])),
                       ref_sigs=list(d.get("ref_sigs", [])),
                       offset=tuple(map(float, d.get("offset", (0.0, 0.0, 0.0)))), angle=d.get("angle", 0.0),
                       rot_axis=d.get("rot_axis", "z"), plane=d.get("plane", "YZ"), copy=d.get("copy", False))


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
    shape: Shape | None = field(default=None, repr=False)  # the shape meshed, for on-demand queries
    face_body: list = field(default_factory=list)  # body (solid) index of each face
    edge_body: list = field(default_factory=list)  # body (solid) index of each edge
    _edge_shapes: list = field(default_factory=list, repr=False)  # build123d edge for each EdgeInfo
    _away: dict = field(default_factory=dict, repr=False)

    def edge_away(self, i: int) -> np.ndarray | None:
        """Unit vector at edge i's middle pointing away from the material: the sum of the
        outward normals of the faces that meet there (convex corner: diagonally out of the
        body; concave: into the empty space). None if it can't be worked out. Cached."""
        if i not in self._away:
            self._away[i] = None
            if self.shape is not None and 0 <= i < len(self._edge_shapes):
                e = self._edge_shapes[i]
                faces = [f for f in self.shape.faces() if any(fe.wrapped.IsSame(e.wrapped) for fe in f.edges())]
                self._away[i] = _edge_out(e, e @ 0.5, [(e, f) for f in faces])
        return self._away[i]


def _vec(v) -> np.ndarray:
    return np.array([v.X, v.Y, v.Z], dtype=float)


def _main_solid(shape: Shape) -> Solid:
    solids = shape.solids()
    if not solids:
        raise ValueError("model has no solid")
    return max(solids, key=lambda s: s.volume)


def _edge_out(edge, mid, candidates) -> np.ndarray | None:
    """Direction away from the material at an edge: the sum of the outward normals of the faces
    that meet there (for a convex corner, the diagonal out of the body; for a concave one, into
    the empty space). None if it can't be worked out."""
    normals, seen = [], []
    for fe, face in candidates:
        if not fe.wrapped.IsSame(edge.wrapped) or any(face.wrapped.IsSame(f.wrapped) for f in seen):
            continue
        seen.append(face)
        try:
            normals.append(_vec(face.normal_at(mid)))
        except Exception:
            continue
    if not normals:
        return None
    out = np.sum(normals, axis=0)
    length = float(np.linalg.norm(out))
    return out / length if length > 1e-9 else None


def _body_index(shape: Shape, kind: str):
    """A function giving the solid index (0 if unknown) of a face or edge of `shape`."""
    found = {}
    for k, solid in enumerate(shape.solids()):
        for sub in getattr(solid, kind)():
            found.setdefault(hash(sub), []).append((sub, k))

    def body(sub) -> int:
        return next((k for other, k in found.get(hash(sub), []) if other.wrapped.IsSame(sub.wrapped)), 0)
    return body


def build_mesh(shape: Shape, tol: float = 0.05) -> MeshData:
    verts, tris, owner, faces, face_body = [], [], [], [], []
    base = 0
    face_of = _body_index(shape, "faces")
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
        face_body.append(face_of(face))
    edges, edge_shapes, edge_body = [], [], []
    edge_of = _body_index(shape, "edges")
    for e in shape.edges():
        n = 2 if e.geom_type == GeomType.LINE else 32
        pts = np.array([_vec(e @ (i / (n - 1))) for i in range(n)])
        edges.append(EdgeInfo(_vec(e @ 0.5), pts))
        edge_shapes.append(e)
        edge_body.append(edge_of(e))
    return MeshData(np.array(verts, np.float32).reshape(-1, 3), np.array(tris, np.uint32).reshape(-1, 3),
                    np.array(owner, np.int32), faces, edges, shape, face_body=face_body, edge_body=edge_body,
                    _edge_shapes=edge_shapes)


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


def _is_seam(topo: refs.Topology, e) -> bool:
    """True when the edge has no corner to round: a seam (the same surface on both sides,
    e.g. down the side of an extruded cylinder) or a smooth join (the faces meet
    tangentially, e.g. where an earlier fillet meets a flat face)."""
    faces = topo.faces_of(e)
    if len(faces) < 2:
        return True
    mid = e @ 0.5
    n0, n1 = (np.asarray(refs._normal_at(f, mid)) for f in faces[:2])
    return abs(float(np.dot(n0, n1))) > 1 - 1e-6


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
        if f.kind == "sweep":
            profile_sketch = self._sketch_for_feature(f.sketch_ref, "sweep profile")
            path_sketch = self._sketch_for_feature(f.path_ref, "sweep path")
            profile_regions = self._selected_regions(profile_sketch, f.profiles)
            if len(profile_regions) != 1:
                raise ValueError("sweep requires exactly one closed profile region")
            paths = [Edge.make_line(path_sketch.plane.to3d(*path_sketch.points[a]),
                                    path_sketch.plane.to3d(*path_sketch.points[b]))
                     for a, b in path_sketch.lines]
            wires = Wire.combine(paths)
            if len(wires) != 1 or not wires[0].is_valid:
                raise ValueError("sweep path must be one connected set of sketch lines")
            face = profile_sketch.plane.b123d().from_local_coords(
                profile_sketch._region_face(profile_regions[0]))
            solid = Solid.sweep(face, wires[0])
            return self._combine(part, solid, f.cut)
        if f.kind == "loft":
            if len(f.section_refs) < 2:
                raise ValueError("loft requires at least two section sketches")
            section_faces = []
            for ref in f.section_refs:
                section_sketch = self._sketch_for_feature(ref, "loft section")
                regions = section_sketch.regions()
                if len(regions) != 1:
                    raise ValueError("each loft section must contain exactly one closed profile")
                section_faces.append(section_sketch.plane.b123d().from_local_coords(
                    section_sketch._region_face(regions[0])))
            solid = loft(section_faces)
            return self._combine(part, solid, f.cut)
        if f.kind == "pushpull":
            if part is None:
                raise ValueError("nothing to push/pull")
            topo = refs.Topology(part)
            sig = f.ref_sigs[0] if f.ref_sigs else {"kind": "face", "center": list(f.face[0]),
                                                     "normal": list(f.face[1])}
            face = refs.resolve_face(topo, sig, "its face")
            # follow the face: store where it is now
            f.ref_sigs = [refs.face_signature(topo, face)]
            f.face = (tuple(map(float, _vec(face.center()))), tuple(map(float, f.face[1])))
            n = np.asarray(f.face[1], float) * f.value
            solid = Solid.extrude(face, Vector(float(n[0]), float(n[1]), float(n[2])))
            return self._combine(part, solid, f.value < 0)
        if f.kind in ("fillet", "chamfer"):
            if part is None:
                raise ValueError("nothing to modify")
            if not f.edges:
                raise ValueError("no edges picked")
            size = "radius" if f.kind == "fillet" else "length"
            if not f.value > 0:
                raise ValueError(f"{f.kind} {size} must be greater than zero")
            topo = refs.Topology(part)
            edges = []
            for k, m in enumerate(f.edges):
                sig = f.ref_sigs[k] if k < len(f.ref_sigs) else {"kind": "edge", "mid": list(m)}
                what = "its edge" if len(f.edges) == 1 else f"edge {k + 1} of {len(f.edges)}"
                e = refs.resolve_edge(topo, sig, what)
                if any(e.wrapped.IsSame(x.wrapped) for x in edges):
                    raise refs.RefError(f"{what} now matches the same edge as another picked edge")
                edges.append(e)
            # follow the edges: store where they are now
            f.ref_sigs = [refs.edge_signature(topo, e) for e in edges]
            f.edges = [tuple(sig["mid"]) for sig in f.ref_sigs]
            f.edge_refs = [refs.edge_path(e) for e in edges]
            if f.kind == "fillet":
                f.edge_dirs = [None if sig["dir"] is None else tuple(sig["dir"]) for sig in f.ref_sigs]
            corners = [e for e in edges if not _is_seam(topo, e)]
            if not corners:
                raise ValueError("picked edge is a seam or a smooth join (no corner) — cannot be rounded")
            solids = part.solids()
            modified: list[Solid] = []
            for s in solids:
                s_edges = [e for e in corners if e in s.edges()]
                if not s_edges:
                    modified.append(s)
                    continue
                try:
                    modified.append(s.fillet(f.value, s_edges) if f.kind == "fillet"
                                    else s.chamfer(f.value, None, s_edges))
                except Exception as e:  # build123d's own message names its API (max_fillet())
                    raise ValueError(f"{f.kind} {size} is too large for the picked edges "
                                     f"or the shape around them") from e
            # Multi-solid parts (e.g. a disconnected model) keep every modified
            # body; fuse into one Part rather than the old Solid(...) container.
            return modified[0] if len(modified) == 1 else modified[0].fuse(*modified[1:]).clean()
        if f.kind in ("move", "mirror"):
            return self._move_bodies(part, f)
        raise ValueError(f"unknown feature {f.kind}")

    AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
    PLANE_NORMALS = {"YZ": "x", "XZ": "y", "XY": "z"}

    def _move_bodies(self, part: Shape | None, f: Feature) -> Shape:
        """Move/rotate or mirror the picked bodies; with `copy` the originals stay too.
        Bodies that end up touching or overlapping join into one."""
        if part is None:
            raise ValueError(f"nothing to {f.kind}")
        if not f.ref_sigs:
            raise ValueError("no body picked")
        if f.kind == "move" and f.rot_axis not in self.AXES:
            raise ValueError("rotation axis must be X, Y or Z")
        if f.kind == "mirror" and f.plane not in self.PLANE_NORMALS:
            raise ValueError("mirror plane must be YZ, XZ or XY")
        solids = part.solids()
        scale = max(float(part.bounding_box().diagonal), 1e-6)
        picked: list[int] = []
        for k, sig in enumerate(f.ref_sigs):
            what = "its body" if len(f.ref_sigs) == 1 else f"body {k + 1} of {len(f.ref_sigs)}"
            i = refs.resolve_body(solids, sig, scale, what)
            if i in picked:
                raise refs.RefError(f"{what} now matches the same body as another picked body")
            picked.append(i)
        f.ref_sigs = [refs.body_signature(solids[i]) for i in picked]  # follow the bodies
        out = [s for j, s in enumerate(solids) if j not in picked]
        for i in picked:
            s = solids[i]
            if f.kind == "move":
                t = s
                if abs(f.angle) > 1e-12:
                    t = t.rotate(Axis(s.center(), self.AXES[f.rot_axis]), f.angle)
                t = t.translate(Vector(*map(float, f.offset)))
            else:
                n = np.asarray(self.AXES[self.PLANE_NORMALS[f.plane]])
                t = s.mirror(Plane(origin=tuple(n * f.value), z_dir=tuple(n)))
            if f.copy:
                out.append(s)
            out.append(t)
        return out[0] if len(out) == 1 else out[0].fuse(*out[1:]).clean()

    def _sketch_for_feature(self, index: int, role: str) -> Sketch:
        if not 0 <= index < len(self.features) or self.features[index].kind != "sketch":
            raise ValueError(f"{role} has no valid sketch")
        sketch = self.features[index].sketch
        sketch.variables = self.variables
        return sketch

    @staticmethod
    def _selected_regions(sketch: Sketch, profiles: list | None):
        regions = sketch.regions()
        if profiles is not None:
            selected = {i for point in profiles if (i := sketch.region_at(*point)) is not None}
            regions = [region for i, region in enumerate(regions) if i in selected]
        return regions

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
