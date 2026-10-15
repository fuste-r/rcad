from __future__ import annotations

import numpy as np
from OpenGL import GL
from PySide6.QtCore import Qt, Signal
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from .camera import Camera, point_segment_dist, ray_plane, ray_triangles
from .kernel import MeshData
from .sketch import Sketch

DRAG_PX = 4


class Viewport(QOpenGLWidget):
    clicked = Signal(float, float, object, object)  # x, y, Qt.MouseButton, Qt.KeyboardModifier
    hovered = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.camera = Camera()
        self.unit_mm = 1.0  # millimetres per display unit; the grid steps in whole display units
        self.mesh: MeshData | None = None
        self.sketch: Sketch | None = None
        self.overlay_sketches: list[Sketch] = []  # sketches on visible layers, drawn faded
        self.region_fill: list[tuple[np.ndarray, tuple]] = []  # (world triangles, RGBA) for highlighted regions
        self.sketch_sel: set[tuple[str, int]] = set()
        self.preview: list[np.ndarray] = []  # world-space polylines for rubber-banding
        self.sel_faces: set[int] = set()
        self.sel_edges: set[int] = set()
        self.hover_face: int | None = None
        self.hover_edge: int | None = None
        self.left_orbits = True
        self._press = None
        self._dragged = False
        self._last = None
        self._tri = np.zeros((0, 3, 3))
        self._verts = np.zeros((0, 3), np.float32)
        self._norms = np.zeros((0, 3), np.float32)
        self._segs = np.zeros((0, 2, 3))
        self._seg_edge = np.zeros(0, int)

    # -- data
    def set_mesh(self, mesh: MeshData | None):
        self.mesh = mesh
        self.sel_faces.clear()
        self.sel_edges.clear()
        self.hover_face = self.hover_edge = None
        if mesh is None or not len(mesh.tris):
            self._tri = np.zeros((0, 3, 3))
            self._verts = np.zeros((0, 3), np.float32)
            self._norms = np.zeros((0, 3), np.float32)
            self._segs = np.zeros((0, 2, 3))
            self._seg_edge = np.zeros(0, int)
        else:
            self._tri = mesh.verts[mesh.tris].astype(float)
            n = np.cross(self._tri[:, 1] - self._tri[:, 0], self._tri[:, 2] - self._tri[:, 0])
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            self._verts = np.ascontiguousarray(self._tri.reshape(-1, 3), dtype=np.float32)
            self._norms = np.ascontiguousarray(np.repeat(n, 3, axis=0), dtype=np.float32)
            segs, owner = [], []
            for i, e in enumerate(mesh.edges):
                segs.append(np.stack([e.polyline[:-1], e.polyline[1:]], axis=1))
                owner += [i] * (len(e.polyline) - 1)
            self._segs = np.concatenate(segs) if segs else np.zeros((0, 2, 3))
            self._seg_edge = np.array(owner, int)
        self.update()

    def frame_all(self):
        if self.mesh is not None and len(self.mesh.verts):
            lo, hi = self.mesh.verts.min(0), self.mesh.verts.max(0)
            self.camera.target = (lo + hi) / 2.0
            self.camera.dist = max(float(np.linalg.norm(hi - lo)) * 1.8, 20.0)
        self.update()

    # -- picking
    def pick_face(self, x, y) -> int | None:
        return self.pick_face_hit(x, y)[0]

    def pick_face_hit(self, x, y):
        """-> (face index, ray distance), or (None, None) when no model face is under the cursor."""
        if self.mesh is None or not len(self._tri):
            return None, None
        o, d = self.camera.ray(x, y, self.width(), self.height())
        k, t = ray_triangles(o, d, self._tri)
        return (None, None) if k is None else (int(self.mesh.tri_face[k]), t)

    def pick_edge(self, x, y, px=7.0) -> int | None:
        if not len(self._segs):
            return None
        w, h = self.width(), self.height()
        a = self.camera.project(self._segs[:, 0], w, h)
        b = self.camera.project(self._segs[:, 1], w, h)
        ok = ~(np.isnan(a).any(1) | np.isnan(b).any(1))
        if not ok.any():
            return None
        dist = np.full(len(a), np.inf)
        dist[ok] = point_segment_dist(np.array([x, y]), a[ok], b[ok])
        k = int(np.argmin(dist))
        return int(self._seg_edge[k]) if dist[k] <= px else None

    def sketch_uv(self, x, y):
        if self.sketch is None:
            return None
        pl = self.sketch.plane
        o, d = self.camera.ray(x, y, self.width(), self.height())
        p = ray_plane(o, d, pl.origin, np.array(pl.normal))
        return None if p is None else pl.to_uv(p)

    def _sketch_world(self):
        pl = self.sketch.plane
        pts = np.array([pl.to3d(*p) for p in self.sketch.points]).reshape(-1, 3)
        return pl, pts

    def pick_sketch(self, x, y, px=8.0):
        """Nearest sketch entity as ('p'|'l'|'c', index); points win over lines over circles."""
        sk = self.sketch
        if sk is None or not sk.points:
            return None
        w, h = self.width(), self.height()
        pl, world = self._sketch_world()
        scr = self.camera.project(world, w, h)
        m = np.array([x, y])
        d = np.linalg.norm(scr - m, axis=1)
        d[np.isnan(d)] = np.inf
        if d.min() <= px:
            return ("p", int(np.argmin(d)))
        best, best_d = None, px
        for i, (a, b) in enumerate(sk.lines):
            if np.isnan(scr[a]).any() or np.isnan(scr[b]).any():
                continue
            dd = point_segment_dist(m, scr[a][None], scr[b][None])[0]
            if dd < best_d:
                best, best_d = ("l", i), dd
        for i, (c, r) in enumerate(sk.circles):
            ring = self.camera.project(np.array(self._circle_world(pl, sk.points[c], r)), w, h)
            ring = ring[~np.isnan(ring).any(1)]
            if len(ring) > 2:
                dd = point_segment_dist(m, ring[:-1], ring[1:]).min()
                if dd < best_d:
                    best, best_d = ("c", i), dd
        return best

    def nearest_sketch_point(self, x, y, px=10.0) -> int | None:
        hit = self.pick_sketch(x, y, px)
        return hit[1] if hit and hit[0] == "p" else None

    @staticmethod
    def _circle_world(pl, center, r, n=64):
        t = np.linspace(0, 2 * np.pi, n + 1)
        return [pl.to3d(center[0] + r * np.cos(a), center[1] + r * np.sin(a)) for a in t]

    # -- GL
    def initializeGL(self):
        GL.glClearColor(0.93, 0.94, 0.96, 1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)  # highlight pass redraws the same triangles at equal depth
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, 1)
        GL.glEnable(GL.GL_LINE_SMOOTH)

    def paintGL(self):
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        w, h = max(self.width(), 1), max(self.height(), 1)
        r = self.devicePixelRatio()
        GL.glViewport(0, 0, int(w * r), int(h * r))
        view, proj = self.camera.matrices(w, h)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadMatrixf(proj.T.astype(np.float32))
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, (0.3, 0.5, 1.0, 0.0))
        GL.glLoadMatrixf(view.T.astype(np.float32))

        self._draw_grid()
        self._draw_model()
        self._draw_sketch()

    def _grid_step(self):
        return self.unit_mm * 10.0 ** np.floor(np.log10(self.camera.dist / 4 / self.unit_mm))

    def _draw_grid(self):
        GL.glDisable(GL.GL_LIGHTING)
        step = self._grid_step()
        n = 10
        pl = self.sketch.plane if self.sketch else None
        GL.glLineWidth(1)
        GL.glColor3f(0.78, 0.8, 0.83)
        GL.glBegin(GL.GL_LINES)
        for i in range(-n, n + 1):
            for a, b in (((i * step, -n * step), (i * step, n * step)), ((-n * step, i * step), (n * step, i * step))):
                if pl:
                    GL.glVertex3f(*pl.to3d(*a))
                    GL.glVertex3f(*pl.to3d(*b))
                else:
                    GL.glVertex3f(a[0], a[1], 0)
                    GL.glVertex3f(b[0], b[1], 0)
        GL.glEnd()
        GL.glLineWidth(2)
        GL.glBegin(GL.GL_LINES)
        ext = step * n
        for col, vec in (((0.85, 0.2, 0.2), (ext, 0, 0)), ((0.2, 0.7, 0.2), (0, ext, 0)), ((0.2, 0.3, 0.9), (0, 0, ext))):
            GL.glColor3f(*col)
            GL.glVertex3f(0, 0, 0)
            GL.glVertex3f(*vec)
        GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_LIGHTING)

    def _tri_range(self, face):
        tf = self.mesh.tri_face
        return int(np.searchsorted(tf, face, "left")), int(np.searchsorted(tf, face, "right"))

    def _draw_model(self):
        if not len(self._verts):
            return
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        GL.glEnableClientState(GL.GL_VERTEX_ARRAY)
        GL.glEnableClientState(GL.GL_NORMAL_ARRAY)
        GL.glVertexPointer(3, GL.GL_FLOAT, 0, self._verts)
        GL.glNormalPointer(GL.GL_FLOAT, 0, self._norms)
        GL.glColor3f(0.62, 0.72, 0.86)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self._verts))
        for faces, col in (({self.hover_face} - {None}, (0.85, 0.85, 0.5)), (self.sel_faces, (1.0, 0.6, 0.2))):
            GL.glColor3f(*col)
            for f in faces:
                a, b = self._tri_range(f)
                GL.glDrawArrays(GL.GL_TRIANGLES, a * 3, (b - a) * 3)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        GL.glDisable(GL.GL_LIGHTING)
        for i, e in enumerate(self.mesh.edges):
            sel, hov = i in self.sel_edges, i == self.hover_edge
            GL.glLineWidth(3 if sel or hov else 1.5)
            GL.glColor3f(*((0.9, 0.1, 0.1) if sel else (0.1, 0.5, 0.9) if hov else (0.15, 0.15, 0.2)))
            GL.glBegin(GL.GL_LINE_STRIP)
            for p in e.polyline:
                GL.glVertex3f(*p)
            GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_LIGHTING)

    def _draw_sketch_geometry(self, sk: Sketch, selected: set, base_color):
        pl = sk.plane
        world = np.array([pl.to3d(*p) for p in sk.points]).reshape(-1, 3)

        def col(kind, i):
            GL.glColor3f(*((1.0, 0.35, 0.1) if (kind, i) in selected else base_color))

        GL.glLineWidth(2.5)
        for i, (a, b) in enumerate(sk.lines):
            col("l", i)
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(*world[a])
            GL.glVertex3f(*world[b])
            GL.glEnd()
        for i, (c, r) in enumerate(sk.circles):
            col("c", i)
            GL.glBegin(GL.GL_LINE_STRIP)
            for p in self._circle_world(pl, sk.points[c], r):
                GL.glVertex3f(*p)
            GL.glEnd()
        GL.glPointSize(8)
        GL.glBegin(GL.GL_POINTS)
        for i, p in enumerate(world):
            col("p", i)
            GL.glVertex3f(*p)
        GL.glEnd()

    def _draw_sketch(self):
        sk = self.sketch
        if sk is None and not self.preview and not self.overlay_sketches and not self.region_fill:
            return
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        if self.region_fill:
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            for tris, color in self.region_fill:
                GL.glColor4f(*color)
                GL.glBegin(GL.GL_TRIANGLES)
                for p in tris.reshape(-1, 3):
                    GL.glVertex3f(*p)
                GL.glEnd()
            GL.glDisable(GL.GL_BLEND)
        for other in self.overlay_sketches:
            if other is not sk:
                self._draw_sketch_geometry(other, set(), (0.45, 0.55, 0.75))
        if sk is not None:
            self._draw_sketch_geometry(sk, self.sketch_sel, (0.05, 0.25, 0.75))
        GL.glColor3f(0.4, 0.4, 0.4)
        GL.glLineWidth(1.5)
        for poly in self.preview:
            GL.glBegin(GL.GL_LINE_STRIP)
            for p in poly:
                GL.glVertex3f(*p)
            GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)

    # -- mouse
    def mousePressEvent(self, e):
        self._press = (e.position(), e.button(), e.modifiers())
        self._last = e.position()
        self._dragged = False

    def mouseMoveEvent(self, e):
        p = e.position()
        self.hovered.emit(p.x(), p.y())
        if self._press is None:
            return
        pos0, btn, mods = self._press
        d = p - self._last
        if (p - pos0).manhattanLength() > DRAG_PX:
            self._dragged = True
        self._last = p
        if not self._dragged:
            return
        alt = bool(mods & Qt.AltModifier)
        if btn == Qt.RightButton or (btn == Qt.LeftButton and (self.left_orbits or alt) and not mods & Qt.ShiftModifier):
            self.camera.orbit(-d.x() * 0.4, -d.y() * 0.4)
        elif btn == Qt.MiddleButton or (btn == Qt.LeftButton and mods & Qt.ShiftModifier):
            self.camera.pan(d.x(), d.y(), self.height())
        self.update()

    def mouseReleaseEvent(self, e):
        if self._press is not None and not self._dragged:
            p = e.position()
            self.clicked.emit(p.x(), p.y(), self._press[1], self._press[2])
        self._press = None

    def wheelEvent(self, e):
        self.camera.zoom(e.angleDelta().y() / 120)
        self.update()
