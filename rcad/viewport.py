from __future__ import annotations

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel

from . import viewcube
from .camera import Camera, FOV_TAN, point_segment_dist, ray_plane, ray_triangles
from .kernel import MeshData
from .sketch import Sketch
from .theme import LIGHT, Theme
from .units import MM, Unit

DRAG_PX = 4


class Viewport(QOpenGLWidget):
    clicked = Signal(float, float, object, object)  # x, y, Qt.MouseButton, Qt.KeyboardModifier
    hovered = Signal(float, float)
    scrolled = Signal(float, float, int, object)  # x, y, angle delta, Qt.KeyboardModifier
    extrude_commit = Signal(float, tuple, tuple)  # distance_mm, center, normal
    sketch_mouse = Signal(str, float, float)  # phase ("press"|"move"|"release"), x, y
    fillet_commit = Signal(float, list)  # radius_mm, [edge midpoints]
    fillet_preview = Signal(float, list)  # live radius_mm, [edge midpoints]

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self._dim_labels = []
        self.camera = Camera()
        self.theme: Theme = LIGHT
        self.cube_hover: str | None = None
        self._cube_press = False
        self._cube_tex: dict[str, int] = {}
        self._cube_tex_key = None
        self.unit_mm = 1.0  # millimetres per display unit; the grid steps in whole display units
        self.unit = MM  # display unit for dimension labels
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
        self._extrude_state = None  # {"face", "center", "normal", "start_screen", "outline"}
        self._extrude_dist = 0.0
        self._extrude_label = QLabel(self)
        self._extrude_label.setStyleSheet(
            "color: #ffffff; background: rgba(0,128,0,180); padding: 2px 6px; "
            "border-radius: 3px; font-weight: bold;")
        self._fillet_state = None  # {"mids", "outward", "start_screen", "radius"}
        self._fillet_label = QLabel(self)
        self._fillet_label.setStyleSheet(
            "color: #ffffff; background: rgba(128,0,128,180); padding: 2px 6px; "
            "border-radius: 3px; font-weight: bold;")

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

    def _fillet_outward(self, edge_idx: int):
        """World unit vector the fillet grows along (perpendicular to the edge, in its face).
        Returns None if the edge's face can't be resolved."""
        m = self.mesh
        if m is None or not len(m.edges) or not (0 <= edge_idx < len(m.edges)):
            return None
        pts = m.edges[edge_idx].polyline
        if len(pts) < 2:
            return None
        d = (pts[-1] - pts[0]).astype(float)
        d = d / max(float(np.linalg.norm(d)), 1e-12)
        mid = np.asarray(m.edges[edge_idx].mid, float)
        w, h = self.width(), self.height()
        s = self.camera.project(mid.reshape(1, 3), w, h)[0]
        face = self.pick_face(float(s[0]), float(s[1]))
        if face is None or m.faces[face].normal is None:
            return None
        n = np.asarray(m.faces[face].normal, float)
        out = np.cross(d, n)
        if float(np.linalg.norm(out)) < 1e-9:
            out = np.cross(n, d)
        out = out / max(float(np.linalg.norm(out)), 1e-12)
        return out

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
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LEQUAL)  # highlight pass redraws the same triangles at equal depth
        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_NORMALIZE)
        GL.glLightModeli(GL.GL_LIGHT_MODEL_TWO_SIDE, 1)

    def _reset_gl_state(self):
        """Qt's own painting leaves texture/blend/array state behind, which tints fixed-function drawing."""
        GL.glUseProgram(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        for cap in (GL.GL_TEXTURE_2D, GL.GL_BLEND, GL.GL_CULL_FACE, GL.GL_STENCIL_TEST, GL.GL_SCISSOR_TEST,
                    GL.GL_LINE_STIPPLE, GL.GL_ALPHA_TEST):
            GL.glDisable(cap)
        for arr in (GL.GL_COLOR_ARRAY, GL.GL_TEXTURE_COORD_ARRAY, GL.GL_NORMAL_ARRAY, GL.GL_VERTEX_ARRAY):
            GL.glDisableClientState(arr)
        GL.glColorMask(GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE, GL.GL_TRUE)
        GL.glBlendEquation(GL.GL_FUNC_ADD)
        GL.glDisable(GL.GL_COLOR_LOGIC_OP)
        GL.glPolygonMode(GL.GL_FRONT_AND_BACK, GL.GL_FILL)

    def paintGL(self):
        self._reset_gl_state()
        GL.glClearColor(*self.theme.background, 1)
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
        self._draw_extrude_preview()
        self._draw_sketch()
        self._draw_cube()

    def _grid_step(self):
        return self.unit_mm * 10.0 ** np.floor(np.log10(self.camera.dist / 4 / self.unit_mm))

    def _draw_grid(self):
        GL.glDisable(GL.GL_LIGHTING)
        step = self._grid_step()
        n = 10
        pl = self.sketch.plane if self.sketch else None
        GL.glLineWidth(1)
        GL.glColor3f(*self.theme.grid)
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
        GL.glColor3f(*self.theme.model)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, len(self._verts))
        for faces, col in (({self.hover_face} - {None}, self.theme.face_hover), (self.sel_faces, self.theme.face_select)):
            GL.glColor3f(*col)
            for f in faces:
                a, b = self._tri_range(f)
                GL.glDrawArrays(GL.GL_TRIANGLES, a * 3, (b - a) * 3)
        GL.glDisableClientState(GL.GL_NORMAL_ARRAY)
        GL.glDisableClientState(GL.GL_VERTEX_ARRAY)
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)

        GL.glDisable(GL.GL_LIGHTING)
        t = self.theme
        for i, e in enumerate(self.mesh.edges):
            sel, hov = i in self.sel_edges, i == self.hover_edge
            GL.glLineWidth(3 if sel or hov else 1.5)
            GL.glColor3f(*(t.edge_select if sel else t.edge_hover if hov else t.edge))
            GL.glBegin(GL.GL_LINE_STRIP)
            for p in e.polyline:
                GL.glVertex3f(*p)
            GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_LIGHTING)

    def _face_outline(self, face_idx):
        """Boundary edges of a face as (vertex_a, vertex_b) pairs of global vertex indices."""
        m = self.mesh
        if m is None or not len(m.tris) or not (0 <= face_idx < len(m.tri_face)):
            return []
        tf, tris = m.tri_face, m.tris
        count: dict[tuple[int, int], int] = {}
        for i in np.where(tf == face_idx)[0]:
            a, b, c = tris[i]
            for e in ((a, b), (b, c), (c, a)):
                k = (min(e), max(e))
                count[k] = count.get(k, 0) + 1
        return [(u, v) for (u, v), n in count.items() if n == 1]

    def _draw_extrude_preview(self):
        """Draw the drag-to-extrude preview: offset face outline + drag axis."""
        if self._extrude_state is None or self.mesh is None:
            self._extrude_label.hide()
            return
        c, n = self._extrude_state["center"], self._extrude_state["normal"]
        d = self._extrude_dist
        if abs(d) < 1e-6:
            return
        GL.glDisable(GL.GL_LIGHTING)
        t = self.theme
        # 1. Drag axis: thick line from face center to offset position
        end = c + d * n
        GL.glLineWidth(3)
        GL.glColor3f(*t.model)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(*c)
        GL.glVertex3f(*end)
        GL.glEnd()
        # 2. Arrowhead (small V at the tip)
        up = np.array([0, 1, 0], float)
        side = np.cross(n, up)
        if np.linalg.norm(side) < 1e-9:
            side = np.array([1, 0, 0], float)
        side /= np.linalg.norm(side)
        L = 0.15 * max(1.0, abs(d))
        b1 = end - L * n + side * L * 0.5
        b2 = end - L * n - side * L * 0.5
        GL.glLineWidth(2)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(*b1)
        GL.glVertex3f(*end)
        GL.glVertex3f(*b2)
        GL.glEnd()
        # 3. Offset silhouette of the selected face (outline of the extruded top)
        verts = self.mesh.verts
        GL.glLineWidth(2)
        GL.glColor3f(0.0, 0.9, 0.6)  # teal for preview
        for a, b in self._extrude_state["outline"]:
            pa = np.asarray(verts[a], float) + d * n
            pb = np.asarray(verts[b], float) + d * n
            GL.glBegin(GL.GL_LINES)
            GL.glVertex3f(*pa)
            GL.glVertex3f(*pb)
            GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_LIGHTING)

        # 4. Distance label (project end point to screen)
        w, h = self.width(), self.height()
        screen = self.camera.project(end, w, h)
        x, y = screen[0], screen[1]
        self._extrude_label.setText(f"{d:.2f} mm")
        self._extrude_label.move(int(x) + 10, max(int(y) - 12, 2))
        self._extrude_label.show()

    def _draw_sketch_geometry(self, sk: Sketch, selected: set, base_color):
        pl = sk.plane
        world = np.array([pl.to3d(*p) for p in sk.points]).reshape(-1, 3)

        def col(kind, i):
            GL.glColor3f(*(self.theme.sketch_select if (kind, i) in selected else base_color))

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
            # keep framebuffer alpha at 1: Qt composites this widget as premultiplied alpha
            GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ZERO, GL.GL_ONE)
            for tris, color in self.region_fill:
                GL.glColor4f(*color)
                GL.glBegin(GL.GL_TRIANGLES)
                for p in tris.reshape(-1, 3):
                    GL.glVertex3f(*p)
                GL.glEnd()
            GL.glDisable(GL.GL_BLEND)
        for other in self.overlay_sketches:
            if other is not sk:
                self._draw_sketch_geometry(other, set(), self.theme.sketch_faded)
        if sk is not None:
            self._draw_sketch_geometry(sk, self.sketch_sel, self.theme.sketch)
        GL.glColor3f(*self.theme.preview)
        GL.glLineWidth(1.5)
        for poly in self.preview:
            GL.glBegin(GL.GL_LINE_STRIP)
            for p in poly:
                GL.glVertex3f(*p)
            GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)

    # -- orientation cube
    def _cube_rect(self):
        return viewcube.rect(self.width())

    def _draw_cube(self):
        w, h, ratio = self.width(), self.height(), self.devicePixelRatio()
        x, y, s = self._cube_rect()
        rot = viewcube.rotation(self.camera)
        t = self.theme
        GL.glViewport(int(x * ratio), int((h - y - s) * ratio), int(s * ratio), int(s * ratio))
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        GL.glOrtho(-viewcube.EXTENT, viewcube.EXTENT, -viewcube.EXTENT, viewcube.EXTENT, -10, 10)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        m = np.eye(4)
        m[:3, :3] = rot
        GL.glLoadMatrixf(m.T.astype(np.float32))
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)
        for name, (_, _, corners) in viewcube.FACES.items():
            GL.glColor3f(*(t.cube_hover if name == self.cube_hover else t.cube_face))
            GL.glBegin(GL.GL_QUADS)
            for c in corners:
                GL.glVertex3f(*c)
            GL.glEnd()
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glColor3f(*t.cube_edge)
        GL.glLineWidth(1.5)
        for _, _, corners in viewcube.FACES.values():
            GL.glBegin(GL.GL_LINE_LOOP)
            for c in corners:
                GL.glVertex3f(*c)
            GL.glEnd()
        GL.glEnable(GL.GL_POINT_SMOOTH)
        GL.glPointSize(10 * ratio)
        GL.glBegin(GL.GL_POINTS)
        for name in viewcube.visible_corners(rot):
            GL.glColor3f(*(t.cube_hover if name == self.cube_hover else t.cube_edge))
            GL.glVertex3f(*(np.asarray(viewcube.CORNERS[name], float) * 1.025))
        GL.glEnd()
        GL.glDisable(GL.GL_POINT_SMOOTH)
        GL.glLineWidth(1)
        self._draw_cube_labels()
        GL.glEnable(GL.GL_LIGHTING)
        GL.glViewport(0, 0, int(w * ratio), int(h * ratio))

    def _cube_textures(self) -> dict:
        """One RGBA label texture per face, rebuilt when the theme changes."""
        if self._cube_tex_key == self.theme.name:
            return self._cube_tex
        if self._cube_tex:
            GL.glDeleteTextures(list(self._cube_tex.values()))
        self._cube_tex = {}
        for name in viewcube.FACES:
            img = QImage(256, 128, QImage.Format_RGBA8888)
            img.fill(Qt.transparent)
            painter = QPainter(img)
            painter.setRenderHint(QPainter.TextAntialiasing)
            font = QFont()
            font.setPixelSize(52)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor.fromRgbF(*self.theme.cube_text))
            painter.drawText(QRectF(0, 0, 256, 128), Qt.AlignCenter, name)
            painter.end()
            tex = GL.glGenTextures(1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, 256, 128, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                            bytes(img.constBits()))
            self._cube_tex[name] = tex
        self._cube_tex_key = self.theme.name
        return self._cube_tex

    def _draw_cube_labels(self):
        textures = self._cube_textures()
        GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ZERO, GL.GL_ONE)
        GL.glColor3f(1, 1, 1)
        for name, (normal, right, _) in viewcube.FACES.items():
            n, r = np.array(normal, float), np.array(right, float)
            up = np.cross(n, r)  # the camera's up when looking straight at this face
            c = n * 1.02
            GL.glBindTexture(GL.GL_TEXTURE_2D, textures[name])
            GL.glBegin(GL.GL_QUADS)
            for (u, v), (sx, sy) in (((0, 1), (-1, -1)), ((1, 1), (1, -1)), ((1, 0), (1, 1)), ((0, 0), (-1, 1))):
                GL.glTexCoord2f(u, v)
                GL.glVertex3f(*(c + r * sx * 0.85 + up * sy * 0.425))
            GL.glEnd()
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glDisable(GL.GL_BLEND)

    def _snap_view(self, face: str):
        normal, right = viewcube.view_for(face)
        self.camera.look_at_plane(self.camera.target, normal, right)
        self.update()

    # -- mouse
    def mousePressEvent(self, e):
        p = e.position()
        self._cube_press = e.button() == Qt.LeftButton and viewcube.inside(p.x(), p.y(), self._cube_rect())
        self._press = (p, e.button(), e.modifiers())
        self._last = p
        self._dragged = False
        # Drag-to-extrude: left-click on a selected planar face starts the drag
        if (e.button() == Qt.LeftButton and not self._cube_press
                and not (e.modifiers() & Qt.ShiftModifier) and self.sel_faces and self.mesh):
            face = self.pick_face(p.x(), p.y())
            if face in self.sel_faces:
                info = self.mesh.faces[face]
                if info.normal is not None:
                    self._extrude_state = {"face": face,
                                           "center": np.array(info.center, float),
                                           "normal": np.array(info.normal, float),
                                           "start_screen": (p.x(), p.y()),
                                           "outline": self._face_outline(face)}
                    self._extrude_dist = 0.0
        # Drag-to-fillet: left drag (no modifier keys) on a picked/selected edge.
        # Takes priority over orbit, so a left drag on an edge fillets instead of rotating.
        if (e.button() == Qt.LeftButton and not self._cube_press and self.sketch is None
                and not (e.modifiers() & (Qt.AltModifier | Qt.ShiftModifier))
                and self.mesh is not None and len(self.mesh.edges)):
            edge = self.pick_edge(p.x(), p.y())
            if edge is not None and (edge in self.sel_edges or not self.sel_edges):
                sel = sorted(self.sel_edges) if self.sel_edges else [edge]
                mids = [tuple(map(float, self.mesh.edges[i].mid)) for i in sel]
                self._fillet_state = {
                    "mids": mids,
                    "outward": self._fillet_outward(sel[0]),
                    "start_screen": (p.x(), p.y()),
                    "radius": 0.0,
                }
                self._fillet_label.hide()
        # Sketch drag: report a plain left press in sketch mode to the app
        if (e.button() == Qt.LeftButton and not self._cube_press
                and self.sketch is not None
                and not (e.modifiers() & (Qt.AltModifier | Qt.ShiftModifier))):
            self.sketch_mouse.emit("press", p.x(), p.y())

    def mouseMoveEvent(self, e):
        p = e.position()
        if self._press is None:
            r = self._cube_rect()
            over = viewcube.inside(p.x(), p.y(), r)
            hover = viewcube.pick(viewcube.rotation(self.camera), p.x(), p.y(), r) if over else None
            if hover != self.cube_hover:
                self.cube_hover = hover
                self.update()
            if not over:
                self.hovered.emit(p.x(), p.y())
            return
        self.hovered.emit(p.x(), p.y())
        pos0, btn, mods = self._press
        d = p - self._last
        if (p - pos0).manhattanLength() > DRAG_PX:
            self._dragged = True
        self._last = p
        if not self._dragged:
            return
        # Sketch drag: report a plain left drag in sketch mode to the app.
        # Takes priority over orbit/pan so a left drag moves geometry, not the camera.
        if (btn == Qt.LeftButton and self.sketch is not None
                and not (mods & (Qt.AltModifier | Qt.ShiftModifier))):
            self.sketch_mouse.emit("move", p.x(), p.y())
            return
        alt = bool(mods & Qt.AltModifier)
        # Drag-to-extrude takes priority: if a face is being dragged, ignore orbit/pan
        if self._extrude_state is not None:
            # Screen-space drag: project cursor movement onto the face normal's
            # screen projection, then convert px to world units at the face depth.
            # (Projecting the cursor ray onto the plane reads ~0 in an ortho view,
            # where the ray offset is parallel to the image plane.)
            c, n = self._extrude_state["center"], self._extrude_state["normal"]
            r, u, b = self.camera.right, self.camera.up, self.camera.back
            nx, ny = float(n @ r), float(n @ u)  # normal in screen space
            p0 = self._extrude_state["start_screen"]
            dx, dy = float(p.x() - p0[0]), float(p0[1] - p.y())  # screen y up
            d_px = dx * nx + dy * ny
            if abs(d_px) < 1e-9:
                self._extrude_dist = 0.0  # face edge-on: no depth component
            else:
                depth = max(1.0, abs(float(np.dot(c - self.camera.eye, b))))
                px_per_world = self.height() / (2.0 * depth * FOV_TAN)
                self._extrude_dist = d_px / px_per_world
        elif self._fillet_state is not None:
            # Screen-space drag: project cursor movement onto the edge's outward
            # (perpendicular, in-face) screen projection; abs() so either side
            # of the edge grows the radius. Convert px to world mm at the edge depth.
            st = self._fillet_state
            c = np.asarray(st["mids"][0], float)
            out = st["outward"]
            p0 = st["start_screen"]
            dx, dy = float(p.x() - p0[0]), float(p0[1] - p.y())  # screen y up
            if out is not None:
                r, u = self.camera.right, self.camera.up
                nx, ny = float(out @ r), float(out @ u)
                d_px = abs(dx * nx + dy * ny)
                depth = max(1.0, abs(float(np.dot(c - self.camera.eye, self.camera.back))))
                px_per_world = self.height() / (2.0 * depth * FOV_TAN)
            else:
                # No resolved face normal: fall back to total screen drag distance.
                d_px = float(np.hypot(dx, dy))
                depth = max(1.0, abs(float(np.dot(c - self.camera.eye, self.camera.back))))
                px_per_world = self.height() / (2.0 * depth * FOV_TAN)
            if d_px < 1e-9:
                st["radius"] = 0.0
            else:
                st["radius"] = d_px / px_per_world
            self.fillet_preview.emit(st["radius"], st["mids"])
            self._show_fillet_label(st)
        elif btn == Qt.RightButton or (btn == Qt.LeftButton and (self.left_orbits or alt or self._cube_press)
                                      and not mods & Qt.ShiftModifier):
            self.camera.orbit(-d.x() * 0.4, -d.y() * 0.4)
        elif btn == Qt.MiddleButton or (btn == Qt.LeftButton and mods & Qt.ShiftModifier):
            self.camera.pan(d.x(), d.y(), self.height())
        self.update()

    def _show_fillet_label(self, st):
        """Place the live fillet-radius label near the first edge's outward point."""
        radius = st["radius"]
        if radius is None or abs(radius) < 1e-4:
            self._fillet_label.hide()
            return
        c = np.asarray(st["mids"][0], float)
        out = st["outward"]
        tip = c + (radius * out if out is not None else np.zeros(3))
        w, h = self.width(), self.height()
        s = self.camera.project(tip.reshape(1, 3), w, h)[0]
        x, y = float(s[0]), float(s[1])
        self._fillet_label.setText(f"R {radius:.2f} mm")
        self._fillet_label.move(int(x) + 10, max(int(y) - 12, 2))
        self._fillet_label.show()

    def mouseReleaseEvent(self, e):
        if self._press is not None and not self._dragged:
            p = e.position()
            if self._cube_press:
                face = viewcube.pick(viewcube.rotation(self.camera), p.x(), p.y(), self._cube_rect())
                if face:
                    self._snap_view(face)
            else:
                self.clicked.emit(p.x(), p.y(), self._press[1], self._press[2])
        # Commit drag-to-extrude
        if self._extrude_state is not None and self._dragged:
            c, n = self._extrude_state["center"], self._extrude_state["normal"]
            self.extrude_commit.emit(self._extrude_dist,
                                    tuple(map(float, c)), tuple(map(float, n)))
            self._extrude_state = None
            self._extrude_dist = 0.0
            self.update()
        # Commit drag-to-fillet
        if self._fillet_state is not None:
            st = self._fillet_state
            if self._dragged and st["radius"] > 1e-4:
                self.fillet_commit.emit(st["radius"], st["mids"])
            self._fillet_state = None
            self._fillet_label.hide()
            self.update()
        # Commit sketch drag
        if (self._press is not None and self._dragged and self._press[1] == Qt.LeftButton
                and self.sketch is not None
                and not (self._press[2] & (Qt.AltModifier | Qt.ShiftModifier))):
            self.sketch_mouse.emit("release", e.position().x(), e.position().y())
        self._press = None
        self._cube_press = False

    def paintEvent(self, e):
        super().paintEvent(e)
        self._sync_dim_labels()

    def _sync_dim_labels(self):
        if self.sketch is None:
            for lbl in self._dim_labels:
                lbl.hide()
            return
        pl = self.sketch.plane
        w, h = self.width(), self.height()
        positions = []
        for c in self.sketch.constraints:
            if c.value is None:
                continue
            try:
                if c.kind == "distance":
                    a, b = self.sketch.points[c.refs[0]], self.sketch.points[c.refs[1]]
                    mid = ((a[0]+b[0])/2, (a[1]+b[1])/2)
                elif c.kind == "spacing":
                    p = self.sketch.points[c.refs[0]]
                    a = self.sketch.points[self.sketch.lines[c.refs[1]][0]]
                    bb = self.sketch.points[self.sketch.lines[c.refs[1]][1]]
                    ax, ay = a; bx, by = bb
                    px_, py_ = p
                    t = ((px_-ax)*(bx-ax)+(py_-ay)*(by-ay))/( (bx-ax)**2+(by-ay)**2 + 1e-12)
                    t=max(0,min(1,t))
                    foot=(ax+t*(bx-ax), ay+t*(by-ay))
                    mid=((p[0]+foot[0])/2,(p[1]+foot[1])/2)
                elif c.kind == "diameter":
                    center = self.sketch.points[self.sketch.circles[c.refs[0]][0]]
                    mid = (center[0]+0.5, center[1])
                else:
                    continue
                pt = np.array(pl.to3d(*mid), dtype=float).reshape(1,3)
                sx, sy = self.camera.project(pt, w, h)[0][:2]
                if np.isnan(sx) or np.isnan(sy):
                    continue
                positions.append((int(sx), int(sy), self.unit.fmt(c.value)))
            except Exception:
                continue
        while len(self._dim_labels) < len(positions):
            lbl = QLabel(self)
            lbl.setStyleSheet("color: #ff3c3c; background: rgba(0,0,0,120); padding: 2px 4px; border-radius: 3px;")
            self._dim_labels.append(lbl)
        for idx, (x, y, text) in enumerate(positions):
            lbl = self._dim_labels[idx]
            lbl.setText(text)
            lbl.move(x, y)
            lbl.adjustSize()
            lbl.show()
            lbl.raise_()
        for idx in range(len(positions), len(self._dim_labels)):
            self._dim_labels[idx].hide()

    def wheelEvent(self, e):
        if e.modifiers() & Qt.ControlModifier:
            self.scrolled.emit(e.position().x(), e.position().y(), e.angleDelta().y(), e.modifiers())
            e.accept()
            self.update()
            return
        self.camera.zoom(e.angleDelta().y() / 120)
        self.update()
