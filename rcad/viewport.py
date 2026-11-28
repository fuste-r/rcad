from __future__ import annotations

import math

import numpy as np
from OpenGL import GL
from PySide6.QtCore import QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QLabel, QRubberBand

from . import datum, viewcube
from .camera import Camera, FOV_TAN, point_segment_dist, ray_plane, ray_triangles
from .kernel import MeshData
from .sketch import Sketch
from .theme import LIGHT, Theme
from .units import MM, Unit
from .value_box import ValueBox

DRAG_PX = 4


class Viewport(QOpenGLWidget):
    clicked = Signal(float, float, object, object)  # x, y, Qt.MouseButton, Qt.KeyboardModifier
    double_clicked = Signal(float, float, object)  # x, y, Qt.KeyboardModifier (left button)
    box_selected = Signal(float, float, float, float, object)  # x0, y0, x1, y1 (sorted), modifiers
    hovered = Signal(float, float)
    scrolled = Signal(float, float, int, object)  # x, y, angle delta, Qt.KeyboardModifier
    extrude_commit = Signal(float, tuple, tuple)  # distance_mm, center, normal
    region_extrude = Signal(float, object)  # distance_mm, the target region_drag returned
    planes_toggled = Signal(bool)  # P: the reference planes were shown (True) or hidden
    sketch_mouse = Signal(str, float, float)  # phase ("press"|"move"|"release"), x, y
    fillet_commit = Signal(float, list, str)  # size_mm, [edge midpoints], "fillet" (push) | "chamfer" (pull)
    fillet_preview = Signal(float, list, str)  # live size_mm, [edge midpoints], kind
    fillet_cancel = Signal()  # Esc during a drag: undo the live fillet
    value_edit = Signal(str, float)  # kind ("extrude"|"fillet"|"chamfer"), new mm typed after the drag
    value_scrub = Signal(str, float)  # kind, live mm while the box left after a drag is scrubbed

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # so a number typed mid-drag reaches keyPressEvent
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
        self.show_planes = True  # the Top, Front and Right reference planes (P toggles them)
        self.hover_plane: str | None = None
        self.sel_plane: str | None = None
        self._datum_half = datum.MIN_HALF
        self._plane_labels: dict[str, QLabel] = {}
        self.snap_mark = None  # {"kind", "at": world point, "guides": [(world a, world b)]} while drawing
        self.sel_faces: set[int] = set()
        self.sel_edges: set[int] = set()
        self.hover_face: int | None = None
        self.hover_edge: int | None = None
        self.left_orbits = True
        self._box_origin = None  # screen point a selection box is drawn from, while one may start
        self._double = False  # the current press is the second press of a double-click
        self.rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._press = None
        self._dragged = False
        self._last = None
        self._tri = np.zeros((0, 3, 3))
        self._verts = np.zeros((0, 3), np.float32)
        self._norms = np.zeros((0, 3), np.float32)
        self._segs = np.zeros((0, 2, 3))
        self._seg_edge = np.zeros(0, int)
        self._extrude_state = None  # {"face", "center", "normal", "start_screen", "outline" | "segments", "region"?}
        # region_drag(x, y, on_arrow) -> {"center", "normal", "segments", ...} for sketch regions a left
        # drag from (x, y) would extrude, or None; set by the app
        self.region_drag = None
        self.region_arrow = None  # (center, normal): the extrude arrow on the picked sketch regions
        self._extrude_dist = 0.0
        self._fillet_state = None  # {"mids", "away", "start_screen", "radius", "kind", "dir"}
        # Value box: shows the dragged size; typing a number sets it. After the drag
        # it stays on the model (``_value_post``) so the new feature's size can be retyped.
        self.variables_source = dict  # callable returning the document's variables (mm)
        self.value_box = ValueBox(self, lambda: self.unit, lambda: self.variables_source())
        self.value_box.edited.connect(self._on_box_edited)
        self.value_box.accepted.connect(self._on_box_accepted)
        self.value_box.cancelled.connect(self._on_box_cancelled)
        self.value_box.scrubbed.connect(self._on_box_scrubbed)
        self.value_box.scrub_finished.connect(self._on_box_scrub_finished)
        self.fields = None  # FieldSet for sketch drawing tools, set by the app
        self.paint_hooks = []  # callables run after each paint to keep overlays beside the model
        self._value_post = None  # {"kind", "base", "dir", "value"} after a committed drag

    # -- data
    def set_mesh(self, mesh: MeshData | None):
        self.mesh = mesh
        self._datum_half = datum.plane_half(None if mesh is None else mesh.verts)
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

    def _edge_tangent(self, edge_idx: int):
        """World unit direction of an edge at its middle, or None."""
        m = self.mesh
        if m is None or not (0 <= edge_idx < len(m.edges)):
            return None
        pts = np.asarray(m.edges[edge_idx].polyline, float)
        if len(pts) < 2:
            return None
        k = len(pts) // 2
        d = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)] if len(pts) > 2 else pts[-1] - pts[0]
        n = float(np.linalg.norm(d))
        return d / n if n > 1e-12 else None

    def _edge_away(self, edge_idx: int):
        """World unit vector pointing away from the material at an edge, or None."""
        m = self.mesh
        if m is None or not (0 <= edge_idx < len(m.edges)):
            return None
        return m.edge_away(edge_idx)

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
        self._draw_planes()
        self._draw_extrude_preview()
        self._draw_region_arrow()
        self._draw_sketch()
        self._draw_origin()
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
        st = self._extrude_state
        if st is None or (self.mesh is None and "segments" not in st):
            return
        c, n = st["center"], st["normal"]
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
        if "segments" in st:
            segs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in st["segments"]]
        else:
            verts = self.mesh.verts
            segs = [(np.asarray(verts[a], float), np.asarray(verts[b], float)) for a, b in st["outline"]]
        GL.glLineWidth(2)
        GL.glColor3f(0.0, 0.9, 0.6)  # teal for preview
        GL.glBegin(GL.GL_LINES)
        for pa, pb in segs:
            GL.glVertex3f(*(pa + d * n))
            GL.glVertex3f(*(pb + d * n))
        for pa in st.get("corners", []):  # a region grows a whole new body: show its sides too
            pa = np.asarray(pa, float)
            GL.glVertex3f(*pa)
            GL.glVertex3f(*(pa + d * n))
        GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_LIGHTING)

    # -- origin and reference planes
    def planes_shown(self) -> bool:
        return self.show_planes and self.sketch is None

    def pick_plane(self, x, y) -> str | None:
        """The reference plane under view point (x, y), or None (also while they are hidden)."""
        if not self.planes_shown():
            return None
        o, d = self.camera.ray(x, y, self.width(), self.height())
        hit = datum.pick(o, d, self._datum_half)
        return None if hit is None else hit[0]

    def toggle_planes(self):
        self.show_planes = not self.show_planes
        if not self.show_planes:
            self.hover_plane = self.sel_plane = None
        self.planes_toggled.emit(self.show_planes)
        self.update()

    def _draw_planes(self):
        """Translucent squares with thin outlines. The model hides what is behind it; the planes
        do not hide each other."""
        if not self.planes_shown():
            return
        t, half = self.theme, self._datum_half
        GL.glDisable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ZERO, GL.GL_ONE)
        GL.glDepthMask(GL.GL_FALSE)
        GL.glEnable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glPolygonOffset(1.0, 1.0)  # a model face lying on a plane stays on top of it
        for name in datum.PLANES:
            alpha = 0.2 if name == self.sel_plane else 0.14 if name == self.hover_plane else 0.07
            GL.glColor4f(*t.datum_fill, alpha)
            GL.glBegin(GL.GL_QUADS)
            for p in datum.corners(name, half):
                GL.glVertex3f(*p)
            GL.glEnd()
        GL.glDisable(GL.GL_POLYGON_OFFSET_FILL)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        for name in datum.PLANES:
            if name == self.sel_plane:
                GL.glColor3f(*t.face_select)
                GL.glLineWidth(2)
            else:
                GL.glColor3f(*(t.datum_hover if name == self.hover_plane else t.datum_edge))
                GL.glLineWidth(1)
            GL.glBegin(GL.GL_LINE_LOOP)
            for p in datum.corners(name, half):
                GL.glVertex3f(*p)
            GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_LIGHTING)

    def _draw_origin(self):
        """A dot in a thin ring at (0, 0, 0), the same size on screen at any zoom, never hidden."""
        o = np.zeros(3)
        s = 1.0 / self._px_per_world(o)
        r, u = self.camera.right * s, self.camera.up * s
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ZERO, GL.GL_ONE)
        GL.glEnable(GL.GL_LINE_SMOOTH)
        GL.glEnable(GL.GL_POLYGON_SMOOTH)
        GL.glColor3f(*self.theme.origin)
        GL.glBegin(GL.GL_TRIANGLE_FAN)  # the dot: 2.5 px radius
        GL.glVertex3f(*o)
        for a in np.linspace(0, 2 * np.pi, 21):
            GL.glVertex3f(*(o + 2.5 * (np.cos(a) * r + np.sin(a) * u)))
        GL.glEnd()
        GL.glLineWidth(1.2)
        GL.glBegin(GL.GL_LINE_LOOP)  # the ring: 7 px radius
        for a in np.linspace(0, 2 * np.pi, 40, endpoint=False):
            GL.glVertex3f(*(o + 7.0 * (np.cos(a) * r + np.sin(a) * u)))
        GL.glEnd()
        GL.glLineWidth(1)
        GL.glDisable(GL.GL_POLYGON_SMOOTH)
        GL.glDisable(GL.GL_LINE_SMOOTH)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)

    def _sync_plane_labels(self):
        """Each plane's name inside its top-left corner; hidden when the plane is seen edge-on."""
        color = "rgb({}, {}, {})".format(*(int(c * 255) for c in self.theme.datum_text))
        w, h = self.width(), self.height()
        for name in datum.PLANES:
            lbl = self._plane_labels.get(name)
            if lbl is None:
                lbl = self._plane_labels[name] = QLabel(name, self)
                lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
            n, _, _ = datum.axes(name)
            facing = abs(float(n @ self.camera.back))
            s = self.camera.project(datum.label_anchor(name, self._datum_half).reshape(1, 3), w, h)[0]
            if not self.planes_shown() or facing < 0.2 or np.isnan(s).any():
                lbl.hide()
                continue
            weight = 600 if name in (self.sel_plane, self.hover_plane) else 400
            lbl.setStyleSheet(f"color: {color}; background: transparent; font-family: sans-serif;"
                              f" font-size: 11px; font-weight: {weight};")
            lbl.adjustSize()
            lbl.move(int(s[0]), int(s[1]))
            lbl.show()
            lbl.lower()  # under value boxes, quick buttons and other overlays

    def _px_per_world(self, point) -> float:
        depth = max(1e-6, abs(float(np.dot(np.asarray(point, float) - self.camera.eye, self.camera.back))))
        return self.height() / (2.0 * depth * FOV_TAN)

    ARROW_PX = 72  # length of the region extrude arrow on screen

    def _arrow_ends(self):
        """World base and tip of the region extrude arrow, or None while it is hidden."""
        if self.region_arrow is None or self._extrude_state is not None or self.sketch is not None:
            return None
        c, n = (np.asarray(v, float) for v in self.region_arrow)
        return c, c + n * (self.ARROW_PX / self._px_per_world(c))

    def on_region_arrow(self, x, y, px=8.0) -> bool:
        ends = self._arrow_ends()
        if ends is None:
            return False
        a, b = self.camera.project(np.array(ends), self.width(), self.height())
        if np.isnan(a).any() or np.isnan(b).any():
            return False
        ab, ap = b - a, np.array([x, y], float) - a
        t = float(np.clip(ap @ ab / max(ab @ ab, 1e-9), 0, 1))
        return float(np.linalg.norm(ap - t * ab)) <= px

    def _draw_region_arrow(self):
        ends = self._arrow_ends()
        if ends is None:
            return
        base, tip = ends
        n = tip - base
        length = float(np.linalg.norm(n))
        n = n / length
        side = np.cross(n, self.camera.back)
        if np.linalg.norm(side) < 1e-6:
            side = np.cross(n, self.camera.up)
        side /= np.linalg.norm(side)
        head = 0.28 * length
        GL.glDisable(GL.GL_LIGHTING)
        GL.glDisable(GL.GL_DEPTH_TEST)  # stays visible on top of the region it rises from
        GL.glColor3f(1.0, 0.55, 0.15)
        GL.glLineWidth(4)
        GL.glBegin(GL.GL_LINES)
        GL.glVertex3f(*base)
        GL.glVertex3f(*(tip - n * head * 0.5))
        GL.glEnd()
        GL.glBegin(GL.GL_TRIANGLES)
        GL.glVertex3f(*tip)
        GL.glVertex3f(*(tip - n * head + side * head * 0.45))
        GL.glVertex3f(*(tip - n * head - side * head * 0.45))
        GL.glEnd()
        GL.glLineWidth(1)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_LIGHTING)

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
        if sk is not None and self.snap_mark is not None:
            self._draw_snap_mark(self.snap_mark)
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

    SNAP_COLOR = (0.15, 0.75, 0.95)

    def _draw_snap_mark(self, mark):
        """A small screen-sized symbol at the snap point (square: point, triangle: midpoint, cross: on a
        curve) and dashed alignment guides."""
        at = np.asarray(mark["at"], float)
        s = 6.0 / self._px_per_world(at)  # half size: 6 px
        r, u = self.camera.right * s, self.camera.up * s
        GL.glColor3f(*self.SNAP_COLOR)
        GL.glLineWidth(1.5)
        if mark["guides"]:
            GL.glEnable(GL.GL_LINE_STIPPLE)
            GL.glLineStipple(2, 0x3333)
            GL.glBegin(GL.GL_LINES)
            for a, b in mark["guides"]:
                GL.glVertex3f(*a)
                GL.glVertex3f(*b)
            GL.glEnd()
            GL.glDisable(GL.GL_LINE_STIPPLE)
        kind = mark["kind"]
        if kind == "point":
            shape = [at - r - u, at + r - u, at + r + u, at - r + u]
        elif kind == "midpoint":
            shape = [at - r - u, at + r - u, at + u * 1.1]
        elif kind in ("on_line", "on_circle"):
            shape = None
            GL.glBegin(GL.GL_LINES)
            for p in (at - r - u, at + r + u, at - r + u, at + r - u):
                GL.glVertex3f(*p)
            GL.glEnd()
        else:  # alignment only: a small dot-like diamond
            shape = [at - r * 0.6, at - u * 0.6, at + r * 0.6, at + u * 0.6]
        if shape is not None:
            GL.glBegin(GL.GL_LINE_LOOP)
            for p in shape:
                GL.glVertex3f(*p)
            GL.glEnd()
        GL.glLineWidth(1)

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
        self.setFocus()
        if self.fields is not None and self.fields.post:
            self.fields.stop()  # a click elsewhere puts away the boxes of the shape just drawn
        if self._typed_live() is not None:
            # A typed value is waiting for Enter: a click on the model confirms it.
            v = self.value_box.value()
            if v is None:
                self._on_box_cancelled()
            else:
                self._on_box_accepted(v)
            return
        if self._value_post is not None:
            self.dismiss_value_box()
        self._cube_press = e.button() == Qt.LeftButton and viewcube.inside(p.x(), p.y(), self._cube_rect())
        self._press = (p, e.button(), e.modifiers())
        self._last = p
        self._dragged = False
        self._box_origin = None
        # Ctrl+left drag in the model draws a selection box (a sketch starts one on empty space: start_box)
        if (e.button() == Qt.LeftButton and not self._cube_press and self.sketch is None
                and e.modifiers() & Qt.ControlModifier):
            self.start_box(p.x(), p.y())
            return
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
        # Drag a sketch region (or its arrow) to extrude it
        if (e.button() == Qt.LeftButton and not self._cube_press and self.sketch is None
                and self._extrude_state is None and self.region_drag is not None
                and not (e.modifiers() & (Qt.AltModifier | Qt.ShiftModifier | Qt.ControlModifier))):
            on_arrow = self.on_region_arrow(p.x(), p.y())
            target = None
            if on_arrow or self.mesh is None or self.pick_edge(p.x(), p.y()) is None:
                target = self.region_drag(p.x(), p.y(), on_arrow)
            if target is not None:
                c, n = np.asarray(target["center"], float), np.asarray(target["normal"], float)
                axis = c  # the arrow's own line, or the line through the pressed point of the region
                if not on_arrow:
                    o, r = self.camera.ray(p.x(), p.y(), self.width(), self.height())
                    hit = ray_plane(o, r, c, n)
                    if hit is not None:
                        axis = np.asarray(hit, float)
                self._extrude_state = {"face": None, "center": c, "normal": n, "axis": axis,
                                       "start_screen": (p.x(), p.y()), "segments": target["segments"],
                                       "corners": target.get("corners", []), "region": target}
                self._extrude_dist = 0.0
                return
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
                    "away": self._edge_away(sel[0]),
                    "tangent": self._edge_tangent(sel[0]),
                    "start_screen": (p.x(), p.y()),
                    "radius": 0.0,
                    "kind": "fillet",
                    "dir": None,  # world direction the value box sits along (follows the drag)
                }
        # Sketch drag: report a plain left press in sketch mode to the app
        if (e.button() == Qt.LeftButton and not self._cube_press
                and self.sketch is not None
                and not (e.modifiers() & (Qt.AltModifier | Qt.ShiftModifier))):
            self.sketch_mouse.emit("press", p.x(), p.y())

    def mouseDoubleClickEvent(self, e):
        """The second press of a double-click: a press like any other, but its release reports
        a double-click instead of a click."""
        self.mousePressEvent(e)
        self._double = e.button() == Qt.LeftButton and self._press is not None

    def start_box(self, x: float, y: float):
        """Let the current left drag draw a selection box from (x, y)."""
        self._box_origin = (float(x), float(y))

    def _box_rect(self, p) -> QRect:
        x0, y0 = self._box_origin
        return QRect(int(min(x0, p.x())), int(min(y0, p.y())), int(abs(p.x() - x0)), int(abs(p.y() - y0)))

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
        if self._box_origin is not None:
            r = self._box_rect(p)
            self.rubber.setGeometry(QRect(r.topLeft(), r.size().expandedTo(QSize(1, 1))))
            self.rubber.show()
            return
        # Sketch drag: report a plain left drag in sketch mode to the app.
        # Takes priority over orbit/pan so a left drag moves geometry, not the camera.
        if (btn == Qt.LeftButton and self.sketch is not None
                and not (mods & (Qt.AltModifier | Qt.ShiftModifier))):
            self.sketch_mouse.emit("move", p.x(), p.y())
            return
        alt = bool(mods & Qt.AltModifier)
        # Drag-to-extrude takes priority: if a face is being dragged, ignore orbit/pan
        if self._extrude_state is not None and self._extrude_state.get("typed"):
            pass  # a typed value overrides the mouse until Enter or Esc
        elif self._fillet_state is not None and self._fillet_state.get("typed"):
            pass
        elif self._extrude_state is not None:
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
                if self._extrude_state.get("region") is not None:
                    # the arrow follows the pointer exactly: the nearest points of the axis and the pointer's ray
                    a = self._extrude_state["axis"]
                    t0, t1 = self._axis_param(a, n, *p0), self._axis_param(a, n, p.x(), p.y())
                    if t0 is not None and t1 is not None:
                        self._extrude_dist = t1 - t0
        elif self._fillet_state is not None:
            # Drag across the edge on screen: toward the body rounds it (fillet), away from
            # the body bevels it (chamfer). Movement along the edge doesn't count.
            st = self._fillet_state
            p0 = st["start_screen"]
            d_world, st["dir"] = self._edge_drag(st, float(p.x() - p0[0]), float(p0[1] - p.y()))
            st["radius"] = abs(d_world)
            st["kind"] = "chamfer" if d_world > 0 else "fillet"
            self.fillet_preview.emit(st["radius"], st["mids"], st["kind"])
        elif btn == Qt.RightButton or (btn == Qt.LeftButton and (self.left_orbits or alt or self._cube_press)
                                      and not mods & Qt.ShiftModifier):
            self.camera.orbit(-d.x() * 0.4, -d.y() * 0.4)
        elif btn == Qt.MiddleButton or (btn == Qt.LeftButton and mods & Qt.ShiftModifier):
            self.camera.pan(d.x(), d.y(), self.height())
        self.update()

    def _edge_drag(self, st, dx: float, dy: float):
        """(signed world distance, box direction) for a screen drag (dx, dy; y up) on an edge.

        Positive is a pull away from the body. The drag counts only across the edge on screen;
        "away" is the side the edge's away-from-material direction shows on. When that barely
        shows (the direction points at the viewer), dragging up the screen counts as pulling."""
        c = np.asarray(st["mids"][0], float)
        away, t = st["away"], st.get("tangent")
        w, h = self.width(), self.height()
        step = max(1e-3 * self.camera.dist, 1e-4)
        s0 = self.camera.project(c.reshape(1, 3), w, h)[0]

        def on_screen(v):  # screen px (y up) per world mm along v, perspective included
            q = self.camera.project((c + v * step).reshape(1, 3), w, h)[0] - s0
            return np.array([q[0], -q[1]]) / step

        depth = max(1.0, abs(float(np.dot(c - self.camera.eye, self.camera.back))))
        px_per_mm = h / (2.0 * depth * FOV_TAN)
        tv = on_screen(t) if t is not None else np.zeros(2)
        n = float(np.hypot(*tv))
        m = np.array([-tv[1], tv[0]]) / n if n > 1e-3 * px_per_mm else np.array([0.0, 1.0])
        across = float(on_screen(away) @ m) if away is not None else 0.0  # px per mm pulled
        if abs(across) >= 0.3 * px_per_mm:
            d_world = (dx * m[0] + dy * m[1]) / across
            box_dir = away * (1.0 if d_world > 0 else -1.0)
        else:  # away barely shows: up the screen is a pull
            up = 1.0 if (m[1] > 1e-6 or (abs(m[1]) <= 1e-6 and m[0] > 0)) else -1.0
            d_world = up * (dx * m[0] + dy * m[1]) / px_per_mm
            box_dir = None
        return d_world, box_dir

    # -- value box (type while dragging)
    def _axis_param(self, c, n, x, y):
        """Distance along the line c + t n to the point nearest the ray through screen (x, y);
        None when the line is seen end-on."""
        o, r = self.camera.ray(x, y, self.width(), self.height())
        o, r = np.asarray(o, float), np.asarray(r, float)
        b = float(n @ r)
        den = float(n @ n) * float(r @ r) - b * b
        if den < 1e-6 * float(r @ r):
            return None
        w0 = c - o
        return float((b * (w0 @ r) - float(r @ r) * (w0 @ n)) / den)

    def _typed_live(self):
        """'extrude' or 'fillet' when a drag's value was typed and awaits Enter, else None."""
        if self._extrude_state is not None and self._extrude_state.get("typed"):
            return "extrude"
        if self._fillet_state is not None and self._fillet_state.get("typed"):
            return "fillet"
        return None

    def _value_info(self):
        """(kind, base point, direction, value mm, live) for the value box, or None."""
        if self._extrude_state is not None and (self._dragged or self._extrude_state.get("typed")):
            s = self._extrude_state
            return "extrude", s["center"], s["normal"], self._extrude_dist, True
        if self._fillet_state is not None and (self._dragged or self._fillet_state.get("typed")):
            s = self._fillet_state
            return s["kind"], np.asarray(s["mids"][0], float), s["dir"], s["radius"], True
        if self._value_post is not None:
            p = self._value_post
            return p["kind"], p["base"], p["dir"], p["value"], False
        return None

    def _sync_value_box(self):
        info = self._value_info()
        box = self.value_box
        if info is None:
            box.hide()
            return
        kind, base, direction, value, live = info
        if live and abs(value) < 1e-4 and not box.typing:
            box.hide()
            return
        tip = np.asarray(base, float)
        if direction is not None:
            tip = tip + value * np.asarray(direction, float)
        s = self.camera.project(tip.reshape(1, 3), self.width(), self.height())[0]
        if np.isnan(s[0]) or np.isnan(s[1]):
            box.hide()
            return
        box.set_kind(kind)
        box.show_value(value)
        x = min(int(s[0]) + 12, max(self.width() - box.width() - 2, 0))
        y = min(max(int(s[1]) - 12, 2), max(self.height() - box.height() - 2, 0))
        box.move(x, y)
        box.show()
        box.raise_()

    def focusNextPrevChild(self, forward):
        # Tab moves between value boxes, never out of the view while one is up.
        if (self.fields is not None and self.fields.active) or self._value_info() is not None:
            return False
        return super().focusNextPrevChild(forward)

    def _sync_fields(self):
        """Keep the boxes of a just-drawn shape next to it as the view moves."""
        f = self.fields
        if f is None or not f.active or not f.post or f.anchor is None:
            return
        s = self.camera.project(np.asarray(f.anchor, float).reshape(1, 3), self.width(), self.height())[0]
        if not (np.isnan(s[0]) or np.isnan(s[1])):
            f.place(float(s[0]), float(s[1]))

    def keyPressEvent(self, e):
        text = e.text()
        f = self.fields
        if f is not None and f.active and not e.modifiers() & (Qt.ControlModifier | Qt.AltModifier):
            if e.key() in (Qt.Key_Tab, Qt.Key_Backtab):
                f.next(e.key() == Qt.Key_Tab and not e.modifiers() & Qt.ShiftModifier)
                return
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                f.accept()
                return
            if text and text.isprintable() and not text.isspace():
                f.type_char(text)
                return
        if (self._value_info() is not None and text and text.isprintable() and not text.isspace()
                and not e.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier)):
            if self._extrude_state is not None:
                self._extrude_state["typed"] = True
            elif self._fillet_state is not None:
                self._fillet_state["typed"] = True
            self.value_box.start_typing(text)
            self.update()
            return
        if e.key() == Qt.Key_P and not e.modifiers() & (Qt.ControlModifier | Qt.AltModifier | Qt.MetaModifier):
            self.toggle_planes()  # (typing into a value box above takes the key first)
            return
        super().keyPressEvent(e)

    def _on_box_edited(self, mm: float):
        kind = self._typed_live()
        if kind == "extrude":
            self._extrude_dist = mm
        elif kind == "fillet" and mm > 1e-4:
            st = self._fillet_state
            st["radius"] = mm
            self.fillet_preview.emit(mm, st["mids"], st["kind"])
        self.update()

    def _on_box_accepted(self, mm: float):
        kind = self._typed_live()
        if kind == "extrude":
            self._extrude_dist = mm
            self._finish_extrude()
        elif kind == "fillet":
            self._fillet_state["radius"] = mm
            self._finish_fillet()
        elif self._value_post is not None:
            self._value_post["value"] = mm
            self.value_edit.emit(self._value_post["kind"], mm)
        self.value_box.end_typing()
        self.setFocus()
        self.update()

    def _on_box_scrubbed(self, mm: float):
        if self._typed_live() is not None:
            self._on_box_edited(mm)
        elif self._value_post is not None:
            self._value_post["value"] = mm
            self.value_scrub.emit(self._value_post["kind"], mm)
            self.update()

    def _on_box_scrub_finished(self, mm: float):
        if self._typed_live() is None and self._value_post is not None:
            self._value_post["value"] = mm
            self.value_edit.emit(self._value_post["kind"], mm)
            self.value_box.end_typing()
            self.setFocus()
            self.update()

    def _on_box_cancelled(self):
        if self._extrude_state is not None:
            self._extrude_state = None
            self._extrude_dist = 0.0
        if self._fillet_state is not None:
            self._fillet_state = None
            self.fillet_cancel.emit()
        self.dismiss_value_box()
        self.setFocus()

    def dismiss_value_box(self):
        """Hide the value box left on the model after a drag."""
        self._value_post = None
        self.value_box.end_typing()
        self.value_box.hide()
        self.update()

    def set_post_value(self, mm: float):
        """Show a different value in the post-drag box (e.g. after the app reverted an edit)."""
        if self._value_post is not None:
            self._value_post["value"] = mm
            self.value_box.end_typing()
            self.update()

    def _finish_extrude(self):
        st = self._extrude_state
        c, n = st["center"], st["normal"]
        d = self._extrude_dist
        self._extrude_state = None
        self._extrude_dist = 0.0
        # Set before emitting so the app can dismiss the box if the commit fails.
        self._value_post = {"kind": "extrude", "base": c, "dir": n, "value": d} if abs(d) >= 0.01 else None
        if st.get("region") is not None:
            self.region_extrude.emit(d, st["region"])
            return
        self.extrude_commit.emit(d, tuple(map(float, c)), tuple(map(float, n)))

    def _finish_fillet(self):
        st = self._fillet_state
        self._fillet_state = None
        r = st["radius"]
        if r > 1e-4:  # (a 0 radius leaves any extrude box from the same press alone)
            self._value_post = {"kind": st["kind"], "base": np.asarray(st["mids"][0], float),
                                "dir": st["dir"], "value": r}
        self.fillet_commit.emit(r, st["mids"], st["kind"])  # 0 rolls back a live preview

    def mouseReleaseEvent(self, e):
        double, self._double = self._double, False
        if self._box_origin is not None and self._press is not None and self._dragged:
            p, (x0, y0) = e.position(), self._box_origin
            self._box_origin = None
            self.rubber.hide()
            mods = self._press[2]
            self._press = None
            self.box_selected.emit(min(x0, p.x()), min(y0, p.y()), max(x0, p.x()), max(y0, p.y()), mods)
            self.update()
            return
        self._box_origin = None
        self.rubber.hide()
        if self._press is not None and not self._dragged:
            p = e.position()
            if self._cube_press:
                face = viewcube.pick(viewcube.rotation(self.camera), p.x(), p.y(), self._cube_rect())
                if face:
                    self._snap_view(face)
            elif double:
                self.double_clicked.emit(p.x(), p.y(), self._press[2])
            else:
                self.clicked.emit(p.x(), p.y(), self._press[1], self._press[2])
        # Commit drag-to-extrude / drag-to-fillet, unless a typed value awaits Enter
        if self._extrude_state is not None and not self._extrude_state.get("typed"):
            if self._dragged:
                self._finish_extrude()
            else:
                self._extrude_state = None
            self.update()
        if self._fillet_state is not None and not self._fillet_state.get("typed"):
            if self._dragged:
                self._finish_fillet()
            else:
                self._fillet_state = None
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
        self._sync_plane_labels()
        self._sync_dim_labels()
        self._sync_value_box()
        self._sync_fields()
        for hook in self.paint_hooks:
            hook()

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
