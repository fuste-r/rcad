from __future__ import annotations

import numpy as np

FOV_TAN = np.tan(np.radians(45) / 2)


def _rot(v, axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)


class Camera:
    """Orientation-matrix camera so any plane can be viewed face-on without gimbal limits."""

    def __init__(self):
        self.target = np.zeros(3)
        self.dist = 120.0
        self.ortho = False  # sketch mode uses an orthographic view so profiles are not distorted
        self.right = np.array([1.0, 0, 0])
        self.up = np.array([0, 1.0, 0])
        self.back = np.array([0, 0, 1.0])
        self.orbit(-35, 55)

    @property
    def eye(self):
        return self.target + self.back * self.dist

    def orbit(self, dx_deg: float, dy_deg: float):
        z = np.array([0, 0, 1.0])
        a = np.radians(dx_deg)
        self.right, self.up, self.back = (_rot(v, z, a) for v in (self.right, self.up, self.back))
        b = np.radians(dy_deg)
        self.up, self.back = _rot(self.up, self.right, b), _rot(self.back, self.right, b)
        self._orthonormalize()

    def _orthonormalize(self):
        self.back /= np.linalg.norm(self.back)
        self.right = np.cross(self.up, self.back)
        self.right /= np.linalg.norm(self.right)
        self.up = np.cross(self.back, self.right)

    def pan(self, dx_px: float, dy_px: float, h: int):
        k = 2 * self.dist * FOV_TAN / max(h, 1)
        self.target += (-self.right * dx_px + self.up * dy_px) * k

    def zoom(self, steps: float):
        self.dist = float(np.clip(self.dist * 0.9 ** steps, 1.0, 1e5))

    def look_at_plane(self, origin, normal, xdir, dist=None):
        self.target = np.asarray(origin, float)
        self.back = np.asarray(normal, float)
        self.right = np.asarray(xdir, float)
        self.up = np.cross(self.back, self.right)
        self._orthonormalize()
        if dist:
            self.dist = dist

    def matrices(self, w: int, h: int):
        a = w / max(h, 1)
        if self.ortho:
            hh = self.dist * FOV_TAN
            hw, far = hh * a, self.dist * 100
            proj = np.array([[1 / hw, 0, 0, 0], [0, 1 / hh, 0, 0], [0, 0, -1 / far, 0], [0, 0, 0, 1]])
        else:
            near, far = self.dist * 0.02, self.dist * 200
            t = near * FOV_TAN
            proj = np.array([[near / (t * a), 0, 0, 0], [0, near / t, 0, 0],
                             [0, 0, -(far + near) / (far - near), -2 * far * near / (far - near)], [0, 0, -1, 0]])
        r = np.stack([self.right, self.up, self.back])
        view = np.eye(4)
        view[:3, :3] = r
        view[:3, 3] = -r @ self.eye
        return view, proj

    def ray(self, x: float, y: float, w: int, h: int):
        nx, ny = 2 * x / w - 1, 1 - 2 * y / h
        if self.ortho:
            hh = self.dist * FOV_TAN
            return self.eye + self.right * nx * hh * w / h + self.up * ny * hh, -self.back
        d = -self.back + self.right * nx * FOV_TAN * w / h + self.up * ny * FOV_TAN
        return self.eye, d / np.linalg.norm(d)

    def project(self, pts: np.ndarray, w: int, h: int) -> np.ndarray:
        """World points (N,3) -> pixel coordinates (N,2); points behind the camera give NaN."""
        view, proj = self.matrices(w, h)
        p = np.c_[pts, np.ones(len(pts))] @ (proj @ view).T
        with np.errstate(divide="ignore", invalid="ignore"):
            nd = p[:, :3] / p[:, 3:4]
        out = np.c_[(nd[:, 0] + 1) * w / 2, (1 - nd[:, 1]) * h / 2]
        out[p[:, 3] <= 0] = np.nan
        return out


def ray_triangles(origin, direction, tri: np.ndarray):
    """Möller–Trumbore against tri (M,3,3); returns (index, t) of the nearest hit or (None, None)."""
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    s = origin - tri[:, 0]
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
    if not hit.any():
        return None, None
    idx = np.where(hit)[0]
    k = idx[np.argmin(t[idx])]
    return int(k), float(t[k])


def ray_plane(origin, direction, plane_origin, normal):
    denom = float(np.dot(direction, normal))
    if abs(denom) < 1e-9:
        return None
    t = float(np.dot(np.asarray(plane_origin) - origin, normal)) / denom
    return origin + direction * t if t > 0 else None


def point_segment_dist(p, a, b):
    """Distances from point p (2,) to segments a-b (S,2)."""
    ab = b - a
    l2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / l2, 0, 1)
    return np.linalg.norm(a + ab * t[:, None] - p, axis=1)
