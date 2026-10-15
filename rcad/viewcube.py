"""Orientation cube: geometry, hit testing and the camera direction for each labelled face (Z is up)."""
from __future__ import annotations

import numpy as np

SIZE = 130  # widget pixels
MARGIN = 12
EXTENT = 1.8  # half-width of the orthographic window around the unit cube

# name -> (outward normal, camera "right" when viewing that face, corners)
FACES = {
    "Top": ((0, 0, 1), (1, 0, 0), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
    "Bottom": ((0, 0, -1), (1, 0, 0), [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)]),
    "Front": ((0, -1, 0), (1, 0, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
    "Back": ((0, 1, 0), (-1, 0, 0), [(1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1)]),
    "Right": ((1, 0, 0), (0, 1, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
    "Left": ((-1, 0, 0), (0, -1, 0), [(-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1)]),
}


def rect(widget_width: int) -> tuple[int, int, int]:
    """(x, y, size) of the cube's square in widget coordinates, top-right corner."""
    return widget_width - SIZE - MARGIN, MARGIN, SIZE


def inside(x: float, y: float, r: tuple[int, int, int]) -> bool:
    return r[0] <= x <= r[0] + r[2] and r[1] <= y <= r[1] + r[2]


def rotation(camera) -> np.ndarray:
    return np.stack([camera.right, camera.up, camera.back])


def to_screen(rot: np.ndarray, v, r: tuple[int, int, int]) -> tuple[float, float]:
    vx, vy, _ = rot @ np.asarray(v, float)
    cx, cy = r[0] + r[2] / 2, r[1] + r[2] / 2
    return cx + vx / EXTENT * r[2] / 2, cy - vy / EXTENT * r[2] / 2


def visible_faces(rot: np.ndarray) -> list[str]:
    """Faces turned toward the viewer."""
    return [n for n, (normal, _, _) in FACES.items() if (rot @ np.asarray(normal, float))[2] > 1e-6]


def _in_polygon(px: float, py: float, poly: list[tuple[float, float]]) -> bool:
    inside_ = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 > py) != (y2 > py) and px < (x2 - x1) * (py - y1) / (y2 - y1) + x1:
            inside_ = not inside_
    return inside_


def pick(rot: np.ndarray, x: float, y: float, r: tuple[int, int, int]) -> str | None:
    for name in visible_faces(rot):
        if _in_polygon(x, y, [to_screen(rot, c, r) for c in FACES[name][2]]):
            return name
    return None


def view_for(name: str):
    """(normal, camera right) for looking at the named face head-on."""
    normal, right, _ = FACES[name]
    return np.array(normal, float), np.array(right, float)
