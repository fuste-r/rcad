"""The origin and the three reference planes (Top, Front, Right) shown in the model view."""
from __future__ import annotations

import math

import numpy as np

# name -> (sketch plane key, normal, in-plane right, in-plane up), Z up as on the view cube
PLANES = {
    "Top": ("XY", (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "Front": ("XZ", (0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "Right": ("YZ", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}

MIN_HALF = 25.0  # half the side of a plane (mm) with no model, or a small one


def plane_half(verts: np.ndarray | None) -> float:
    """Half the planes' side: big enough to reach past the model, rounded up to 1, 2 or 5 x 10^n so
    it does not change with every small edit."""
    if verts is None or not len(verts):
        return MIN_HALF
    need = 0.6 * float(np.abs(np.asarray(verts, float)).max()) * 2
    if need <= MIN_HALF:
        return MIN_HALF
    exp = 10 ** math.floor(math.log10(need))
    for m in (1, 2, 5, 10):
        if m * exp >= need:
            return float(m * exp)
    return float(10 * exp)


def axes(name: str):
    _, n, u, v = PLANES[name]
    return np.array(n), np.array(u), np.array(v)


def corners(name: str, half: float) -> np.ndarray:
    """The plane's four corners (4, 3), counter-clockwise from the bottom-left."""
    _, u, v = axes(name)
    return np.array([(-u - v), (u - v), (u + v), (-u + v)]) * half


def label_anchor(name: str, half: float, inset: float = 0.05) -> np.ndarray:
    """Just inside the plane's top-left corner, where its name goes."""
    _, u, v = axes(name)
    return (-u + v) * half + (u - v) * half * inset


def pick(origin, direction, half: float, names=PLANES) -> tuple[str, float] | None:
    """(name, distance along the ray) of the nearest plane the ray hits inside its square, or None."""
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    best = None
    for name in names:
        n, u, v = axes(name)
        den = float(n @ d)
        if abs(den) < 1e-9:
            continue  # the ray runs along the plane
        t = -float(n @ o) / den
        if t <= 0:
            continue
        p = o + t * d
        if abs(float(p @ u)) <= half and abs(float(p @ v)) <= half and (best is None or t < best[1]):
            best = (name, t)
    return best
