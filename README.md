# rCad

A small parametric 3D CAD app for Linux, built on [build123d](https://github.com/gumyr/build123d) (OpenCascade), [SolveSpace](https://solvespace.com/)'s `libslvs` constraint solver, and PySide6.

- Sketch on the XY/XZ/YZ planes or on any planar face (line, rectangle, circle)
- Geometric and dimensional constraints solved by libslvs
- Extrude (add or cut), face push/pull, fillet, chamfer
- Feature history that rebuilds on edit; save/load `.rcad` projects (JSON); export STEP and STL

## Install

Python 3.10+ is required.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`py-slvs` ships no wheels for recent Python versions, so `pip` builds it from source and needs SWIG. If that fails:

```sh
pip install swig scikit-build cmake ninja setuptools wheel
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install --no-build-isolation py-slvs
pip install -e ".[dev]"
```

## Run

```sh
rcad            # or: python -m rcad
```

On Wayland with NVIDIA, if the OpenGL context fails to create, try `QT_QPA_PLATFORM=xcb rcad`.

## Controls

| Action | Input |
| --- | --- |
| Orbit (3D view) | left or right drag |
| Pan | middle drag, or Shift + left drag |
| Zoom | mouse wheel |
| Standard view (Top, Bottom, Front, Back, Left, Right) | click a face of the cube in the top-right corner; drag the cube to orbit |
| Multi-select | Ctrl + click |
| Cancel / clear selection | Esc |

Workflow: choose a plane (or select a face and choose "Selected face"), **New Sketch**, draw, add constraints, **Finish Sketch**, then **Extrude**. Select a face for **Push/Pull**, or edges for **Fillet**/**Chamfer**. Double-click a feature to edit it.

## Limitations

Faces and edges are re-found by geometry, not topological IDs, so an upstream change can leave a later feature unable to find its target. Push/pull supports planar faces only. There is no undo/redo yet.

## Tests

```sh
pytest
```
