# rCad

A small parametric 3D CAD app for Linux, built on [build123d](https://github.com/gumyr/build123d) (OpenCascade), [SolveSpace](https://solvespace.com/)'s `libslvs` constraint solver, and PySide6.

- Sketch on the XY/XZ/YZ planes or on any planar face (line, rectangle, circle)
- Geometric and dimensional constraints solved by libslvs
- Extrude, revolve, sweep, and loft (add or cut), face push/pull, fillet, chamfer
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
| Push/pull a face | select it, then drag it |
| Fillet or chamfer an edge | drag the edge across itself: toward the body (push) rounds it, away from the body (pull) bevels it; you can switch direction mid-drag. Dragging a fillet's rounded edge resizes it, and pulling it turns it into a chamfer. If the body side can't be seen (the edge faces you), dragging up the screen counts as a pull |
| Exact size while dragging a face or edge | type a value (units, maths and variables work: `2in + 5mm`, `wall * 2`), then Enter or click; Esc cancels |
| Change the size just made | click the value box left on the model and type, or drag it sideways to scrub (Shift: fine, Ctrl: coarse) |
| Exact size while drawing a line, rectangle, or circle | after the first click, type a size; Tab / Shift+Tab move between sizes (length and angle, width and height, diameter); Enter or a click places it. The sizes stay on the new shape (or on a line chain's last segment after Esc): retype or scrub them to resize it, click elsewhere to put them away |

Every modelling tool is in the line along the bottom of the view: **Sketch**, **Create**, and **Modify** in the model, and **Draw**, **Constrain**, **Modify**, snap, **Extrude**, and **Exit sketch** while sketching (file, variables, undo, units, and theme stay in the top bar). Workflow: **Sketch ▾** and pick a plane (or select a face and pick "On selected face"), draw, add constraints, **✓ Exit sketch**, then **Create ▾** → **Extrude**, **Revolve**, **Sweep**, or **Loft**. Revolve uses the sketch U or V axis; sweep uses a connected line sketch as its path; loft uses two or more ordered sketch sections. Select a face for **Modify ▾ → Push/Pull**, or edges for **Fillet**/**Chamfer**. Extrude, Revolve, Push/Pull, Fillet, Chamfer, sketch dimensions, and editing any of them open a settings bar along the bottom of the view: the model previews as you type or drag a value sideways, click an option word (Auto/Add/Cut, the axis) to switch it, then ✓ or Enter to keep it, ✕ or Esc to cancel. Double-click a feature to edit it. Every number field, in dialogs and on the model, accepts units (`2in`, `5cm`), arithmetic (`40/3 + 2`), and variables (`wall * 2`); plain numbers are in the display unit, and variables are millimetres.

## Limitations

Faces and edges are re-found by geometry, not topological IDs, so an upstream change can leave a later feature unable to find its target. Push/pull supports planar faces only.

## Tests

```sh
pytest
```
