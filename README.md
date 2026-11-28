# rCad

A small parametric 3D CAD app for Linux, built on [build123d](https://github.com/gumyr/build123d) (OpenCascade), [SolveSpace](https://solvespace.com/)'s `libslvs` constraint solver, and PySide6.

- Sketch on the XY/XZ/YZ planes or on any planar face (line, rectangle, circle, arc), with construction geometry for reference
- Curves split each other where they cross: overlapping shapes and lines drawn across a shape make separate regions, each of which can be picked, extruded, or revolved on its own
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
| Reference planes | the Top (XY), Front (XZ), and Right (YZ) planes through the origin highlight under the pointer; click one (where no part of the model is in the way) and use ✏️ Sketch to sketch on it. P hides or shows them |
| Multi-select | Ctrl + click |
| Whole body / whole outline | double-click a face or edge (a sketch line selects every line connected to it); Ctrl adds |
| Move or mirror a body | select it (double-click), then ✥ Move (X/Y/Z offsets, a rotation about X, Y or Z through its centre) or ⇋ Mirror (across the YZ, XZ or XY plane at an offset); Keep: Move/Copy. Bodies that end up touching or overlapping join |
| Select everything | Ctrl + A (every body, or everything in the sketch) |
| Box select | in the model Ctrl + drag (adds the visible edges wholly inside); in a sketch drag from empty space (points, lines and circles wholly inside; Ctrl adds) |
| Cancel / clear selection | Esc (a tag at the top of the view names what is selected) |
| Step back | right-click (without dragging): cancels an open tool, then drops a half-drawn shape, then leaves the drawing tool, then leaves the sketch. On something (or with something selected) it opens that selection's tools at the pointer instead |
| Next step for a selection | use the quick buttons beside it: a face offers Sketch and Push/Pull, edges Fillet and Chamfer (and 🗑 for a fillet's edge), a highlighted region Extrude and Revolve, sketch geometry the constraints that fit it; ⋯ holds the rest |
| Push/pull a face | select it, then drag it |
| Extrude a sketch region | click it (an arrow appears on it), then drag the region or the arrow; type a size while dragging as with push/pull. While regions are picked, a plain click on another region of the same sketch adds it and clicking a picked one drops it. The result adds or cuts depending on where it goes |
| Fillet or chamfer an edge | drag the edge across itself: toward the body (push) rounds it, away from the body (pull) bevels it; you can switch direction mid-drag. Dragging a fillet's rounded edge resizes it, and pulling it turns it into a chamfer. If the body side can't be seen (the edge faces you), dragging up the screen counts as a pull |
| Exact size while dragging a face or edge | type a value (units, maths and variables work: `2in + 5mm`, `wall * 2`), then Enter or click; Esc cancels |
| Change the size just made | click the value box left on the model and type, or drag it sideways to scrub (Shift: fine, Ctrl: coarse) |
| Change a sketch dimension | click its label on the sketch (hovering shows what it measures in red): the settings bar opens with its value and the sketch follows as you type or drag the value sideways; Enter keeps it, Esc puts it back. Right-click the label to delete the dimension. Double-clicking it in the Constraints list does the same |
| Constraint marks | each constraint without a dimension shows a small faint symbol beside what it holds (H/V horizontal and vertical, ∥ parallel, ⊥ perpendicular, = equal, ● coincident, ∈ on a line/circle/arc, M midpoint, T tangent, ⇋ symmetry, Fix). Hover one to see its geometry, click to select it, then Delete (or right-click it) to remove it. Constrain ▾ → Show constraint marks hides them; while a drawing tool is active they let clicks through |
| Refused constraints | a constraint that contradicts or repeats others, or that would shrink a line, circle or arc to nothing, is not added: the status bar says why and the constraints it clashes with turn red for a few seconds |
| Draw an arc | ◠ Arc: click the centre, then the start (this sets the radius), then move round either way and click where it ends. Type R for the radius before the start, ∠ for the angle it sweeps (negative: clockwise) before the end. Tangent, Radius and Concentric are in Constrain; a line and an arc (or two arcs) can be tangent where they share an end |
| Tangent arc | ◡ Tangent arc: click the end of a line or arc and the arc leaves it smoothly, bending toward the pointer; click where it ends (R and ∠ can be typed). It goes on from its end until Esc. Choosing Tangent arc or Line in the middle of a chain carries on from the chain's end, and a line drawn on from an arc's end snaps straight along its tangent. Every joint drawn this way is held tangent |
| Construction geometry | select lines, circles or arcs and press Q (or Modify ▾ → Construction): they turn dashed and still work for snapping and constraints, but never make regions. Q again makes them normal |
| Snapping while drawing | points snap to existing points, line midpoints, lines and circles, and to being level with or straight above another point (or the line's start), shown by a marker, dashed guides, and the status bar. The snap becomes a constraint (coincident, midpoint, point on line or circle, horizontal/vertical), so the shape stays attached when the sketch changes. Otherwise points land on the grid, whose step is the Snap value scaled up (×2, ×5, ×10, …) when zoomed out. Hold Ctrl to place a point on the grid only, with no snapping or constraints |
| Exact size while drawing a line, rectangle, or circle | after the first click, type a size; Tab / Shift+Tab move between sizes (length and angle, width and height, diameter); Enter or a click places it. The sizes stay on the new shape (or on a line chain's last segment after Esc): retype or scrub them to resize it, click elsewhere to put them away |

Every modelling tool is in the line along the bottom of the view: **Sketch**, **Create**, and **Modify** in the model, and **Draw**, **Constrain**, **Modify**, snap, **Extrude**, and **Exit sketch** while sketching (file, variables, undo, units, and theme stay in the top bar). Workflow: **Sketch ▾** and pick a plane (or select a face and pick "On selected face"), draw, add constraints, **✓ Exit sketch**, then **Create ▾** → **Extrude**, **Revolve**, **Sweep**, or **Loft**. Revolve uses the sketch U or V axis; sweep uses a connected line sketch as its path; loft uses two or more ordered sketch sections. Select a face for **Modify ▾ → Push/Pull**, or edges for **Fillet**/**Chamfer**. Select a body for **Move body** or **Mirror body**. Extrude, Revolve, Push/Pull, Fillet, Chamfer, Move, Mirror, sketch dimensions, and editing any of them open a settings bar along the bottom of the view: the model previews as you type or drag a value sideways, click an option word (Auto/Add/Cut, the axis) to switch it, then ✓ or Enter to keep it, ✕ or Esc to cancel. Double-click a feature to edit it. Every number field, in dialogs and on the model, accepts units (`2in`, `5cm`), arithmetic (`40/3 + 2`), and variables (`wall * 2`); plain numbers are in the display unit, and variables are millimetres.

## Saving and recovery

Saving writes the project in one step: the new file is written beside the old one and swapped in only when it is complete, so a crash or a full disk never leaves a half-written project, and the previous version is kept as `<name>.rcad.bak`. Every minute, unsaved changes are also copied to a recovery folder (`~/.local/share/rcad/recovery` on Linux), never to the project itself. If rCad closes without saving, the next start offers to restore that copy (Restore, Discard, or Not now). Saving, opening another project, or closing normally removes it.

## Limitations

Push/pull faces and fillet/chamfer edges are found again after upstream edits by a geometric signature (type, position, size, direction, neighbouring faces), so they follow a face or edge that grows or moves along its normal, such as a height or width change. When the reference can't be found, or two faces or edges match equally well, the feature fails with a message in the feature list instead of using a different face or edge; one that moved sideways to a new place has to be redone. Moved and mirrored bodies are found again by their centre and size, so they follow a body that grows or shifts by less than its own extent. A sketch on a face keeps its plane where it was drawn. Push/pull supports planar faces only.

## Tests

```sh
pytest
```
