# rCad TODO

## P0: Correctness and stability

- [x] Fix expression-constraint creation: append the new constraint before solving, then verify it survives rebuild, undo/redo, and save/load.
- [x] Make variable edits re-solve dependent expressions, rebuild the model, and mark the document dirty. Test editing a variable and confirming the updated geometry is saved.
- [x] Scope sketch undo/redo history to the sketch being edited, or clear/switch histories safely when entering another sketch. Test undo and redo after moving between sketches.
- [x] Fix fillet result-edge ownership for circular and mixed edge selections. Keep metadata associated with each selected edge instead of relying on a shortened parallel direction list. Verify multiple fillets remain independently editable after rebuild and save/load.
- [ ] Replace midpoint-only face and edge references with stable references or deterministic geometric signatures. When an upstream edit removes or makes a reference ambiguous, report which feature needs repair instead of silently binding to a different entity.
- [x] Format live extrude and fillet labels in the selected display units, matching sketch dimensions and dialogs.
- [ ] Expand edge-operation regression coverage: circular edges, mixed circular and straight selections, seams, disconnected solids, invalid radii, edits to earlier features, undo/redo, and save/load.

## P1: Interaction and sketching

- [ ] Unify direct manipulation for extrude, push/pull, fillet, and chamfer with live previews and precise numeric entry; reduce the split between drag interactions and dialogs.
- [ ] Add smart snapping and optional auto-constraints for endpoints, midpoints, centers, tangencies, and alignments.
- [ ] Make displayed sketch dimensions editable on-canvas and open the corresponding constraint editor when selected.
- [ ] Offer context-aware actions based on the current selection, and add direct drag editing for chamfer to complement the existing fillet interaction.
- [ ] Add sketch arcs, trim/split/offset tools, construction geometry, and projected model edges. Support closed profiles that combine lines, arcs, and circles.
- [ ] Add common sketch constraints such as angle, radius, tangent, and concentric, with solver status and expression support where applicable.

## P1: Feel and flow (ideas from superGeometry.app)

Interaction ideas taken from superGeometry.app's README (Nov 2026). Several of them describe how to do the P1 interaction items above.

- [ ] Type while dragging: every drag or draw shows a value box by the pointer that stays after release. Typing a number goes into it, Tab moves to the next size, Enter confirms, and dragging the box left or right scrubs the value. Every number field accepts units (`2in`, `5cm`), arithmetic (`40/3 + 2`), and variables (`wall * 2`).
- [ ] Replace feature dialogs with a tool bar that becomes the active tool's settings: values you can tap and type or drag, option words to toggle, a tick to finish and a cross to cancel.
- [ ] Quick buttons beside the current selection: its two likeliest next tools (flat face → Sketch, Push/Pull; edge → Fillet, Chamfer; body → Move, Mirror), then Delete and More. Show only the tools that apply to the selection.
- [ ] Hover previews: with the pointer over a tool button, draw a dashed preview of what it would do to the current selection before the tool starts.
- [ ] Sketch regions: overlapping closed shapes split into separately selectable regions. Clicking a region shows the extrude arrow right away, the region itself can be dragged, more regions can be added while the distance is still 0, and construction geometry never makes regions.
- [ ] Auto extrude operation: decide new body, add, or cut from the drag direction (on a plane, up is a new body and down is a cut; on a face, out adds and in cuts), with an explicit override. Start at the last-used distance.
- [ ] Right-click steps back: first a half-finished entity or pick, then the active tool, then out of the sketch. With something selected, right-click opens its tools at the pointer.
- [ ] Selection shortcuts: double-click selects the whole body, Ctrl+A selects everything, drag a box on empty space to select several, and a tag names what is picked.
- [ ] Navigation: mouse presets matching other CAD programs (Onshape, Fusion, SolidWorks, Blender, FreeCAD, and others), a settable orbit centre, 90° turn buttons on the view cube, double-click the cube to fit, and saved views.
- [ ] Smooth onboarding: a new project opens in a sketch on XY looking down with Line active, and leaving a sketch swings to a three-quarter view.
- [ ] Strict constraints: refuse a constraint or typed value that contradicts the existing ones, with a message, instead of distorting the sketch. Draw constraint marks small, faint, and colour-coded by kind, and keep them from catching clicks meant for geometry.
- [ ] Grid snapping whose step follows the zoom level (for example 1, 5, or 10 mm).
- [ ] Variable tags on the model: show where each variable acts, and let a tag be clicked to type a value or dragged to change it with the model following live.
- [ ] History panel: hovering a step outlines the faces and edges it uses, steps can be grouped and folded, an eye suppresses a step, and a failed step turns red and says why.
- [ ] Run rebuilds off the UI thread, with a "Working…" indicator and a Stop button that cancels the rebuild and reverts the change that started it.
- [ ] A complete keyboard shortcut set (S sketch, L line, R rectangle, C circle, E extrude, F fillet, T trim, M move, and others) and a short built-in tutorial that walks through a first part.

## P1: Modeling

- [x] Add a revolve feature using a sketch profile and selectable axis. Cover add/cut modes, rebuilds, and project-file round trips.
- [x] Add sweep and loft features with selectable paths/sections, add/cut modes, rebuilds, and project-file round trips.
- [ ] Add shell, offset-face, and thicken operations with clear limits for unsupported OpenCascade cases.
- [ ] Add explicit boolean intersection and improve body-aware union/subtract workflows. Keep disconnected solids independently selectable and editable.
- [ ] Add construction planes and axes, then parametric move/rotate/mirror/pattern/split-body features.
- [ ] Extend push/pull beyond planar faces where the geometry kernel supports a predictable result. Otherwise, make unsupported face types explicit in the UI and error messages.
- [ ] Extrude taper: a far-end scale percentage, so a pyramid or cone takes one step.
- [ ] Revolve with travel along the axis for threads and springs (angles above 360°).
- [ ] Hole tool with ISO screw presets (M2 to M12; plain, counterbore, countersink), through or to a depth, with editable sizes and positions measured from the nearest edges.
- [ ] Thread tool for round faces and holes, using the ISO coarse pitch by default, with a left-hand option and a lead-in at free ends.
- [ ] Move/rotate gimbals for bodies (with a Copy option) and for faces. Delete or replace a face, fillet, or chamfer so the neighbouring faces extend until they meet.
- [ ] More sketch tools: splines (including freehand strokes cleaned up into a few lines, arcs, and curves), ellipses, polygons, sketch fillet and chamfer, sketch move/rotate/scale, and linear and circular sketch patterns.
- [ ] Per-body colour and transparency, plus isolate.
- [ ] Export 3MF (keeping body names and colours) and export sketches as DXF, with construction geometry on its own layer.

## P2: Workflow and inspection

- [ ] Extend feature history with rollback, reorder, and rename operations, while preserving dependency checks and recovery from invalid downstream features.
- [ ] Replace the flat Layers list with a structured items panel for bodies, sketches, planes, and folders, including visibility controls.
- [ ] Add measurement tools for distance, angle, area, volume, and bounding boxes, plus a draggable section plane.
- [ ] Add display styles for smooth shading and hidden edges, and improve selection with box selection and tangent-chain/loop picking.
- [ ] Add import for STEP and IGES solids, BREP, STL meshes, and DXF/SVG sketch geometry, with unit handling and explicit multi-object behavior.
- [ ] Add atomic project saves, autosave, and crash recovery.
- [ ] Update the README controls and workflow documentation to cover undo/redo, variables, expressions, and the implemented direct-manipulation tools.
- [ ] Improve recovery from failed history features: identify the failing feature in the tree, preserve the last valid model, and make it straightforward to edit or suppress the failure.
- [ ] Add end-to-end smoke coverage for opening, editing, saving, and exporting a representative project, alongside the existing focused unit and GUI tests.

## P3: Larger projects

- [ ] Add 2D drawings with generated views and dimensions.
- [ ] Add assemblies with components and joints.
- [ ] Add rendering with materials and lighting.
- [ ] Add sketch text and reference images, then evaluate pen and touch input.

## Foundations and release readiness

- [ ] Replace immediate-mode OpenGL rendering with buffered geometry and shaders, keeping selection and overlays responsive on larger models.
- [ ] Split the large application module into focused UI, interaction, and document modules without changing behavior.
- [ ] Review dependency licensing, including the SolveSpace/libslvs integration, and choose and document rCad's license before distributing builds.