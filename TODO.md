# rCad TODO

## P0: Correctness and stability

- [ ] Fix expression-constraint creation: append the new constraint before solving, then verify it survives rebuild, undo/redo, and save/load.
- [ ] Make variable edits re-solve dependent expressions, rebuild the model, and mark the document dirty. Test editing a variable and confirming the updated geometry is saved.
- [ ] Scope sketch undo/redo history to the sketch being edited, or clear/switch histories safely when entering another sketch. Test undo and redo after moving between sketches.
- [ ] Fix fillet result-edge ownership for circular and mixed edge selections. Keep metadata associated with each selected edge instead of relying on a shortened parallel direction list. Verify multiple fillets remain independently editable after rebuild and save/load.
- [ ] Replace midpoint-only face and edge references with stable references or deterministic geometric signatures. When an upstream edit removes or makes a reference ambiguous, report which feature needs repair instead of silently binding to a different entity.
- [ ] Format live extrude and fillet labels in the selected display units, matching sketch dimensions and dialogs.
- [ ] Expand edge-operation regression coverage: circular edges, mixed circular and straight selections, seams, disconnected solids, invalid radii, edits to earlier features, undo/redo, and save/load.

## P1: Interaction and sketching

- [ ] Unify direct manipulation for extrude, push/pull, fillet, and chamfer with live previews and precise numeric entry; reduce the split between drag interactions and dialogs.
- [ ] Add smart snapping and optional auto-constraints for endpoints, midpoints, centers, tangencies, and alignments.
- [ ] Make displayed sketch dimensions editable on-canvas and open the corresponding constraint editor when selected.
- [ ] Offer context-aware actions based on the current selection, and add direct drag editing for chamfer to complement the existing fillet interaction.
- [ ] Add sketch arcs, trim/split/offset tools, construction geometry, and projected model edges. Support closed profiles that combine lines, arcs, and circles.
- [ ] Add common sketch constraints such as angle, radius, tangent, and concentric, with solver status and expression support where applicable.

## P1: Modeling

- [x] Add a revolve feature using a sketch profile and selectable axis. Cover add/cut modes, rebuilds, and project-file round trips.
- [ ] Add sweep and loft features with selectable paths/sections, add/cut modes, rebuilds, and project-file round trips.
- [ ] Add shell, offset-face, and thicken operations with clear limits for unsupported OpenCascade cases.
- [ ] Add explicit boolean intersection and improve body-aware union/subtract workflows. Keep disconnected solids independently selectable and editable.
- [ ] Add construction planes and axes, then parametric move/rotate/mirror/pattern/split-body features.
- [ ] Extend push/pull beyond planar faces where the geometry kernel supports a predictable result. Otherwise, make unsupported face types explicit in the UI and error messages.

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