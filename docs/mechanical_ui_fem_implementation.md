# Mechanical UI + FEM Implementation Notes

This document summarizes the implementation delivered from `mechanical_ui_fem_roadmap.md`.

## Implemented Architecture

- `SelectionManager` added in `mech/ui/selection_manager.py`.
  - Supports modes: `object`, `face`, `edge`, `vertex`, `body`, `component`.
  - Handles replace/add/toggle/remove logic.
  - Tracks `hover`, `active item`, multi-selection and filtering (`hidden/locked/layer/type`).
  - Synchronizes viewport/tree/property updates through Qt signals.

- `FEMStudyManager` added in `mech/engine/fem_study.py`.
  - Study lifecycle: create, activate, remove.
  - Data handling: participating bodies, FEM role, materials, contacts, BCs, loads, mesh config, solver config.
  - Validation engine (pre-solve checklist).
  - Solve simulation workflow with progress/result events.
  - Full serialization/restoration integrated in `SceneEngine`.

- `SceneEngine` integration updated in `mech/engine/scene_engine.py`.
  - New `engine.fem` manager.
  - `serialize_state()` / `restore_state()` include `fem_state`.
  - Object metadata normalization now includes layer/material/FEM/default component fields.

## UI/UX Changes

- Ribbon-style top UI with grouped tabs:
  - `Project`, `Edit`, `View`, `Selection`, `Modeling`, `FEM`, `Import / Export`.
- Left panel reorganized into:
  - `Scene`, `Components`, `Layers`, `Studies`.
- Right contextual panel reorganized into:
  - `Properties`, `Transform`, `Material`, `FEM`, `BCs`, `Loads`, `Mesh`, `Solve`, `Results`.
- Bottom dock reorganized into:
  - `Log`, `Measurements`, `Selection Info`, `Diagnostics`, `Solver Console`.

## Viewport and Selection Visuals

- `ViewportPyVista` updated with:
  - Floating quick toolbar (camera/fit/grid/axes/render/section/screenshot).
  - Hover tracking and hover signal.
  - Selection overlays (active/secondary) and hover outline overlays.
  - Ctrl-click selection toggle.
  - Ctrl+Alt-click visibility toggle command.
  - Alt-click quick boundary command (Shift+Alt for force).

## Scene Tree

- Columns expanded to:
  - `Name`, `Type`, `Visible`, `Locked`, `Layer`, `Material`, `FEM Role`, `Solve`.
- Existing imported multi-body assembly grouping retained.
- Visibility toggle still controlled by checkboxes.

## FEM Workflow in UI

- New actions added to `MechanicsPage`:
  - study creation/activation/removal
  - include bodies
  - assign material
  - add BC/load
  - mesh config + mesh generation
  - solver config
  - validation
  - solve run + results refresh
- Validation statuses shown in solve panel and solver console.

## Test Coverage Added

- `tests/mech/test_selection_manager.py`
  - mode/toggle logic
  - face pick mapping to parent object
  - locked filter behavior

- `tests/mech/test_fem_study_manager.py`
  - full study validation/solve flow
  - FEM state serialize/restore roundtrip

- `tests/mech/test_modeler_ux_smoke.py`
  - expanded required method coverage for new components/FEM actions
