# Mechanical FreeCAD Integration

Date: 2026-02-21

## Scope

This repository now includes a provider-based mechanical kernel integration for FreeCAD/OCCT with safe fallback mode.

Implemented components:

- `mech/mechanical/interfaces.py`: stable `MechanicalKernel` contract.
- `mech/mechanical/providers/null_provider.py`: no-crash fallback provider.
- `mech/mechanical/providers/freecad_provider.py`: in-process FreeCAD provider.
- `mech/mechanical/diagnostics.py`: capability/doctor detection.
- `mech/mechanical/worker/freecad_headless_worker.py`: JSON worker entrypoint for headless mode.
- `tools/mechanical_doctor.py`: CLI doctor report generator.

Integrated components:

- `mech/engine/scene_engine.py` now auto-selects provider (`freecad` or `null`) and exposes diagnostics/capabilities.
- `mech/ui/page_mechanics.py` includes backend diagnostics panel, doctor actions, and STEP export support.
- `mech/ui/context_menu.py` includes Tube primitive, STEP export, and validate/heal actions.
- `mech/ui/scene_tree.py` now supports assembly hierarchy and visibility toggles by body/group.

## Modeling UX Additions (2026-02-24)

- New `Boundaries` tab in Mechanics UI:
  - apply boundary conditions to selected bodies (`fixed`, `force`, `pressure`, `displacement`, `symmetry`, `contact`);
  - optional picked-point association for local boundary intent;
  - boundary listing, JSON copy, targeted clear/remove operations;
  - viewport boundary overlay toggle for selected bodies.
- Scene engine boundary model:
  - persistent `boundaries` state with undo/redo command support;
  - automatic cleanup when bodies are deleted;
  - per-object boundary summary helpers.
- Direct mouse commands on picked piece:
  - `Ctrl+Click`: toggle visibility;
  - `Alt+Click`: quick fixed boundary;
  - `Shift+Alt+Click`: quick force boundary.
- Multi-body import handling:
  - imported assemblies are split into body objects (kernel and fallback paths);
  - bodies receive import metadata (`import_asset_id`, body index/count);
  - tree presents grouped assembly node with visibility control.

## Modeler Pro Parity Updates (2026-02-24)

- Layer system integrated in engine and UI:
  - create/rename/delete layers;
  - assign selected bodies to layer;
  - per-layer visibility and color controls;
  - layer column/filter in scene tree.
- Grid controls expanded:
  - configurable grid enabled/step/size/color from `Scene` tab and preferences;
  - snap step now stays aligned with grid step.
- Face/edge interactive editing:
  - left-click supports object/face/edge pick mode;
  - selected face/edge can be offset directly from `Transform` tab;
  - picked cell/point metadata is propagated to boundary assignment.
- Selection-to-analysis workflow:
  - selected body can be opened in a dedicated dynamic analysis tab;
  - optional auto-open on left-click selection.
- Precision import/retessellation:
  - import uses configurable triangulation `deflection`;
  - kernel-backed objects can be re-tessellated interactively (`Retessellate selected`).
- FreeCAD connection controls:
  - backend reconnect action added in `Backend` tab.
- Legacy tab interoperability:
  - old CTk mechanical tab now launches Modeler Pro with runtime mesh payload (`--runtime-json`) when available;
  - imported AEDT/OBJ body metadata (color/group/material) is preserved at startup.
- UI ergonomics:
  - standard Qt icons applied on primary buttons;
  - new dropdown menu with common CAD actions (import/export, undo/redo, selection/view/backend).

## Capabilities

When FreeCAD is available (`import FreeCAD` + `import Part`):

- Primitives: box, cylinder, sphere, cone, tube.
- Transform: translate/rotate/scale.
- Booleans: union, cut, common.
- Import: STEP/STP, STL, IGES/IGS.
- Export: STEP/STP, STL, IGES/IGS.
- Triangulate/validate/heal.

When FreeCAD is not available:

- App remains operational with local mesh fallback.
- Mechanical tab remains usable.
- Doctor report explains missing capabilities.

## Installation

Python dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_mechanical_freecad.ps1
```

Optional FreeCAD install (Windows + winget):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_mechanical_freecad.ps1 -InstallFreeCAD
```

## Doctor

Run backend diagnostics and save report:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/mechanical_doctor.ps1
```

Or direct Python command:

```powershell
venv\Scripts\python.exe tools\mechanical_doctor.py --json out\mechanical_doctor_report.json
```

Important report fields:

- `freecad.inprocess_available`: true only when `import FreeCAD` and `import Part` succeed in the app Python.
- `freecad.headless_available`: true when `FreeCADCmd` is detected.
- `freecad.python_abi_match`: true when app Python minor version matches `FreeCADCmd` Python minor version.
- `freecad.import_errors`: import failure details for `FreeCAD` / `Part` / `Fem`.

## Compatibility Note (Windows)

FreeCAD 0.21 commonly ships with Python 3.8. If the app runs on Python 3.12, in-process imports will fail with DLL/ABI mismatch. In this case:

- `inprocess_available = false`
- `headless_available = true` (if `FreeCADCmd` exists)
- doctor summary includes `freecad:abi-mismatch`

This is expected and does not block app startup. The mechanical module keeps fallback behavior active.

## Tests

Mechanical marker tests:

```powershell
venv\Scripts\python.exe -m pytest -q -m mechanical --maxfail=1
```

FreeCAD-only tests are gated by marker/availability checks:

```powershell
venv\Scripts\python.exe -m pytest -q tests\mech\test_freecad_provider_gated.py --maxfail=1
```

## Quality Gate

`run_quality_gate.ps1` now includes:

- compile checks
- pytest full suite
- core coverage
- docs auto-check
- mechanical doctor report
- optional mechanical marker run (`-WithMechanical`)
- optional external AEDT run (`-WithExternalAedt`)
