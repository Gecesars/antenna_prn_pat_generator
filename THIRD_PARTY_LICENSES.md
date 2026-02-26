# Third-Party Licenses

This project integrates or interoperates with third-party tools and libraries. You must comply with each tool's license terms when distributing binaries or automation that depends on them.

## FreeCAD

- Project: FreeCAD
- Website: https://www.freecad.org/
- Typical license in distribution: LGPL-2.0-or-later (core), with additional component licenses.
- Usage in this project: optional external geometric kernel runtime (`mech/mechanical/providers/freecad_provider.py`).

## Open CASCADE Technology (OCCT)

- Project: Open CASCADE Technology
- Website: https://www.opencascade.com/open-cascade-technology/
- Typical license: LGPL-2.1 with OCCT exception.
- Usage in this project: indirectly via FreeCAD geometric operations.

## Notes

- This repository does not statically link FreeCAD/OCCT.
- Integration is capability-gated and optional; the app runs in fallback mode without FreeCAD.
- Always verify exact license text from the installed version you distribute.
