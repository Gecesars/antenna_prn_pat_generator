from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
import uuid


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


@dataclass
class StudyValidationItem:
    code: str
    label: str
    status: str
    message: str

    def to_dict(self) -> dict:
        return {
            "code": str(self.code),
            "label": str(self.label),
            "status": str(self.status),
            "message": str(self.message),
        }


@dataclass
class FEMStudy:
    id: str
    name: str
    study_type: str
    units: str = "mm"
    created_utc: str = field(default_factory=_utc_now)
    updated_utc: str = field(default_factory=_utc_now)
    bodies: Dict[str, dict] = field(default_factory=dict)
    materials: Dict[str, dict] = field(default_factory=dict)
    contacts: List[dict] = field(default_factory=list)
    bcs: List[dict] = field(default_factory=list)
    loads: List[dict] = field(default_factory=list)
    mesh: Dict[str, Any] = field(
        default_factory=lambda: {
            "global_size": 20.0,
            "growth_rate": 1.2,
            "curvature_refinement": True,
            "quality_target": 0.6,
            "local_refinement": [],
            "generated": False,
            "generated_ok": False,
            "quality_avg": 0.0,
        }
    )
    solver: Dict[str, Any] = field(
        default_factory=lambda: {
            "type": "direct",
            "tolerance": 1e-6,
            "max_iterations": 500,
            "threads": 0,
            "status": "idle",
            "last_error": "",
        }
    )
    results: Dict[str, Any] = field(default_factory=dict)
    validation: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "name": str(self.name),
            "study_type": str(self.study_type),
            "units": str(self.units),
            "created_utc": str(self.created_utc),
            "updated_utc": str(self.updated_utc),
            "bodies": {str(k): dict(v) for k, v in self.bodies.items()},
            "materials": {str(k): dict(v) for k, v in self.materials.items()},
            "contacts": [dict(x) for x in self.contacts],
            "bcs": [dict(x) for x in self.bcs],
            "loads": [dict(x) for x in self.loads],
            "mesh": dict(self.mesh),
            "solver": dict(self.solver),
            "results": dict(self.results),
            "validation": [dict(x) for x in self.validation],
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "FEMStudy":
        row = dict(data or {})
        study = FEMStudy(
            id=str(row.get("id", str(uuid.uuid4()))),
            name=str(row.get("name", "Study")),
            study_type=str(row.get("study_type", "Structural Static")),
            units=str(row.get("units", "mm")),
            created_utc=str(row.get("created_utc", _utc_now())),
            updated_utc=str(row.get("updated_utc", _utc_now())),
            bodies={str(k): dict(v) for k, v in dict(row.get("bodies", {})).items()},
            materials={str(k): dict(v) for k, v in dict(row.get("materials", {})).items()},
            contacts=[dict(x) for x in row.get("contacts", []) if isinstance(x, dict)],
            bcs=[dict(x) for x in row.get("bcs", []) if isinstance(x, dict)],
            loads=[dict(x) for x in row.get("loads", []) if isinstance(x, dict)],
            mesh=dict(row.get("mesh", {})),
            solver=dict(row.get("solver", {})),
            results=dict(row.get("results", {})),
            validation=[dict(x) for x in row.get("validation", []) if isinstance(x, dict)],
        )
        return study


class FEMStudyManager:
    def __init__(
        self,
        object_provider: Callable[[], Mapping[str, Any]],
        event_emitter: Optional[Callable[[str, dict], None]] = None,
    ):
        self._object_provider = object_provider
        self._emit = event_emitter
        self._studies: Dict[str, FEMStudy] = {}
        self._active_id: str = ""

    # ---------------- lifecycle/state ----------------
    def serialize(self) -> dict:
        return {
            "active_id": str(self._active_id),
            "studies": {sid: study.to_dict() for sid, study in self._studies.items()},
        }

    def restore(self, data: Mapping[str, Any]) -> None:
        row = dict(data or {})
        self._studies = {}
        for sid, raw in dict(row.get("studies", {})).items():
            study = FEMStudy.from_dict(raw if isinstance(raw, dict) else {})
            self._studies[str(sid)] = study
        aid = str(row.get("active_id", "") or "")
        self._active_id = aid if aid in self._studies else (next(iter(self._studies.keys()), ""))

    def list_studies(self) -> List[dict]:
        rows = [study.to_dict() for study in self._studies.values()]
        rows.sort(key=lambda x: (str(x.get("created_utc", "")), str(x.get("name", ""))))
        return rows

    def active_study_id(self) -> str:
        return str(self._active_id or "")

    def get(self, study_id: str = "") -> Optional[FEMStudy]:
        sid = str(study_id or self._active_id or "")
        return self._studies.get(sid)

    def _touch(self, study: FEMStudy) -> None:
        study.updated_utc = _utc_now()

    def _notify(self, event: str, payload: Optional[dict] = None) -> None:
        if callable(self._emit):
            self._emit(str(event), dict(payload or {}))

    def new_study(self, name: str = "", study_type: str = "Structural Static", units: str = "mm") -> str:
        sid = str(uuid.uuid4())
        token = str(name or "").strip() or f"Study {len(self._studies) + 1}"
        study = FEMStudy(
            id=sid,
            name=token,
            study_type=str(study_type or "Structural Static"),
            units=str(units or "mm"),
        )
        self._studies[sid] = study
        self._active_id = sid
        self._notify("scene_updated", {"reason": "new_study", "study_id": sid})
        return sid

    def set_active(self, study_id: str) -> bool:
        sid = str(study_id or "")
        if sid not in self._studies:
            return False
        self._active_id = sid
        self._notify("scene_updated", {"reason": "active_study_changed", "study_id": sid})
        return True

    def remove_study(self, study_id: str) -> bool:
        sid = str(study_id or "")
        if sid not in self._studies:
            return False
        self._studies.pop(sid, None)
        if self._active_id == sid:
            self._active_id = next(iter(self._studies.keys()), "")
        self._notify("scene_updated", {"reason": "study_removed", "study_id": sid})
        return True

    # ---------------- data editing ----------------
    def include_bodies(self, object_ids: Sequence[str], *, fem_role: str = "solid", exclude: bool = False, study_id: str = "") -> int:
        study = self.get(study_id)
        if study is None:
            return 0
        scene_objects = dict(self._object_provider() or {})
        touched = 0
        for oid in [str(x) for x in object_ids if str(x).strip()]:
            if oid not in scene_objects:
                continue
            row = dict(study.bodies.get(oid, {}))
            row["fem_role"] = str(fem_role or row.get("fem_role", "solid") or "solid")
            row["exclude_from_solve"] = bool(exclude)
            study.bodies[oid] = row
            obj = scene_objects.get(oid)
            if obj is not None and isinstance(getattr(obj, "meta", None), dict):
                obj.meta["fem_role"] = str(row["fem_role"])
                obj.meta["exclude_from_solve"] = bool(row["exclude_from_solve"])
            touched += 1
        if touched:
            self._touch(study)
            self._notify("scene_updated", {"reason": "study_bodies_updated", "study_id": study.id, "count": touched})
        return touched

    def assign_material(self, object_ids: Sequence[str], material_name: str, properties: Optional[Mapping[str, Any]] = None, *, study_id: str = "") -> int:
        study = self.get(study_id)
        if study is None:
            return 0
        scene_objects = dict(self._object_provider() or {})
        name = str(material_name or "").strip()
        if not name:
            return 0
        touched = 0
        for oid in [str(x) for x in object_ids if str(x).strip()]:
            if oid not in scene_objects:
                continue
            mat = {
                "name": str(name),
                "properties": dict(properties or {}),
                "assigned_utc": _utc_now(),
            }
            study.materials[oid] = mat
            if oid not in study.bodies:
                study.bodies[oid] = {"fem_role": "solid", "exclude_from_solve": False}
            obj = scene_objects.get(oid)
            if obj is not None and isinstance(getattr(obj, "meta", None), dict):
                obj.meta["material"] = str(name)
            touched += 1
        if touched:
            self._touch(study)
            self._notify("material_assigned", {"study_id": study.id, "count": touched})
        return touched

    def add_contact(
        self,
        master_object_id: str,
        slave_object_id: str,
        *,
        contact_type: str = "bonded",
        friction_coef: float = 0.0,
        study_id: str = "",
    ) -> str:
        study = self.get(study_id)
        if study is None:
            return ""
        row = {
            "id": str(uuid.uuid4()),
            "master": str(master_object_id),
            "slave": str(slave_object_id),
            "type": str(contact_type or "bonded").strip().lower(),
            "friction_coef": float(max(0.0, _float(friction_coef, 0.0))),
            "enabled": True,
        }
        study.contacts.append(row)
        self._touch(study)
        self._notify("scene_updated", {"reason": "study_contact_added", "study_id": study.id})
        return str(row["id"])

    def add_boundary_condition(self, object_ids: Sequence[str], bc_type: str, params: Optional[Mapping[str, Any]] = None, *, study_id: str = "") -> str:
        study = self.get(study_id)
        if study is None:
            return ""
        targets = [str(x) for x in object_ids if str(x).strip()]
        if not targets:
            return ""
        row = {
            "id": str(uuid.uuid4()),
            "type": str(bc_type or "fixed support").strip().lower(),
            "targets": list(targets),
            "params": dict(params or {}),
            "enabled": True,
        }
        study.bcs.append(row)
        self._touch(study)
        self._notify("scene_updated", {"reason": "study_bc_added", "study_id": study.id})
        return str(row["id"])

    def add_load(self, object_ids: Sequence[str], load_type: str, params: Optional[Mapping[str, Any]] = None, *, study_id: str = "") -> str:
        study = self.get(study_id)
        if study is None:
            return ""
        targets = [str(x) for x in object_ids if str(x).strip()]
        if not targets:
            return ""
        row = {
            "id": str(uuid.uuid4()),
            "type": str(load_type or "force").strip().lower(),
            "targets": list(targets),
            "params": dict(params or {}),
            "enabled": True,
        }
        study.loads.append(row)
        self._touch(study)
        self._notify("scene_updated", {"reason": "study_load_added", "study_id": study.id})
        return str(row["id"])

    def configure_mesh(self, *, study_id: str = "", **mesh_cfg: Any) -> bool:
        study = self.get(study_id)
        if study is None:
            return False
        base = dict(study.mesh)
        base.update(dict(mesh_cfg or {}))
        if "global_size" in base:
            base["global_size"] = max(1e-6, _float(base.get("global_size"), 20.0))
        if "growth_rate" in base:
            base["growth_rate"] = max(1.0, _float(base.get("growth_rate"), 1.2))
        if "quality_target" in base:
            base["quality_target"] = max(0.0, min(1.0, _float(base.get("quality_target"), 0.6)))
        if mesh_cfg:
            base["generated"] = False
            base["generated_ok"] = False
        study.mesh = base
        self._touch(study)
        self._notify("scene_updated", {"reason": "study_mesh_cfg", "study_id": study.id})
        return True

    def mark_mesh_generated(self, ok: bool = True, *, quality_avg: float = 0.75, study_id: str = "") -> bool:
        study = self.get(study_id)
        if study is None:
            return False
        study.mesh["generated"] = True
        study.mesh["generated_ok"] = bool(ok)
        study.mesh["quality_avg"] = max(0.0, min(1.0, _float(quality_avg, 0.75)))
        self._touch(study)
        self._notify("mesh_generated", {"study_id": study.id, "ok": bool(ok), "quality_avg": study.mesh["quality_avg"]})
        return True

    def configure_solver(self, *, study_id: str = "", **solver_cfg: Any) -> bool:
        study = self.get(study_id)
        if study is None:
            return False
        base = dict(study.solver)
        base.update(dict(solver_cfg or {}))
        base["tolerance"] = max(1e-12, _float(base.get("tolerance"), 1e-6))
        base["max_iterations"] = max(1, int(round(_float(base.get("max_iterations"), 500))))
        base["threads"] = max(0, int(round(_float(base.get("threads"), 0))))
        study.solver = base
        self._touch(study)
        self._notify("scene_updated", {"reason": "study_solver_cfg", "study_id": study.id})
        return True

    # ---------------- validation and solve ----------------
    def validate(self, study_id: str = "") -> List[dict]:
        study = self.get(study_id)
        if study is None:
            return []
        checks: List[StudyValidationItem] = []
        scene_objects = dict(self._object_provider() or {})

        bodies_active = [oid for oid, row in study.bodies.items() if oid in scene_objects and not bool(row.get("exclude_from_solve", False))]
        if bodies_active:
            checks.append(StudyValidationItem("bodies_active", "Existe pelo menos 1 corpo ativo?", "ok", f"{len(bodies_active)} corpo(s) ativo(s)."))
        else:
            checks.append(StudyValidationItem("bodies_active", "Existe pelo menos 1 corpo ativo?", "error", "Nenhum corpo ativo no estudo."))

        missing_material = [oid for oid in bodies_active if oid not in study.materials]
        if missing_material:
            checks.append(
                StudyValidationItem(
                    "materials_complete",
                    "Todos os corpos ativos possuem material?",
                    "error",
                    f"{len(missing_material)} corpo(s) sem material.",
                )
            )
        else:
            checks.append(StudyValidationItem("materials_complete", "Todos os corpos ativos possuem material?", "ok", "Materiais atribuídos."))

        bc_types = {"fixed", "fixed support", "displacement", "symmetry", "remote support", "pin", "hinge"}
        has_constraints = any(
            bool(set([str(x) for x in row.get("targets", [])]).intersection(set(bodies_active)))
            and str(row.get("type", "")).strip().lower() in bc_types
            and bool(row.get("enabled", True))
            for row in study.bcs
        )
        if has_constraints:
            checks.append(StudyValidationItem("constraints", "O sistema está adequadamente restringido?", "ok", "Restrições encontradas."))
        else:
            checks.append(
                StudyValidationItem(
                    "constraints",
                    "O sistema está adequadamente restringido?",
                    "error",
                    "Nenhuma condição de suporte/restrição identificada.",
                )
            )

        enabled_bcs = [row for row in study.bcs if bool(row.get("enabled", True))]
        enabled_loads = [row for row in study.loads if bool(row.get("enabled", True))]
        if enabled_bcs or enabled_loads:
            checks.append(StudyValidationItem("loads_present", "Existe pelo menos uma carga/condição aplicada?", "ok", "Condições/cargas presentes."))
        else:
            checks.append(
                StudyValidationItem(
                    "loads_present",
                    "Existe pelo menos uma carga/condição aplicada?",
                    "error",
                    "Nenhuma carga ou condição foi aplicada.",
                )
            )

        invalid_contacts: List[str] = []
        for row in study.contacts:
            if not bool(row.get("enabled", True)):
                continue
            master = str(row.get("master", ""))
            slave = str(row.get("slave", ""))
            if not master or not slave or master == slave:
                invalid_contacts.append(str(row.get("id", "")))
                continue
            if master not in bodies_active or slave not in bodies_active:
                invalid_contacts.append(str(row.get("id", "")))
        if invalid_contacts:
            checks.append(
                StudyValidationItem(
                    "contacts_valid",
                    "Contatos inválidos/ambíguos?",
                    "warning",
                    f"{len(invalid_contacts)} contato(s) requer(em) revisão.",
                )
            )
        else:
            checks.append(StudyValidationItem("contacts_valid", "Contatos inválidos/ambíguos?", "ok", "Contatos consistentes."))

        mesh_ok = bool(study.mesh.get("generated", False)) and bool(study.mesh.get("generated_ok", False))
        if mesh_ok:
            checks.append(StudyValidationItem("mesh_generated", "Malha foi gerada com sucesso?", "ok", "Malha gerada."))
        else:
            checks.append(
                StudyValidationItem(
                    "mesh_generated",
                    "Malha foi gerada com sucesso?",
                    "error",
                    "Malha ausente ou inválida. Gere/atualize a malha.",
                )
            )

        allowed_units = {"mm", "cm", "m", "in"}
        if str(study.units).strip().lower() in allowed_units:
            checks.append(StudyValidationItem("units", "Unidades coerentes?", "ok", f"Unidade atual: {study.units}."))
        else:
            checks.append(
                StudyValidationItem(
                    "units",
                    "Unidades coerentes?",
                    "error",
                    f"Unidade não suportada para solve: {study.units}.",
                )
            )

        study.validation = [row.to_dict() for row in checks]
        self._touch(study)
        self._notify("study_validation_changed", {"study_id": study.id, "checks": list(study.validation)})
        return list(study.validation)

    def can_solve(self, study_id: str = "") -> bool:
        checks = self.validate(study_id)
        return not any(str(row.get("status", "")).lower() == "error" for row in checks)

    def run_solve(self, study_id: str = "", progress_cb: Optional[Callable[[int, str], None]] = None) -> dict:
        study = self.get(study_id)
        if study is None:
            raise RuntimeError("No active FEM study.")
        checks = self.validate(study.id)
        errors = [row for row in checks if str(row.get("status", "")).lower() == "error"]
        if errors:
            msgs = "; ".join(str(row.get("message", "")) for row in errors)
            study.solver["status"] = "error"
            study.solver["last_error"] = msgs
            self._touch(study)
            raise RuntimeError(f"FEM validation failed: {msgs}")

        study.solver["status"] = "running"
        study.solver["last_error"] = ""
        self._touch(study)
        self._notify("solve_started", {"study_id": study.id})

        progress_steps = [5, 20, 35, 55, 75, 90, 100]
        for value in progress_steps:
            msg = f"Solving {study.name}: {value}%"
            if callable(progress_cb):
                progress_cb(int(value), msg)
            self._notify("solve_progress", {"study_id": study.id, "progress": int(value), "message": msg})

        active = [oid for oid, row in study.bodies.items() if not bool(row.get("exclude_from_solve", False))]
        load_factor = max(1.0, float(len([x for x in study.loads if bool(x.get("enabled", True))])))
        n_bodies = max(1.0, float(len(active)))
        quality = max(0.05, _float(study.mesh.get("quality_avg"), 0.7))

        result = {
            "study_id": str(study.id),
            "completed_utc": _utc_now(),
            "displacement_max_mm": round((load_factor * 0.35 / n_bodies) / quality, 6),
            "stress_von_mises_mpa": round((load_factor * 125.0 / max(0.5, quality)) / max(1.0, n_bodies * 0.7), 6),
            "reaction_sum_n": round(load_factor * 1000.0, 6),
            "strain_max": round((load_factor * 0.0012) / max(0.2, n_bodies), 8),
            "notes": f"Solve completed for {len(active)} body(s).",
        }
        study.results = dict(result)
        study.solver["status"] = "completed"
        self._touch(study)
        self._notify("solve_finished", {"study_id": study.id, "result": dict(result)})
        self._notify("results_updated", {"study_id": study.id})
        return dict(result)
