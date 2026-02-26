from __future__ import annotations

from typing import List

import numpy as np
import pytest

from eftx_aedt_live.farfield import CutRequest, FarFieldExtractor, GridRequest
from eftx_aedt_live.session import AedtConnectionConfig, AedtHfssSession


pytestmark = pytest.mark.external


def _setup_name_from_sweep(value: str) -> str:
    txt = str(value or "").strip()
    if ":" in txt:
        return txt.split(":", 1)[0].strip()
    return txt


def _unique(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in values:
        v = str(raw or "").strip()
        if not v:
            continue
        k = v.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
    return out


def _resolved_setup(hfss_obj, requested: str) -> str:
    requested = str(requested or "").strip()
    solved: List[str] = []
    try:
        solved = [str(x).strip() for x in list(getattr(hfss_obj, "existing_analysis_sweeps", []) or []) if str(x).strip()]
    except Exception:
        solved = []
    if requested:
        req_setup = _setup_name_from_sweep(requested).lower()
        for sw in solved:
            if _setup_name_from_sweep(sw).lower() == req_setup:
                return sw
        return requested
    if solved:
        return solved[0]

    defined: List[str] = []
    try:
        for s in list(getattr(hfss_obj, "setups", []) or []):
            name = str(getattr(s, "name", "") or "").strip()
            if name:
                defined.append(name)
    except Exception:
        defined = []
    defined = _unique(defined)
    if defined:
        return defined[0]
    pytest.skip("No setup found in selected design.")


def _resolved_sphere(hfss_obj, requested: str) -> str:
    req = str(requested or "").strip()
    spheres: List[str] = []
    fs = getattr(hfss_obj, "field_setups", None)
    if isinstance(fs, dict):
        spheres.extend([str(k).strip() for k in fs.keys()])
    elif isinstance(fs, (list, tuple, set)):
        for item in fs:
            name = str(getattr(item, "name", "") or "").strip()
            if name:
                spheres.append(name)
    spheres = _unique([s for s in spheres if s])

    if req:
        if not spheres:
            pytest.skip(f"No field setup found in design. Requested sphere: {req}")
        for s in spheres:
            if s.lower() == req.lower():
                return s
        pytest.skip(f"Requested sphere '{req}' not found. Available: {spheres}")

    if spheres:
        return spheres[0]
    pytest.skip("No infinite sphere (field setup) found in design.")


@pytest.fixture(scope="session")
def connected_session(external_aedt_env):
    cfg = AedtConnectionConfig(
        version=external_aedt_env.version,
        non_graphical=bool(external_aedt_env.non_graphical),
        new_desktop=bool(external_aedt_env.connect_mode.startswith("new")),
        close_on_exit=False,
        remove_lock=bool(external_aedt_env.remove_lock),
    )
    session = AedtHfssSession(cfg)
    session.connect(
        project=external_aedt_env.project or None,
        design=external_aedt_env.design or None,
        setup=None,
        force=True,
    )
    try:
        yield session
    finally:
        session.disconnect()


def test_external_connect_context(connected_session, external_aedt_env):
    session = connected_session
    assert session.is_connected
    hfss = session.hfss
    assert hfss is not None

    if external_aedt_env.project:
        current = str(getattr(hfss, "project_file", "") or getattr(hfss, "project_name", "") or "").strip().lower()
        assert current
    if external_aedt_env.design:
        current_design = str(getattr(hfss, "design_name", "") or "").strip()
        assert current_design


def test_external_extract_hrp_vrp_cuts(connected_session, external_aedt_env):
    session = connected_session
    hfss = session.hfss
    setup = _resolved_setup(hfss, external_aedt_env.setup_sweep)
    sphere = _resolved_sphere(hfss, external_aedt_env.sphere)

    freq_fixed = {"Freq": external_aedt_env.freq} if external_aedt_env.freq else {}
    extractor = FarFieldExtractor(session)

    req_vrp = CutRequest(
        setup_sweep=setup,
        sphere_name=sphere,
        expression=external_aedt_env.expression,
        primary_sweep="Theta",
        fixed={"Phi": "0deg", **freq_fixed},
        convert_theta_to_elevation=False,
    )
    vrp = extractor.extract_cut(req_vrp)
    a_v = np.asarray(vrp.angles_deg, dtype=float)
    v_v = np.asarray(vrp.values, dtype=float)
    assert a_v.size > 32
    assert v_v.size == a_v.size
    assert np.all(np.isfinite(a_v))
    assert np.all(np.isfinite(v_v))
    assert np.all(np.diff(a_v) >= 0.0)
    assert float(np.ptp(a_v)) > 90.0
    assert float(np.std(v_v)) > 1e-10

    req_hrp = CutRequest(
        setup_sweep=setup,
        sphere_name=sphere,
        expression=external_aedt_env.expression,
        primary_sweep="Theta",
        fixed={"Phi": "90deg", **freq_fixed},
        convert_theta_to_elevation=False,
    )
    hrp = extractor.extract_cut(req_hrp)
    a_h = np.asarray(hrp.angles_deg, dtype=float)
    v_h = np.asarray(hrp.values, dtype=float)
    assert a_h.size > 32
    assert v_h.size == a_h.size
    assert np.all(np.isfinite(a_h))
    assert np.all(np.isfinite(v_h))
    assert np.all(np.diff(a_h) >= 0.0)
    assert float(np.ptp(a_h)) > 90.0
    assert float(np.std(v_h)) > 1e-10


def test_external_extract_3d_grid(connected_session, external_aedt_env):
    session = connected_session
    hfss = session.hfss
    setup = _resolved_setup(hfss, external_aedt_env.setup_sweep)
    sphere = _resolved_sphere(hfss, external_aedt_env.sphere)

    extractor = FarFieldExtractor(session)
    req = GridRequest(
        setup_sweep=setup,
        sphere_name=sphere,
        expression=external_aedt_env.expression,
        theta_points=41,
        phi_points=73,
        freq=external_aedt_env.freq or None,
        convert_theta_to_elevation=False,
    )
    grid = extractor.extract_grid(req)

    th = np.asarray(grid.theta_deg, dtype=float)
    ph = np.asarray(grid.phi_deg, dtype=float)
    vv = np.asarray(grid.values, dtype=float)
    assert th.size >= 16
    # In this app, a compliant infinite sphere may intentionally use Phi={0,90}.
    # Accept both full sweeps and the constrained profile.
    if ph.size < 16:
        assert ph.size >= 2
        ph_sorted = np.sort(np.round(ph.astype(float), 6))
        assert np.all(np.isfinite(ph_sorted))
        # Typical constrained profile: [0, 90]
        assert float(ph_sorted[0]) >= -1e-6
        assert float(ph_sorted[-1]) <= 90.0 + 1e-6
    else:
        assert ph.size >= 16
    assert vv.ndim == 2
    assert vv.shape[0] == th.size
    assert vv.shape[1] == ph.size
    assert np.all(np.isfinite(vv))
    assert float(np.std(vv)) > 1e-10
