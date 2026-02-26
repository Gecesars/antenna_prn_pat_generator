from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


def audit_enabled() -> bool:
    raw = str(os.environ.get("DEBUG_AUDIT", "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _jsonable(value):
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def emit_audit(event: str, *, logger: Optional[logging.Logger] = None, **fields) -> None:
    """Emit audit log entry only when DEBUG_AUDIT=1.

    In normal execution this function is intentionally no-op.
    """
    if not audit_enabled():
        return
    log = logger or logging.getLogger("eftx.audit")
    payload: Dict[str, object] = {
        "event": str(event or "unknown"),
        "ts": time.time(),
    }
    for k, v in (fields or {}).items():
        payload[str(k)] = _jsonable(v)
    try:
        log.info("AUDIT %s", json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except Exception:
        pass


@contextmanager
def audit_span(
    event: str,
    *,
    logger: Optional[logging.Logger] = None,
    threshold_ms: float = 0.0,
    **fields,
) -> Iterator[None]:
    """Timed audit span with automatic status and elapsed_ms."""
    if not audit_enabled():
        yield
        return

    t0 = time.perf_counter()
    emit_audit(f"{event}:start", logger=logger, **fields)
    status = "ok"
    error_text = ""
    try:
        yield
    except Exception as exc:
        status = "error"
        error_text = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if float(elapsed_ms) >= float(threshold_ms):
            emit_audit(
                f"{event}:end",
                logger=logger,
                status=status,
                elapsed_ms=round(float(elapsed_ms), 3),
                error=error_text,
                **fields,
            )

