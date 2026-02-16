from __future__ import annotations

import logging
import time
from typing import Optional


class PerfTracer:
    """Lightweight latency tracer for hot paths.

    Logs only slow events (dt > threshold_s) to avoid flooding.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, threshold_s: float = 0.010):
        self.logger = logger or logging.getLogger("eftx.perf")
        self.threshold_s = float(threshold_s)

    def start(self) -> float:
        return time.perf_counter()

    def log_if_slow(self, tag: str, t0: float, extra: str = "", threshold_s: Optional[float] = None) -> float:
        dt = time.perf_counter() - float(t0)
        thr = self.threshold_s if threshold_s is None else float(threshold_s)
        if dt > thr:
            msg = f"{tag} dt={dt*1000.0:.2f}ms"
            if extra:
                msg += f" | {extra}"
            try:
                self.logger.info(msg)
            except Exception:
                pass
        return dt


DEFAULT_TRACER = PerfTracer()
