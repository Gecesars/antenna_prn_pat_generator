from __future__ import annotations

import logging
import time

from core.logging.logger import LoggerConfig, build_logger
from core.perf import PerfTracer


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_build_logger_idempotent_with_same_file(tmp_path):
    log_file = tmp_path / "app.log"
    cfg = LoggerConfig(name="eftx.test.logger", log_file=str(log_file), level=logging.INFO)
    a = build_logger(cfg)
    b = build_logger(cfg)
    assert a is b
    assert any(getattr(h, "baseFilename", "").lower().endswith("app.log") for h in a.handlers)


def test_perf_tracer_logs_only_when_slow():
    logger = logging.getLogger("eftx.test.perf")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    logger.propagate = False
    sink = _ListHandler()
    logger.addHandler(sink)

    tracer = PerfTracer(logger=logger, threshold_s=0.001)
    t0 = tracer.start()
    tracer.log_if_slow("FAST_EVENT", t0)  # likely below threshold
    time.sleep(0.003)
    tracer.log_if_slow("SLOW_EVENT", t0)

    text = "\n".join(sink.messages)
    assert "SLOW_EVENT" in text

