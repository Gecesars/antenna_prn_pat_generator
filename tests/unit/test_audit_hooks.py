from __future__ import annotations

import logging

from core.audit import audit_span, emit_audit


class _ListHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _logger(name: str):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers = []
    log.propagate = False
    h = _ListHandler()
    log.addHandler(h)
    return log, h


def test_emit_audit_is_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("DEBUG_AUDIT", raising=False)
    log, h = _logger("eftx.test.audit.disabled")
    emit_audit("UNIT_EVENT_DISABLED", logger=log, value=1)
    assert h.messages == []


def test_emit_audit_emits_when_enabled(monkeypatch):
    monkeypatch.setenv("DEBUG_AUDIT", "1")
    log, h = _logger("eftx.test.audit.enabled")
    emit_audit("UNIT_EVENT_ENABLED", logger=log, value=7)
    assert h.messages
    assert any("UNIT_EVENT_ENABLED" in m for m in h.messages)


def test_audit_span_emits_start_end(monkeypatch):
    monkeypatch.setenv("DEBUG_AUDIT", "1")
    log, h = _logger("eftx.test.audit.span")
    with audit_span("SPAN_CASE", logger=log):
        pass
    text = "\n".join(h.messages)
    assert "SPAN_CASE:start" in text
    assert "SPAN_CASE:end" in text

