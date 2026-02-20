from __future__ import annotations

import threading
import traceback
from typing import Any, Callable, Optional


try:
    from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal  # type: ignore
except Exception:  # pragma: no cover
    QObject = object  # type: ignore
    QRunnable = object  # type: ignore
    QThreadPool = None  # type: ignore
    Signal = None  # type: ignore


if Signal is not None:
    class WorkerSignals(QObject):
        progress = Signal(int, int, str)
        log = Signal(str)
        result = Signal(object)
        error = Signal(str)
        canceled = Signal()
        finished = Signal()

        def __init__(self):
            super().__init__()
else:
    class WorkerSignals:  # pragma: no cover
        def __init__(self):
            self.progress = None
            self.log = None
            self.result = None
            self.error = None
            self.canceled = None
            self.finished = None


class PipelineRunnable(QRunnable):
    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()
        self.cancel_event = threading.Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        try:
            if self.cancel_event.is_set():
                if getattr(self.signals, "canceled", None) is not None:
                    self.signals.canceled.emit()
                return

            def _progress(cur: int, tot: int, label: str):
                if getattr(self.signals, "progress", None) is not None:
                    self.signals.progress.emit(int(cur), int(tot), str(label))

            def _cancelled() -> bool:
                return bool(self.cancel_event.is_set())

            result = self.fn(*self.args, progress_cb=_progress, cancel_check=_cancelled, **self.kwargs)
            if self.cancel_event.is_set():
                if getattr(self.signals, "canceled", None) is not None:
                    self.signals.canceled.emit()
                return
            if getattr(self.signals, "result", None) is not None:
                self.signals.result.emit(result)
        except Exception:
            if getattr(self.signals, "error", None) is not None:
                self.signals.error.emit(traceback.format_exc())
        finally:
            if getattr(self.signals, "finished", None) is not None:
                self.signals.finished.emit()


class QtPipelineExecutor:
    def __init__(self):
        if QThreadPool is None:
            raise RuntimeError("PySide6 is not available.")
        self.pool = QThreadPool.globalInstance()

    def submit(self, runnable: PipelineRunnable):
        self.pool.start(runnable)
