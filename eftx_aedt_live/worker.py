from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class WorkerResult:
    ok: bool
    value: Any = None
    error: Optional[str] = None


class TkWorker:
    """Run blocking tasks in a background thread and marshal results to Tk via `after()`.

    This utility avoids freezing the UI while AEDT is connecting or retrieving data.

    Usage:
      worker = TkWorker(tk_root_or_widget)
      worker.run(task_fn, on_done)
    """

    def __init__(self, tk_widget):
        self.tk = tk_widget
        self._busy = False
        self._lock = threading.Lock()

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return bool(self._busy)

    def run(self, fn: Callable[[], Any], on_done: Callable[[WorkerResult], None]):
        with self._lock:
            if self._busy:
                self.tk.after(0, lambda: on_done(WorkerResult(ok=False, error="Operation already in progress.")))
                return
            self._busy = True

        def _thread():
            try:
                v = fn()
                res = WorkerResult(ok=True, value=v)
            except Exception as e:
                res = WorkerResult(ok=False, error=str(e))
            def _finish():
                with self._lock:
                    self._busy = False
                on_done(res)
            self.tk.after(0, _finish)
        t = threading.Thread(target=_thread, daemon=True)
        t.start()
