from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Iterable, List


def _parse_local_address(addr: str) -> tuple[str, int]:
    text = str(addr or "").strip()
    if not text:
        return "", 0
    text = text.strip("[]")
    if ":" not in text:
        return text, 0
    host, port_txt = text.rsplit(":", 1)
    try:
        port = int(port_txt)
    except Exception:
        port = 0
    return host.strip("[]"), port


def _normalize_status(value: str) -> str:
    low = str(value or "").strip().upper()
    if low == "LISTENING":
        return "LISTEN"
    return low


def _list_task_processes() -> list[dict]:
    # Pure shim mode: do not spawn OS commands from GUI runtime.
    return []


def _pid_cmdline(pid: int) -> list[str]:
    # Pure shim mode: no external process probing.
    _ = pid
    return []


def _iter_net_connections_tcp() -> Iterable[SimpleNamespace]:
    # Pure shim mode: no external command execution.
    return []


def create_psutil_shim_module() -> ModuleType:
    mod = ModuleType("psutil")
    mod.__dict__["__file__"] = "<eftx_psutil_shim>"
    mod.__dict__["__package__"] = "psutil"
    mod.__dict__["__version__"] = "0.0-shim"

    class Error(Exception):
        pass

    class NoSuchProcess(Error):
        pass

    class AccessDenied(Error):
        pass

    class ZombieProcess(Error):
        pass

    mod.Error = Error
    mod.NoSuchProcess = NoSuchProcess
    mod.AccessDenied = AccessDenied
    mod.ZombieProcess = ZombieProcess

    mod.STATUS_RUNNING = "running"
    mod.STATUS_IDLE = "idle"
    mod.STATUS_SLEEPING = "sleeping"
    mod.STATUS_DISK_SLEEP = "disk-sleep"
    mod.STATUS_DEAD = "dead"
    mod.STATUS_PARKED = "parked"

    class Process:
        def __init__(self, pid: int):
            try:
                self.pid = int(pid)
            except Exception:
                raise NoSuchProcess(pid)

        def name(self) -> str:
            for row in _list_task_processes():
                if row.get("pid") == self.pid:
                    return str(row.get("name") or "")
            return ""

        def status(self) -> str:
            return mod.STATUS_RUNNING

        def cmdline(self) -> list[str]:
            return _pid_cmdline(self.pid)

        def net_connections(self, kind: str | None = None):
            return [c for c in _iter_net_connections_tcp() if int(getattr(c, "pid", -1)) == self.pid]

        def kill(self):
            # No-op in shim mode.
            return None

        terminate = kill

        def is_running(self) -> bool:
            for row in _list_task_processes():
                if row.get("pid") == self.pid:
                    return True
            return False

    def process_iter(attrs=None):
        procs = []
        for row in _list_task_processes():
            try:
                procs.append(Process(int(row.get("pid"))))
            except Exception:
                continue
        return procs

    def net_connections(kind: str = "inet"):
        if str(kind or "").lower() not in {"inet", "tcp"}:
            return []
        return list(_iter_net_connections_tcp())

    mod.Process = Process
    mod.process_iter = process_iter
    mod.net_connections = net_connections
    return mod
