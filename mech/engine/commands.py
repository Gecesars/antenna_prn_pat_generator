from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, List


class EngineStateProvider(Protocol):
    def serialize_state(self) -> dict: ...
    def restore_state(self, state: dict, emit: bool = True) -> None: ...


class Command(Protocol):
    label: str
    def do(self, engine: EngineStateProvider) -> None: ...
    def undo(self, engine: EngineStateProvider) -> None: ...


@dataclass
class SnapshotCommand:
    label: str
    action: Callable[[EngineStateProvider], None]
    _before: Optional[dict] = None
    _after: Optional[dict] = None

    def do(self, engine: EngineStateProvider) -> None:
        if self._before is None:
            self._before = engine.serialize_state()
            self.action(engine)
            self._after = engine.serialize_state()
            return
        if self._after is not None:
            engine.restore_state(self._after, emit=True)

    def undo(self, engine: EngineStateProvider) -> None:
        if self._before is not None:
            engine.restore_state(self._before, emit=True)


class CommandStack:
    def __init__(self):
        self._undo: List[Command] = []
        self._redo: List[Command] = []

    def execute(self, cmd: Command, engine: EngineStateProvider) -> None:
        cmd.do(engine)
        self._undo.append(cmd)
        self._redo.clear()

    def undo(self, engine: EngineStateProvider) -> bool:
        if not self._undo:
            return False
        cmd = self._undo.pop()
        cmd.undo(engine)
        self._redo.append(cmd)
        return True

    def redo(self, engine: EngineStateProvider) -> bool:
        if not self._redo:
            return False
        cmd = self._redo.pop()
        cmd.do(engine)
        self._undo.append(cmd)
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @property
    def undo_label(self) -> str:
        if not self._undo:
            return ""
        return str(getattr(self._undo[-1], "label", "") or "")

    @property
    def redo_label(self) -> str:
        if not self._redo:
            return ""
        return str(getattr(self._redo[-1], "label", "") or "")
