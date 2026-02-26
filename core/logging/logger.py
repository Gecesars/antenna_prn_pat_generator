from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LoggerConfig:
    name: str = "eftx"
    level: int = logging.INFO
    max_bytes: int = 2_000_000
    backup_count: int = 5
    encoding: str = "utf-8"
    console: bool = False
    log_file: Optional[str] = None


def _default_log_file(name: str) -> str:
    app_dir = Path(os.path.expanduser("~")) / ".eftx_converter" / "logs"
    app_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{name.replace('.', '_')}.log"
    return str(app_dir / fname)


def _same_file(handler: logging.Handler, path: str) -> bool:
    base = getattr(handler, "baseFilename", None)
    if not base:
        return False
    try:
        return os.path.normcase(os.path.abspath(base)) == os.path.normcase(os.path.abspath(path))
    except Exception:
        return False


def build_logger(config: LoggerConfig) -> logging.Logger:
    """Build or reuse a rotating logger with optional console output.

    The function is idempotent for the same logger name and file path.
    """
    logger = logging.getLogger(config.name)
    logger.setLevel(int(config.level))

    log_file = config.log_file or _default_log_file(config.name)
    Path(os.path.dirname(os.path.abspath(log_file))).mkdir(parents=True, exist_ok=True)

    has_file = any(_same_file(h, log_file) for h in logger.handlers)
    if not has_file:
        handler = RotatingFileHandler(
            log_file,
            maxBytes=int(config.max_bytes),
            backupCount=int(config.backup_count),
            encoding=config.encoding,
        )
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)

    wants_console = bool(config.console or str(os.environ.get("EFTX_LOG_STDOUT", "")).strip() == "1")
    has_console = any(isinstance(h, logging.StreamHandler) and not isinstance(h, RotatingFileHandler) for h in logger.handlers)
    if wants_console and (not has_console):
        stream = logging.StreamHandler(stream=sys.stdout)
        stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(stream)

    logger.propagate = False
    return logger

