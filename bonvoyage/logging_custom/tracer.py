import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path


_loggers: dict[str, logging.Logger] = {}


def _get_logger(session_id: str) -> logging.Logger:
    if session_id in _loggers:
        return _loggers[session_id]

    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(f"bonvoyage.{session_id}")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    handler = logging.FileHandler(logs_dir / f"session_{session_id}.jsonl")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    _loggers[session_id] = logger
    return logger


def log_event(
    session_id: str,
    step: int,
    phase: str,
    event: str,
    **kwargs,
) -> None:
    record = {
        "session_id": session_id,
        "step": step,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "event": event,
        **kwargs,
    }
    logger = _get_logger(session_id)
    logger.info(json.dumps(record, default=str))
