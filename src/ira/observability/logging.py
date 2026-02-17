from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional


def setup_json_logger(service_name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(service_name)
    logger.setLevel(level.upper())
    logger.handlers.clear()
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level.upper())

    class JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            payload: dict[str, Any] = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
            }
            # Attach common structured fields if present
            for key in ("request_id", "event", "path", "method"):
                if hasattr(record, key):
                    payload[key] = getattr(record, key)
            if record.exc_info:
                payload["exc_info"] = self.formatException(record.exc_info)
            return json.dumps(payload, ensure_ascii=False)

    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    request_id: Optional[str] = None,
    **fields: Any,
) -> None:
    extra = {"event": event}
    if request_id:
        extra["request_id"] = request_id
    extra.update(fields)
    logger.info(event, extra=extra)
