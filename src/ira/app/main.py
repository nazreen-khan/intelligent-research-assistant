from __future__ import annotations

from fastapi import FastAPI, Request
from ira.settings import settings
from ira.observability.logging import setup_json_logger, log_event
from ira.observability.middleware import RequestIdMiddleware

logger = setup_json_logger(service_name=settings.service_name, level=settings.log_level)

app = FastAPI(title="Intelligent Research Assistant", version="0.1.0")
app.add_middleware(RequestIdMiddleware)


@app.get("/health")
def health(request: Request):
    request_id = getattr(request.state, "request_id", "unknown")
    log_event(logger, event="health_check", request_id=request_id, path="/health", method="GET")
    return {
        "status": "ok",
        "service": settings.service_name,
        "env": settings.app_env,
        "version": "0.1.0",
        "request_id": request_id,
    }
