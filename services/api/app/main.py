"""KORTEX AI — FastAPI application entry point."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import audit, auth, checkout, demo, discrepancies, products, sessions, system, vision
from app.core.config import get_settings
from app.core.errors import DomainError
from app.db import session as db_session
from app.db.init_db import init_db
from app.db.seed import seed_core
from app.services import pipeline
from app.websocket import routes as ws_routes
from app.websocket.hub import hub

log = logging.getLogger("kortex")


async def _sweeper(interval: float) -> None:
    """Finalise pickups whose scan grace window has expired."""

    def tick() -> int:
        with db_session.SessionLocal() as db:
            n = pipeline.finalize_pending(db)
            db.commit()
            return n

    while True:
        try:
            n = await asyncio.to_thread(tick)
            if n:
                log.info("sweeper finalised %d observation(s)", n)
        except Exception:  # pragma: no cover - keep the loop alive
            log.exception("sweeper tick failed")
        await asyncio.sleep(interval)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db()
    with db_session.SessionLocal() as db:
        seed_core(db)
    hub.bind_loop(asyncio.get_running_loop())
    task = None
    if settings.sweep_interval_seconds > 0:
        task = asyncio.create_task(_sweeper(settings.sweep_interval_seconds))
    if settings.uses_insecure_defaults:
        log.warning("KORTEX is running with a development secret key. Set KORTEX_SECRET_KEY in .env.")
    if settings.vision_simulator:
        from vision_simulator import start as start_vision_sim
        start_vision_sim(db_session.SessionLocal)
        log.info("Vision simulator enabled (KORTEX_VISION_SIMULATOR=true)")
    yield
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    if settings.vision_simulator:
        from vision_simulator import stop as stop_vision_sim
        stop_vision_sim()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="KORTEX AI API",
        version="1.0.0",
        description=(
            "Scan It. Pick It. Verify It. — Uncertainty-aware scan-and-go reconciliation API "
            "(hackathon prototype; not production-certified)."
        ),
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_origin_regex=settings.cors_origin_regex or None,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Session-Token"],
    )

    @app.exception_handler(DomainError)
    async def domain_error(_req: Request, exc: DomainError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_req: Request, exc: RequestValidationError):
        fields = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]} for e in exc.errors()
        ]
        message = fields[0]["message"] if fields else "Invalid request"
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_failed", "message": message, "details": {"fields": fields}}},
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_req: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": f"http_{exc.status_code}", "message": str(exc.detail), "details": {}}},
        )

    @app.get("/", include_in_schema=False)
    async def root():
        return HTMLResponse(
            """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>KORTEX AI — Backend API</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #090d16; color: #f1f5f9; display: flex; align-items: center; justify-content: center; min-height: 100vh; padding: 1.5rem; }
    .card { background: #131b2e; border: 1px solid #1e293b; border-radius: 16px; padding: 2.5rem; max-width: 520px; width: 100%; text-align: center; box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); }
    .badge { display: inline-flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); padding: 6px 14px; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; margin-bottom: 1.25rem; }
    .dot { width: 8px; height: 8px; background: #10b981; border-radius: 50%; box-shadow: 0 0 8px #10b981; }
    h1 { font-size: 1.85rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.5rem; letter-spacing: -0.02em; }
    p { color: #94a3b8; font-size: 0.95rem; line-height: 1.6; margin-bottom: 1.75rem; }
    .actions { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
    .btn { display: inline-flex; align-items: center; gap: 6px; padding: 10px 18px; border-radius: 8px; font-size: 0.9rem; font-weight: 600; text-decoration: none; transition: all 0.15s ease; }
    .btn-primary { background: #2563eb; color: #ffffff; }
    .btn-primary:hover { background: #1d4ed8; }
    .btn-secondary { background: #1e293b; color: #cbd5e1; border: 1px solid #334155; }
    .btn-secondary:hover { background: #334155; color: #ffffff; }
  </style>
</head>
<body>
  <div class="card">
    <div class="badge"><span class="dot"></span> API Service Active</div>
    <h1>KORTEX AI API</h1>
    <p>FastAPI backend service is live and running. Use the links below to test the API endpoints or view system health.</p>
    <div class="actions">
      <a href="/docs" class="btn btn-primary">Swagger Docs (/docs)</a>
      <a href="/api/health" class="btn btn-secondary">Health Check</a>
    </div>
  </div>
</body>
</html>"""
        )

    for r in (auth, products, sessions, checkout, discrepancies, vision, demo, audit, system):
        app.include_router(r.router)
    app.include_router(ws_routes.router)
    return app


app = create_app()
