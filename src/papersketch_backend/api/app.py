from __future__ import annotations

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from papersketch_backend.api.routes.papersketch import router as papersketch_router
from papersketch_backend.api.routes.compare import router as compare_router
from papersketch_backend.api.routes.structure import router as structure_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="PaperSketch Backend API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS: useful during local dev (Flask UI, connector testing, etc.)
    allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in allowed_origins if o.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Serve local dev images from /static
    # write figure images into ./static/<paper_hash>/page_3.png
    static_dir = os.getenv("STATIC_DIR", "static")
    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    # API routes
    app.include_router(papersketch_router, prefix="/api/v1", tags=["papersketch"])
    app.include_router(compare_router, prefix="/api/v1", tags=["compare"])
    app.include_router(structure_router, prefix="/api/v1", tags=["structure"])

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    return app


app = create_app()
