"""FastAPI application factory."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .image_client import StableDiffusionClient
from .llm_client import DeepSeekClient
from .model_downloader import ensure_models
from .orchestrator import ChatOrchestrator
from .routes import router

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Instantiate the FastAPI application."""

    settings = settings or get_settings()
    app = FastAPI(
        title="Multimodal Chatbot",
        version="0.1.0",
        description="Offline-friendly multimodal assistant powered by OpenVINO runtimes.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    project_root = Path(__file__).resolve().parents[2]
    static_dir = project_root / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    else:
        logger.warning("Static directory not found at %s", static_dir)

    llm_client = DeepSeekClient(settings)
    image_client = StableDiffusionClient(settings)
    orchestrator = ChatOrchestrator(settings, llm_client, image_client)

    app.state.settings = settings
    app.state.llm_client = llm_client
    app.state.image_client = image_client
    app.state.orchestrator = orchestrator

    @app.on_event("startup")
    async def startup_event() -> None:
        logger.info("Starting Multimodal Chatbot service")
        if settings.auto_download_models:
            try:
                await ensure_models(settings)
            except Exception:  # noqa: BLE001
                logger.exception("Model download failed during startup")

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        logger.info("Stopping Multimodal Chatbot service")
        await llm_client.aclose()
        await image_client.aclose()

    @app.get("/", response_class=HTMLResponse)
    async def root_index() -> HTMLResponse:
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return HTMLResponse(
            content=(
                "<h1>Multimodal Chatbot</h1>"
                "<p>Static UI not found. Visit /docs for the API explorer.</p>"
            ),
            status_code=200,
        )

    app.include_router(router)
    return app

