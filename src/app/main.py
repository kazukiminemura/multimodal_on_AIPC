from __future__ import annotations

import logging

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .models import HealthStatus, ImageGenerationRequest, ImageGenerationResponse
from .services import StableDiffusionService

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = Settings()
    settings.ensure_directories()

    service = StableDiffusionService(settings=settings)
    if settings.auto_download_models:
        try:
            service.ensure_model_snapshot()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Model download failed: %s", exc)

    app = FastAPI(
        title="Stable Diffusion Service",
        description="Generate images from text prompts via Stable Diffusion.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    router = APIRouter()

    @router.post("/image", response_model=ImageGenerationResponse)
    async def generate_image(payload: ImageGenerationRequest) -> ImageGenerationResponse:
        try:
            return await service.generate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to generate image")
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @router.get("/health", response_model=HealthStatus)
    async def health() -> HealthStatus:
        status = "ok" if service.models_cached() or settings.use_mocks else "degraded"
        return HealthStatus(
            status=status,  # type: ignore[arg-type]
            use_mocks=settings.use_mocks,
            models_cached=service.models_cached(),
            model_repo=settings.stable_diffusion_repo_id,
            generated_images_dir=str(settings.generated_images_dir),
            models_cache_dir=str(settings.models_cache_dir / "stable-diffusion"),
        )

    app.include_router(router)

    if settings.static_dir.exists():
        app.mount(
            "/static",
            StaticFiles(directory=str(settings.static_dir)),
            name="static",
        )

        index_path = settings.static_dir / "index.html"

        @app.get("/", include_in_schema=False)
        async def root() -> FileResponse:  # pragma: no cover
            return FileResponse(index_path)

    app.mount(
        "/generated",
        StaticFiles(directory=str(settings.generated_images_dir), check_dir=False),
        name="generated",
    )

    return app


app = create_app()
