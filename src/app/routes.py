"""API routes for the multimodal chatbot."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from .config import Settings, get_settings
from .model_downloader import _is_populated
from .orchestrator import ChatOrchestrator
from .prompt_manager import PromptManager
from .schemas import ChatRequest, ChatResponse, HealthStatus

logger = logging.getLogger(__name__)

router = APIRouter()


def get_orchestrator(request: Request) -> ChatOrchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise RuntimeError("Orchestrator not configured")
    return orchestrator


def get_app_settings(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        settings = get_settings()
    return settings


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    chat_request: ChatRequest,
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
) -> ChatResponse:
    return await orchestrator.handle_chat(chat_request)


@router.get("/health", response_model=HealthStatus)
async def health_endpoint(
    request: Request,
    settings: Settings = Depends(get_app_settings),
) -> HealthStatus:
    models_root = settings.models_cache_dir
    deepseek_cached = _is_populated(models_root / "deepseek")
    diffusion_cached = _is_populated(models_root / "stable-diffusion")
    cached_models = deepseek_cached and (diffusion_cached or not settings.enable_image_generation)

    details = {
        "deepseek_cached": deepseek_cached,
        "stable_diffusion_cached": diffusion_cached,
        "models_cache_dir": str(models_root),
    }

    return HealthStatus(
        status="ok",
        use_mocks=settings.use_mocks,
        auto_download_models=settings.auto_download_models,
        enable_image_generation=settings.enable_image_generation,
        cached_models=cached_models,
        details=details,
    )


@router.get("/debug/llm")
async def debug_llm_endpoint(
    request: Request,
    settings: Settings = Depends(get_app_settings),
):
    llm_client = getattr(request.app.state, "llm_client", None)
    if llm_client is None:
        raise HTTPException(status_code=503, detail="LLM client unavailable")

    prompt_manager = PromptManager()
    messages = prompt_manager.build_messages([], "Diagnostic ping from /debug/llm.")

    try:
        payload = await llm_client.generate(messages)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM debug call failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {"messages": messages, "response": payload, "used_mocks": llm_client.uses_mocks}

