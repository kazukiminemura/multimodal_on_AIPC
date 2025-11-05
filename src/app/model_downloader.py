"""Utilities for downloading model snapshots via Hugging Face."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import Settings

logger = logging.getLogger(__name__)


def _is_populated(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


async def ensure_models(settings: Settings, force: bool = False) -> dict:
    """Ensure required models are available locally. Returns a status dictionary."""

    statuses = {}
    if settings.use_mocks and not force and not settings.auto_download_models:
        logger.info("Skipping model download because mocks are enabled.")
        statuses["skipped"] = True
        return statuses

    deepseek_dir = settings.models_cache_dir / "deepseek"
    diffusion_dir = settings.models_cache_dir / "stable-diffusion"
    deepseek_dir.mkdir(parents=True, exist_ok=True)
    diffusion_dir.mkdir(parents=True, exist_ok=True)

    statuses["deepseek"] = await _ensure_model(
        repo_id=settings.deepseek_repo_id,
        target_dir=deepseek_dir,
        token=settings.huggingface_token,
        force=force,
    )
    statuses["stable_diffusion"] = await _ensure_model(
        repo_id=settings.stable_diffusion_repo_id,
        target_dir=diffusion_dir,
        token=settings.huggingface_token,
        force=force,
    )
    return statuses


async def _ensure_model(repo_id: str, target_dir: Path, token: str | None, force: bool) -> dict:
    """Download a single model snapshot when needed."""

    already_present = _is_populated(target_dir)
    status = {
        "repo_id": repo_id,
        "path": str(target_dir),
        "downloaded": False,
        "skipped": False,
    }

    if already_present and not force:
        status["skipped"] = True
        return status

    logger.info("Downloading %s into %s", repo_id, target_dir)
    await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    status["downloaded"] = True
    return status

