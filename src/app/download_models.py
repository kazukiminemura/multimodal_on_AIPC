"""Async entrypoint used by the download_models CLI wrapper."""

from __future__ import annotations

import asyncio
import logging

from .config import get_settings
from .model_downloader import ensure_models

logger = logging.getLogger(__name__)


async def download_all(force: bool = False) -> dict:
    """Download all required models."""

    settings = get_settings()
    statuses = await ensure_models(settings, force=force)
    return statuses


def run(force: bool = False) -> dict:
    """Synchronous helper to invoke downloads from a CLI context."""

    logger.info("Starting model download (force=%s)", force)
    return asyncio.run(download_all(force=force))

