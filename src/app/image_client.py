"""Stable Diffusion client abstraction."""

from __future__ import annotations

import logging
from typing import Any, Dict

import httpx

from .config import Settings
from .schemas import ImagePrompt, ImageResult

logger = logging.getLogger(__name__)


class StableDiffusionClient:
    """Async client for interacting with a Stable Diffusion inference server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.uses_mocks = settings.use_mocks or not settings.enable_image_generation
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(self._settings.request_timeout_seconds)
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def generate(self, prompt: ImagePrompt) -> ImageResult:
        """Trigger image generation for the supplied prompt."""

        if self.uses_mocks:
            return self._mock_result(prompt)

        client = await self._get_client()
        payload: Dict[str, Any] = {
            "prompt": prompt.prompt,
            "negative_prompt": prompt.negative_prompt,
            "num_inference_steps": prompt.num_inference_steps,
            "guidance_scale": prompt.guidance_scale,
            "seed": prompt.seed,
        }
        if prompt.width:
            payload["width"] = prompt.width
        if prompt.height:
            payload["height"] = prompt.height

        response = await client.post(str(self._settings.stable_diffusion_endpoint), json=payload)
        response.raise_for_status()
        data = response.json()

        job_id = data.get("id") or data.get("job_id") or "sd-job"
        urls = data.get("data") or data.get("image_urls") or []
        if isinstance(urls, dict):
            urls = urls.get("urls", [])

        urls = [str(url) for url in urls]
        if not urls:
            logger.warning("Stable Diffusion response missing URLs; returning placeholder")
            urls = ["/static/mock-image.svg"]

        return ImageResult(job_id=str(job_id), urls=urls, provider="stable-diffusion")

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _mock_result(self, prompt: ImagePrompt) -> ImageResult:
        """Return a canned Stable Diffusion result."""

        del prompt
        return ImageResult(
            job_id="mock-job-1234",
            urls=["/static/mock-image.svg"],
            provider="mock-stable-diffusion",
        )


async def shutdown_client(client: StableDiffusionClient) -> None:
    """Helper for FastAPI shutdown events."""

    await client.aclose()

