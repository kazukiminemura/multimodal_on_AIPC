"""Client for interacting with the DeepSeek OpenVINO endpoint (or mocks)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

from .config import Settings
from .orchestrator import parse_llm_string

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Thin async wrapper around an OpenAI-compatible chat completion endpoint."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.uses_mocks = settings.use_mocks
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            timeout = httpx.Timeout(self._settings.request_timeout_seconds)
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def generate(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a chat completion payload."""

        if self.uses_mocks:
            return self._mock_response(messages)

        client = await self._get_client()
        payload = {
            "model": self._settings.deepseek_repo_id,
            "messages": messages,
            "stream": False,
        }

        try:
            response = await client.post(str(self._settings.deepseek_endpoint), json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            logger.error("DeepSeek request failed; falling back to catbot", exc_info=True)
            if self._settings.enable_catbot_fallback:
                return self._catbot_response(messages)
            raise RuntimeError("DeepSeek endpoint request failed") from exc

        try:
            message = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            logger.error("Unexpected DeepSeek payload", extra={"payload": data})
            raise RuntimeError("Invalid DeepSeek response structure") from exc

        return parse_llm_string(message)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _mock_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Return a deterministic mock payload for offline development."""

        user_message = messages[-1]["content"]
        suffix = "..." if len(user_message) > 200 else ""
        assistant_text = (
            "This is a mock response. You said: "
            f"{user_message[:200]}{suffix}"
        )

        payload: Dict[str, Any] = {"assistant_response": assistant_text}

        if any(keyword in user_message.lower() for keyword in ("image", "illustration", "picture")):
            payload["image_prompt"] = {
                "prompt": "Dreamy watercolor illustration of a friendly robot and a curious child.",
                "num_inference_steps": 24,
                "guidance_scale": 7.5,
            }

        return payload

    def _catbot_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Light-hearted fallback when the real endpoint fails."""

        user_message = messages[-1]["content"]
        assistant_text = (
            "Meow! The main engine is snoozing, but CatBot is here. "
            "Here's what I heard: "
            f"{user_message}"
        )
        return {"assistant_response": assistant_text}


async def shutdown_client(client: DeepSeekClient) -> None:
    """Helper used in FastAPI shutdown hooks."""

    await client.aclose()

