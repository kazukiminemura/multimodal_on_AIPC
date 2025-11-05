"""Conversation orchestration across LLM and diffusion clients."""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from typing import Deque, Dict, List

from .config import Settings
from .prompt_manager import PromptManager
from .schemas import ChatRequest, ChatResponse, ImagePrompt

logger = logging.getLogger(__name__)


class ConversationStore:
    """In-memory bounded history keyed by user identifier."""

    def __init__(self, limit: int) -> None:
        self._store: Dict[str, Deque[dict]] = defaultdict(lambda: deque(maxlen=limit * 2))

    def get_history(self, user_id: str) -> List[dict]:
        return list(self._store[user_id])

    def append(self, user_id: str, role: str, content: str) -> None:
        self._store[user_id].append({"role": role, "content": content})

    def reset(self, user_id: str) -> None:
        if user_id in self._store:
            del self._store[user_id]


class ChatOrchestrator:
    """Coordinates prompt assembly, LLM calls, and optional image generation."""

    def __init__(
        self,
        settings: Settings,
        llm_client,
        image_client=None,
    ) -> None:
        self._settings = settings
        self._prompt_manager = PromptManager()
        self._llm_client = llm_client
        self._image_client = image_client
        self._store = ConversationStore(settings.conversation_history_limit)

    async def handle_chat(self, request: ChatRequest) -> ChatResponse:
        history = self._store.get_history(request.user_id)
        messages = self._prompt_manager.build_messages(history, request.message)

        logger.info("Dispatching LLM request", extra={"user_id": request.user_id})
        try:
            llm_payload = await self._llm_client.generate(messages)
        except Exception:  # noqa: BLE001
            logger.exception("LLM request failed")
            return ChatResponse(
                assistant_response=(
                    "The DeepSeek inference endpoint is unavailable. "
                    "Please ensure the local DeepSeek server is running and try again."
                ),
                used_mocks=True,
            )

        response = self._parse_llm_payload(llm_payload)

        assistant_text = response.assistant_response
        self._store.append(request.user_id, "user", request.message)
        self._store.append(request.user_id, "assistant", assistant_text)

        if (
            response.image_prompt
            and self._settings.enable_image_generation
            and self._image_client is not None
        ):
            image_result = await self._image_client.generate(response.image_prompt)
            response.image_job_id = image_result.job_id
            response.image_urls = image_result.urls
        else:
            response.image_prompt = None

        response.used_mocks = bool(
            getattr(self._llm_client, "uses_mocks", False)
            or getattr(self._image_client, "uses_mocks", False)
        )
        return response

    def _parse_llm_payload(self, payload) -> ChatResponse:
        """Parse the LLM payload into a ChatResponse."""

        if isinstance(payload, str):
            payload = parse_llm_string(payload)

        if "assistant_response" in payload:
            assistant_response = payload["assistant_response"]
            image_prompt_data = payload.get("image_prompt")
        else:
            assistant_response = payload.get("content") or payload.get("message") or ""
            image_prompt_data = payload.get("image_prompt")

        if not assistant_response and "choices" in payload:
            assistant_response = payload["choices"][0]["message"]["content"]

        if not assistant_response:
            logger.warning("LLM payload missing assistant response; defaulting to empty string.")

        response = ChatResponse(assistant_response=assistant_response)

        if image_prompt_data:
            try:
                response.image_prompt = ImagePrompt.model_validate(image_prompt_data)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to parse image prompt; ignoring", exc_info=True)

        return response


def parse_llm_string(content: str) -> dict:
    """Attempt to parse a structured JSON blob from the LLM response string."""

    stripped = content.strip()
    if not stripped:
        return {"assistant_response": ""}

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    return {"assistant_response": stripped}

