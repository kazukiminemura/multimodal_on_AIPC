"""Prompt construction utilities for the chatbot."""

from __future__ import annotations

from typing import Iterable, List, Mapping


class PromptManager:
    """Assembles system/developer prompts with conversation history."""

    def __init__(self) -> None:
        self._system_prompt = (
            "You are an offline-friendly multimodal assistant running on user-controlled hardware. "
            "Provide helpful, concise answers. When appropriate, propose an `image_prompt` JSON "
            "object describing imagery that would complement your response."
        )
        self._developer_prompt = (
            "If you decide an illustration would help, respond with valid JSON under the key "
            "`image_prompt` alongside your natural language reply. Only request imagery that aligns "
            "with the user's instructions and avoid disallowed or unsafe content."
        )

    def build_messages(
        self,
        history: Iterable[Mapping[str, str]],
        user_message: str,
    ) -> List[dict]:
        """Return an ordered list of messages suitable for the LLM client."""

        messages: List[dict] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "system", "name": "developer", "content": self._developer_prompt},
        ]

        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

