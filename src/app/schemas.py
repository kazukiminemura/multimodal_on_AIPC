"""Shared Pydantic models for request/response payloads."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat message from a client."""

    user_id: str = Field(..., min_length=1, max_length=256)
    message: str = Field(..., min_length=1)


class ImagePrompt(BaseModel):
    """Prompt payload returned by the LLM when image generation is requested."""

    prompt: str
    negative_prompt: Optional[str] = None
    num_inference_steps: int = Field(default=24, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=0.0, le=50.0)
    seed: Optional[int] = Field(default=None, ge=0)
    width: Optional[int] = Field(default=None, ge=64)
    height: Optional[int] = Field(default=None, ge=64)


class ImageResult(BaseModel):
    """Metadata returned by the diffusion client."""

    job_id: str
    urls: List[str]
    provider: str = Field(default="stable-diffusion")


class ChatResponse(BaseModel):
    """Response returned by the `/chat` endpoint."""

    assistant_response: str
    image_prompt: Optional[ImagePrompt] = None
    image_job_id: Optional[str] = None
    image_urls: List[str] = Field(default_factory=list)
    used_mocks: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HealthStatus(BaseModel):
    """Status payload for the `/health` endpoint."""

    status: str
    use_mocks: bool
    auto_download_models: bool
    enable_image_generation: bool
    cached_models: bool
    details: dict = Field(default_factory=dict)

