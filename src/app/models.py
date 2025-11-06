from __future__ import annotations

import datetime as dt
from typing import Literal, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1024)
    negative_prompt: Optional[str] = Field(default=None, max_length=1024)
    num_inference_steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    seed: Optional[int] = Field(default=None, ge=0)

    @field_validator("width", "height")
    @classmethod
    def ensure_multiple_of_8(cls, value: int) -> int:
        if value % 8 != 0:
            raise ValueError("width and height must be multiples of 8")
        return value


class ImageGenerationResponse(BaseModel):
    job_id: str
    urls: list[HttpUrl | str]
    provider: Literal["stable-diffusion"]
    used_mocks: bool
    created_at: dt.datetime
    metadata: dict[str, str | int | float | None]


class HealthStatus(BaseModel):
    status: Literal["ok", "degraded"]
    use_mocks: bool
    models_cached: bool
    model_repo: str
    generated_images_dir: str
    models_cache_dir: str
