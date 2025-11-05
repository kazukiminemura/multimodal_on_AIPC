"""Application configuration powered by environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Feature toggles
    use_mocks: bool = Field(default=False, alias="USE_MOCKS")
    auto_download_models: bool = Field(default=True, alias="AUTO_DOWNLOAD_MODELS")
    enable_image_generation: bool = Field(default=False, alias="ENABLE_IMAGE_GENERATION")
    enable_catbot_fallback: bool = Field(default=False, alias="ENABLE_CATBOT_FALLBACK")

    # Runtime configuration
    request_timeout_seconds: float = Field(default=30.0, alias="REQUEST_TIMEOUT")
    conversation_history_limit: int = Field(default=10, alias="CONVERSATION_HISTORY_LIMIT")

    # Model + inference endpoints
    deepseek_endpoint: str = Field(
        default="http://localhost:8001/v1/chat/completions", alias="DEEPSEEK_ENDPOINT"
    )
    stable_diffusion_endpoint: str = Field(
        default="http://localhost:8002/v1/images/generations",
        alias="STABLE_DIFFUSION_ENDPOINT",
    )

    # Hugging Face repos + cache paths
    deepseek_repo_id: str = Field(
        default="OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov", alias="DEEPSEEK_REPO_ID"
    )
    stable_diffusion_repo_id: str = Field(
        default="OpenVINO/stable-diffusion-v1-5-int8-ov",
        alias="STABLE_DIFFUSION_REPO_ID",
    )
    models_cache_dir: Path = Field(default=Path("data/models"), alias="MODELS_CACHE_DIR")
    huggingface_token: Optional[str] = Field(default=None, alias="HUGGINGFACE_TOKEN")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings()
    settings.models_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings

