from __future__ import annotations

from pathlib import Path

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from environment variables or .env."""

    use_mocks: bool = Field(
        False,
        description="Return placeholder assets instead of contacting the inference server.",
    )
    auto_download_models: bool = Field(
        True,
        description="Automatically download model snapshots during startup.",
    )
    models_cache_dir: Path = Field(
        Path("data/models"),
        description="Directory where downloaded model snapshots are stored.",
    )
    generated_images_dir: Path = Field(
        Path("data/generated"),
        description="Directory that stores generated image files.",
    )
    static_dir: Path = Field(
        Path("static"),
        description="Directory containing static assets exposed by the service.",
    )
    stable_diffusion_repo_id: str = Field(
        "OpenVINO/stable-diffusion-v1-5-int8-ov",
        description="Hugging Face repository that hosts the Stable Diffusion OpenVINO weights.",
    )
    huggingface_token: str | None = Field(
        default=None,
        description="Optional token used to authenticate against Hugging Face.",
    )
    request_timeout: float = Field(
        45.0,
        ge=1.0,
        description="Timeout for inference requests, in seconds.",
    )
    base_url: HttpUrl | None = Field(
        default=None,
        description="Optional base URL forwarded to the frontend to build absolute links.",
    )
    openvino_device: str = Field(
        "GPU",
        description="OpenVINO device identifier used for inference (e.g., 'GPU', 'CPU', 'AUTO').",
    )

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    def ensure_directories(self) -> None:
        """Create required directories if they do not exist yet."""
        self.models_cache_dir.mkdir(parents=True, exist_ok=True)
        self.generated_images_dir.mkdir(parents=True, exist_ok=True)
