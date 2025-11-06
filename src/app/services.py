from __future__ import annotations

import asyncio
import datetime as dt
import logging
import uuid
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download
from optimum.intel import OVStableDiffusionPipeline
from PIL import Image

from .config import Settings
from .models import ImageGenerationRequest, ImageGenerationResponse

logger = logging.getLogger(__name__)


class StableDiffusionService:
    """Handles Stable Diffusion inference and response generation."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipeline: OVStableDiffusionPipeline | None = None
        self._compiled_size: tuple[int, int] | None = None
        self._model_dir: Path = self.settings.models_cache_dir / "stable-diffusion"

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        job_id = f"sd-job-{uuid.uuid4().hex[:8]}"
        created_at = dt.datetime.utcnow().replace(microsecond=0, tzinfo=dt.timezone.utc)

        if self.settings.use_mocks:
            url = self._mount_static_asset("mock-image.svg")
            metadata = {"mode": "mock", "prompt": request.prompt}
            return ImageGenerationResponse(
                job_id=job_id,
                urls=[url],
                provider="stable-diffusion",
                used_mocks=True,
                created_at=created_at,
                metadata=metadata,
            )

        image = await asyncio.to_thread(self._run_inference, request)
        image_path = self._store_image(job_id, image)

        metadata = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": request.width,
            "height": request.height,
            "guidance_scale": request.guidance_scale,
            "num_inference_steps": request.num_inference_steps,
            "seed": request.seed,
            "model": self.settings.stable_diffusion_repo_id,
            "device": self.settings.openvino_device,
        }

        url = self._relative_image_url(image_path)

        return ImageGenerationResponse(
            job_id=job_id,
            urls=[url],
            provider="stable-diffusion",
            used_mocks=False,
            created_at=created_at,
            metadata=metadata,
        )

    def _run_inference(self, request: ImageGenerationRequest) -> Image.Image:
        pipeline = self._prepare_pipeline(request.width, request.height)
        generator = None
        if request.seed is not None:
            generator = np.random.RandomState(request.seed)

        result = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
        )
        if not result.images:
            raise RuntimeError("Stable Diffusion pipeline returned no images.")
        return result.images[0]

    def _prepare_pipeline(self, width: int, height: int) -> OVStableDiffusionPipeline:
        pipeline = self._load_pipeline()
        target_shape = (width, height)
        if self._compiled_size != target_shape:
            pipeline.reshape(
                batch_size=1,
                height=height,
                width=width,
                num_images_per_prompt=1,
            )
            pipeline.compile()
            self._compiled_size = target_shape
        return pipeline

    def _load_pipeline(self) -> OVStableDiffusionPipeline:
        if self._pipeline is not None:
            return self._pipeline

        model_source = self._model_dir if self.models_cached() else self.settings.stable_diffusion_repo_id
        self._pipeline = OVStableDiffusionPipeline.from_pretrained(
            model_source,
            compile=False,
            use_auth_token=self.settings.huggingface_token,
        )
        try:
            self._pipeline.to(self.settings.openvino_device)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to place pipeline on device '%s': %s. Falling back to CPU.",
                self.settings.openvino_device,
                exc,
            )
            self._pipeline.to("CPU")
        return self._pipeline

    def _store_image(self, job_id: str, image: Image.Image) -> Path:
        self.settings.generated_images_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.settings.generated_images_dir / f"{job_id}.png"
        image.save(output_path)
        return output_path

    def _relative_image_url(self, path: Path) -> str:
        base_url = self.settings.base_url.rstrip("/") if self.settings.base_url else ""
        filename = path.name
        relative = f"generated/{filename}"
        if base_url:
            return f"{base_url}/{relative}"
        return f"/{relative}"

    def _mount_static_asset(self, filename: str) -> str:
        asset_path = self.settings.static_dir / filename
        if not asset_path.exists():
            raise FileNotFoundError(f"Static asset '{filename}' not found in {asset_path.parent}")

        base_url = self.settings.base_url.rstrip("/") if self.settings.base_url else ""
        relative = asset_path.as_posix().replace("\\", "/")
        if base_url:
            return f"{base_url}/{relative}"
        return f"/{relative}"

    def ensure_model_snapshot(self) -> None:
        """Download the configured model snapshot if requested."""
        if not self.settings.auto_download_models:
            return

        cache_dir = self._model_dir
        if cache_dir.exists() and any(cache_dir.rglob("*.xml")):
            return

        cache_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.settings.stable_diffusion_repo_id,
            local_dir=cache_dir,
            local_dir_use_symlinks=False,
            token=self.settings.huggingface_token,
            resume_download=True,
        )

    def models_cached(self) -> bool:
        cache_dir = self._model_dir
        return cache_dir.exists() and any(cache_dir.rglob("*.xml"))
