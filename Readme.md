# Stable Diffusion Service
Generate images from text prompts with a FastAPI backend powered by OpenVINO (`OpenVINO/stable-diffusion-v1-5-int8-ov`) and a lightweight browser UI for prompt submission.

## Getting Started

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
# optional: download model snapshot locally (requires HF token for gated repos)
python download_models.py
setx HF_TOKEN "<your-hugging-face-token>"   # or use a .env file
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` to use the bundled interface, or issue API calls directly:

```http
POST /image
Content-Type: application/json

{
  "prompt": "Moody watercolor of neon-lit Kyoto streets at night",
  "negative_prompt": "low detail, blurry",
  "num_inference_steps": 28,
  "guidance_scale": 8.0,
  "width": 512,
  "height": 512
}
```

Successful responses contain the generated image URLs along with metadata:

```jsonc
{
  "job_id": "sd-job-001",
  "urls": ["/static/mock-image.svg"],
  "provider": "stable-diffusion",
  "used_mocks": true,
  "created_at": "2025-05-01T12:00:00Z"
}
```

## Configuration

Settings are read from environment variables (or `.env`). The most relevant options are:

| Variable | Default | Description |
| --- | --- | --- |
| `USE_MOCKS` | `false` | Return the bundled placeholder image instead of calling the inference backend |
| `AUTO_DOWNLOAD_MODELS` | `true` | Download the Stable Diffusion snapshot automatically at startup (if not cached) |
| `MODELS_CACHE_DIR` | `data/models` | Directory used to cache Stable Diffusion model snapshots |
| `GENERATED_IMAGES_DIR` | `data/generated` | Location used to persist generated PNG files |
| `STABLE_DIFFUSION_REPO_ID` | `OpenVINO/stable-diffusion-v1-5-int8-ov` | Hugging Face repository that hosts the Stable Diffusion OpenVINO weights |
| `HUGGINGFACE_TOKEN` | *(unset)* | Token for gated repositories if required |
| `REQUEST_TIMEOUT` | `45.0` | HTTP timeout used by the client in seconds |
| `BASE_URL` | *(unset)* | Optional base URL used when building absolute asset links |
| `OPENVINO_DEVICE` | `GPU` | Target device for inference (`GPU`, `CPU`, `AUTO`, etc.). Falls back to CPU if unavailable. |

Use the CLI helper to manage model downloads manually:

```powershell
python download_models.py          # force download
python download_models.py --no-force
```

## API Surface

- `POST /image` — accept an image-generation prompt and return resulting image URLs.
- `GET /health` — report service readiness, cache status, and runtime configuration.
- Static assets (`/static/*`) host the single-page interface built with vanilla JS.

## Development Notes

- Mock mode uses `static/mock-image.svg` to keep the UI functioning without an inference server.
- Model downloads rely on `huggingface_hub.snapshot_download`; cached assets live under `MODELS_CACHE_DIR/stable-diffusion`.
- Generated images are persisted under `GENERATED_IMAGES_DIR` and exposed at `/generated/<file>`.
- The runtime uses `optimum-intel`'s `OVStableDiffusionPipeline`; first-call compilation per resolution can take a few seconds.
- By default the pipeline targets Intel GPU. If the device is missing, the server logs a warning and runs on CPU.
