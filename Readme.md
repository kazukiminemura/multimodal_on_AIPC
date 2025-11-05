# OpenVINO Stable Diffusion Service
Generate images locally with **OpenVINO/stable-diffusion-v1-5-int8-ov**. This project contains a FastAPI backend with an OpenVINO-aware model downloader plus a lightweight browser UI for prompt submission.

## Getting Started

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py       # optional; forces model downloads
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
| `USE_MOCKS` | `false` | Return a canned placeholder image instead of contacting the inference server |
| `AUTO_DOWNLOAD_MODELS` | `true` | Download the Stable Diffusion snapshot automatically at startup |
| `MODELS_CACHE_DIR` | `data/models` | Directory used to cache OpenVINO model files |
| `STABLE_DIFFUSION_ENDPOINT` | `http://localhost:8002/v1/images/generations` | Stable Diffusion inference endpoint (OpenAI-compatible POST API) |
| `STABLE_DIFFUSION_REPO_ID` | `OpenVINO/stable-diffusion-v1-5-int8-ov` | Hugging Face repository that hosts the quantized weights |
| `HUGGINGFACE_TOKEN` | *(unset)* | Token for gated repositories if required |
| `REQUEST_TIMEOUT` | `30.0` | HTTP timeout used by the client in seconds |

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
- Adjust prompt defaults or UI behavior via `static/index.html` and `static/chat.js`.
