

# Multimodal Chatbot

This project implements an orchestrator that combines DeepSeek-R1-Distill-Qwen-1.5B for conversational reasoning with Stable Diffusion 3 Medium for on-demand image generation.

## Getting Started

```shell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --app-dir src --reload
```

The API exposes a `/chat` endpoint that accepts:

```json
{
  "user_id": "demo-user",
  "message": "Generate a serene landscape painting."
}
```

When the language model decides an image is needed, it returns prompts that are forwarded to the Stable Diffusion backend. The default configuration uses mock responses; set real endpoints via environment variables defined in `src/app/config.py`.

Use `GET /health` to confirm service readiness and check whether mocks are enabled and models are cached.

## Web UI

After starting the server, open `http://127.0.0.1:8000/` to use the built-in chat UI.  
The interface stores recent exchanges in your browser session only and shows generated image thumbnails when the model supplies `image_urls`.

## Model Downloads

On startup the application downloads the required model weights from Hugging Face when:
- `USE_MOCKS=false`
- `AUTO_DOWNLOAD_MODELS=true` (default)

You can customize the cache location and repositories with the following environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MODELS_CACHE_DIR` | `data/models` | Local directory to store model weights |
| `DEEPSEEK_REPO_ID` | `OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov` | Hugging Face repo for the LLM |
| `STABLE_DIFFUSION_REPO_ID` | `OpenVINO/stable-diffusion-v1-5-int8-ov` | Hugging Face repo for Stable Diffusion |
| `HUGGINGFACE_TOKEN` | empty | Optional token for private model access |

To prefetch weights ahead of time, run:

```shell
python download_models.py
```

This script forces a download even if `USE_MOCKS=true`, so ensure you have the required Hugging Face token exported beforehand.

Set `AUTO_DOWNLOAD_MODELS=false` to disable automatic downloads (e.g., when using pre-provisioned volumes).
