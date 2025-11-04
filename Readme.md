

# Multimodal Chatbot

This project implements an orchestrator that combines DeepSeek-R1-Distill-Qwen-1.5B for conversational reasoning with Stable Diffusion v1.5 for on-demand image generation.

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

## Model Downloads

On startup the application downloads the required model weights from Hugging Face when:
- `USE_MOCKS=false`
- `AUTO_DOWNLOAD_MODELS=true` (default)

You can customize the cache location and repositories with the following environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `MODELS_CACHE_DIR` | `models` | Local directory to store model weights |
| `DEEPSEEK_REPO_ID` | `DeepSeek-AI/DeepSeek-R1-Distill-Qwen-1.5B-INT4` | Hugging Face repo for the LLM |
| `STABLE_DIFFUSION_REPO_ID` | `runwayml/stable-diffusion-v1-5` | Hugging Face repo for Stable Diffusion |
| `HUGGINGFACE_TOKEN` | empty | Optional token for private model access |

Set `AUTO_DOWNLOAD_MODELS=false` to disable automatic downloads (e.g., when using pre-provisioned volumes).
