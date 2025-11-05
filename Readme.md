# Multimodal Chatbot
- `GET /debug/llm` — smoke-tests the DeepSeek endpoint with a diagnostic prompt.
- `POST /image` — direct Stable Diffusion endpoint for prompt-driven image tests.

An offline-friendly multimodal assistant that pairs **OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int4-ov** for chat with **OpenVINO/stable-diffusion-v1-5-int8-ov** for on-demand image synthesis. The service runs entirely on user-controlled hardware and ships with a lightweight single-page UI.

## Getting Started

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_models.py       # optional; forces model downloads
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/` to use the chat interface, or interact with the API directly.

```http
POST /chat
Content-Type: application/json

{
  "user_id": "demo-user",
  "message": "Create a short bedtime story with a dreamy illustration."
}
```

The orchestrator maintains per-user context, calls the DeepSeek model, and—when an `image_prompt` is returned—invokes Stable Diffusion to generate imagery. Responses contain:

```jsonc
{
  "assistant_response": "…",
  "image_prompt": { "prompt": "…", "num_inference_steps": 24, ... },
  "image_job_id": "mock-job-1234",
  "image_urls": ["/static/mock-image.svg"],
  "used_mocks": true,
  "created_at": "2025-05-01T12:00:00Z"
}
```

### Text-only quickstart (DeepSeek only)

If you want to verify the chat experience before wiring up Stable Diffusion, leave image generation disabled (the default) and point the service at your DeepSeek endpoint:

```powershell
# optional: stay in mock mode while preparing the DeepSeek runtime
set USE_MOCKS=true

# once your OpenAI-compatible DeepSeek server is reachable:
set USE_MOCKS=false
set DEEPSEEK_ENDPOINT=http://127.0.0.1:8001/v1/chat/completions
set ENABLE_IMAGE_GENERATION=false

uvicorn main:app --reload
```

Use the `/debug/llm` endpoint to send a diagnostic prompt and confirm the DeepSeek server responds before trying the UI.

## Configuration

All settings are loaded from environment variables (or `.env`) via `src/app/config.py`. Defaults now assume real inference endpoints are available so the chatbot attempts to call the actual models unless you toggle them off explicitly.

| Variable | Default | Purpose |
| --- | --- | --- |
| `USE_MOCKS` | `false` | Return canned responses instead of calling inference servers |
| `AUTO_DOWNLOAD_MODELS` | `true` | Automatically fetch missing model snapshots on startup |
| `MODELS_CACHE_DIR` | `data/models` | Root directory for cached OpenVINO weights |
| `DEEPSEEK_ENDPOINT` | `http://localhost:8001/v1/chat/completions` | OpenAI-compatible chat endpoint |
| `STABLE_DIFFUSION_ENDPOINT` | `http://localhost:8002/v1/images/generations` | Diffusion endpoint |
| `DEEPSEEK_REPO_ID` | `OpenVINO/DeepSeek-R1-Distill-Qwen-1.5B-int8-ov` | Hugging Face repo for the LLM |
| `STABLE_DIFFUSION_REPO_ID` | `OpenVINO/stable-diffusion-v1-5-int8-ov` | Hugging Face repo for diffusion |
| `ENABLE_IMAGE_GENERATION` | `false` | Enable Stable Diffusion image generation calls |
| `HUGGINGFACE_TOKEN` | *(unset)* | Token required for gated downloads |
| `ENABLE_CATBOT_FALLBACK` | `false` | Enable playful local fallback when the LLM endpoint is unavailable |

Use the bundled CLI to stage weights ahead of time. It relies on the Hugging Face `hf download` command (bundled with recent `huggingface-hub` releases`) and bypasses the mock guard unless `--no-force` is supplied.

```powershell
python download_models.py          # force download
python download_models.py --no-force
```

## API Surface

- `POST /chat` — core multimodal endpoint. Accepts a `ChatRequest` and returns a `ChatResponse`.
- `GET /health` — reports service status, mock usage, and whether model snapshots are cached (`details` map).
- Static assets mounted at `/static/*` serve the bundled UI (`static/index.html`, `chat.js`, `style.css`).

## Development Notes

- Conversation history is kept in-memory with a configurable turn limit (`CONVERSATION_HISTORY_LIMIT`, default 10).
- Mock mode produces deterministic text and overlays image placeholders from `static/mock-image.svg`.
- When `USE_MOCKS=false`, ensure the configured DeepSeek and Stable Diffusion endpoints expose OpenAI-compatible JSON APIs.
- Extend the pipeline by swapping clients in `src/app/llm_client.py` and `src/app/image_client.py`, or by persisting history in `src/app/orchestrator.py`.
- If an endpoint goes offline, enable CatBot fallback (`ENABLE_CATBOT_FALLBACK=true`) when you prefer graceful copy over raw errors.
