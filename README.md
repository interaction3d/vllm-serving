# vLLM FastAPI Serving

A minimal FastAPI server to serve an open-source LLM using vLLM. Defaults to `gpt-oss-20b` but can be configured via environment variables.

## Prerequisites
- Python 3.10+
- GPU with recent CUDA drivers recommended for performance (vLLM supports CPU but is slow).

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure
Environment variables (use `.env` or export):
- `MODEL_NAME` (default: `gpt-oss-20b`) – any HuggingFace or compatible model path
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `TP_DEGREE` (default: `1`) – tensor parallelism degree
- `MAX_MODEL_LEN` (default: `8192`) – max sequence length

Create a `.env` file:
```env
MODEL_NAME=gpt-oss-20b
HOST=0.0.0.0
PORT=8000
TP_DEGREE=1
MAX_MODEL_LEN=8192
```

## Run
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

vLLM will lazy-load the model on first request.

## API
- `POST /generate` – single-turn generation
- `POST /chat` – multi-turn chat with messages array
- `GET /health` – liveness

### Generate
```bash
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Write a short poem about the ocean",
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

### Chat
```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain tensors simply."}
    ],
    "temperature": 0.3,
    "max_tokens": 256
  }'
```

## Notes
- This server uses vLLM's `LLM` and `SamplingParams` directly for speed.
- Adjust `tensor_parallel_size` (`TP_DEGREE`) for multi-GPU setups.
- For production, consider: request timeouts, auth, rate limits, logging, and observability. 