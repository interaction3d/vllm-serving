# vLLM FastAPI Serving

A minimal FastAPI server to serve an open-source LLM using vLLM. Defaults to `Qwen/Qwen2.5-1.5B-Instruct` but can be configured via environment variables.

## Prerequisites
- Python 3.10+
- GPU with recent CUDA drivers recommended for performance (vLLM supports CPU but is slow).
- Google Cloud Platform account with Cloud Run and Artifact Registry enabled
- Docker installed locally

## Configure
Environment variables (use `.env` or export):
- `MODEL_NAME` (default: `Qwen/Qwen2.5-1.5B-Instruct`) – any HuggingFace or compatible model path
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `TP_DEGREE` (default: `1`) – tensor parallelism degree
- `MAX_MODEL_LEN` (default: `8192`) – max sequence length

Create a `.env` file:
```env
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
HOST=0.0.0.0
PORT=8000
TP_DEGREE=1
MAX_MODEL_LEN=8192
```

## Deploy to Cloud Run

### 1. Build the Docker image
```bash
./build.sh
```

The `build.sh` script performs the following operations:
- Submits the local codebase to Google Cloud Build using `gcloud builds submit`
- Cloud Build pulls the vLLM OpenAI base image (`vllm/vllm-openai:gptoss`) and builds the Docker image remotely
- Tags the built image directly in Artifact Registry as `us-central1-docker.pkg.dev/image-classification-terraform/vllm/vllm-gpt-oss-20b-v3:latest`
- No local Docker build or push operations required - everything happens in the cloud

### 2. Deploy to Cloud Run
```bash
./deploy.sh \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --memory 32Gi \
  --cpu 8 \
  --gpu-type nvidia-l4 \
  --gpu-count 1 \
  --max-tokens 8192 \
  --tp-degree 1 \
  --timeout 3600 \
  --port 8080 \
  --max-instances 1 \
  --service-name vllm-serving \
  --region us-central1 \
  --hf-token your_huggingface_token_here
```

The `deploy.sh` script executes a `gcloud run deploy` command with configurable parameters via command line flags:
- **Image**: Pulls from Artifact Registry (`us-central1-docker.pkg.dev/image-classification-terraform/vllm/vllm-gpt-oss-20b-v3:latest`)
- **Compute**: Configurable vCPUs (default: 8), memory (default: 32GB) with GPU acceleration (default: 1x NVIDIA L4)
- **Scaling**: Configurable maximum instances (default: 1) with concurrency limit of 1 request per instance
- **Network**: Configurable port (default: 8080) with unauthenticated public access
- **Environment**: Configurable model parameters including model name, max tokens, tensor parallelism degree, CUDA device
- **Timeout**: Configurable timeout (default: 3600 seconds) for long-running inference tasks

**Available flags:**
- `-m, --model` - Model name (default: Qwen/Qwen2.5-1.5B-Instruct)
- `-M, --memory` - Memory allocation (default: 32Gi)
- `-c, --cpu` - CPU count (default: 8)
- `-g, --gpu-type` - GPU type (default: nvidia-l4)
- `-G, --gpu-count` - GPU count (default: 1)
- `-t, --max-tokens` - Max tokens (default: 8192)
- `-T, --timeout` - Request timeout in seconds (default: 3600)
- `-s, --service-name` - Cloud Run service name (default: vllm-serving)
- `-r, --region` - Cloud Run region (default: us-central1)
- `-H, --hf-token` - HuggingFace token (required for private models)

**Other examples:**
```bash
# Deploy with custom model and resources
./deploy.sh --model microsoft/DialoGPT-medium --memory 16Gi --cpu 4

# Deploy with T4 GPU instead of L4
./deploy.sh -m gpt2 -M 8Gi -c 2 -g nvidia-t4

# Show all available options
./deploy.sh --help
```

The service will be deployed with L4 GPU acceleration and will lazy-load the model on first request.

## API
- `POST /generate` – single-turn generation
- `POST /chat` – multi-turn chat with messages array
- `GET /health` – liveness

### Generate
```bash
curl -X POST https://vllm-serving-710725879078.us-central1.run.app/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about the ocean",
    "temperature": 0.7,
    "max_tokens": 128
  }' | jq
```

**Example Response:**
```json
{
  "text": ": The vast, endless expanse of blue, A wondrous sight that never ends. The waves crash against the shore, A symphony of sound that never ceases.\nThe salt air invades your nostrils, A reminder of the salty sea. The sun sets in the horizon, A moment of beauty that never fades.\nThe ocean holds the breath of all, Its power and majesty never fails. It's a place of mystery and wonder, A place where dreams take flight. The ocean is a beloved treasure, A place that never fails to touch our hearts. Its beauty and strength we will never forget, and its mysteries we",
  "num_tokens": 128
}
```

### Chat
```bash
curl -X POST https://vllm-serving-710725879078.us-central1.run.app/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain tensors simply."}
    ],
    "temperature": 0.3,
    "max_tokens": 256
  }' | jq
```

**Example Response:**
```json
{
  "text": "Tensors are mathematical objects that generalize scalars, vectors, and matrices to higher dimensions. They are used to describe physical quantities that have both magnitude and direction, and can be represented as multi-dimensional arrays of numbers.\n\nIn simple terms, a tensor is a mathematical object that can be used to describe physical quantities that have both magnitude and direction, and can be represented as multi-dimensional arrays of numbers. For example, a vector is a type of tensor that has only one dimension, and represents a quantity that has both magnitude and direction. A matrix is a type of tensor that has two dimensions, and represents a quantity that has multiple directions or components.\n\nTensors can be used to describe a wide range of physical phenomena, including stress and strain in materials, electromagnetic fields, and fluid flow. They are also used in many areas of science and engineering, including physics, engineering, and computer science.",
  "num_tokens": 179
}
```

## Notes
- This server uses vLLM's `LLM` and `SamplingParams` directly for speed.
- Adjust `tensor_parallel_size` (`TP_DEGREE`) for multi-GPU setups.
- For production, consider: request timeouts, auth, rate limits, logging, and observability. 