#!/bin/bash

# Exit on any error
set -e

# Default values
PROJECT_ID="image-classification-terraform"
REGION="us-central1"
REPO="vllm"
IMAGE_NAME="vllm-gpt-oss-20b-v3"
IMAGE_TAG="chattemplate"
SERVICE_NAME="vllm-serving"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
MEMORY="32Gi"
CPU="8"
GPU_TYPE="nvidia-l4"
GPU_COUNT="1"
MAX_TOKENS="8192"
TP_DEGREE="1"
TIMEOUT="3600"
PORT="8080"
MAX_INSTANCES="1"
HF_TOKEN="TOKEN"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Deploy vLLM service to Cloud Run with GPU acceleration"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_NAME       Model name (default: $MODEL_NAME)"
    echo "  -M, --memory MEMORY           Memory allocation (default: $MEMORY)"
    echo "  -c, --cpu CPU_COUNT           CPU count (default: $CPU)"
    echo "  -g, --gpu-type GPU_TYPE       GPU type (default: $GPU_TYPE)"
    echo "  -G, --gpu-count GPU_COUNT     GPU count (default: $GPU_COUNT)"
    echo "  -t, --max-tokens MAX_TOKENS   Max tokens (default: $MAX_TOKENS)"
    echo "  -p, --tp-degree TP_DEGREE     Tensor parallelism degree (default: $TP_DEGREE)"
    echo "  -T, --timeout TIMEOUT         Request timeout in seconds (default: $TIMEOUT)"
    echo "  -P, --port PORT               Service port (default: $PORT)"
    echo "  -i, --max-instances INSTANCES Max instances (default: $MAX_INSTANCES)"
    echo "  -s, --service-name NAME       Cloud Run service name (default: $SERVICE_NAME)"
    echo "  -r, --region REGION           Cloud Run region (default: $REGION)"
    echo "  -H, --hf-token TOKEN          HuggingFace token (default: configured token)"
    echo "  -h, --help                    Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --model microsoft/DialoGPT-medium --memory 16Gi --cpu 4"
    echo "  $0 -m gpt2 -M 8Gi -c 2 -g nvidia-t4 --hf-token your_token_here"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -M|--memory)
            MEMORY="$2"
            shift 2
            ;;
        -c|--cpu)
            CPU="$2"
            shift 2
            ;;
        -g|--gpu-type)
            GPU_TYPE="$2"
            shift 2
            ;;
        -G|--gpu-count)
            GPU_COUNT="$2"
            shift 2
            ;;
        -t|--max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -p|--tp-degree)
            TP_DEGREE="$2"
            shift 2
            ;;
        -T|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -P|--port)
            PORT="$2"
            shift 2
            ;;
        -i|--max-instances)
            MAX_INSTANCES="$2"
            shift 2
            ;;
        -s|--service-name)
            SERVICE_NAME="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -H|--hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Display configuration
echo "Deploying with configuration:"
echo "  Service Name: $SERVICE_NAME"
echo "  Region: $REGION"
echo "  Model: $MODEL_NAME"
echo "  Memory: $MEMORY"
echo "  CPU: $CPU"
echo "  GPU: $GPU_COUNT x $GPU_TYPE"
echo "  Max Tokens: $MAX_TOKENS"
echo "  Tensor Parallelism: $TP_DEGREE"
echo "  Timeout: ${TIMEOUT}s"
echo "  Port: $PORT"
echo "  Max Instances: $MAX_INSTANCES"
echo ""

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
  --image=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG} \
  --region=$REGION \
  --platform=managed \
  --gpu=$GPU_COUNT \
  --gpu-type=$GPU_TYPE \
  --memory=$MEMORY \
  --cpu=$CPU \
  --concurrency=1 \
  --timeout=$TIMEOUT \
  --port=$PORT \
  --set-env-vars=MODEL_NAME=$MODEL_NAME,TP_DEGREE=$TP_DEGREE,MAX_MODEL_LEN=$MAX_TOKENS,TRUST_REMOTE_CODE=false,VLLM_DEVICE=cuda,HUGGING_FACE_HUB_TOKEN=$HF_TOKEN \
  --max-instances=$MAX_INSTANCES \
  --allow-unauthenticated

echo ""
echo "Deployment complete!"
echo "Service URL: $(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')"