PROJECT_ID="image-classification-terraform"
REGION="us-central1"
REPO="vllm"
IMAGE_NAME="vllm-gpt-oss-20b-v3"
IMAGE_TAG="chattemplate"

# gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com --project "$PROJECT_ID"

# gcloud artifacts repositories create "$REPO" \
#   --repository-format=docker \
#   --location="$REGION" \
#   --description="Docker repo for vLLM serving" \
#   --project "$PROJECT_ID" || echo "Repo may already exist"

gcloud builds submit /Users/tyler/Desktop/code/vllm-serving \
  --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --project "$PROJECT_ID"