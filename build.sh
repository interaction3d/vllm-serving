#!/bin/bash

# Exit on any error
set -e

# Default values
PROJECT_ID="image-classification-terraform"
REGION="us-central1"
REPO="vllm"
IMAGE_NAME="vllm-gpt-oss-20b-v3"
IMAGE_TAG="chattemplate"
SOURCE_DIR="/Users/tyler/Desktop/code/vllm-serving"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Build and push Docker image to Google Cloud Artifact Registry"
    echo ""
    echo "Options:"
    echo "  -p, --project-id PROJECT_ID   GCP Project ID (default: $PROJECT_ID)"
    echo "  -r, --region REGION           Registry region (default: $REGION)"
    echo "  -R, --repo REPO               Repository name (default: $REPO)"
    echo "  -i, --image-name IMAGE_NAME   Image name (default: $IMAGE_NAME)"
    echo "  -t, --tag IMAGE_TAG           Image tag (default: $IMAGE_TAG)"
    echo "  -s, --source-dir SOURCE_DIR   Source directory (default: $SOURCE_DIR)"
    echo "  -h, --help                    Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --tag latest --image-name vllm-serving"
    echo "  $0 -t v1.0 -i my-vllm -r us-west1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project-id)
            PROJECT_ID="$2"
            shift 2
            ;;
        -r|--region)
            REGION="$2"
            shift 2
            ;;
        -R|--repo)
            REPO="$2"
            shift 2
            ;;
        -i|--image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -s|--source-dir)
            SOURCE_DIR="$2"
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
echo "Building with configuration:"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Repository: $REPO"
echo "  Image Name: $IMAGE_NAME"
echo "  Image Tag: $IMAGE_TAG"
echo "  Source Directory: $SOURCE_DIR"
echo ""

echo "Checking and enabling required services..."

# Check if Artifact Registry API is enabled
if ! gcloud services list --enabled --filter="name:artifactregistry.googleapis.com" --format="value(name)" --project="$PROJECT_ID" | grep -q artifactregistry.googleapis.com; then
    echo "Enabling Artifact Registry API..."
    gcloud services enable artifactregistry.googleapis.com --project="$PROJECT_ID"
else
    echo "Artifact Registry API already enabled"
fi

# Check if Cloud Build API is enabled
if ! gcloud services list --enabled --filter="name:cloudbuild.googleapis.com" --format="value(name)" --project="$PROJECT_ID" | grep -q cloudbuild.googleapis.com; then
    echo "Enabling Cloud Build API..."
    gcloud services enable cloudbuild.googleapis.com --project="$PROJECT_ID"
else
    echo "Cloud Build API already enabled"
fi

echo "Checking and creating Artifact Registry repository..."

# Check if repository exists
if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
    echo "Creating Artifact Registry repository..."
    gcloud artifacts repositories create "$REPO" \
      --repository-format=docker \
      --location="$REGION" \
      --description="Docker repo for vLLM serving" \
      --project="$PROJECT_ID"
else
    echo "Artifact Registry repository already exists"
fi

echo "Building and pushing image..."
gcloud builds submit "$SOURCE_DIR" \
  --tag "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --project "$PROJECT_ID"

echo "Build complete!"
echo "Image: ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${IMAGE_TAG}"