#!/bin/bash

# Exit on any error
set -e

# Default values
PROJECT_ID="image-classification-terraform"

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Enable Google Cloud Trace API for the project"
    echo ""
    echo "Options:"
    echo "  -p, --project-id PROJECT_ID   GCP Project ID (default: $PROJECT_ID)"
    echo "  -h, --help                    Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --project-id my-project"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -p|--project-id)
            PROJECT_ID="$2"
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
echo "Enabling Cloud Trace for project: $PROJECT_ID"
echo ""

# Check if Cloud Trace API is enabled
if ! gcloud services list --enabled --filter="name:cloudtrace.googleapis.com" --format="value(name)" --project="$PROJECT_ID" | grep -q cloudtrace.googleapis.com; then
    echo "Enabling Cloud Trace API..."
    gcloud services enable cloudtrace.googleapis.com --project="$PROJECT_ID"
    echo "Cloud Trace API enabled successfully!"
else
    echo "Cloud Trace API already enabled"
fi

echo ""
echo "Cloud Trace setup complete!"
echo "You can view traces at: https://console.cloud.google.com/traces/list?project=$PROJECT_ID" 