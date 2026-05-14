#!/usr/bin/env bash

# No actual GCP-specific implementation yet, 
#but this script will build the Docker image and deploy to Cloud Run when ready.

# Deploy payback-assistant to Cloud Run.
# Usage: ./scripts/deploy_gcp.sh [PROJECT_ID] [REGION]

set -euo pipefail

PROJECT=${1:-${GCP_PROJECT_ID:-""}}
REGION=${2:-"europe-west1"}
SERVICE="payback-assistant"
IMAGE="gcr.io/${PROJECT}/${SERVICE}"

if [[ -z "$PROJECT" ]]; then
  echo "Error: GCP_PROJECT_ID not set. Pass it as first argument or set env var."
  exit 1
fi

echo "Building and pushing image to $IMAGE ..."
gcloud builds submit --tag "$IMAGE" .

echo "Deploying to Cloud Run ($REGION) ..."
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --platform managed \
  --region "$REGION" \
  --allow-unauthenticated \
  --port 8080 \
  --set-env-vars "DEFAULT_LLM_PROVIDER=${DEFAULT_LLM_PROVIDER:-claude}" \
  --set-secrets "ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,GOOGLE_API_KEY=GOOGLE_API_KEY:latest"

echo "Deployment complete."
gcloud run services describe "$SERVICE" --region "$REGION" --format "value(status.url)"
