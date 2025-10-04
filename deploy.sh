#!/bin/bash

# Set variables
PROJECT_ID=$(gcloud config get-value project)
SERVICE_NAME="rag-minimal"
REGION="us-central1"

echo "Deploying to project: $PROJECT_ID"

# Build and deploy to Cloud Run
gcloud run deploy $SERVICE_NAME \
  --source . \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 5 \
  --min-instances 0 \
  --timeout 300 \
  --set-env-vars PROJECT_ID=$PROJECT_ID \
  --set-env-vars GCP_REGION=$REGION

echo "Deployment complete!"
gcloud run services describe $SERVICE_NAME --region $REGION --format "value(status.url)"
