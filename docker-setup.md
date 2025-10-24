#frontend build
docker build -t gcr.io/lcrai2025/lcr-frontend:latest -f Dockerfile.frontend .

#frontend push
docker push gcr.io/lcrai2025/lcr-frontend:latest

gcloud run deploy lcr-frontend \
  --image gcr.io/lcrai2025/lcr-frontend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="BACKEND_URL=REPLACE_BACKEND_URL,GOOGLE_API_KEY=API_KEY" \
  --memory=1Gi \
  --cpu=1 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=3 \
  --project=lcrai2025


# backend build
docker build -t gcr.io/lcrai2025/lcr-backend:latest -f Dockerfile.backend .

#backend push
docker push gcr.io/lcrai2025/lcr-backend:latest


# Cloud run
gcloud run deploy lcr-backend \
  --image gcr.io/lcrai2025/lcr-backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="DATABASE_URL=postgresql://postgres:DB_PASSWD/ragdb?host=/cloudsql/lcrai2025:us-central1:rag-postgres,GOOGLE_API_KEY=API_KEY,CHAT_MODEL=gemini-2.0-flash-exp,EMBED_MODEL=models/text-embedding-004,RETRIEVE_K=5" \
  --add-cloudsql-instances=lcrai2025:us-central1:rag-postgres \
  --memory=2Gi \
  --cpu=2 \
  --timeout=300 \
  --min-instances=0 \
  --max-instances=3 \
  --project=lcrai2025
