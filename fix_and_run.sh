#!/bin/bash

echo "🔧 Fixing RAG System Issues..."

# 1. Install missing dependencies
echo "📦 Installing dependencies..."
pip install google-cloud-aiplatform
pip install vertexai
pip install pgvector
pip install psycopg2-binary

# 2. Set up environment variables
echo "⚙️ Setting up environment..."
export GCP_PROJECT_ID="lcrailcrai-472418"  # Replace with your project ID
export GOOGLE_APPLICATION_CREDENTIALS="./gcp-key.json"
export DATABASE_URL="postgresql://raguser:lcrai_2025@localhost:5432/ragdb_dev"

# 3. Initialize database
echo "🗄️ Initializing database..."
python -c "from app.models.database import create_tables; create_tables()"

# 4. Start the API server
echo "🚀 Starting API server..."
cd app
PYTHONPATH=. python main.py

echo "✅ RAG System fixed and running!"