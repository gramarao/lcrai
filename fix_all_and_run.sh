#!/bin/bash

echo "🔧 Comprehensive RAG System Fix Script..."


# 2. Install missing dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install google-cloud-aiplatform vertexai pgvector psycopg2-binary
pip install sqlalchemy alembic fastapi uvicorn streamlit

# 3. Set up environment variables
echo "⚙️ Setting up environment..."
if [ -z "$GCP_PROJECT_ID" ]; then
    echo "⚠️  Please set GCP_PROJECT_ID environment variable"
    echo "   export GCP_PROJECT_ID='your-actual-project-id'"
    exit 1
fi

if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "⚠️  Please set GOOGLE_APPLICATION_CREDENTIALS environment variable"
    echo "   export GOOGLE_APPLICATION_CREDENTIALS='/path/to/service-account.json'"
    exit 1
fi

# 4. Test database connection
echo "🗄️ Testing database connection..."
if ! pg_isready -h localhost -p 5433 -U raguser; then
    echo "❌ PostgreSQL is not running or not accessible"
    echo "   Please start PostgreSQL and ensure user 'raguser' exists"
    exit 1
fi


# 7. Test the fixed system
echo "🧪 Testing system..."
python3 test_fixes.py

if [ $? -eq 0 ]; then
    echo "🎉 All fixes applied successfully!"
    echo "🚀 Starting API server..."
    cd app
    python3 main.py
else
    echo "❌ Tests failed. Please check the error messages above."
    exit 1
fi