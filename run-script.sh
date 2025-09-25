#!/bin/bash

# RAG Service Startup Script
# Starts: Cloud SQL Proxy > Uvicorn (FastAPI) > Streamlit (UI)
# Usage: ./run-script.sh [start|restart|stop]

# =====================================
# CONFIGURATION - UPDATE THESE VALUES
# =====================================

# Cloud SQL Proxy (for PostgreSQL with pgvector)
PROXY_INSTANCE="lcrai-472418:us-central1:rag-postgres"  # Format: project:region:instance
PROXY_PORT=5433

# Uvicorn (FastAPI/RAG service)
UVICORN_PORT=8000
UVICORN_APP="api.app.main:app"  # Replace with your FastAPI app (e.g., app.main:app)
UVICORN_HOST="0.0.0.0"
UVICORN_ENV="lcrai_venv"  # Virtual env path (relative to project root)

# Streamlit (UI)
STREAMLIT_PORT=8501
STREAMLIT_APP="streamlit_app.py"  # Replace with your Streamlit file (e.g., ui.py)
STREAMLIT_HOST="0.0.0.0"
STREAMLIT_ENV="lcrai_venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =====================================
# FUNCTIONS
# =====================================

# Kill processes on specific port
kill_port() {
    local port=$1
    local name=$2
    local pids=$(lsof -t -i:$port 2>/dev/null)
    
    if [ ! -z "$pids" ]; then
        echo -e "${YELLOW}🔄 Stopping $name on port $port...${NC}"
        for pid in $pids; do
            kill -9 $pid 2>/dev/null
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Killed PID $pid for $name${NC}"
            fi
        done
        sleep 1
    else
        echo -e "${YELLOW}⚠️  No $name process found on port $port${NC}"
    fi
}

# Check if command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo -e "${RED}❌ Error: $1 is not installed${NC}"
        echo "Install it with: sudo apt install $1"
        exit 1
    fi
}

# Activate virtual environment
activate_venv() {
    local env_name=$1
    local env_path=$(pwd)/$env_name
    
    if [ -d "$env_path" ]; then
        if [ -f "$env_path/bin/activate" ]; then
            source "$env_path/bin/activate"
            echo -e "${GREEN}✅ Activated virtual environment: $env_name${NC}"
        else
            echo -e "${RED}❌ Virtual environment $env_path/bin/activate not found${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠️  Virtual environment $env_path not found, skipping...${NC}"
    fi
}

# =====================================
# MAIN LOGIC
# =====================================

# Check required commands
check_command "./cloud-sql-proxy"
check_command "uvicorn"
check_command "streamlit"
check_command "lsof"

# Parse arguments
ACTION=${1:-start}
shift

case $ACTION in
    "start")
        echo -e "${GREEN}🚀 Starting RAG Services...${NC}"
        ;;
    "stop")
        echo -e "${YELLOW}🛑 Stopping RAG Services...${NC}"
        kill_port $PROXY_PORT "Cloud SQL Proxy"
        kill_port $UVICORN_PORT "Uvicorn"
        kill_port $STREAMLIT_PORT "Streamlit"
        echo -e "${GREEN}✅ All services stopped${NC}"
        exit 0
        ;;
    "restart")
        echo -e "${YELLOW}🔄 Restarting RAG Services...${NC}"
        kill_port $PROXY_PORT "Cloud SQL Proxy"
        kill_port $UVICORN_PORT "Uvicorn"
        kill_port $STREAMLIT_PORT "Streamlit"
        sleep 3
        ;;
    "status")
        echo -e "${GREEN}📊 RAG Services Status:${NC}"
        echo "Cloud SQL Proxy (port $PROXY_PORT): $([ ! -z "$(lsof -t -i:$PROXY_PORT)" ] && echo "🟢 Running" || echo "🔴 Stopped")"
        echo "Uvicorn (port $UVICORN_PORT): $([ ! -z "$(lsof -t -i:$UVICORN_PORT)" ] && echo "🟢 Running" || echo "🔴 Stopped")"
        echo "Streamlit (port $STREAMLIT_PORT): $([ ! -z "$(lsof -t -i:$STREAMLIT_PORT)" ] && echo "🟢 Running" || echo "🔴 Stopped")"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Invalid action: $ACTION${NC}"
        echo "Usage: $0 [start|stop|restart|status]"
        exit 1
        ;;
esac

# =====================================
# START SERVICES IN ORDER
# =====================================

# 1. Start Cloud SQL Proxy
echo -e "${YELLOW}🌐 Starting Cloud SQL Proxy...${NC}"
./cloud-sql-proxy $PROXY_INSTANCE --port $PROXY_PORT &
PROXY_PID=$!
echo -e "${GREEN}✅ Cloud SQL Proxy started (PID: $PROXY_PID) on port $PROXY_PORT${NC}"

# Wait for proxy to be ready
echo -e "${YELLOW}⏳ Waiting for database connection...${NC}"
sleep 5

# Test database connection (optional)
if command -v psql &> /dev/null; then
    if PGPASSWORD=your_password ::q!psql -h localhost -p $PROXY_PORT -U your_username -d your_database -c "SELECT 1;" &> /dev/null; then
        echo -e "${GREEN}✅ Database connection verified${NC}"
    else
        echo -e "${YELLOW}⚠️  Database connection test failed (check credentials)${NC}"
    fi
fi

# 2. Start Uvicorn (FastAPI/RAG service)
echo -e "${YELLOW}⚙️  Starting Uvicorn (FastAPI)...${NC}"
activate_venv "$UVICORN_ENV"
cd "$(dirname "$0")"  # Ensure we're in project root

# Start uvicorn in background
uvicorn $UVICORN_APP --host $UVICORN_HOST --port $UVICORN_PORT --reload &
UVICORN_PID=$!
echo -e "${GREEN}✅ Uvicorn started (PID: $UVICORN_PID) on http://localhost:$UVICORN_PORT${NC}"

# Wait for API to be ready
sleep 3
if curl -f http://localhost:$UVICORN_PORT/health 2>/dev/null || curl -f http://localhost:$UVICORN_PORT/ 2>/dev/null; then
    echo -e "${GREEN}✅ API health check passed${NC}"
else
    echo -e "${YELLOW}⚠️  API health check failed (check your /health endpoint)${NC}"
fi

# 3. Start Streamlit (UI)
echo -e "${YELLOW}🎨 Starting Streamlit UI...${NC}"
deactivate  # Exit uvicorn venv if we activated it
activate_venv "$STREAMLIT_ENV"

# Start Streamlit in foreground (so script doesn't exit)
echo -e "${GREEN}🌟 Streamlit UI ready at http://localhost:$STREAMLIT_PORT${NC}"
echo -e "${GREEN}📱 API available at http://localhost:$UVICORN_PORT${NC}"
echo -e "${GREEN}💾 Database via proxy on port $PROXY_PORT${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop Streamlit (other services will continue)${NC}"

streamlit run $STREAMLIT_APP \
    --server.port $STREAMLIT_PORT \
    --server.address $STREAMLIT_HOST \
    --server.headless true \
    --logger.level info

# Cleanup on exit
echo -e "${YELLOW}🛑 Streamlit stopped, cleaning up...${NC}"
kill $PROXY_PID $UVICORN_PID 2>/dev/null
echo -e "${GREEN}✅ All services stopped${NC}"
