#!/bin/bash

# manage-services.sh
# Combined script to start/stop services with logging and parameterized control

# Create logs directory if it doesn't exist
mkdir -p logs

# Log files
CLOUD_SQL_LOG="logs/cloud_sql.log"
UVICORN_LOG="logs/uvicorn.log"
STREAMLIT_LOG="logs/streamlit.log"

# Function to start services
start_service() {
    case $1 in
        cloud_sql)
            echo "Starting Cloud SQL Proxy..."
            nohup ./cloud-sql-proxy lcrai-472418:us-central1:rag-postgres --port 5432 2 >> "$CLOUD_SQL_LOG" 2>&1 &
            echo "Cloud SQL Proxy started (PID: $!). Check logs at $CLOUD_SQL_LOG"
            sleep 5  # Wait for Cloud SQL Proxy to initialize
            ;;
        uvicorn)
            echo "Starting Uvicorn (FastAPI)..."
            nohup uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120 >> "$UVICORN_LOG" 2>&1 &
            echo "Uvicorn started (PID: $!). Check logs at $UVICORN_LOG"
            sleep 3  # Wait for Uvicorn to initialize
            ;;
        streamlit)
            echo "Starting Streamlit..."
            nohup streamlit run streamlit_app_new_fb.py --server.port=8501 >> "$STREAMLIT_LOG" 2>&1 &
            echo "Streamlit started (PID: $!). Check logs at $STREAMLIT_LOG"
            sleep 3  # Wait for Streamlit to initialize
            ;;
        all)
            echo "Starting all services..."
            start_service cloud_sql
            start_service uvicorn
            start_service streamlit
            ;;
        *)
            echo "Invalid service: $1. Use 'cloud_sql', 'uvicorn', 'streamlit', or 'all'."
            exit 1
            ;;
    esac
}

# Function to stop services
stop_service() {
    case $1 in
        cloud_sql)
            echo "Stopping Cloud SQL Proxy..."
            pkill -f "cloud-sql-proxy.*rag-postgres" && echo "Cloud SQL Proxy stopped" || echo "No Cloud SQL Proxy process found"
            ;;
        uvicorn)
            echo "Stopping Uvicorn..."
            pkill -f "uvicorn.*app.api.main:app" && echo "Uvicorn stopped" || echo "No Uvicorn process found"
            ;;
        streamlit)
            echo "Stopping Streamlit..."
            pkill -f "streamlit.*streamlit_app__new_fb.py" && echo "Streamlit stopped" || echo "No Streamlit process found"
            ;;
        all)
            echo "Stopping all services..."
            stop_service cloud_sql
            stop_service uvicorn
            stop_service streamlit
            ;;
        *)
            echo "Invalid service: $1. Use 'cloud_sql', 'uvicorn', 'streamlit', or 'all'."
            exit 1
            ;;
    esac
}

# Main script
if [ $# -lt 2 ]; then
    echo "Usage: $0 {start|stop} {cloud_sql|uvicorn|streamlit|all}"
    exit 1
fi

ACTION=$1
SERVICE=$2

case $ACTION in
    start)
        start_service "$SERVICE"
        ;;
    stop)
        stop_service "$SERVICE"
        ;;
    *)
        echo "Invalid action: $ACTION. Use 'start' or 'stop'."
        exit 1
        ;;
esac

echo "Done."
