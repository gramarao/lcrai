import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

class Settings:
    # Google Cloud Configuration
    DB_USER = os.getenv("DB_USER", "postgres")  # Cloud SQL database user
    DB_PASSWORD = os.getenv("DB_PASSWORD", "lcrai_2025")  # Set this in environment or directly
    DB_HOST = os.getenv("DB_HOST", "localhost")  # Proxy host
    DB_PORT = os.getenv("DB_PORT", "5432")  # Proxy port
    DB_NAME = os.getenv("DB_NAME", "ragdb")  # Database name
    GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "lcrai-472418")
    GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")
    
    # Database Configuration
    DATABASE_URL: str = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # RAG Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    MAX_SOURCES: int = int(os.getenv("MAX_SOURCES", "5"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()