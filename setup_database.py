import psycopg2
from config.settings import settings

def setup_database():
    # Connect to Cloud SQL instance
    conn = psycopg2.connect(
        host="127.0.0.1",  # Use Cloud SQL Proxy
        database=settings.DB_NAME,
        user=settings.DB_USER,
        password=settings.DB_PASSWORD,
        port=5433
    )
    
    cur = conn.cursor()
    
    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    
    print("Database setup complete!")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    setup_database()
