from sqlalchemy import text
from app.models.database import engine

def create_contact_requests_table():
    """Create contact_requests table"""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS contact_requests (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                query_id UUID NOT NULL,
                name VARCHAR NOT NULL,
                email VARCHAR NOT NULL,
                phone VARCHAR,
                company VARCHAR,
                preferred_time VARCHAR NOT NULL,
                additional_notes TEXT,
                original_query TEXT NOT NULL,
                original_response TEXT,
                status VARCHAR DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT NOW(),
                contacted_at TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_contact_requests_status ON contact_requests(status);
            CREATE INDEX IF NOT EXISTS idx_contact_requests_created_at ON contact_requests(created_at);
            CREATE INDEX IF NOT EXISTS idx_contact_requests_email ON contact_requests(email);
        """))
        conn.commit()
        print("✓ contact_requests table created successfully")

if __name__ == "__main__":
    create_contact_requests_table()
