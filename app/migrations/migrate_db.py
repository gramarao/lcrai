"""
Create missing database tables
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from app.models.database import Base, SystemConfig, ModelFeedback

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found")

engine = create_engine(DATABASE_URL)

def migrate():
    """Create missing tables"""
    print("🔄 Creating missing tables...")
    
    # Enable pgvector extension
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    
    # Create all tables (only missing ones will be created)
    Base.metadata.create_all(bind=engine)
    
    print("✅ Database migration complete!")
    
    # Initialize default config
    from sqlalchemy.orm import Session
    db = Session(engine)
    
    try:
        config = db.query(SystemConfig).filter(
            SystemConfig.key == "selected_model"
        ).first()
        
        if not config:
            config = SystemConfig(
                key="selected_model",
                value="gemini-2.5-flash",
                description="Default Gemini model"
            )
            db.add(config)
            db.commit()
            print("✅ Default config initialized!")
        else:
            print("ℹ️  Config already exists")
    except Exception as e:
        print(f"❌ Config error: {e}")
        db.rollback()
    finally:
        db.close()

def create_performance_indexes():
    """Create indexes for better query performance"""
    try:
            # Initialize default config
        from sqlalchemy.orm import Session
        db = Session(engine)

        # Vector similarity index (HNSW for better performance)
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
            ON rag_document_chunks 
            USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 16, ef_construction = 64);
        """))
        
        # Session lookup index
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id 
            ON rag_chat_sessions(session_id);
        """))
        
        # Message timestamp index for history queries
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_timestamp 
            ON rag_chat_messages(session_id, timestamp DESC);
        """))
        
        # Document status index
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_documents_status 
            ON rag_documents(status);
        """))
        
        # Source lookup index
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_message_sources_message_id 
            ON rag_message_sources(message_id);
        """))
        
        db.commit()
        print("✅ Performance indexes created")
        
    except Exception as e:
        print(f"⚠️  Index creation warning: {e}")
        db.rollback()


if __name__ == "__main__":
    #migrate()
    create_performance_indexes()
