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

if __name__ == "__main__":
    migrate()
