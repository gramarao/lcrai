"""
Initialize feedback configuration in database
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from app.models.database import SystemConfig

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def setup_feedback_config():
    """Add feedback configuration to system_config table"""
    db = Session(engine)
    
    try:
        # Check if system_config table exists
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'system_config'
                )
            """))
            table_exists = result.fetchone()[0]
            
            if not table_exists:
                print("❌ system_config table doesn't exist!")
                print("Run: python migrate_db.py first")
                return
        
        print("✅ system_config table exists")
        
        # Add feedback_enabled config
        fb_enabled = db.query(SystemConfig).filter(
            SystemConfig.key == "feedback_enabled"
        ).first()
        
        if not fb_enabled:
            fb_enabled = SystemConfig(
                key="feedback_enabled",
                value="True",
                description="Enable feedback-enhanced retrieval"
            )
            db.add(fb_enabled)
            print("✅ Added feedback_enabled config")
        else:
            print("ℹ️  feedback_enabled already exists")
        
        # Add feedback_weight config
        fb_weight = db.query(SystemConfig).filter(
            SystemConfig.key == "feedback_weight"
        ).first()
        
        if not fb_weight:
            fb_weight = SystemConfig(
                key="feedback_weight",
                value="0.3",
                description="Weight of feedback in retrieval (0.0-1.0)"
            )
            db.add(fb_weight)
            print("✅ Added feedback_weight config")
        else:
            print("ℹ️  feedback_weight already exists")
        
        db.commit()
        print("\n✅ Feedback configuration initialized successfully!")
        
        # Display current config
        print("\n📊 Current Configuration:")
        configs = db.query(SystemConfig).all()
        for config in configs:
            print(f"  • {config.key}: {config.value}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    setup_feedback_config()
