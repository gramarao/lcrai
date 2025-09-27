# migrations/add_feedback_system.py
from app.models.database import engine, Base
from sqlalchemy import text

def upgrade():
    """Add feedback and quality tables with UUID consistency"""
    
    # Drop existing tables to avoid conflicts
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS feedback_sources CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS user_feedback CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS quality_metrics CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS query_performance CASCADE"))
        conn.commit()

    # Create new tables
    Base.metadata.create_all(bind=engine)
    
    # Add indexes for performance
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_feedback_session_created 
            ON user_feedback(session_id, created_at DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_feedback_rating_created 
            ON user_feedback(rating, created_at DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_feedback_quality_score 
            ON user_feedback(overall_quality DESC)
        """))
        
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_query_performance_pattern 
            ON query_performance(query_pattern, avg_rating DESC)
        """))
        
        conn.commit()
    
    print("✅ Feedback system tables created successfully!")

def downgrade():
    """Drop feedback and quality tables"""
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS feedback_sources CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS user_feedback CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS quality_metrics CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS query_performance CASCADE"))
        conn.commit()
    
    print("✅ Feedback system tables dropped successfully!")

if __name__ == "__main__":
    upgrade()