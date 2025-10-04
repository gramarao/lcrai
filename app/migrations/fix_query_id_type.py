"""
Fix query_id column type in user_feedback table
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def fix_query_id_type():
    """Convert query_id from VARCHAR to UUID"""
    with engine.connect() as conn:
        try:
            # Check current type
            result = conn.execute(text("""
                SELECT data_type 
                FROM information_schema.columns 
                WHERE table_name = 'user_feedback' 
                AND column_name = 'query_id'
            """))
            
            current_type = result.fetchone()
            print(f"Current query_id type: {current_type[0] if current_type else 'Not found'}")
            
            if current_type and current_type[0] != 'uuid':
                print("Converting query_id to UUID type...")
                
                # First, ensure all existing values are valid UUIDs or NULL
                conn.execute(text("""
                    UPDATE user_feedback 
                    SET query_id = NULL 
                    WHERE query_id !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                """))
                
                # Drop unique constraint if exists
                conn.execute(text("""
                    ALTER TABLE user_feedback 
                    DROP CONSTRAINT IF EXISTS user_feedback_query_id_key
                """))
                
                # Convert column type
                conn.execute(text("""
                    ALTER TABLE user_feedback 
                    ALTER COLUMN query_id TYPE UUID USING query_id::UUID
                """))
                
                # Re-add unique constraint
                conn.execute(text("""
                    ALTER TABLE user_feedback 
                    ADD CONSTRAINT user_feedback_query_id_key UNIQUE (query_id)
                """))
                
                conn.commit()
                print("✅ query_id converted to UUID successfully!")
            else:
                print("✅ query_id is already UUID type")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            conn.rollback()

if __name__ == "__main__":
    fix_query_id_type()
