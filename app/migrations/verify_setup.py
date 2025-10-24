#!/usr/bin/env python3
"""
Diagnostic script to verify database setup
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect

load_dotenv()

def verify_setup():
    """Verify database setup step by step"""
    
    print("🔍 Database Setup Verification")
    print("=" * 60)
    
    # 1. Check environment variables
    print("\n1. Environment Variables:")
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        # Mask password for security
        masked_url = db_url.split('@')[0].split(':')[0] + ':****@' + db_url.split('@')[1] if '@' in db_url else db_url
        print(f"   DATABASE_URL: {masked_url}")
    else:
        print("   ❌ DATABASE_URL not set")
        return False
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    print(f"   GEMINI_API_KEY: {'✅ Set' if gemini_key else '❌ Not set'}")
    
    # 2. Test database connection
    print("\n2. Database Connection:")
    try:
        from database import engine, SessionLocal
        db = SessionLocal()
        db.execute(text("SELECT version();"))
        result = db.execute(text("SELECT version();")).fetchone()
        print(f"   ✅ Connected to: {result[0].split(',')[0]}")
        db.close()
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return False
    
    # 3. Check pgvector extension
    print("\n3. pgvector Extension:")
    try:
        db = SessionLocal()
        result = db.execute(text(
            "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector');"
        )).scalar()
        
        if result:
            print("   ✅ pgvector extension is installed")
        else:
            print("   ❌ pgvector extension not found")
            print("   Try: CREATE EXTENSION vector;")
        db.close()
    except Exception as e:
        print(f"   ❌ Check failed: {e}")
        return False
    
    # 4. Check existing tables
    print("\n4. Existing Tables:")
    try:
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        print(f"   Total tables: {len(all_tables)}")
        
        rag_tables = [t for t in all_tables if t.startswith('rag_') or t == 'system_config']
        if rag_tables:
            print(f"   RAG tables: {', '.join(rag_tables)}")
        else:
            print("   No RAG tables found (this is OK for first run)")
            
    except Exception as e:
        print(f"   ❌ Check failed: {e}")
        return False
    
    # 5. Check if database.py has required models
    print("\n5. Database Models:")
    try:
        import database as db_models
        
        required_models = [
            'SystemConfig',
            'Document', 
            'DocumentChunk',
            'ChatSession',
            'ChatMessage',
            'MessageSource',
            'ModelFeedback'
        ]
        
        for model_name in required_models:
            if hasattr(db_models, model_name):
                print(f"   ✅ {model_name}")
            else:
                print(f"   ❌ {model_name} not found")
                
    except Exception as e:
        print(f"   ❌ Check failed: {e}")
        return False
    
    # 6. Check Base metadata
    print("\n6. SQLAlchemy Metadata:")
    try:
        from database import Base
        table_names = [table.name for table in Base.metadata.sorted_tables]
        print(f"   Registered tables: {len(table_names)}")
        rag_registered = [t for t in table_names if t.startswith('rag_') or t == 'system_config']
        if rag_registered:
            print(f"   RAG tables registered: {', '.join(rag_registered)}")
        else:
            print("   ⚠️  No RAG tables registered in metadata")
            
    except Exception as e:
        print(f"   ❌ Check failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ Verification complete!")
    print("\nIf all checks passed, run: python migrate_database.py")
    return True

if __name__ == "__main__":
    verify_setup()
