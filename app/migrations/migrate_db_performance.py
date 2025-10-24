# Add to migrate_database.py

def create_performance_indexes(db):
    """Create indexes for better query performance"""
    try:
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
