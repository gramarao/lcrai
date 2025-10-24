-- 1. Enable pgvector (requires superuser)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create tables manually (if needed)
-- Run this only if Base.metadata.create_all() fails

-- System config table
CREATE TABLE IF NOT EXISTS system_config (
    id SERIAL PRIMARY KEY,
    key VARCHAR UNIQUE NOT NULL,
    value TEXT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Documents table
CREATE TABLE IF NOT EXISTS rag_documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR UNIQUE NOT NULL,
    file_type VARCHAR,
    file_size INTEGER,
    content_hash VARCHAR UNIQUE,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR DEFAULT 'processed'
);

-- Document chunks with vectors
CREATE TABLE IF NOT EXISTS rag_document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES rag_documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),  -- Gemini embedding dimension
    feedback_score FLOAT DEFAULT 0.0,
    retrieval_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat sessions
CREATE TABLE IF NOT EXISTS rag_chat_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chat messages
CREATE TABLE IF NOT EXISTS rag_chat_messages (
    id SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES rag_chat_sessions(id) ON DELETE CASCADE,
    query TEXT NOT NULL,
    response TEXT,
    model_used VARCHAR,
    response_time_ms FLOAT,
    quality_score FLOAT,
    user_rating INTEGER,
    user_feedback TEXT,
    feedback_timestamp TIMESTAMP,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Message sources
CREATE TABLE IF NOT EXISTS rag_message_sources (
    id SERIAL PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES rag_chat_messages(id) ON DELETE CASCADE,
    chunk_id INTEGER NOT NULL REFERENCES rag_document_chunks(id) ON DELETE CASCADE,
    source_name VARCHAR NOT NULL,
    similarity_score FLOAT,
    rank_position INTEGER
);

-- Model feedback
CREATE TABLE IF NOT EXISTS rag_model_feedback (
    id SERIAL PRIMARY KEY,
    question_index INTEGER NOT NULL,
    question TEXT NOT NULL,
    model_name VARCHAR NOT NULL,
    rating INTEGER NOT NULL,
    comment TEXT,
    comparison_session VARCHAR,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Create indexes
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON rag_document_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_session_id ON rag_chat_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_timestamp ON rag_chat_messages(session_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status ON rag_documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON rag_documents(filename);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON rag_document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_message_sources_message_id ON rag_message_sources(message_id);

-- 4. Insert default config
INSERT INTO system_config (key, value, description, create_at, updated_at)
VALUES ('selected_model', 'gemini-2.5-flash-lite', 'Default Gemini model for RAG', NOW(), NOW())
ON CONFLICT (key) DO NOTHING;
