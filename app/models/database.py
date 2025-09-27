from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from pgvector.sqlalchemy import Vector
from datetime import datetime
import os
import time
from config.settings import settings
import uuid

# Database configuration
DATABASE_URL = settings.DATABASE_URL
engine = create_engine(DATABASE_URL, pool_size=10, max_overflow=20, pool_timeout=30, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

start_time = time.time()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    filename = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(Text)
    content_hash = Column(String(32), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(768))
    doc_metadata = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    
    document = relationship("Document", back_populates="chunks")
    
    __table_args__ = ({"extend_existing": True},)

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String, index=True)
    query_id = Column(String, index=True)
    question = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    rating = Column(Integer)
    thumbs_up = Column(Boolean)
    feedback_text = Column(Text)
    relevance_score = Column(Float, default=0.0)
    accuracy_score = Column(Float, default=0.0)
    completeness_score = Column(Float, default=0.0)
    overall_quality = Column(Float, default=0.0)
    response_time = Column(Float)
    tokens_used = Column(Integer)
    sources_count = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)
    
    sources = relationship("FeedbackSource", back_populates="feedback")

class FeedbackSource(Base):
    __tablename__ = "feedback_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    feedback_id = Column(UUID(as_uuid=True), ForeignKey("user_feedback.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.id"), nullable=False)
    relevance_score = Column(Float, default=0.0)
    
    feedback = relationship("UserFeedback", back_populates="sources")
    document = relationship("Document")
    chunk = relationship("DocumentChunk")

class QualityMetrics(Base):
    __tablename__ = "quality_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.now, index=True)
    avg_rating = Column(Float, default=0.0)
    total_queries = Column(Integer, default=0)
    positive_feedback_ratio = Column(Float, default=0.0)
    avg_response_time = Column(Float, default=0.0)
    avg_relevance_score = Column(Float, default=0.0)
    daily_active_users = Column(Integer, default=0)
    total_documents_used = Column(Integer, default=0)

class QueryPerformance(Base):
    __tablename__ = "query_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    query_hash = Column(String, unique=True, index=True)
    query_pattern = Column(String, index=True)
    avg_rating = Column(Float, default=0.0)
    total_queries = Column(Integer, default=0)
    success_rate = Column(Float, default=0.0)
    best_documents = Column(JSON)
    optimal_sources_count = Column(Integer, default=3)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
    
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
            ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100)
        """))
        conn.commit()

if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully!")