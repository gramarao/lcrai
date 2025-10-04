from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, Boolean, Integer, ForeignKey, Index
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from pydantic import BaseModel

import uuid
import os

from pgvector.sqlalchemy import Vector

# Database URL with connection pooling
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql://user:password@localhost:5432/rag_db"
)

# Critical: Gemini text-embedding-004 produces 768-dimensional vectors
EMBEDDING_DIMENSION = 768

# Enhanced engine configuration
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=20,           # Increase pool size for multiple workers
    max_overflow=30,        # Allow overflow connections
    pool_timeout=30,        # Connection timeout
    pool_recycle=1800,      # Recycle connections every 30 minutes
    pool_pre_ping=True,     # Validate connections
    echo=False              # Set to True only for debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    filename = Column(String, index=True, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(768), nullable=True)  # This should be Vector, not ARRAY(Float)
    chunk_index = Column(Integer, nullable=False)
    doc_metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    content_hash = Column(String(32))
    
    document = relationship("Document", back_populates="chunks")

class UserFeedback(Base):
    __tablename__ = "user_feedback"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    session_id = Column(String, index=True, nullable=False)
    query_id = Column(UUID(as_uuid=True), unique=True, nullable=False)
    question = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    rating = Column(Integer)  # 1-5 rating
    thumbs_up = Column(Boolean)
    feedback_text = Column(Text)
    response_time = Column(Float)
    tokens_used = Column(Integer)
    sources_count = Column(Integer)
    
    # Quality score fields
    relevance_score = Column(Float)
    accuracy_score = Column(Float)
    completeness_score = Column(Float)
    coherence_score = Column(Float)
    citation_score = Column(Float)
    overall_quality_score = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    sources = relationship("FeedbackSource", back_populates="feedback", cascade="all, delete-orphan")

class FeedbackSource(Base):
    __tablename__ = "feedback_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    feedback_id = Column(UUID(as_uuid=True), ForeignKey("user_feedback.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), nullable=False)
    relevance_score = Column(Float, default=0.0)
    
    feedback = relationship("UserFeedback", back_populates="sources")

class QueryPerformance(Base):
    __tablename__ = "query_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    query_hash = Column(String, index=True, nullable=False)
    query_pattern = Column(String, index=True, nullable=False)
    response_time = Column(Float)
    tokens_used = Column(Integer)
    sources_count = Column(Integer)
    quality_scores = Column(JSONB)
    best_documents = Column(ARRAY(UUID(as_uuid=True)))
    optimal_sources_count = Column(Integer, default=3)
    created_at = Column(DateTime, default=datetime.utcnow)

class SystemConfig(Base):
    """System configuration table for storing settings"""
    __tablename__ = "system_config"
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, nullable=False, index=True)
    value = Column(Text, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<SystemConfig(key={self.key}, value={self.value})>"

class QualityScores(BaseModel):
    relevance: float
    accuracy: float
    completeness: float
    coherence: float
    citation: float


class ModelFeedback(Base):
    """Model comparison feedback for tuning"""
    __tablename__ = "model_feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    question_index = Column(Integer, nullable=False)
    question = Column(Text, nullable=False)
    model_name = Column(String(255), nullable=False, index=True)
    rating = Column(Integer, nullable=False)  # 1-5 stars
    comment = Column(Text, nullable=True)
    comparison_session = Column(String(255), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes for analytics queries
    __table_args__ = (
        Index('idx_model_rating', 'model_name', 'rating'),
        Index('idx_comparison_session', 'comparison_session'),
    )
    
    def __repr__(self):
        return f"<ModelFeedback(model={self.model_name}, rating={self.rating})>"