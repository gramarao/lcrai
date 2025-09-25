from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
from datetime import datetime
import os
import time
from config.settings import settings  # Import settings

# Database configuration
DATABASE_URL = settings.DATABASE_URL  # Use settings.py for consistency

# Create engine with connection pooling
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
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    doc_metadata = Column(Text)  # JSON string for additional metadata
    content_hash = Column(String(32), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(Vector(768))  # text-embedding-004 produces 768-dim vectors
    doc_metadata = Column(Text)  # JSON string for additional metadata
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationship to document
    document = relationship("Document", back_populates="chunks")
    
    # Create index for vector similarity search
    __table_args__ = (
        {"extend_existing": True}
    )

# Create tables
def create_tables():
    """Create all tables"""
    Base.doc_metadata.create_all(bind=engine)
    
    # Create vector extension and index
    with engine.connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("""
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
            ON document_chunks USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100)
        """)
        conn.commit()

if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully!")
