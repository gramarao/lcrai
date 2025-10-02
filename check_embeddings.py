# check_embeddings.py
import os
from sqlalchemy import create_engine, text


from app.models.database import SessionLocal, Document
from config.settings import settings

# Print DATABASE_URL for debugging
print(f"Using DATABASE_URL: {settings.DATABASE_URL}")

db = SessionLocal()


def check_embedding_dimensions():
    # Check embedding dimensions
    result = db.execute(text("""
        SELECT 
            id,
            array_length(embedding, 1) as dimension,
            created_at
        FROM document_chunks 
        WHERE embedding IS NOT NULL
        ORDER BY created_at DESC
        LIMIT 10
    """)).fetchall()
    
    print("Recent embeddings:")
    for row in result:
        print(f"ID: {row.id}, Dimension: {row.dimension}, Created: {row.created_at}")
    
    # Check for dimension inconsistencies
    dimension_summary = conn.execute(text("""
        SELECT 
            array_length(embedding, 1) as dimension,
            COUNT(*) as count
        FROM document_chunks
        WHERE embedding IS NOT NULL
        GROUP BY array_length(embedding, 1)
        ORDER BY count DESC
    """)).fetchall()
    
    print("\nDimension distribution:")
    for row in dimension_summary:
        print(f"Dimension {row.dimension}: {row.count} chunks")
        
    # Check for problematic embeddings
    bad_embeddings = conn.execute(text("""
        SELECT COUNT(*) as bad_count
        FROM document_chunks
        WHERE embedding IS NOT NULL 
        AND array_length(embedding, 1) != 768
    """)).fetchone()
    
    if bad_embeddings and bad_embeddings.bad_count > 0:
        print(f"\n⚠️  WARNING: Found {bad_embeddings.bad_count} chunks with incorrect dimensions!")
        return False
    else:
        print("\n✅ All embeddings have correct dimensions")
        return True

if __name__ == "__main__":
    check_embedding_dimensions()
