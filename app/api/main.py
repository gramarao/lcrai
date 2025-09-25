from fastapi import FastAPI, APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError
from app.models.database import Document, DocumentChunk, get_db
import logging
import time
from datetime import datetime
from app.services.rag_service_googleai import rag_service
import json

# Initialize FastAPI app
app = FastAPI(title="RAG Application")
router = APIRouter()
logger = logging.getLogger(__name__)
start_time = time.time()

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("""
            SELECT 
                (SELECT COUNT(*) FROM documents) AS doc_count,
                (SELECT COUNT(*) FROM document_chunks WHERE embedding IS NOT NULL) AS chunk_count,
                pg_size_pretty(pg_total_relation_size('document_chunks')) AS chunk_size
        """)).first()
        
        doc_count, chunk_count, chunk_size = result if result else (0, 0, "0 B")
        logger.debug(f"Health check: {doc_count} docs, {chunk_count} chunks, size {chunk_size}")
        
        return {
            "status": "healthy",
            "service": "RAG",
            "timestamp": datetime.now().isoformat(),
            "document_count": doc_count,
            "chunk_count": chunk_count,
            "embedding_storage": chunk_size or "0 B",
            "uptime": time.time() - start_time
        }
    except SQLAlchemyError as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/documents")
async def get_documents(db: Session = Depends(get_db)):
    try:
        documents = db.query(Document).all()
        return [
            {
                "id": str(doc.id),  # Convert UUID to string
                "filename": doc.filename,
                "created_at": doc.created_at.isoformat(),
                "content_length": len(doc.content or "")  # Add content length
            } for doc in documents
        ]
    except Exception as e:
        logger.error(f"Failed to fetch documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch documents: {str(e)}")

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        doc_id = rag_service.store_document(file.filename, content)
        return {"document_id": str(doc_id)}  # Convert UUID to string
    except Exception as e:
        logger.error(f"Failed to upload document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, db: Session = Depends(get_db)):
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
        db.query(Document).filter(Document.id == doc_id).delete()
        db.commit()
        logger.info(f"Deleted document ID: {doc_id}")
        return {"message": "Document deleted successfully"}
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to delete document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@router.post("/query")
async def query(payload: dict, db: Session = Depends(get_db)):
    try:
        question = payload.get("question")
        doc_ids = payload.get("doc_ids", [])
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
        result = await rag_service.search_and_generate(question, max_sources=3, doc_ids=doc_ids)
        return result
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.post("/query_stream")
async def query_stream(payload: dict, db: Session = Depends(get_db)):
    async def stream_generator():
        try:
            question = payload.get("question")
            doc_ids = payload.get("doc_ids", [])
            if not question:
                raise HTTPException(status_code=400, detail="Question is required")
            async for chunk in rag_service.stream_search_and_generate(question, max_sources=3, doc_ids=doc_ids):
                yield json.dumps(chunk) + "\n"  # Add newline for streaming
        except Exception as e:
            logger.error(f"Streaming query failed: {str(e)}")
            error_chunk = {"type": "error", "message": f"Stream failed: {str(e)}", "is_complete": True}
            yield json.dumps(error_chunk) + "\n"
            yield json.dumps({"type": "stream_end", "timestamp": datetime.now().isoformat()}) + "\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

# Include router in the app
app.include_router(router, prefix="/api")