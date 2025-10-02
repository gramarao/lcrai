from fastapi import FastAPI, APIRouter, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool
import asyncio
import time
import logging
from datetime import datetime
from app.models.database import (
    Document, DocumentChunk, UserFeedback, FeedbackSource, 
    QueryPerformance, get_db, Base, engine
)
from app.services.rag_service_googleai import rag_service
import json
import uuid

# Pydantic V2 imports
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Application")
router = APIRouter()
start_time = time.time()

# Pydantic V2 Models
class FeedbackSourceSchema(BaseModel):
    document_id: str = Field(..., description="Document UUID as string")
    id: str = Field(..., description="Chunk UUID as string")
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    filename: Optional[str] = ""
    chunk_index: Optional[int] = 0
    
    @field_validator('document_id', 'id')
    @classmethod
    def validate_uuid_strings(cls, v: str) -> str:
        """Validate UUID format for document_id and id fields"""
        if not v or not str(v).strip():
            raise ValueError("UUID field cannot be empty")
        try:
            # Test if it's a valid UUID
            uuid.UUID(str(v).strip())
            return str(v).strip()
        except ValueError:
            raise ValueError(f"Invalid UUID format: '{v}'")
    
    @field_validator('relevance_score')
    @classmethod
    def validate_relevance_score(cls, v: float) -> float:
        """Ensure relevance score is between 0 and 1"""
        if not isinstance(v, (int, float)):
            raise ValueError("Relevance score must be a number")
        return float(v)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "4b4066bd-d053-43b3-8833-9a88c0ef7ea0",
                "id": "4d8875a5-8324-41c4-80bf-dd088b277aae",
                "relevance_score": 0.85,
                "filename": "document.pdf",
                "chunk_index": 1
            }
        }
    }

class QualityScoresSchema(BaseModel):
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence: float = Field(default=0.0, ge=0.0, le=1.0)
    citation: float = Field(default=0.0, ge=0.0, le=1.0)
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @field_validator('relevance', 'accuracy', 'completeness', 'coherence', 'citation', 'overall')
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Ensure all scores are between 0 and 1"""
        if not isinstance(v, (int, float)):
            raise ValueError("Score must be a number")
        return float(max(0.0, min(1.0, v)))

class FeedbackRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Session identifier")
    query_id: str = Field(..., min_length=1, description="Query identifier")
    question: str = Field(..., min_length=1, description="User question")
    response: str = Field(..., min_length=1, description="Assistant response")
    rating: Optional[int] = Field(None, ge=1, le=5, description="Rating from 1 to 5")
    thumbs_up: Optional[bool] = None
    feedback_text: Optional[str] = ""
    response_time: Optional[float] = Field(None, ge=0.0)
    tokens_used: Optional[int] = Field(None, ge=0)
    sources_count: Optional[int] = Field(None, ge=0)
    sources: List[FeedbackSourceSchema] = Field(default_factory=list)
    quality_scores: Optional[QualityScoresSchema] = None
    
    @field_validator('query_id')
    @classmethod
    def validate_query_id_uuid(cls, v: str) -> str:
        """Validate query_id is a valid UUID"""
        if not v or not v.strip():
            raise ValueError("Query ID cannot be empty")
        try:
            uuid.UUID(v.strip())
            return str(v).strip()
        except ValueError:
            raise ValueError(f"query_id must be a valid UUID, got: '{v}'")
    
    @field_validator('feedback_text', mode='before')
    @classmethod
    def handle_none_feedback_text(cls, v) -> str:
        """Convert None feedback_text to empty string"""
        return v if v is not None else ""
    
    @field_validator('response_time', mode='before')
    @classmethod
    def validate_response_time(cls, v) -> Optional[float]:
        """Validate response_time is a positive number"""
        if v is None:
            return None
        try:
            val = float(v)
            return val if val >= 0 else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    @field_validator('tokens_used', mode='before')
    @classmethod
    def validate_tokens_used(cls, v) -> Optional[int]:
        """Validate tokens_used is a non-negative integer"""
        if v is None:
            return None
        try:
            val = int(v)
            return val if val >= 0 else 0
        except (ValueError, TypeError):
            return 0
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "session-123",
                "query_id": "550e8400-e29b-41d4-a716-446655440000",
                "question": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence...",
                "rating": 4,
                "thumbs_up": True,
                "feedback_text": "Good explanation",
                "response_time": 2.5,
                "tokens_used": 150,
                "sources_count": 2,
                "sources": [
                    {
                        "document_id": "4b4066bd-d053-43b3-8833-9a88c0ef7ea0",
                        "id": "4d8875a5-8324-41c4-80bf-dd088b277aae",
                        "relevance_score": 0.85,
                        "filename": "ml_guide.pdf",
                        "chunk_index": 0
                    }
                ],
                "quality_scores": {
                    "relevance": 0.8,
                    "accuracy": 0.9,
                    "completeness": 0.7,
                    "coherence": 0.8,
                    "citation": 0.6,
                    "overall": 0.76
                }
            }
        }
    }

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    doc_ids: List[str] = Field(default_factory=list, description="Document IDs to search in")
    session_id: Optional[str] = None
    query_id: Optional[str] = None
    
    @field_validator('doc_ids')
    @classmethod
    def validate_doc_ids(cls, v: List[str]) -> List[str]:
        """Validate document IDs are valid UUIDs"""
        if not v:
            return []
        
        valid_ids = []
        for doc_id in v:
            if doc_id and str(doc_id).strip():
                try:
                    uuid.UUID(str(doc_id).strip())
                    valid_ids.append(str(doc_id).strip())
                except ValueError:
                    # Skip invalid UUIDs but don't fail
                    logger.warning(f"Invalid document ID skipped: {doc_id}")
                    continue
        
        return valid_ids

# Add request timing middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    method = request.method
    path = request.url.path
    
    try:
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(f"{method} {path} -> {response.status_code} ({duration:.1f}ms)")
        return response
    except Exception as e:
        duration = (time.time() - start) * 1000
        logger.exception(f"ERROR {method} {path} after {duration:.1f}ms: {e}")
        raise

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/checked.")

@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("SELECT 1")).first()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - start_time,
            "rag_service": "initialized" if rag_service else "failed"
        }
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    """Get all documents"""
    logger.info("Fetching documents (sync)")
    try:
        start_time = time.time()
        
        documents = db.query(
            Document.id,
            Document.filename, 
            Document.created_at,
            Document.content
        ).all()
        
        result = [
            {
                "id": str(doc.id),
                "filename": doc.filename,
                "created_at": doc.created_at.isoformat(),
                "content_length": len(doc.content or "")
            } for doc in documents
        ]
        
        duration = time.time() - start_time
        logger.info(f"Fetched {len(documents)} documents in {duration:.2f}s")
        return result
        
    except Exception as e:
        logger.exception("Failed to fetch documents")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    
    logger.info(f"Uploading document: {file.filename}")
    try:
        content = await file.read()
        doc_id = await run_in_threadpool(rag_service.store_document, file.filename, content)
        logger.info(f"Document uploaded: {file.filename} -> {doc_id}")
        return {"document_id": str(doc_id)}
        
    except Exception as e:
        logger.exception(f"Upload failed: {file.filename}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    logger.info(f"Deleting document: {doc_id}")
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        
        db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
        db.query(FeedbackSource).filter(FeedbackSource.document_id == doc_id).delete()
        db.query(Document).filter(Document.id == doc_id).delete()
        
        db.commit()
        logger.info(f"Document deleted: {doc_id}")
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        db.rollback()
        logger.exception(f"Delete failed: {doc_id}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query(payload: QueryRequest):
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    if not payload.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    logger.info(f"Processing query: {payload.question[:50]}...")
    
    def sync_search_and_generate():
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                rag_service.search_and_generate(
                    payload.question, 
                    max_sources=3, 
                    doc_ids=payload.doc_ids
                )
            )
        finally:
            loop.close()
    
    result = await run_in_threadpool(sync_search_and_generate)
    logger.info(f"Query completed: {len(result.get('sources', []))} sources")
    return result

@router.post("/query_stream")
async def query_stream(payload: QueryRequest):
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    if not payload.question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    logger.info(f"Starting stream: {payload.question[:50]}... (query_id: {payload.query_id})")
    
    async def stream_generator():
        try:
            chunk_count = 0
            async for chunk in rag_service.stream_search_and_generate(
                payload.question,
                max_sources=3,
                doc_ids=payload.doc_ids,
                query_id=payload.query_id,
                session_id=payload.session_id
            ):
                chunk_count += 1
                if chunk.get("type") == "complete":
                    sources_count = len(chunk.get("sources", []))
                    quality = chunk.get("quality_scores", {}).get("overall", 0)
                    logger.info(f"Stream complete: {sources_count} sources, quality: {quality:.2f}")
                
                yield json.dumps(chunk) + "\n"
            
            logger.info(f"Stream finished: {chunk_count} chunks sent")
            
        except Exception as e:
            logger.exception(f"Stream failed: {payload.query_id}")
            error_chunk = {
                "type": "error", 
                "message": f"Stream failed: {str(e)}", 
                "is_complete": True
            }
            yield json.dumps(error_chunk) + "\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, db: Session = Depends(get_db)):
    """Submit feedback with Pydantic V2 validation"""
    logger.info(f"Processing feedback for query_id: {feedback.query_id}")
    
    try:
        # Validate query_id UUID
        try:
            query_uuid = uuid.UUID(feedback.query_id)
        except ValueError as e:
            logger.error(f"Invalid query_id UUID: {feedback.query_id}")
            raise HTTPException(status_code=422, detail=f"Invalid query_id format: {e}")
        
        # Create feedback entry
        new_feedback = UserFeedback(
            session_id=feedback.session_id,
            query_id=query_uuid,
            question=feedback.question,
            response=feedback.response,
            rating=feedback.rating,
            thumbs_up=feedback.thumbs_up,
            feedback_text=feedback.feedback_text,
            response_time=feedback.response_time,
            tokens_used=feedback.tokens_used,
            sources_count=feedback.sources_count,
            created_at=datetime.now()
        )
        
        # Add quality scores if provided
        if feedback.quality_scores:
            new_feedback.relevance_score = feedback.quality_scores.relevance
            new_feedback.accuracy_score = feedback.quality_scores.accuracy
            new_feedback.completeness_score = feedback.quality_scores.completeness
            new_feedback.coherence_score = feedback.quality_scores.coherence
            new_feedback.citation_score = feedback.quality_scores.citation
            new_feedback.overall_quality_score = feedback.quality_scores.overall
            logger.info(f"Added quality scores: overall={new_feedback.overall_quality_score:.2f}")
        
        db.add(new_feedback)
        db.flush()
        
        # Add sources
        sources_added = 0
        for i, source_data in enumerate(feedback.sources):
            try:
                feedback_source = FeedbackSource(
                    feedback_id=new_feedback.id,
                    document_id=uuid.UUID(source_data.document_id),
                    chunk_id=uuid.UUID(source_data.id),  # Use 'id' field
                    relevance_score=source_data.relevance_score
                )
                db.add(feedback_source)
                sources_added += 1
            except ValueError as e:
                logger.error(f"Invalid source UUID {i+1}: {e}")
                continue
        
        db.commit()
        logger.info(f"Feedback stored: {sources_added}/{len(feedback.sources)} sources")
        return {"message": "Feedback submitted successfully", "feedback_id": str(new_feedback.id)}
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception("Feedback storage failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedback_stats")
def feedback_stats(db: Session = Depends(get_db)):
    try:
        result = db.execute(text("""
            SELECT 
                COALESCE(AVG(rating), 0) as avg_rating,
                COALESCE(SUM(CASE WHEN thumbs_up THEN 1 ELSE 0 END)::float / NULLIF(COUNT(*), 0), 0) as positive_ratio,
                COALESCE(AVG(overall_quality_score), 0) as avg_quality,
                COUNT(*) as total_feedback
            FROM user_feedback
            WHERE rating IS NOT NULL
        """)).first()
        
        if result:
            avg_rating, positive_ratio, avg_quality, total_feedback = result
            return {
                "avg_rating": round(float(avg_rating or 0.0), 2),
                "positive_ratio": round(float(positive_ratio or 0.0), 2),
                "avg_quality": round(float(avg_quality or 0.0), 2),
                "total_feedback": int(total_feedback or 0)
            }
        return {"avg_rating": 0.0, "positive_ratio": 0.0, "avg_quality": 0.0, "total_feedback": 0}
        
    except Exception as e:
        logger.exception("Failed to fetch feedback stats")
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoint
@router.get("/debug/verify_methods")
def verify_methods():
    """Verify RAG service methods are available"""
    if not rag_service:
        return {"error": "RAG service is None"}
    
    result = {
        "service_type": str(type(rag_service)),
        "service_id": id(rag_service),
        "initialized": getattr(rag_service, '_initialized', False),
        "methods": {}
    }
    
    required_methods = ['stream_search_and_generate', 'search_and_generate', 'store_document']
    
    for method_name in required_methods:
        if hasattr(rag_service, method_name):
            method = getattr(rag_service, method_name)
            result["methods"][method_name] = {
                "exists": True,
                "callable": callable(method),
                "type": str(type(method))
            }
        else:
            result["methods"][method_name] = {"exists": False}
    
    return result

app.include_router(router, prefix="/api")
