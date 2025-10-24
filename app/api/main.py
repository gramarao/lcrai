import os
import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, text, Integer
import numpy as np

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document as LC_Document
import uuid
from sqlalchemy import text, bindparam
# Import your existing database components
from app.models.database import SessionLocal, engine, Base
import app.models.database as db_models

# At the top of main.py, after imports
from collections import defaultdict
import threading

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from types import SimpleNamespace

from fastapi.security import OAuth2PasswordRequestForm

import logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)


from app.api.auth import ( 
    verify_password, get_password_hash, create_access_token, 
    get_current_active_user, require_role, ACCESS_TOKEN_EXPIRE_MINUTES
)


# Set seed for consistent detection
DetectorFactory.seed = 0


# Global cache for previous scores (thread-safe)
_score_cache = defaultdict(float)
_score_cache_lock = threading.Lock()


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Backend with PostgreSQL + pgvector",
    description="Production RAG system using Gemini and pgvector",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501", 
        "https://lcr-frontend-24415162879.us-central1.run.app",
        "*"  # Keep this for now during testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-pro")
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "5"))
EMBEDDING_DIMENSION = 768  # Gemini text-embedding-004

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def detect_language(text: str) -> str:
    """
    Detect language of input text
    Returns ISO 639-1 language code (e.g., 'en', 'es', 'fr', 'hi', 'zh-cn')
    """
    try:
        lang_code = detect(text)
        
        # Map common language codes to full names for better prompting
        language_map = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'ko': 'Korean',
            'zh-cn': 'Simplified Chinese',
            'zh-tw': 'Traditional Chinese',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'bn': 'Bengali',
            'ta': 'Tamil',
            'te': 'Telugu',
            'mr': 'Marathi',
            'ur': 'Urdu',
            'vi': 'Vietnamese',
            'th': 'Thai',
            'nl': 'Dutch',
            'pl': 'Polish',
            'tr': 'Turkish',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
        }
        
        language_name = language_map.get(lang_code, 'English')
        
        print(f"🌍 Detected language: {language_name} ({lang_code})")
        
        return language_name
        
    except LangDetectException as e:
        print(f"⚠️  Language detection failed: {e}, defaulting to English")
        return 'English'
    except Exception as e:
        print(f"⚠️  Unexpected error in language detection: {e}")
        return 'English'


def chunk_text(text: str, source: str) -> List[str]:
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if chunk.strip()]

def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from various file formats"""
    file_ext = os.path.splitext(filename)[1].lower()
    
    try:
        if file_ext in ['.txt', '.md', '.csv', '.json']:
            return file_content.decode('utf-8', errors='ignore')
        elif file_ext == '.pdf':
            try:
                import PyPDF2
                import io
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
            except ImportError:
                return file_content.decode('utf-8', errors='ignore')
        else:
            return file_content.decode('utf-8', errors='ignore')
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return ""


async def vector_similarity_search(
    query: str, 
    db: Session, 
    limit: int = RETRIEVE_K
) -> List[Tuple[db_models.DocumentChunk, float]]:
    """Optimized pgvector similarity search with caching and performance improvements"""
    try:
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Optimized query with explicit index usage
        # Using cosine distance for better semantic similarity
        results = db.execute(
            text("""
                SELECT 
                    dc.id,
                    dc.document_id,
                    dc.chunk_index,
                    dc.content,
                    dc.doc_metadata,
                    d.filename,
                    (dc.embedding <=> CAST(:query_embedding AS vector)) as distance
                FROM document_chunks dc
                JOIN documents d ON dc.document_id = d.id
                ORDER BY dc.embedding <=> CAST(:query_embedding AS vector)
                LIMIT :limit
            """),
            {
                "query_embedding": str(query_embedding),
                "limit": limit
            }
        ).fetchall()

        # Convert results to chunks with similarity scores
        processed_results = []
        for row in results:

            chunk = db_models.DocumentChunk()
            chunk.id = row.id
            chunk.document_id = row.document_id
            chunk.chunk_index = row.chunk_index
            chunk.content = row.content
            chunk.doc_metadata = row.doc_metadata
            chunk._filename = row.filename
            
            similarity = max(0.0, 1.0 - float(row.distance))
            processed_results.append((chunk, similarity))
            
        
        db.commit()
        return processed_results
        
    except Exception as e:
        print(f"Vector search error: {e}")
        db.rollback()
        return []



async def feedback_enhanced_search(
    query: str,
    db: Session,
    limit: int = RETRIEVE_K,
    feedback_weight: float = 0.3
) -> List[Tuple[db_models.DocumentChunk, float]]:
    """
    Enhanced vector search that incorporates user feedback
    """
    try:
        print(f"🔍 Starting feedback-enhanced search...")
        
        # Step 1: Get base similarity results
        query_embedding = embeddings.embed_query(query)
        
        results = db.query(
            db_models.DocumentChunk,
            (db_models.DocumentChunk.embedding.cosine_distance(query_embedding)).label("distance")
        ).join(
            db_models.Document
        ).order_by(
            text("distance")
        ).limit(limit * 3).all()  # Get 3x results for reranking
        
        print(f"✅ Retrieved {len(results)} candidate chunks")
        
        # Step 2: Get feedback scores for each chunk
        chunk_feedback_scores = {}
        
        for chunk, distance in results:
            # Query feedback for this chunk - FIXED
            feedback_stats = db.query(
                func.avg(db_models.UserFeedback.rating).label('avg_rating'),
                func.count(db_models.UserFeedback.id).label('feedback_count'),
                func.count(
                    func.nullif(db_models.UserFeedback.thumbs_up, False)
                ).label('thumbs_up_count')
            ).join(
                db_models.FeedbackSource,
                db_models.FeedbackSource.feedback_id == db_models.UserFeedback.id
            ).filter(
                db_models.FeedbackSource.chunk_id == chunk.id
            ).first()
            
            if feedback_stats and feedback_stats.feedback_count and feedback_stats.feedback_count > 0:
                # Calculate feedback score (0.0 to 1.0)
                avg_rating = float(feedback_stats.avg_rating or 0)
                feedback_count = int(feedback_stats.feedback_count or 0)
                thumbs_up_count = int(feedback_stats.thumbs_up_count or 0)
                
                # Normalize rating to 0-1 scale
                normalized_rating = (avg_rating - 1) / 4.0  # 1-5 scale to 0-1
                
                # Thumbs up ratio
                thumbs_up_ratio = thumbs_up_count / feedback_count if feedback_count > 0 else 0.5
                
                # Combined feedback score with confidence based on count
                confidence = min(feedback_count / 10.0, 1.0)  # Max confidence at 10 feedbacks
                feedback_score = (normalized_rating * 0.7 + thumbs_up_ratio * 0.3) * confidence
                
                chunk_feedback_scores[chunk.id] = {
                    'score': feedback_score,
                    'count': feedback_count,
                    'avg_rating': avg_rating,
                    'thumbs_up_count': thumbs_up_count
                }
                
                print(f"  📊 Chunk {str(chunk.id)[:8]}: feedback_score={feedback_score:.3f} (n={feedback_count}, avg={avg_rating:.1f}, 👍={thumbs_up_count})")
            else:
                # No feedback yet - neutral score
                chunk_feedback_scores[chunk.id] = {
                    'score': 0.5,  # Neutral
                    'count': 0,
                    'avg_rating': 3.0,
                    'thumbs_up_count': 0
                }
        
        # Step 3: Combine similarity with feedback
        enhanced_results = []
        
        for chunk, distance in results:
            base_similarity = max(0.0, 1.0 - distance)
            feedback_data = chunk_feedback_scores.get(chunk.id, {'score': 0.5})
            feedback_score = feedback_data['score']
            
            # Weighted combination
            enhanced_similarity = (
                (1.0 - feedback_weight) * base_similarity +
                feedback_weight * feedback_score
            )
            
            enhanced_results.append((
                chunk, 
                enhanced_similarity, 
                base_similarity, 
                feedback_score,
                feedback_data.get('count', 0)
            ))
        
        # Step 4: Re-rank by enhanced similarity
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        final_results = [
            (chunk, enhanced_sim) 
            for chunk, enhanced_sim, base_sim, fb_score, fb_count in enhanced_results[:limit]
        ]

        print(f"✅ Returning {len(final_results)} feedback-enhanced results")

        # Compare with previous scores
        with _score_cache_lock:
            for chunk, enhanced_sim, base_sim, fb_score, fb_count in enhanced_results[:limit]:
                chunk_key = str(chunk.id)
                prev_score = _score_cache.get(chunk_key, base_sim)
                score_change = enhanced_sim - prev_score
                
                # Update cache
                _score_cache[chunk_key] = enhanced_sim
                
                # Determine arrow based on change
                if abs(score_change) < 0.001:
                    arrow = "➡️"
                    change_text = "no change"
                elif score_change > 0:
                    arrow = "⬆️" if score_change < 0.05 else "⬆️⬆️"
                    change_text = f"+{score_change:.3f}"
                else:
                    arrow = "⬇️" if score_change > -0.05 else "⬇️⬇️"
                    change_text = f"{score_change:.3f}"
                
                print(f"  {arrow} {chunk.document.filename[:35]}: "
                      f"final={enhanced_sim:.3f} ({change_text} from previous)")        
        
        
        # Log the impact of feedback
        # Log the impact of feedback - IMPROVED LOGIC
        print("📊 Top results with feedback impact:")
        
        # Store previous scores for comparison (you'd need to persist this)
        # For now, we'll compare feedback contribution
        
        for i, (chunk, enhanced_sim, base_sim, fb_score, fb_count) in enumerate(enhanced_results[:limit], 1):
            fb_data = chunk_feedback_scores.get(chunk.id, {})
            boost = enhanced_sim - base_sim
            
            # NEW LOGIC: Show arrow based on feedback quality and count
            if fb_count == 0:
                boost_indicator = "🆕"  # New (no feedback yet)
                fb_status = "NEW"
            elif fb_score > 0.7:
                boost_indicator = "⬆️⬆️"  # Excellent feedback
                fb_status = "✅ EXCELLENT"
            elif fb_score > 0.5:
                boost_indicator = "⬆️"     # Good feedback
                fb_status = "✅ GOOD"
            elif fb_score > 0.3:
                boost_indicator = "➡️"     # Mixed feedback
                fb_status = "⚠️ MIXED"
            elif fb_score > 0.1:
                boost_indicator = "⬇️"     # Poor feedback
                fb_status = "⚠️ POOR"
            else:
                boost_indicator = "⬇️⬇️"  # Very poor feedback
                fb_status = "❌ BAD"
            
            # Calculate feedback strength (how much it matters)
            feedback_contribution = feedback_weight * fb_score
            similarity_contribution = (1.0 - feedback_weight) * base_sim
            
            print(f"  {i}. {boost_indicator} {fb_status} {chunk.document.filename[:35]}: "
                  f"final={enhanced_sim:.3f} "
                  f"(sim={base_sim:.3f}[{similarity_contribution:.3f}] + "
                  f"fb={fb_score:.3f}[{feedback_contribution:.3f}], "
                  f"n={fb_count}, avg={fb_data.get('avg_rating', 0):.1f}⭐)")
        
        return final_results
        
    except Exception as e:
        import traceback
        print(f"❌ Feedback-enhanced search error:")
        print(traceback.format_exc())
        print("⚠️  Falling back to basic vector search")
        # Fallback to basic search
        return await vector_similarity_search(query, db, limit)


async def compute_quality_metrics(
    query: str, 
    answer: str, 
    retrieved_chunks: List[Tuple[db_models.DocumentChunk, float]],
    db: Session
) -> Dict[str, Any]:
    """Compute quality metrics optimized for feedback tuning"""
    
    # 1. Retrieval relevance (average similarity)
    if retrieved_chunks:
        retrieval_relevance = float(np.mean([sim for _, sim in retrieved_chunks[:3]]))
    else:
        retrieval_relevance = 0.0
    
    # 2. Context coverage
    context_text = " ".join([chunk.content for chunk, _ in retrieved_chunks])
    answer_tokens = set(answer.lower().split())
    context_tokens = set(context_text.lower().split())
    
    coverage = len(answer_tokens & context_tokens) / len(answer_tokens) if answer_tokens else 0.0
    
    # 3. Source diversity
    unique_sources = len(set(chunk.document.filename for chunk, _ in retrieved_chunks if chunk.document))
    
    # 4. Response characteristics
    response_length = len(answer.split())
    
    # 5. Overall quality score
    overall_score = (
        0.4 * retrieval_relevance +
        0.3 * min(1.0, coverage) +
        0.2 * min(unique_sources / 3.0, 1.0) +
        0.1 * min(response_length / 100.0, 1.0)
    )
    
    return {
        "score": round(overall_score, 3),
        "retrieval_relevance": round(retrieval_relevance, 3),
        "coverage": round(coverage, 3),
        "sources_used": unique_sources,
        "response_length": response_length,
        "notes": "Baseline metrics - ready for feedback tuning"
    }

def get_current_model_from_db(db: Session) -> str:
    """Get current selected model"""
    try:
        config = db.query(db_models.SystemConfig).filter(
            db_models.SystemConfig.key == "selected_model"
        ).first()
        return config.value if config else DEFAULT_CHAT_MODEL
    except:
        return DEFAULT_CHAT_MODEL

def set_current_model_in_db(db: Session, model: str) -> bool:
    """Set current model"""
    try:
        config = db.query(db_models.SystemConfig).filter(
            db_models.SystemConfig.key == "selected_model"
        ).first()
        
        if config:
            config.value = model
            config.updated_at = datetime.utcnow()
        else:
            config = db_models.SystemConfig(
                key="selected_model",
                value=model,
                description="Selected Gemini model for RAG responses"
            )
            db.add(config)
        
        db.commit()
        return True
    except Exception as e:
        print(f"Error setting model: {e}")
        db.rollback()
        return False

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None
    session_id: Optional[str] = None  # Add session support
    session_label: Optional[str] = None  # Add this

class DeleteDocumentsRequest(BaseModel):
    documents: List[str]

class ModelSelectionRequest(BaseModel):
    model: str

class QuestionGenerationRequest(BaseModel):
    document: str
    num_questions: int = 5

class FeedbackRequest(BaseModel):
    feedback: List[Dict[str, Any]]

class FeedbackSubmission(BaseModel):
    query_id: str
    question: str
    response: str
    thumbs_up: Optional[bool] = None
    rating: Optional[int] = None
    feedback_text: Optional[str] = ""
    quality_scores: Optional[Dict[str, Any]] = {}
    sources: Optional[List[Dict[str, Any]]] = []
    response_time: Optional[float] = 0
    model_used: Optional[str] = "unknown"

# Add this Pydantic model with your other models (around line ~200)
class FeedbackConfig(BaseModel):
    feedback_enabled: bool = True
    feedback_weight: float = 0.3

class QualityScores(BaseModel):
    relevance: float
    accuracy: float
    completeness: float
    coherence: float
    citation: float

class ContactRequestSubmit(BaseModel):
    query_id: str
    name: str
    email: str
    phone: Optional[str] = None
    company: Optional[str] = None
    preferred_time: str
    additional_notes: Optional[str] = None
    original_query: str
    original_response: str


@app.post("/quality/analyze")
async def analyze_quality(scores: QualityScores):
    """Analyze quality scores and provide feedback"""
    try:
        # Define weights
        weights = {
            'relevance': 0.25,
            'accuracy': 0.30,
            'completeness': 0.20,
            'coherence': 0.15,
            'citation': 0.10
        }
        
        # Calculate overall score
        overall_score = (
            scores.relevance * weights['relevance'] +
            scores.accuracy * weights['accuracy'] +
            scores.completeness * weights['completeness'] +
            scores.coherence * weights['coherence'] +
            scores.citation * weights['citation']
        )
        
        # Grade calculation
        def get_grade(score):
            if score >= 0.9:
                return 'A+'
            elif score >= 0.85:
                return 'A'
            elif score >= 0.8:
                return 'A-'
            elif score >= 0.75:
                return 'B+'
            elif score >= 0.7:
                return 'B'
            elif score >= 0.65:
                return 'B-'
            elif score >= 0.6:
                return 'C+'
            elif score >= 0.55:
                return 'C'
            else:
                return 'D'
        
        overall_grade = get_grade(overall_score)
        overall_percentage = overall_score * 100
        
        # Component breakdown
        component_breakdown = {}
        for component, score in scores.dict().items():
            component_breakdown[component] = {
                'score': score,
                'percentage': score * 100,
                'grade': get_grade(score),
                'weight': weights[component]
            }
        
        # Improvement suggestions
        suggestions = []
        score_dict = scores.dict()
        
        # Find lowest scoring components
        sorted_scores = sorted(score_dict.items(), key=lambda x: x[1])
        
        for component, score in sorted_scores[:2]:  # Bottom 2
            if score < 0.7:
                suggestions.append(f"Focus on improving {component} (currently {score:.2f})")
        
        if scores.relevance < 0.8:
            suggestions.append("Ensure retrieved documents closely match the query")
        
        if scores.accuracy < 0.8:
            suggestions.append("Verify factual correctness of responses")
        
        if scores.completeness < 0.8:
            suggestions.append("Provide more comprehensive answers covering all aspects")
        
        if scores.citation < 0.7:
            suggestions.append("Include more source citations in responses")
        
        return {
            "analysis": {
                "overall_score": round(overall_score, 3),
                "overall_grade": overall_grade,
                "overall_percentage": round(overall_percentage, 1),
                "component_breakdown": component_breakdown,
                "improvement_suggestions": suggestions if suggestions else ["Great job! All metrics are strong."]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/quality/attributes")
async def get_quality_attributes():
    """Get quality attribute definitions"""
    return {
        "quality_attributes": {
            "relevance": {
                "description": "How well the retrieved documents match the user's query",
                "calculation": "Cosine similarity between query and document embeddings",
                "scale": "0.0 (not relevant) to 1.0 (highly relevant)",
                "weight": 0.25
            },
            "accuracy": {
                "description": "Factual correctness of the response based on source documents",
                "calculation": "Overlap between response and source content, fact verification",
                "scale": "0.0 (inaccurate) to 1.0 (completely accurate)",
                "weight": 0.30
            },
            "completeness": {
                "description": "How thoroughly the response addresses all aspects of the query",
                "calculation": "Coverage of query topics in the response",
                "scale": "0.0 (incomplete) to 1.0 (comprehensive)",
                "weight": 0.20
            },
            "coherence": {
                "description": "Logical flow and readability of the response",
                "calculation": "Sentence structure, transitions, and overall clarity",
                "scale": "0.0 (incoherent) to 1.0 (perfectly coherent)",
                "weight": 0.15
            },
            "citation": {
                "description": "Proper attribution of information to source documents",
                "calculation": "Presence and accuracy of source references",
                "scale": "0.0 (no citations) to 1.0 (well-cited)",
                "weight": 0.10
            }
        },
        "calculation_method": "Weighted average of all quality attributes",
        "total_weight": 1.0
    }


@app.post("/contact-requests")
async def submit_contact_request(
    request: ContactRequestSubmit,
    current_user: db_models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Submit contact request for expert consultation"""
    try:
        # Convert query_id string to UUID, or generate new one if empty
        try:
            query_uuid = uuid.UUID(request.query_id) if request.query_id else uuid.uuid4()
        except (ValueError, AttributeError):
            query_uuid = uuid.uuid4()
        
        # Create contact request record
        contact_request = db_models.ContactRequest(
            query_id=query_uuid,  # Use the UUID
            name=request.name,
            email=request.email,
            phone=request.phone,
            company=request.company,
            preferred_time=request.preferred_time,
            additional_notes=request.additional_notes,
            original_query=request.original_query,
            original_response=request.original_response[:1000] if request.original_response else "",
            status="pending"
        )
        
        db.add(contact_request)
        db.commit()
        db.refresh(contact_request)
        
        # Log for monitoring
        print(f"\n{'='*60}")
        print(f"📞 NEW CONTACT REQUEST")
        print(f"{'='*60}")
        print(f"ID: {contact_request.id}")
        print(f"Query ID: {query_uuid}")
        print(f"Name: {request.name}")
        print(f"Email: {request.email}")
        print(f"Phone: {request.phone or 'Not provided'}")
        print(f"Company: {request.company or 'Not provided'}")
        print(f"Preferred Time: {request.preferred_time}")
        print(f"Question: {request.original_query}")
        if request.additional_notes:
            print(f"Notes: {request.additional_notes}")
        print(f"{'='*60}\n")
        
        return {
            "status": "success",
            "message": "Contact request received. An expert will reach out within 24 hours.",
            "request_id": str(contact_request.id)
        }
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error saving contact request: {e}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to save contact request: {str(e)}")

@app.get("/contact-requests")
async def get_contact_requests(
    status: Optional[str] = None,
    current_user: db_models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """Get contact requests (admin only)"""
    
    # Only allow admin/superuser
    if current_user.role not in ["admin", "superuser"]:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        query = db.query(db_models.ContactRequest)
        
        if status:
            query = query.filter(db_models.ContactRequest.status == status)
        
        requests = query.order_by(desc(db_models.ContactRequest.created_at)).limit(100).all()
        
        return {
            "total": len(requests),
            "requests": [
                {
                    "id": str(req.id),
                    "name": req.name,
                    "email": req.email,
                    "phone": req.phone,
                    "company": req.company,
                    "preferred_time": req.preferred_time,
                    "original_query": req.original_query,
                    "status": req.status,
                    "created_at": req.created_at.isoformat()
                }
                for req in requests
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize system configuration"""
    db = SessionLocal()
    try:
        # Ensure pgvector extension is enabled
        db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        
        # Set default model if not exists
        config = db.query(db_models.SystemConfig).filter(
            db_models.SystemConfig.key == "selected_model"
        ).first()
        
        if not config:
            db.add(db_models.SystemConfig(
                key="selected_model",
                value=DEFAULT_CHAT_MODEL,
                description="Default Gemini model"
            ))
        
        db.commit()
        print(f"✅ System initialized with model: {get_current_model_from_db(db)}")
        
    except Exception as e:
        print(f"❌ Startup error: {e}")
        db.rollback()
    finally:
        db.close()

# API Endpoints
@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """System health check"""
    try:
        doc_count = db.query(db_models.Document).count()
        chunk_count = db.query(db_models.DocumentChunk).count()
        current_model = get_current_model_from_db(db)
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "documents": doc_count,
            "chunks": chunk_count,
            "current_model": current_model,
            "embedding_dimension": EMBEDDING_DIMENSION
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/chat")
async def chat_endpoint(
    request: ChatRequest, 
    current_user: db_models.User = Depends(get_current_active_user),
    db: Session = Depends(get_db)):
    """Enhanced RAG chat with session persistence and conversation history"""
    start_time = time.time()
    timing_breakdown = {}
    
    try:
        model_to_use = request.model or get_current_model_from_db(db)
        
        # Get or create session
        t_session = time.time()
        session = None
        if hasattr(request, 'session_id') and request.session_id:
            session = db.query(db_models.ChatSession).filter(
                db_models.ChatSession.session_id == request.session_id
            ).first()
        
        if not session:
            session = db_models.ChatSession(user_id=current_user.id)
            db.add(session)
            db.flush()
        
        if request.session_label and request.session_label != session.session_label:
            session.session_label = request.session_label
        
        session.last_activity = datetime.utcnow()
        timing_breakdown['session_setup'] = (time.time() - t_session) * 1000
        
        # Get conversation history
        t_history = time.time()
        previous_messages = db.query(db_models.ChatMessage).filter(
            db_models.ChatMessage.session_id == session.id
        ).order_by(
            desc(db_models.ChatMessage.timestamp)
        ).limit(10).all()
        previous_messages.reverse()
        timing_breakdown['history_fetch'] = (time.time() - t_history) * 1000
        
        # Create current message record
        chat_message = db_models.ChatMessage(
            session_id=session.id,
            query=request.query,
            model_used=model_to_use,
            timestamp=datetime.utcnow()
        )
        db.add(chat_message)
        db.flush()
        
        # Vector similarity search
        t_vector = time.time()
        retrieved_chunks = await vector_similarity_search(request.query, db, RETRIEVE_K)
        timing_breakdown['vector_search'] = (time.time() - t_vector) * 1000
        
        # 🔍 ENCODING CHECK 1 - From Database
        print(f"\n{'='*60}")
        print(f"🔍 ENCODING CHECK 1 - Retrieved from Database")
        print(f"Query: {request.query}")
        print(f"Retrieved {len(retrieved_chunks) if retrieved_chunks else 0} chunks")
        
        if not retrieved_chunks:
            answer = "I don't have enough context to answer your question. Please upload relevant documents."
            quality = {"score": 0, "notes": "No relevant documents found"}
            sources = []
            sources_count = 0
            print(f"⚠️ No chunks retrieved")
        else:
            # Log database chunks
            for i, (chunk, similarity) in enumerate(retrieved_chunks[:2]):
                chunk_preview = chunk.content[:300].replace('\n', ' ')
                print(f"  Chunk {i+1} (similarity: {similarity:.3f}):")
                print(f"    Preview: {chunk_preview}...")
                
                # Check for I-5 pattern
                if 'I-5' in chunk.content:
                    i5_start = chunk.content.find('I-5')
                    i5_section = chunk.content[i5_start:i5_start+30]
                    print(f"    ⚠️ Found I-5 pattern: '{i5_section}'")
                    print(f"    🔬 Hex: {i5_section.encode('unicode-escape').decode('ascii')}")
            
            # Build context from retrieved documents
            t_context = time.time()
            context_parts = []
            for chunk, similarity in retrieved_chunks:
                source = chunk._filename
                context_parts.append(f"[Source: {source}]\n{chunk.content}")
            context = "\n\n---\n\n".join(context_parts)
            timing_breakdown['context_build'] = (time.time() - t_context) * 1000
            
            # 📤 ENCODING CHECK 2 - Sending to Gemini
            print(f"\n📤 ENCODING CHECK 2 - Context Prepared for Gemini")
            context_preview = context[:400].replace('\n', ' ')
            print(f"  Context preview: {context_preview}...")
            
            if 'I-5' in context:
                i5_start = context.find('I-5')
                i5_section = context[i5_start:i5_start+30]
                print(f"  ⚠️ I-5 in context: '{i5_section}'")
                print(f"  🔬 Hex: {i5_section.encode('unicode-escape').decode('ascii')}")
            
            # Build conversation history for context
            conversation_history = []
            for prev_msg in previous_messages[-4:]:
                if prev_msg.query:
                    conversation_history.append({
                        "role": "user",
                        "content": prev_msg.query
                    })
                if prev_msg.response:
                    conversation_history.append({
                        "role": "assistant",
                        "content": prev_msg.response
                    })
            
            # Generate response with enforced system prompt
            t_llm = time.time()
            llm = ChatGoogleGenerativeAI(
                model=model_to_use, 
                temperature=0.1,
                top_p=0.95,
                convert_system_message_to_human=True
            )
            
            system_prompt = """You are a professional RAG assistant. Follow these rules strictly:

1. ONLY answer based on the provided CONTEXT below
2. ALWAYS respond in the SAME LANGUAGE as the user's question
3. If the user asks in English, respond in English
4. If the user asks in French, respond in French
5. If the answer is not in the context, say "I don't have that information in the available documents"
6. Be concise and accurate
7. Cite sources when relevant

DO NOT translate the user's question or change languages unless explicitly asked."""

            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            if conversation_history:
                messages.extend(conversation_history)
            
            user_message = f"""CONTEXT from documents:
{context}

---

USER QUESTION: {request.query}

Remember: Answer in the same language as the question above."""

            messages.append({"role": "user", "content": user_message})
            
            response = await llm.ainvoke(messages)
            answer = response.content
            timing_breakdown['llm_generation'] = (time.time() - t_llm) * 1000
            
            # 📥 ENCODING CHECK 3 - Received from Gemini
            print(f"\n📥 ENCODING CHECK 3 - Response from Gemini")
            answer_preview = answer[:400].replace('\n', ' ')
            print(f"  Answer preview: {answer_preview}...")
            
            if 'I-5' in answer:
                i5_start = answer.find('I-5')
                i5_section = answer[i5_start:i5_start+30]
                print(f"  ⚠️ I-5 in answer: '{i5_section}'")
                print(f"  🔬 Hex: {i5_section.encode('unicode-escape').decode('ascii')}")
                
                # Check if corruption happened
                if 'Ƒѵ' in i5_section or 'ƒ' in i5_section:
                    print(f"  ❌ CORRUPTION DETECTED in LLM response!")
            
            print(f"{'='*60}\n")
            
            # Compute quality metrics
            t_quality = time.time()
            quality = await compute_quality_metrics(request.query, answer, retrieved_chunks, db)
            quality["sources_used"] = len(retrieved_chunks)
            timing_breakdown['quality_metrics'] = (time.time() - t_quality) * 1000
            
            # Save sources
            t_sources = time.time()
            for i, (chunk, similarity) in enumerate(retrieved_chunks):
                source_record = db_models.MessageSource(
                    message_id=chat_message.id,
                    chunk_id=chunk.id,
                    source_name=chunk._filename,
                    similarity_score=similarity,
                    rank_position=i + 1
                )
                db.add(source_record)
            timing_breakdown['save_sources'] = (time.time() - t_sources) * 1000
            
            sources = [
                {
                    "source": chunk._filename,
                    "similarity": round(similarity, 3),
                    "chunk_preview": chunk.content[:150] + "..." if len(chunk.content) > 150 else chunk.content
                }
                for chunk, similarity in retrieved_chunks
            ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Update message with response
        chat_message.response = answer
        chat_message.response_time_ms = latency_ms
        chat_message.quality_score = quality.get("score", 0)
        
        t_commit = time.time()
        db.commit()
        timing_breakdown['db_commit'] = (time.time() - t_commit) * 1000
        
        # Log timing breakdown
        print(f"⏱️ TIMING BREAKDOWN (Total: {latency_ms:.0f}ms):")
        for key, value in timing_breakdown.items():
            print(f"  {key}: {value:.0f}ms ({value/latency_ms*100:.1f}%)")
        
        # 🚀 FINAL ENCODING CHECK - Before sending to frontend
        print(f"\n🚀 FINAL CHECK - Response being sent to frontend:")
        print(f"  Answer length: {len(answer)} chars")
        if 'I-5' in answer:
            i5_start = answer.find('I-5')
            i5_section = answer[i5_start:i5_start+30]
            print(f"  I-5 section: '{i5_section}'")
        
        return {
            "answer": answer,
            "latency_ms": round(latency_ms, 2),
            "timing_breakdown": timing_breakdown,
            "quality": quality,
            "sources": sources,
            "sources_count": len(sources),
            "model_used": model_to_use,
            "session_id": session.session_id,
            "conversation_length": len(previous_messages) + 1,
            "query_id": str(chat_message.id)
        }
        
    except Exception as e:
        db.rollback()
        print(f"❌ Chat error: {e}")
        import traceback
        print(traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Chat processing failed: {str(e)}",
                "answer": "I apologize, but I encountered an error processing your request.",
                "latency_ms": (time.time() - start_time) * 1000,
                "quality": {"score": 0},
                "sources": [],
                "sources_count": 0 
            }
        )

# Admin-only endpoints
@app.get("/admin/users")
async def list_users(
    current_user: db_models.User = Depends(require_role(["admin", "superuser"])),
    db: Session = Depends(get_db)
):
    users = db.query(db_models.User).all()
    return [{"username": u.username, "email": u.email, "role": u.role} for u in users]

@app.get("/admin/cache-stats")
async def get_cache_stats():
    """Get LRU cache statistics"""
    if hasattr(get_cached_embedding, 'cache_info'):
        info = get_cached_embedding.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
            "hit_rate": round(info.hits / (info.hits + info.misses), 2) if (info.hits + info.misses) > 0 else 0
        }
    return {"error": "Cache not configured"}

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    """Upload and process documents - FIXED for your schema"""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        uploaded_files = []
        total_chunks = 0
        
        for file in files:
            content = await file.read()
            filename = file.filename or "unknown"
            
            # Extract text
            text = extract_text_from_file(content, filename)
            
            if not text.strip():
                continue
            
            # Create content hash
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            
            # Check for duplicates
            existing_doc = db.query(db_models.Document).filter(
                (db_models.Document.filename == filename) |
                (db_models.Document.content_hash == content_hash)
            ).first()
            
            if existing_doc:
                print(f"Skipping duplicate: {filename}")
                continue
            
            # Save document with content
            doc_record = db_models.Document(
                id=uuid.uuid4(),
                filename=filename,
                content=text,  # Save full content
                content_hash=content_hash,
                created_at=datetime.utcnow()
            )
            db.add(doc_record)
            db.flush()
            
            # Process chunks
            text_chunks = chunk_text(text, filename)
            
            # Batch embed chunks for efficiency
            chunk_embeddings = embeddings.embed_documents(text_chunks)
            
            for i, (chunk_content, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
                chunk_hash = hashlib.md5(chunk_content.encode()).hexdigest()
                
                chunk_record = db_models.DocumentChunk(
                    id=uuid.uuid4(),
                    document_id=doc_record.id,
                    chunk_index=i,
                    content=chunk_content,
                    embedding=embedding,
                    content_hash=chunk_hash,
                    doc_metadata={"source": filename, "chunk_size": len(chunk_content)},
                    created_at=datetime.utcnow()
                )
                db.add(chunk_record)
                total_chunks += 1
            
            uploaded_files.append(filename)
        
        if uploaded_files:
            db.commit()
            return {
                "message": f"Successfully uploaded {len(uploaded_files)} documents",
                "uploaded_count": len(uploaded_files),
                "chunks_added": total_chunks,
                "documents": uploaded_files
            }
        else:
            raise HTTPException(status_code=400, detail="No new documents to upload (all duplicates)")
            
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/documents")
async def list_documents(db: Session = Depends(get_db)):
    """List all documents"""
    try:
        documents = db.query(db_models.Document).order_by(db_models.Document.filename).all()
        return [doc.filename for doc in documents]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents")
async def delete_documents(request: DeleteDocumentsRequest, db: Session = Depends(get_db)):
    """Delete documents and associated chunks with manual cascade"""
    try:
        deleted_count = 0
        deleted_filenames = []
        
        for filename in request.documents:
            doc = db.query(db_models.Document).filter(
                db_models.Document.filename == filename
            ).first()
            
            if doc:
                # Get all chunk IDs before deletion
                chunk_ids = [chunk.id for chunk in doc.chunks]
                
                if chunk_ids:
                    # Delete records that reference chunks
                    db.query(db_models.FeedbackSource).filter(
                        db_models.FeedbackSource.chunk_id.in_(chunk_ids)
                    ).delete(synchronize_session=False)
                    
                    db.query(db_models.MessageSource).filter(
                        db_models.MessageSource.chunk_id.in_(chunk_ids)
                    ).delete(synchronize_session=False)
                
                # Delete document (cascade will handle chunks)
                db.delete(doc)
                
                deleted_count += 1
                deleted_filenames.append(filename)
        
        db.commit()
        
        return {
            "message": f"Deleted {deleted_count} documents and all related data",
            "deleted_count": deleted_count,
            "deleted_sources": deleted_filenames
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/admin/selected-model")
async def get_selected_model(db: Session = Depends(get_db)):
    """Get currently selected model"""
    current_model = get_current_model_from_db(db)
    
    config = db.query(db_models.SystemConfig).filter(
        db_models.SystemConfig.key == "selected_model"
    ).first()
    
    last_updated = config.updated_at.isoformat() if config and config.updated_at else "Never"
    
    return {
        "model": current_model,
        "available_models": [
            "gemini-2.5-flash",           # Current - Fast & efficient
            "gemini-2.5-pro",             # Most capable
            "gemini-2.5-flash-lite"
        ],
        "last_updated": last_updated
    }

@app.post("/admin/set-model")
async def set_selected_model(request: ModelSelectionRequest, db: Session = Depends(get_db)):
    """Set system model"""
    available_models = [
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest"
    ]
    
    if request.model not in available_models:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model. Available: {available_models}"
        )
    
    success = set_current_model_in_db(db, request.model)
    
    if success:
        return {
            "success": True,
            "model": request.model,
            "message": f"Model updated to {request.model}"
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to update model")

@app.post("/admin/generate-questions")
async def generate_questions(request: QuestionGenerationRequest, db: Session = Depends(get_db)):
    """Generate test questions from document"""
    try:
        doc = db.query(db_models.Document).filter(
            db_models.Document.filename == request.document
        ).first()
        
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document '{request.document}' not found")
        
        chunks = db.query(db_models.DocumentChunk).filter(
            db_models.DocumentChunk.document_id == doc.id
        ).order_by(db_models.DocumentChunk.chunk_index).all()
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Document has no content")
        
        # Combine chunks for question generation
        full_content = " ".join([chunk.content for chunk in chunks])[:6000]
        
        current_model = get_current_model_from_db(db)
        
        llm = ChatGoogleGenerativeAI(model=current_model, temperature=0.3)
        
        prompt = f"""
Based on this document content, generate {request.num_questions} specific questions that can be answered from the text.
Return as JSON array: ["Question 1?", "Question 2?", ...]

Document: {request.document}
Content: {full_content}

Requirements:
1. Questions should be answerable from the content
2. Cover different aspects
3. Vary in complexity
4. Be clear and specific
"""
        
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        
        # Parse JSON response
        import re
        json_match = re.search(r'\[.*?\]', response.content, re.DOTALL)
        
        if json_match:
            questions = json.loads(json_match.group())
            if isinstance(questions, list):
                return {
                    "questions": questions[:request.num_questions],
                    "document": request.document,
                    "generated_count": len(questions[:request.num_questions])
                }
        
        # Fallback questions
        fallback_questions = [
            f"What are the main topics in {request.document}?",
            f"What are the key findings in {request.document}?",
            f"What conclusions are drawn in {request.document}?",
            f"What recommendations are made in {request.document}?",
            f"What methodology is discussed in {request.document}?"
        ]
        
        return {
            "questions": fallback_questions[:request.num_questions],
            "document": request.document,
            "generated_count": request.num_questions,
            "note": "Used fallback questions"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission, db: Session = Depends(get_db)):
    """Save user feedback for response improvement - FIXED UUID handling"""
    try:
        print(f"📝 Receiving feedback for query: {feedback.query_id}")
        
        # Convert query_id string to UUID for database query
        try:
            query_uuid = uuid.UUID(feedback.query_id)
        except ValueError:
            # If it's not a valid UUID, use it as string
            query_uuid = feedback.query_id
        
        print(f"query uuid: {query_uuid}")
        # Check if feedback already exists for this query
        # Use string comparison since query_id is stored as VARCHAR in your schema
        print(f"query_uuid value: {query_uuid}")
        print(f"query_uuid type: {type(query_uuid)}")
        print(f"UserFeedback.query_id column type: {db_models.UserFeedback.query_id.type}")
        existing_feedback = db.query(db_models.UserFeedback).filter(
            db_models.UserFeedback.query_id == query_uuid
        ).first()
        
        if existing_feedback:
            # Update existing feedback
            print(f"🔄 Updating existing feedback")
            existing_feedback.rating = feedback.rating
            existing_feedback.thumbs_up = feedback.thumbs_up
            existing_feedback.feedback_text = feedback.feedback_text
            
            if feedback.quality_scores:
                existing_feedback.relevance_score = feedback.quality_scores.get("retrieval_relevance", 0)
                existing_feedback.accuracy_score = feedback.quality_scores.get("score", 0)
                existing_feedback.completeness_score = feedback.quality_scores.get("coverage", 0)
                existing_feedback.overall_quality_score = feedback.quality_scores.get("score", 0)
            
            feedback_record = existing_feedback
        else:
            # Create new feedback record
            print(f"✨ Creating new feedback record")
            
            # Generate session ID from timestamp
            session_id = f"session_{int(time.time())}"
            
            feedback_record = db_models.UserFeedback(
                id=uuid.uuid4(),
                session_id=session_id,
                query_id=query_uuid,  # Store as UUID
                question=feedback.question,
                response=feedback.response,
                rating=feedback.rating,
                thumbs_up=feedback.thumbs_up,
                feedback_text=feedback.feedback_text,
                response_time=feedback.response_time,
                tokens_used=0,
                sources_count=len(feedback.sources),
                relevance_score=feedback.quality_scores.get("retrieval_relevance", 0) if feedback.quality_scores else 0,
                accuracy_score=feedback.quality_scores.get("score", 0) if feedback.quality_scores else 0,
                completeness_score=feedback.quality_scores.get("coverage", 0) if feedback.quality_scores else 0,
                coherence_score=feedback.quality_scores.get("score", 0) if feedback.quality_scores else 0,
                citation_score=feedback.quality_scores.get("retrieval_relevance", 0) if feedback.quality_scores else 0,
                overall_quality_score=feedback.quality_scores.get("score", 0) if feedback.quality_scores else 0,
                created_at=datetime.utcnow()
            )
            db.add(feedback_record)
            db.flush()
            
            # Save source references
            if feedback.sources:
                for source in feedback.sources:
                    try:
                        # Safely extract UUIDs
                        doc_id = source.get("document_id")
                        chunk_id = source.get("chunk_id")
                        
                        if doc_id and chunk_id:
                            source_record = db_models.FeedbackSource(
                                id=uuid.uuid4(),
                                feedback_id=feedback_record.id,
                                document_id=uuid.UUID(doc_id) if isinstance(doc_id, str) else doc_id,
                                chunk_id=uuid.UUID(chunk_id) if isinstance(chunk_id, str) else chunk_id,
                                relevance_score=source.get("similarity", 0)
                            )
                            db.add(source_record)
                    except Exception as e:
                        print(f"⚠️  Could not save source reference: {e}")
                        continue
        
        db.commit()
        print(f"✅ Feedback saved successfully!")
        
        return {
            "success": True,
            "message": "Feedback recorded successfully",
            "feedback_id": str(feedback_record.id),
            "query_id": feedback.query_id
        }
        
    except Exception as e:
        import traceback
        print(f"❌ Error saving feedback:")
        print(traceback.format_exc())
        db.rollback()
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to save feedback: {str(e)}"
        )

@app.get("/feedback/stats")
async def get_feedback_stats(days: int = 30, db: Session = Depends(get_db)):
    """Get feedback statistics for a time period"""
    try:
        from datetime import timedelta
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get feedback in date range
        feedback_records = db.query(db_models.UserFeedback).filter(
            db_models.UserFeedback.created_at >= start_date,
            db_models.UserFeedback.created_at <= end_date
        ).all()
        
        total_feedback = len(feedback_records)
        
        if total_feedback == 0:
            return {
                "total_feedback": 0,
                "average_rating": 0,
                "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                "average_quality_scores": {},
                "date_range": {
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat()
                }
            }
        
        # Calculate statistics
        ratings = [fb.rating for fb in feedback_records if fb.rating]
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        
        # Rating distribution
        rating_dist = {i: 0 for i in range(1, 6)}
        for rating in ratings:
            if rating in rating_dist:
                rating_dist[rating] += 1
        
        # Average quality scores
        relevance_scores = [fb.relevance_score for fb in feedback_records if fb.relevance_score]
        accuracy_scores = [fb.accuracy_score for fb in feedback_records if fb.accuracy_score]
        completeness_scores = [fb.completeness_score for fb in feedback_records if fb.completeness_score]
        
        avg_quality = {
            "relevance": sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            "accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0,
            "completeness": sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        }
        
        return {
            "total_feedback": total_feedback,
            "average_rating": round(avg_rating, 2),
            "rating_distribution": rating_dist,
            "average_quality_scores": avg_quality,
            "date_range": {
                "from": start_date.isoformat(),
                "to": end_date.isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/feedback/analytics")
async def get_feedback_analytics(db: Session = Depends(get_db)):
    """Get feedback analytics for system improvement"""
    try:
        # Overall statistics
        total_feedback = db.query(db_models.UserFeedback).count()
        
        positive_feedback = db.query(db_models.UserFeedback).filter(
            db_models.UserFeedback.thumbs_up == True
        ).count()
        
        avg_rating = db.query(func.avg(db_models.UserFeedback.rating)).scalar() or 0
        
        # Rating distribution
        rating_dist = db.query(
            db_models.UserFeedback.rating,
            func.count(db_models.UserFeedback.id).label('count')
        ).group_by(db_models.UserFeedback.rating).all()
        
        # Average quality scores
        avg_relevance = db.query(func.avg(db_models.UserFeedback.relevance_score)).scalar() or 0
        avg_accuracy = db.query(func.avg(db_models.UserFeedback.accuracy_score)).scalar() or 0
        avg_completeness = db.query(func.avg(db_models.UserFeedback.completeness_score)).scalar() or 0
        
        # Recent negative feedback for improvement
        negative_feedback = db.query(db_models.UserFeedback).filter(
            (db_models.UserFeedback.thumbs_up == False) | 
            (db_models.UserFeedback.rating <= 2)
        ).order_by(desc(db_models.UserFeedback.created_at)).limit(10).all()
        
        return {
            "summary": {
                "total_feedback": total_feedback,
                "positive_feedback": positive_feedback,
                "positive_rate": round(positive_feedback / total_feedback * 100, 2) if total_feedback > 0 else 0,
                "avg_rating": round(float(avg_rating), 2),
                "avg_relevance": round(float(avg_relevance), 3),
                "avg_accuracy": round(float(avg_accuracy), 3),
                "avg_completeness": round(float(avg_completeness), 3)
            },
            "rating_distribution": [
                {"rating": r.rating, "count": r.count}
                for r in rating_dist
            ],
            "recent_negative_feedback": [
                {
                    "query_id": str(fb.query_id),
                    "question": fb.question[:100],
                    "rating": fb.rating,
                    "comment": fb.feedback_text,
                    "timestamp": fb.created_at.isoformat()
                }
                for fb in negative_feedback
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")


@app.post("/admin/feedback")
async def save_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    """Save model comparison feedback"""
    try:
        saved_count = 0
        comparison_session = f"session_{int(time.time())}"
        
        for entry in request.feedback:
            feedback = db_models.ModelFeedback(
                question_index=entry.get("question_idx", 0),
                question=entry.get("question", ""),
                model_name=entry.get("model", "unknown"),
                rating=entry.get("rating", 0),
                comment=entry.get("comment", ""),
                comparison_session=comparison_session,
                timestamp=datetime.utcnow()
            )
            db.add(feedback)
            saved_count += 1
        
        db.commit()
        
        return {
            "success": True,
            "saved_entries": saved_count,
            "comparison_session": comparison_session,
            "message": "Feedback saved successfully"
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

# Replace the feedback config endpoints
@app.get("/admin/feedback-config")
async def get_feedback_config(db: Session = Depends(get_db)):
    """Get current feedback configuration"""
    try:
        fb_enabled_config = db.query(db_models.SystemConfig).filter(
            db_models.SystemConfig.key == "feedback_enabled"
        ).first()
        
        fb_weight_config = db.query(db_models.SystemConfig).filter(
            db_models.SystemConfig.key == "feedback_weight"
        ).first()
        
        feedback_enabled = True  # Default
        feedback_weight = 0.5    # Default
        
        if fb_enabled_config:
            feedback_enabled = fb_enabled_config.value.lower() == "true"
        
        if fb_weight_config:
            try:
                feedback_weight = float(fb_weight_config.value)
            except ValueError:
                feedback_weight = 0.3
        
        return {
            "feedback_enabled": feedback_enabled,
            "feedback_weight": feedback_weight
        }
        
    except Exception as e:
        print(f"⚠️  Error loading feedback config, using defaults: {e}")
        # Return defaults if error
        return {
            "feedback_enabled": True,
            "feedback_weight": 0.3
        }

@app.post("/admin/feedback-config")
async def set_feedback_config(config: FeedbackConfig, db: Session = Depends(get_db)):
    """Configure feedback-enhanced retrieval"""
    try:
        print(f"📝 Updating feedback config: enabled={config.feedback_enabled}, weight={config.feedback_weight}")
        
        # Update or create feedback_enabled
        fb_enabled = db.query(db_models.SystemConfig).filter(
            db_models.SystemConfig.key == "feedback_enabled"
        ).first()
        
        if fb_enabled:
            fb_enabled.value = str(config.feedback_enabled)
            fb_enabled.updated_at = datetime.utcnow()
        else:
            fb_enabled = db_models.SystemConfig(
                key="feedback_enabled",
                value=str(config.feedback_enabled),
                description="Enable/disable feedback-enhanced retrieval",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(fb_enabled)
        
        # Update or create feedback_weight
        fb_weight = db.query(db_models.SystemConfig).filter(
            db_models.SystemConfig.key == "feedback_weight"
        ).first()
        
        if fb_weight:
            fb_weight.value = str(config.feedback_weight)
            fb_weight.updated_at = datetime.utcnow()
        else:
            fb_weight = db_models.SystemConfig(
                key="feedback_weight",
                value=str(config.feedback_weight),
                description="Weight of feedback in retrieval (0.0-1.0)",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(fb_weight)
        
        db.commit()
        print("✅ Feedback config updated successfully")
        
        return {
            "success": True,
            "feedback_enabled": config.feedback_enabled,
            "feedback_weight": config.feedback_weight,
            "message": "Feedback configuration updated"
        }
        
    except Exception as e:
        print(f"❌ Error updating feedback config: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update config: {str(e)}")

@app.get("/admin/analytics")
async def get_analytics(db: Session = Depends(get_db)):
    """Get system analytics"""
    try:
        # Basic statistics
        doc_count = db.query(db_models.Document).count()
        chunk_count = db.query(db_models.DocumentChunk).count()
        
        # Check if UserFeedback table exists and has data
        try:
            message_count = db.query(db_models.UserFeedback).count()
            
            # Quality metrics (only if we have feedback)
            if message_count > 0:
                avg_quality = db.query(func.avg(db_models.UserFeedback.overall_quality_score)).scalar() or 0
                avg_response_time = db.query(func.avg(db_models.UserFeedback.response_time)).scalar() or 0
            else:
                avg_quality = 0
                avg_response_time = 0
        except Exception as e:
            print(f"⚠️  UserFeedback query error: {e}")
            message_count = 0
            avg_quality = 0
            avg_response_time = 0
        
        # Model usage - safely handle if no feedback exists
        try:
            # Try to get model usage from UserFeedback
            model_usage_query = db.query(
                func.count(db_models.UserFeedback.id).label('count')
            ).group_by(db_models.UserFeedback.id).all()
            
            model_usage = []
        except Exception as e:
            print(f"⚠️  Model usage query error: {e}")
            model_usage = []
        
        # Feedback stats - safely handle if ModelFeedback doesn't exist
        try:
            feedback_stats = db.query(
                db_models.ModelFeedback.model_name,
                func.avg(db_models.ModelFeedback.rating).label('avg_rating'),
                func.count(db_models.ModelFeedback.id).label('count')
            ).group_by(db_models.ModelFeedback.model_name).all()
        except Exception as e:
            print(f"⚠️  ModelFeedback query error: {e}")
            feedback_stats = []
        
        return {
            "system_status": {
                "current_model": get_current_model_from_db(db),
                "documents": doc_count,
                "chunks": chunk_count,
                "messages": message_count,
                "avg_quality": round(float(avg_quality), 3),
                "avg_response_time_ms": round(float(avg_response_time), 2)
            },
            "model_usage": [
                {"model": "gemini-2.5-flash", "count": message_count}
            ] if message_count > 0 else [],
            "feedback_stats": [
                {
                    "model": stat.model_name,
                    "avg_rating": round(float(stat.avg_rating), 2),
                    "count": stat.count
                }
                for stat in feedback_stats
            ] if feedback_stats else [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Analytics error:")
        print(error_details)
        
        # Return minimal data instead of failing
        return JSONResponse(
            status_code=200,  # Return 200 with minimal data instead of 500
            content={
                "system_status": {
                    "current_model": "gemini-2.5-flash",
                    "documents": 0,
                    "chunks": 0,
                    "messages": 0,
                    "avg_quality": 0.0,
                    "avg_response_time_ms": 0.0
                },
                "model_usage": [],
                "feedback_stats": [],
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


@app.get("/sessions/summaries")
async def get_session_summaries(
    current_user: db_models.User = Depends(require_role(["admin", "superuser"])),
    db: Session = Depends(get_db)):
    """Get summaries of last 10 sessions"""
    sessions = db.query(db_models.ChatSession).order_by(
        desc(db_models.ChatSession.last_activity)
    ).limit(10).all()
    
    summaries = []
    for session in sessions:
        messages = db.query(db_models.ChatMessage).filter(
            db_models.ChatMessage.session_id == session.id
        ).order_by(db_models.ChatMessage.timestamp).all()
        
        # Generate summary from first and last messages
        if messages:
            first_q = messages[0].query[:100]
            summary = f"Started with: '{first_q}...'"
            if len(messages) > 1:
                last_q = messages[-1].query[:100]
                summary += f"\nEnded with: '{last_q}...'"
        else:
            summary = "No messages"
        
        display_name = session.session_label if session.session_label else session.created_at.strftime('%Y-%m-%d %H:%M')
        
        summaries.append({
            "session_id": session.session_id,
            "display_name": display_name,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(messages),
            "summary": summary
        })
    
    return summaries

@app.get("/sessions/{session_id}")
async def get_session_history(session_id: str, db: Session = Depends(get_db)):
    """Get conversation history for a session"""
    try:
        session = db.query(db_models.ChatSession).filter(
            db_models.ChatSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        messages = db.query(db_models.ChatMessage).filter(
            db_models.ChatMessage.session_id == session.id
        ).order_by(db_models.ChatMessage.timestamp).all()
        
        return {
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": len(messages),
            "messages": [
                {
                    "id": msg.id,
                    "query": msg.query,
                    "response": msg.response,
                    "model_used": msg.model_used,
                    "quality_score": msg.quality_score,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in messages
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: Session = Depends(get_db)):
    """Delete a session and its messages"""
    try:
        session = db.query(db_models.ChatSession).filter(
            db_models.ChatSession.session_id == session_id
        ).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        db.delete(session)  # Cascade will delete messages
        db.commit()
        
        return {
            "success": True,
            "message": f"Session {session_id} deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, db: Session = Depends(get_db)):
    """Get all messages for a session"""
    session = db.query(db_models.ChatSession).filter(
        db_models.ChatSession.session_id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = db.query(db_models.ChatMessage).filter(
        db_models.ChatMessage.session_id == session.id
    ).order_by(db_models.ChatMessage.timestamp).all()
    
    return [
        {
            "query": m.query,
            "response": m.response,
            "timestamp": m.timestamp.isoformat(),
            "quality_score": m.quality_score
        }
        for m in messages
    ]

@app.get("/admin/system-stats")
async def get_system_stats(
    current_user: db_models.User = Depends(require_role(["admin", "superuser"])),
    db: Session = Depends(get_db)):
    """Get real-time system statistics"""
    from datetime import timedelta
    
    # Total sessions
    total_sessions = db.query(db_models.ChatSession).count()
    
    # Active sessions (last 24 hours)
    cutoff = datetime.utcnow() - timedelta(hours=24)
    active_sessions = db.query(db_models.ChatSession).filter(
        db_models.ChatSession.last_activity > cutoff
    ).count()
    
    # Average response time
    avg_time = db.query(func.avg(db_models.ChatMessage.response_time_ms)).scalar()
    avg_time_seconds = (avg_time / 1000) if avg_time else 0
    
    # Most accessed document
    top_doc = db.query(
        db_models.MessageSource.source_name,
        func.count(db_models.MessageSource.id).label('count')
    ).group_by(
        db_models.MessageSource.source_name
    ).order_by(desc('count')).first()
    
    return {
        "total_sessions": total_sessions,
        "active_sessions": active_sessions,
        "avg_response_time": round(avg_time_seconds, 1),
        "top_document": top_doc.source_name if top_doc else "None",
        "top_document_count": top_doc.count if top_doc else 0
    }

@app.patch("/sessions/{session_id}/label")
async def update_session_label(session_id: str, request: dict, db: Session = Depends(get_db)):
    """Update session label"""
    session = db.query(db_models.ChatSession).filter(
        db_models.ChatSession.session_id == session_id
    ).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session.session_label = request.get("label", "")
    db.commit()
    
    return {"success": True, "label": session.session_label}


@app.get("/sessions")
async def list_sessions(db: Session = Depends(get_db)):
    """List all chat sessions with labels"""
    sessions = db.query(
        db_models.ChatSession.session_id,
        db_models.ChatSession.session_label,
        db_models.ChatSession.created_at,
        db_models.ChatSession.last_activity,
        func.count(db_models.ChatMessage.id).label('message_count')
    ).outerjoin(
        db_models.ChatMessage
    ).group_by(
        db_models.ChatSession.id
    ).order_by(
        desc(db_models.ChatSession.last_activity)
    ).limit(20).all()
    
    return [
        {
            "session_id": s.session_id,
            "session_label": s.session_label,
            "created_at": s.created_at.isoformat(),
            "last_activity": s.last_activity.isoformat(),
            "message_count": s.message_count
        }
        for s in sessions
    ]

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(db_models.User).filter(db_models.User.username == form_data.username).first()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

@app.post("/register")
async def register(username: str, email: str, password: str, full_name: str = "", db: Session = Depends(get_db)):
    """Register new user"""
    if db.query(db_models.User).filter(db_models.User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    
    user = db_models.User(
        username=username,
        email=email,
        hashed_password=get_password_hash(password),
        full_name=full_name,
        role="user"
    )
    db.add(user)
    db.commit()
    return {"message": "User created successfully"}

from functools import lru_cache
import hashlib

# Add at module level
@lru_cache(maxsize=1000)
def get_cached_embedding(text_hash: str):
    """Cache embeddings to avoid redundant API calls"""
    # This is a placeholder - actual embedding happens in the calling function
    return None

def get_text_hash(text: str) -> str:
    """Generate hash for text caching"""
    return hashlib.md5(text.encode()).hexdigest()

async def vector_similarity_search_cached(
    query: str, 
    db: Session, 
    limit: int = RETRIEVE_K
) -> List[Tuple[db_models.DocumentChunk, float]]:
    """Vector search with query caching"""
    
    query_hash = get_text_hash(query)
    
    # Check if we've seen this exact query recently
    cached_embedding = get_cached_embedding(query_hash)
    
    if cached_embedding is None:
        # Generate new embedding
        query_embedding = embeddings.embed_query(query)
        # Cache it (in production, use Redis or similar)
        get_cached_embedding.__wrapped__(query_hash, query_embedding)
    else:
        query_embedding = cached_embedding
    
    # Rest of the search logic...
    # (Use the optimized query from Solution 3)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
