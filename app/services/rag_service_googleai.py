import logging
from typing import List, Dict, AsyncGenerator, Optional, Any
from sqlalchemy.orm import Session
from app.models.database import SessionLocal, Document, DocumentChunk
from datetime import datetime
import time
import hashlib
from contextlib import contextmanager
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel
import uuid
from app.services.quality_dashboard import QualityDashboard
from app.services.model_manager import model_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        logger.info("=== RAGService Initialization Starting ===")
        
        # Initialize all attributes first to ensure object is always valid
        self.embedding_model = None
        self.llm_model = None
        self.quality_evaluator = None
        self.response_optimizer = None
        self._initialization_error = None
        self._initialized = False
        
        try:
            # Step 1: Initialize Vertex AI
            logger.info("Step 1: Initializing Vertex AI...")
            vertexai.init(project="lcrai-472418", location="us-central1")
            logger.info("✅ Vertex AI initialized")
            
            # Step 2: Load embedding model
            logger.info("Step 2: Loading embedding model...")
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            logger.info("✅ Embedding model loaded")
            
            # Step 3: Load LLM model
            logger.info("Step 3: Loading LLM model...")
            self.llm_model = GenerativeModel("gemini-2.5-flash")
            logger.info("✅ LLM model loaded")
            
            # Step 4: Initialize quality evaluator (with fallback)
            logger.info("Step 4: Initializing quality evaluator...")
            try:
                from app.services.quality_evaluator import QualityEvaluator
                self.quality_evaluator = QualityEvaluator()
                logger.info("✅ Quality evaluator initialized")
            except Exception as e:
                logger.warning(f"Quality evaluator initialization failed: {e}")
                self.quality_evaluator = None
            
            # Step 5: Initialize response optimizer (with fallback)
            logger.info("Step 5: Initializing response optimizer...")
            try:
                from app.services.response_optimizer import ResponseOptimizer
                self.response_optimizer = ResponseOptimizer()
                logger.info("✅ Response optimizer initialized")
            except Exception as e:
                logger.warning(f"Response optimizer initialization failed: {e}")
                self.response_optimizer = None

            logger.info("Step 6: Initializing quality dashboard...")

            self.quality_dashboard = QualityDashboard()

            self._initialized = True
            logger.info("=== RAGService Initialization Complete ===")
            
        except Exception as e:
            self._initialization_error = str(e)
            logger.exception("=== RAGService Initialization FAILED ===")
            # Don't raise - let the object exist but track the error
    
    async def _evaluate_quality_with_dashboard(self, question: str, response: str, 
                                             sources: List[Dict], tone_reference: Optional[Dict] = None) -> Dict[str, any]:
        """Enhanced quality evaluation using the dashboard"""
        try:
            # Get your existing quality scores (relevance, accuracy, etc.)
            raw_scores = self.quality_evaluator.evaluate_response(question, response, sources)
            
            # Use dashboard to calculate weighted scores and detailed breakdown
            dashboard_analysis = self.quality_dashboard.calculate_weighted_score(raw_scores)
            
            # Add tone adherence if available
            if tone_reference:
                tone_score = await self._evaluate_tone_adherence(response, tone_reference)
                raw_scores['tone_adherence'] = tone_score
                dashboard_analysis['tone_analysis'] = {
                    'tone_score': tone_score,
                    'tone_reference_used': tone_reference['filename']
                }
            
            return {
                'raw_scores': raw_scores,
                'dashboard_analysis': dashboard_analysis,
                'detailed_breakdown': dashboard_analysis['component_breakdown'],
                'improvement_suggestions': dashboard_analysis['improvement_suggestions'],
                'overall_grade': dashboard_analysis['overall_grade']
            }
            
        except Exception as e:
            logger.error(f"Dashboard quality evaluation failed: {e}")
            return self._get_fallback_quality_scores()
    
    def _check_initialized(self):
        """Check if service is properly initialized"""
        if not self._initialized:
            error_msg = f"RAGService not properly initialized: {self._initialization_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        if not self.embedding_model or not self.llm_model:
            raise RuntimeError("Required models not loaded")

    @contextmanager
    def get_db_session(self):
        """Context manager for database sessions"""
        db = SessionLocal()
        try:
            yield db
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _get_default_quality_scores(self) -> Dict[str, float]:
        """Get default quality scores when evaluator is not available"""
        return {
            "relevance": 0.5,
            "accuracy": 0.5,
            "completeness": 0.5,
            "coherence": 0.5,
            "citation": 0.5,
            "overall": 0.5
        }

    def _vector_search(self, question_embedding: List[float], limit: int = 3, doc_ids: Optional[List[str]] = None) -> List[Dict]:
        """Perform vector similarity search with pgvector support"""
        logger.info(f"Starting vector search (limit={limit}, doc_ids={doc_ids})")
        
        # Validate embedding dimensions
        if not question_embedding:
            raise ValueError("Question embedding is empty")
        
        if len(question_embedding) != 768:
            raise ValueError(f"Invalid embedding dimension: got {len(question_embedding)}, expected 768")
        
        with self.get_db_session() as db:
            try:
                query = db.query(
                    DocumentChunk.content,
                    Document.filename,
                    DocumentChunk.chunk_index,
                    DocumentChunk.id.label('id'),
                    Document.id.label('document_id')
                ).join(Document, Document.id == DocumentChunk.document_id)
                
                # Filter by document IDs if provided
                if doc_ids:
                    try:
                        normalized_ids = [uuid.UUID(doc_id) for doc_id in doc_ids if doc_id]
                        if normalized_ids:
                            query = query.filter(Document.id.in_(normalized_ids))
                            logger.info(f"Filtering by {len(normalized_ids)} document IDs")
                    except ValueError as e:
                        logger.error(f"Invalid document ID format: {e}")
                
                # Use pgvector's cosine_distance method
                try:
                    query = query.order_by(DocumentChunk.embedding.cosine_distance(question_embedding))
                    logger.info("✅ Using pgvector cosine distance")
                except Exception as e:
                    logger.warning(f"Vector search failed, using fallback: {e}")
                    query = query.order_by(DocumentChunk.id)
                
                results = query.limit(limit).all()
                
                search_results = []
                for row in results:
                    search_results.append({
                        "id": str(row.id),
                        "document_id": str(row.document_id),
                        "content": row.content or "",
                        "filename": row.filename or "Unknown",
                        "chunk_index": row.chunk_index or 0
                    })
                
                logger.info(f"Vector search completed: found {len(search_results)} results")
                return search_results
                
            except Exception as e:
                logger.exception("Vector search failed")
                raise

    def _build_context(self, search_results: List[Dict]) -> str:
        """Build context string from search results"""
        context_parts = []
        for result in search_results:
            content = (result.get('content') or '')[:500]
            filename = result.get('filename', 'Unknown')
            chunk_idx = result.get('chunk_index', 0)
            context_parts.append(f"[Source: {filename} (Chunk {chunk_idx})]\n{content}")
        
        return "\n\n".join(context_parts)

    def _create_prompt(self, context: str, question: str) -> str:
        """Create optimized prompt for better responses"""
        return f"""Answer the question based on the provided context. If the context doesn't contain enough information, acknowledge this limitation clearly.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    async def search_and_generate(self, question: str, max_sources: int = 3, doc_ids: Optional[List[str]] = None) -> Dict:
        """Search and generate response with quality evaluation"""
        logger.info(f"Processing query: {question[:100]}...")
        
        # Check initialization
        self._check_initialized()
        
        try:
            start_time = time.time()
            
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")
            
            # Get embeddings
            logger.info("Generating embeddings")
            embeddings_result = self.embedding_model.get_embeddings([question.strip()])
            if not embeddings_result or not embeddings_result[0].values:
                raise ValueError("Failed to generate embeddings")
            
            question_embedding = embeddings_result[0].values
            
            # Validate dimensions
            if len(question_embedding) != 768:
                raise ValueError(f"Invalid embedding dimension: got {len(question_embedding)}, expected 768")
            
            # Perform search
            search_results = self._vector_search(question_embedding, limit=max_sources, doc_ids=doc_ids)
            
            if not search_results:
                return {
                    "answer": "No relevant information found in the selected documents.",
                    "sources": [],
                    "response_time": time.time() - start_time,
                    "tokens_used": 0,
                    "quality_scores": self._get_default_quality_scores()
                }
            
            # Build context and generate response
            context = self._build_context(search_results)
            prompt = self._create_prompt(context, question)
            
            logger.info("Generating LLM response")
            response = self.llm_model.generate_content(
                contents=prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 1000}
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from LLM")
            
            answer = response.text
            tokens_used = response.usage_metadata.total_token_count if response.usage_metadata else 0
            
            # Evaluate quality
            try:
                if self.quality_evaluator:
                    quality_scores = self.quality_evaluator.evaluate_response(question, answer, search_results)
                else:
                    quality_scores = self._get_default_quality_scores()
            except Exception as e:
                logger.error(f"Quality evaluation failed: {e}")
                quality_scores = self._get_default_quality_scores()
            
            return {
                "answer": answer,
                "sources": search_results,
                "response_time": time.time() - start_time,
                "tokens_used": tokens_used,
                "quality_scores": quality_scores
            }
            
        except Exception as e:
            logger.exception("Search and generate failed")
            raise

    async def stream_search_and_generate(self, question: str, max_sources: int = 3, 
                                     query_id: Optional[str] = None, 
                                     session_id: Optional[str] = None,
                                     doc_ids: Optional[List[str]] = None,
                                     model_key: str = "gemini-1.5-pro") -> AsyncGenerator[Dict, None]:
        """Stream search and generate with quality evaluation"""
        logger.info(f"=== STREAMING QUERY START ===")
        logger.info(f"Question: {question[:100]}...")
        logger.info(f"Query ID: {query_id}")
        logger.info(f"Session ID: {session_id}")
        tone_reference = None
        # FIXED: Ensure query_id is UUID
        if not query_id:
            query_id = str(uuid.uuid4())
        
        try:
            # Check initialization FIRST
            self._check_initialized()
            logger.info("✅ RAG service initialization check passed")
            
            start_time = time.time()
            
            yield {
                "type": "metadata",
                "question": question,
                "query_id": query_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "status": "starting"
            }
            
            if not question or not question.strip():
                logger.warning("Empty question provided")
                yield {
                    "type": "error",
                    "message": "Question cannot be empty",
                    "is_complete": True
                }
                return
            
            # Get embeddings
            logger.info("Generating embeddings for streaming query")
            try:
                embeddings_result = self.embedding_model.get_embeddings([question.strip()])
                if not embeddings_result or not embeddings_result[0].values:
                    raise ValueError("Failed to generate embeddings")
                
                question_embedding = embeddings_result[0].values
                
                if len(question_embedding) != 768:
                    yield {
                        "type": "error",
                        "message": f"Invalid embedding dimension: got {len(question_embedding)}, expected 768",
                        "is_complete": True
                    }
                    return
                
                logger.info("✅ Embeddings generated successfully")
                
            except Exception as e:
                logger.exception("Embedding generation failed")
                yield {
                    "type": "error",
                    "message": f"Failed to generate embeddings: {str(e)}",
                    "is_complete": True
                }
                return
            
            # Perform search
            logger.info("Performing vector search")
            try:
                search_results = self._vector_search(question_embedding, limit=max_sources, doc_ids=doc_ids)
            except Exception as e:
                logger.exception("Vector search failed")
                yield {
                    "type": "error",
                    "message": f"Search failed: {str(e)}",
                    "is_complete": True
                }
                return
            
            if not search_results:
                logger.warning("No search results found")
                yield {
                    "type": "complete",
                    "full_response": "No relevant information found in the selected documents.",
                    "sources": [],
                    "response_time": time.time() - start_time,
                    "tokens_used": 0,
                    "quality_scores": self._get_default_quality_scores(),
                    "is_complete": True
                }
                return
            
            # Build context and stream response
            logger.info(f"Building context from {len(search_results)} results")
            context = self._build_context(search_results)
            prompt = self._create_prompt(context, question)
            
            logger.info("Starting LLM streaming")
            try:
                response_stream = self.llm_model.generate_content(
                    contents=prompt,
                    generation_config={"temperature": 0.1, "max_output_tokens": 1000},
                    stream=True
                )
                
                full_response = ""
                chunk_count = 0
                tokens_used = 0
                
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        chunk_count += 1
                        yield {
                            "type": "content",
                            "chunk": chunk.text,
                            "chunk_num": chunk_count,
                            "timestamp": datetime.now().isoformat()
                        }
                    
                    if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                        tokens_used = chunk.usage_metadata.total_token_count
                
                # Evaluate quality
                logger.info("Evaluating response quality")
                try:                
                    # Get your existing quality scores first (keep this!)
                    quality_scores = await self._evaluate_quality_with_tone(question, full_response, search_results, tone_reference)
                    
                    # THEN enhance with dashboard analysis
                    if hasattr(self, 'quality_dashboard'):
                        try:
                            # Extract raw scores for dashboard analysis
                            raw_scores = {
                                'relevance': quality_scores.get('relevance', 0.5),
                                'accuracy': quality_scores.get('accuracy', 0.5),
                                'completeness': quality_scores.get('completeness', 0.5),
                                'coherence': quality_scores.get('coherence', 0.5),
                                'citation': quality_scores.get('citation', 0.5)
                            }
                            
                            # Get dashboard analysis
                            dashboard_analysis = self.quality_dashboard.calculate_weighted_score(raw_scores)
                            
                            # Add dashboard data to quality_scores (don't replace!)
                            quality_scores['dashboard_analysis'] = dashboard_analysis
                            quality_scores['overall_grade'] = dashboard_analysis['overall_grade']
                            quality_scores['improvement_suggestions'] = dashboard_analysis['improvement_suggestions']
                            
                            # Yield dashboard-specific data
                            yield {
                                "type": "quality_analysis",
                                "overall_score": dashboard_analysis['overall_score'],
                                "overall_grade": dashboard_analysis['overall_grade'],
                                "component_scores": dashboard_analysis['component_breakdown'],
                                "improvement_suggestions": dashboard_analysis['improvement_suggestions'],
                                "calculation_formula": dashboard_analysis['calculation_formula']
                            }
                            
                        except Exception as e:
                            logger.error(f"Dashboard analysis failed: {e}")
                            # Continue with original quality_scores if dashboard fails
                    
                        # Quality evaluation (add this before the final "complete" yield)
                        logger.info("Evaluating response quality")
                        try:
                            # Use basic quality evaluation
                            quality_scores = self._evaluate_basic_quality(question, full_response, search_results)
                            
                            # Yield quality analysis
                            yield {
                                "type": "quality_analysis",
                                "overall_score": quality_scores.get('overall_score', 0.5),
                                "overall_percentage": quality_scores.get('overall_percentage', 50.0),
                                "overall_grade": quality_scores.get('overall_grade', 'C'),
                                "component_scores": quality_scores.get('component_breakdown', {}),
                                "improvement_suggestions": quality_scores.get('improvement_suggestions', []),
                                "raw_scores": quality_scores.get('raw_scores', {})
                            }
                        except Exception as quality_error:
                            logger.error(f"Quality evaluation failed: {quality_error}")
                            # Yield simple quality data
                            yield {
                                "type": "quality_analysis",
                                "overall_score": 0.6,
                                "overall_percentage": 60.0,
                                "overall_grade": 'C+',
                                "component_scores": {},
                                "improvement_suggestions": ["Response generated successfully"],
                                "raw_scores": {'relevance': 0.6, 'accuracy': 0.6, 'completeness': 0.6, 'coherence': 0.6, 'citation': 0.6}
                            } 
                        # Keep your existing complete response structure
                        yield {
                            "type": "complete",
                            "full_response": full_response,
                            "sources": search_results,
                            "tone_reference": tone_reference,  # This should be defined from earlier in your method
                            "response_time": time.time() - start_time,
                            "tokens_used": tokens_used,
                            "quality_scores": quality_scores,  # This preserves your original structure + dashboard enhancements
                            "is_complete": True
                        }
                except Exception as e:
                    logger.exception("Evaluation failed")
                    yield {
                        "type": "error",
                        "message": f"Response generation failed: {str(e)}",
                        "is_complete": True
                    }
                logger.info(f"✅ Streaming completed in {time.time() - start_time:.2f}s")
                
            except Exception as e:
                logger.exception("LLM streaming failed")
                yield {
                    "type": "error",
                    "message": f"Response generation failed: {str(e)}",
                    "is_complete": True
                }
                
        except Exception as e:
            logger.exception("=== STREAMING QUERY FAILED ===")
            yield {
                "type": "error",
                "message": f"Stream failed: {str(e)}",
                "is_complete": True
            }

    def store_document(self, filename: str, file_content: bytes) -> str:
        """Store document with comprehensive error handling"""
        self._check_initialized()
        
        logger.info(f"Storing document: {filename}")
        with self.get_db_session() as db:
            try:
                text = file_content.decode('utf-8', errors='ignore')
                if not text.strip():
                    raise ValueError("Document content is empty")
                
                text_hash = hashlib.md5(text.encode()).hexdigest()
                existing_doc = db.query(Document).filter(Document.content_hash == text_hash).first()
                if existing_doc:
                    logger.warning(f"Document already exists: {existing_doc.id}")
                    return str(existing_doc.id)
                
                doc = Document(
                    filename=filename,
                    content=text,
                    content_hash=text_hash
                )
                db.add(doc)
                db.flush()
                doc_id = doc.id
                
                chunks = self.chunk_text(text)
                if not chunks:
                    raise ValueError("No chunks generated")
                
                successful_chunks = 0
                for i, chunk_text in enumerate(chunks):
                    try:
                        embedding_result = self.embedding_model.get_embeddings([chunk_text])[0]
                        chunk = DocumentChunk(
                            document_id=doc_id,
                            content=chunk_text,
                            embedding=embedding_result.values,
                            chunk_index=i,
                            doc_metadata={"filename": filename, "chunk_range": f"{i}-{i+len(chunk_text.split())}"}
                        )
                        db.add(chunk)
                        successful_chunks += 1
                    except Exception as e:
                        logger.error(f"Failed to create chunk {i}: {e}")
                        continue
                
                if successful_chunks == 0:
                    raise ValueError("Failed to create any chunks")
                
                db.commit()
                logger.info(f"Successfully stored {filename} with {successful_chunks} chunks")
                return str(doc_id)
                
            except Exception as e:
                logger.exception("Error storing document")
                raise

    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Text chunking with overlap"""
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            chunk = text[start:end]
            
            if len(chunk.strip()) > 50:
                chunks.append(chunk.strip())
            
            start = end - overlap
            if start >= text_length:
                break
        
        return chunks

    async def _evaluate_quality_with_tone(self, question: str, response: str, sources: List[Dict], 
                                    tone_reference: Optional[Dict] = None) -> Dict[str, float]:
        """Enhanced quality evaluation with dashboard analysis (tone analysis disabled for now)"""
        try:
            # Get base quality scores from your existing evaluator
            base_scores = self.quality_evaluator.evaluate_response(question, response, sources)
            
            # Add dashboard analysis if available
            if hasattr(self, 'quality_dashboard') and self.quality_dashboard:
                try:
                    # Extract raw scores for dashboard analysis
                    raw_scores = {
                        'relevance': base_scores.get('relevance', 0.5),
                        'accuracy': base_scores.get('accuracy', 0.5),
                        'completeness': base_scores.get('completeness', 0.5),
                        'coherence': base_scores.get('coherence', 0.5),
                        'citation': base_scores.get('citation', 0.5)
                    }
                    
                    # Get dashboard analysis
                    dashboard_analysis = self.quality_dashboard.calculate_weighted_score(raw_scores)
                    
                    # Add dashboard data to base_scores
                    base_scores['dashboard_analysis'] = dashboard_analysis
                    base_scores['overall_grade'] = dashboard_analysis['overall_grade']
                    base_scores['improvement_suggestions'] = dashboard_analysis['improvement_suggestions']
                    base_scores['component_breakdown'] = dashboard_analysis['component_breakdown']
                    
                    logger.info(f"Dashboard analysis completed - Overall grade: {dashboard_analysis['overall_grade']}")
                    
                except Exception as e:
                    logger.error(f"Dashboard analysis failed: {e}")
                    # Continue with base scores only
            
            # Add tone reference info if provided (just for information, no analysis yet)
            if tone_reference:
                base_scores['tone_reference_used'] = tone_reference.get('filename', 'Unknown')
                base_scores['tone_authority_level'] = tone_reference.get('authority_level', 0)
            
            return base_scores
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            # Return default scores
            return {
                'relevance': 0.5, 'accuracy': 0.5, 'completeness': 0.5, 
                'coherence': 0.5, 'citation': 0.5, 'overall': 0.5
            }

    async def stream_search_and_generate_with_quality(
        self,
        question: str,
        max_sources: int = 3,
        query_id: Optional[str] = None,
        session_id: Optional[str] = None,
        doc_ids: Optional[List[str]] = None,
        model_key: str = "gemini-1.5-pro"
    ) -> AsyncGenerator[Dict, None]:
        """
        Enhanced streaming with model selection and unified quality evaluation.
        Maintains all existing functionality while adding model choice capability.
        """
        logger.info("=== ENHANCED STREAMING START ===")
        logger.info(f"Model: {model_key}")
        logger.info(f"Question: {question[:100]}...")
        
        # Initialize
        if not query_id:
            query_id = str(uuid.uuid4())
        
        try:
            # Basic initialization check (keep your existing logic)
            if hasattr(self, "_check_initialized"):
                self._check_initialized()
            
            start_time = time.time()
            
            # Metadata yield (matches your existing structure)
            yield {
                "type": "metadata",
                "question": question,
                "query_id": query_id,
                "session_id": session_id,
                "model_used": model_key,
                "timestamp": datetime.now().isoformat(),
                "status": "starting"
            }
            
            # Validate input
            if not question or not question.strip():
                yield {"type": "error", "message": "Question cannot be empty", "is_complete": True}
                return
            
            # 1. EMBEDDINGS (keep your exact existing logic)
            logger.info("Generating embeddings")
            try:
                embeddings_result = self.embedding_model.get_embeddings([question.strip()])
                if not embeddings_result or not getattr(embeddings_result[0], "values", None):
                    raise ValueError("Failed to generate embeddings")
                
                question_embedding = embeddings_result[0].values
                
                if len(question_embedding) != 768:
                    yield {
                        "type": "error",
                        "message": f"Invalid embedding dimension: got {len(question_embedding)}, expected 768",
                        "is_complete": True
                    }
                    return
                
                logger.info("✅ Embeddings generated successfully")
                
            except Exception as e:
                logger.exception("Embedding generation failed")
                yield {"type": "error", "message": f"Failed to generate embeddings: {str(e)}", "is_complete": True}
                return
            
            # 2. VECTOR SEARCH (keep your exact existing logic)
            logger.info("Performing vector search")
            try:
                search_results = self._vector_search(question_embedding, limit=max_sources, doc_ids=doc_ids)
            except Exception as e:
                logger.exception("Vector search failed")
                yield {"type": "error", "message": f"Search failed: {str(e)}", "is_complete": True}
                return
            
            # Handle no search results with proper quality scoring
            if not search_results:
                logger.warning("No search results found")
                no_data_quality = self._create_no_data_quality_scores()
                
                yield {"type": "quality_analysis", **no_data_quality}
                yield {
                    "type": "complete",
                    "full_response": "No relevant information found in the selected documents.",
                    "sources": [],
                    "response_time": time.time() - start_time,
                    "tokens_used": 0,
                    "model_used": model_key,
                    "quality_scores": no_data_quality,
                    "query_id": query_id,
                    "session_id": session_id,
                    "is_complete": True
                }
                return
            
            # Yield sources early (matches your existing pattern)
            yield {"type": "sources", "sources": search_results, "count": len(search_results)}
            
            # 3. CONTEXT AND PROMPT (keep your existing logic)
            logger.info(f"Building context from {len(search_results)} results")
            context = self._build_context(search_results)
            prompt = self._create_prompt(context, question)
            
            # 4. GENERATION WITH MODEL SELECTION
            logger.info(f"Starting generation with {model_key}")
            full_response = ""
            tokens_used = 0
            
            try:
                # Try model manager first (if available)
                if MODEL_MANAGER_AVAILABLE and model_manager:
                    logger.info("Using model manager for generation")
                    chunk_count = 0
                    async for gen_chunk in model_manager.generate_response(model_key, prompt):
                        if gen_chunk.get("type") == "content":
                            content = gen_chunk.get("content", "")
                            full_response += content
                            tokens_used += gen_chunk.get("tokens", 0)
                            chunk_count += 1
                            
                            # Yield content chunk (matches your existing structure)
                            yield {
                                "type": "content",
                                "chunk": content,
                                "chunk_num": chunk_count,
                                "timestamp": datetime.now().isoformat()
                            }
                        elif gen_chunk.get("type") == "error":
                            # Fall back to direct model
                            raise RuntimeError(gen_chunk.get("message", "Model manager generation error"))
                else:
                    # Force fallback if model manager not available
                    raise RuntimeError("Model manager not available")
                    
            except Exception as manager_error:
                # FALLBACK: Use your existing direct model approach
                logger.warning(f"Model manager failed ({manager_error}), using fallback to direct model")
                try:
                    response_stream = self.llm_model.generate_content(
                        contents=prompt,
                        generation_config={"temperature": 0.1, "max_output_tokens": 1000},
                        stream=True
                    )
                    
                    chunk_count = 0
                    for chunk in response_stream:
                        if hasattr(chunk, 'text') and chunk.text:
                            full_response += chunk.text
                            chunk_count += 1
                            
                            # Yield content chunk (matches your existing structure)
                            yield {
                                "type": "content",
                                "chunk": chunk.text,
                                "chunk_num": chunk_count,
                                "timestamp": datetime.now().isoformat()
                            }
                        
                        if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                            tokens_used = getattr(chunk.usage_metadata, 'total_token_count', tokens_used)
                            
                except Exception as fallback_error:
                    logger.exception("Fallback generation also failed")
                    yield {
                        "type": "error", 
                        "message": f"Response generation failed: {str(fallback_error)}", 
                        "is_complete": True
                    }
                    return
            
            # Validate response
            if not full_response.strip():
                yield {"type": "error", "message": "Generated empty response", "is_complete": True}
                return
            
            # 5. UNIFIED QUALITY EVALUATION (single evaluation, no duplication)
            logger.info("Evaluating response quality")
            try:
                quality_analysis = self._unified_quality_evaluation(question, full_response, search_results)
                
                # Single quality analysis yield
                yield {
                    "type": "quality_analysis",
                    "overall_score": quality_analysis.get('overall_score', 0.5),
                    "overall_percentage": quality_analysis.get('overall_percentage', 50.0),
                    "overall_grade": quality_analysis.get('overall_grade', 'C'),
                    "component_breakdown": quality_analysis.get('component_breakdown', {}),
                    "improvement_suggestions": quality_analysis.get('improvement_suggestions', []),
                    "raw_scores": quality_analysis.get('raw_scores', {}),
                    "calculation_formula": quality_analysis.get('calculation_formula', '')
                }
                
            except Exception as quality_error:
                logger.error(f"Quality evaluation failed: {quality_error}")
                # Simple fallback quality data
                quality_analysis = {
                    "overall_score": 0.6,
                    "overall_percentage": 60.0,
                    "overall_grade": "C+",
                    "component_breakdown": {},
                    "improvement_suggestions": ["Quality evaluation temporarily unavailable"],
                    "raw_scores": {'relevance': 0.6, 'accuracy': 0.6, 'completeness': 0.6, 'coherence': 0.6, 'citation': 0.6},
                    "calculation_formula": "Default scoring"
                }
                yield {"type": "quality_analysis", **quality_analysis}
            
            # 6. FINAL COMPLETION (matches your existing structure)
            yield {
                "type": "complete",
                "full_response": full_response,
                "sources": search_results,
                "response_time": time.time() - start_time,
                "tokens_used": tokens_used,
                "model_used": model_key,
                "quality_scores": quality_analysis,
                "query_id": query_id,
                "session_id": session_id,
                "is_complete": True
            }
            
            logger.info(f"✅ Enhanced streaming completed in {time.time() - start_time:.2f}s using {model_key}")
            
        except Exception as e:
            logger.exception("=== ENHANCED STREAMING FAILED ===")
            yield {
                "type": "error",
                "message": f"Stream failed: {str(e)}",
                "model_used": model_key,
                "is_complete": True
            }

    async def _generate_response_fallback(self, prompt: str) -> str:
        """Fallback response generation"""
        try:
            # Try to use your existing model
            if hasattr(self, 'model') and self.model:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        top_p=0.8,
                        top_k=40,
                        max_output_tokens=2048,
                        candidate_count=1
                    )
                )
                return response.text if hasattr(response, 'text') else str(response)
            
            # Direct Google AI call as fallback
            import google.generativeai as genai
            model = genai.GenerativeModel('gemini-1.5-pro')
            response = model.generate_content(prompt)
            return response.text if hasattr(response, 'text') else str(response)
            
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    async def _stream_generate_response(self, prompt: str) -> AsyncGenerator[Dict, None]:
        """Stream response generation using existing methods"""
        try:
            # Use your existing generation logic but with streaming
            response_text = ""
            
            # If you have an existing generate method, adapt it for streaming
            if hasattr(self, 'model') and self.model:
                try:
                    # Generate with your existing model
                    response = self.model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.3,
                            top_p=0.8,
                            top_k=40,
                            max_output_tokens=2048,
                            candidate_count=1
                        ),
                        stream=True
                    )
                    
                    for chunk in response:
                        if hasattr(chunk, 'text') and chunk.text:
                            content = chunk.text
                            response_text += content
                            yield {
                                "type": "content",
                                "content": content,
                                "tokens": len(content.split())
                            }
                    
                except Exception as model_error:
                    logger.error(f"Model generation failed: {model_error}")
                    # Fallback to non-streaming generation
                    try:
                        response = self.model.generate_content(prompt)
                        if hasattr(response, 'text') and response.text:
                            content = response.text
                            yield {
                                "type": "content",
                                "content": content,
                                "tokens": len(content.split())
                            }
                        else:
                            raise Exception("No response generated")
                    except Exception as fallback_error:
                        logger.error(f"Fallback generation failed: {fallback_error}")
                        yield {
                            "type": "error",
                            "message": "Failed to generate response",
                            "is_complete": True
                        }
            else:
                # If no model available, return error
                yield {
                    "type": "error",
                    "message": "Generation model not available",
                    "is_complete": True
                }
                
        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield {
                "type": "error",
                "message": f"Generation failed: {str(e)}",
                "is_complete": True
            }

    def _build_enhanced_context(self, search_results: List[Dict], question: str) -> str:
        """Build enhanced context for better responses"""
        context_parts = []
        
        # Add question context
        context_parts.append(f"User Question: {question}\n")
        
        # Add sources with better formatting
        context_parts.append("Relevant Information Sources:")
        for i, result in enumerate(search_results, 1):
            filename = result.get('filename', 'Unknown')
            content = result.get('content', '')
            relevance = result.get('relevance_score', 0)
            
            # Limit content length but ensure it's substantial
            if len(content) > 800:
                content = content[:800] + "..."
            
            context_parts.append(f"""
    Source {i} [{filename}] (Relevance: {relevance:.2f}):
    {content}
    """)
        
        return "\n".join(context_parts)

    def _create_comprehensive_prompt(self, context: str, question: str) -> str:
        """Create a prompt that encourages comprehensive responses"""
        return f"""You are an expert assistant providing detailed, accurate information. Please provide a comprehensive answer that:

    1. Directly addresses all aspects of the user's question
    2. Uses specific information from the provided sources
    3. Provides detailed explanations with examples when relevant
    4. Maintains professional tone while being accessible
    5. Cites relevant sources when making specific claims
    6. Includes practical implications or next steps when appropriate

    IMPORTANT: Provide a thorough response (aim for 200-400 words when the question warrants it). Be specific and detailed rather than brief.

    CONTEXT:
    {context}

    USER QUESTION: {question}

    Please provide a detailed, comprehensive response:"""

    async def _evaluate_response_quality(self, question: str, response: str, sources: List[Dict]) -> Dict[str, Any]:
        """Evaluate response quality with dashboard integration"""
        try:
            # Use existing quality evaluator if available
            if hasattr(self, 'quality_evaluator') and self.quality_evaluator:
                base_scores = self.quality_evaluator.evaluate_response(question, response, sources)
            else:
                # Use basic evaluation
                base_scores = self._basic_quality_evaluation(question, response, sources)
            
            # Use quality dashboard if available
            if hasattr(self, 'quality_dashboard') and self.quality_dashboard:
                try:
                    raw_scores = {
                        'relevance': base_scores.get('relevance', 0.5),
                        'accuracy': base_scores.get('accuracy', 0.5),
                        'completeness': base_scores.get('completeness', 0.5),
                        'coherence': base_scores.get('coherence', 0.5),
                        'citation': base_scores.get('citation', 0.5)
                    }
                    
                    dashboard_analysis = self.quality_dashboard.calculate_weighted_score(raw_scores)
                    
                    # Merge dashboard analysis with base scores
                    result = {**base_scores}
                    result.update({
                        'overall_score': dashboard_analysis['overall_score'],
                        'overall_percentage': dashboard_analysis['overall_percentage'],
                        'overall_grade': dashboard_analysis['overall_grade'],
                        'component_breakdown': dashboard_analysis['component_breakdown'],
                        'improvement_suggestions': dashboard_analysis['improvement_suggestions']
                    })
                    
                    logger.info(f"Quality analysis completed - Grade: {result['overall_grade']}")
                    return result
                    
                except Exception as dashboard_error:
                    logger.error(f"Dashboard analysis failed: {dashboard_error}")
            
            # Add basic overall calculation if not from dashboard
            if 'overall_score' not in base_scores:
                base_scores['overall_score'] = sum(base_scores.get(k, 0.5) for k in ['relevance', 'accuracy', 'completeness', 'coherence', 'citation']) / 5
                base_scores['overall_percentage'] = base_scores['overall_score'] * 100
                base_scores['overall_grade'] = self._score_to_grade(base_scores['overall_score'])
            
            return base_scores
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return self._get_default_quality_scores()

    def _basic_quality_evaluation(self, question: str, response: str, sources: List[Dict]) -> Dict[str, float]:
        """Basic quality evaluation with stricter scoring for poor responses"""
        
        # Check for "no data" responses
        no_data_indicators = [
            "no relevant information",
            "no data available", 
            "cannot find",
            "don't have information",
            "no sources found",
            "unable to find",
            "no information available",
            "couldn't find"
        ]
        
        response_lower = response.lower()
        is_no_data_response = any(indicator in response_lower for indicator in no_data_indicators)
        
        # If it's a "no data" response, score very low
        if is_no_data_response:
            return {
                'relevance': 0.3,
                'accuracy': 0.3,
                'completeness': 0.2,
                'coherence': 0.5,
                'citation': 0.1,
                'overall': 0.28
            }
        
        # Normal scoring for actual responses
        word_count = len(response.split())
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Basic relevance
        relevance = min(1.0, len(question_words & response_words) / len(question_words)) if question_words else 0.5
        relevance = max(0.3, relevance)
        
        # Completeness based on length
        completeness = min(1.0, word_count / 150)
        completeness = max(0.3, completeness)
        
        # Accuracy based on sources
        accuracy = 0.8 if len(sources) > 0 else 0.4
        
        # Coherence
        sentence_count = len([s for s in response.split('.') if s.strip()])
        coherence = min(1.0, max(0.4, word_count / (sentence_count * 10))) if sentence_count > 0 else 0.5
        
        # Citation
        citation = 0.7 if sources else 0.3
        
        return {
            'relevance': relevance,
            'accuracy': accuracy,
            'completeness': completeness,
            'coherence': coherence,
            'citation': citation,
            'overall': (relevance + accuracy + completeness + coherence + citation) / 5
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.95: return 'A+'
        elif score >= 0.90: return 'A'
        elif score >= 0.85: return 'A-'
        elif score >= 0.80: return 'B+'
        elif score >= 0.75: return 'B'
        elif score >= 0.70: return 'B-'
        elif score >= 0.65: return 'C+'
        elif score >= 0.60: return 'C'
        elif score >= 0.55: return 'C-'
        elif score >= 0.50: return 'D'
        else: return 'F'


    def _evaluate_basic_quality(self, question: str, response: str, sources: List[Dict]) -> Dict[str, Any]:
        """Simple quality evaluation that works with your existing code"""
        try:
            word_count = len(response.split())
            question_words = set(question.lower().split())
            response_words = set(response.lower().split())
            
            # Basic scoring
            relevance = min(1.0, len(question_words & response_words) / len(question_words)) if question_words else 0.5
            relevance = max(0.3, relevance)
            
            completeness = min(1.0, word_count / 100)
            completeness = max(0.4, completeness)
            
            accuracy = 0.8 if len(sources) > 0 else 0.6
            coherence = min(1.0, max(0.5, word_count / 50))
            citation = 0.7 if sources else 0.5
            
            overall_score = (relevance + accuracy + completeness + coherence + citation) / 5
            
            raw_scores = {
                'relevance': relevance,
                'accuracy': accuracy,
                'completeness': completeness,
                'coherence': coherence,
                'citation': citation
            }
            
            result = {
                'overall_score': overall_score,
                'overall_percentage': overall_score * 100,
                'overall_grade': self._score_to_grade(overall_score),
                'raw_scores': raw_scores,
                'improvement_suggestions': []
            }
            
            # Try dashboard if available
            if hasattr(self, 'quality_dashboard') and self.quality_dashboard:
                try:
                    dashboard_analysis = self.quality_dashboard.calculate_weighted_score(raw_scores)
                    result.update({
                        'overall_score': dashboard_analysis['overall_score'],
                        'overall_percentage': dashboard_analysis['overall_percentage'],
                        'overall_grade': dashboard_analysis['overall_grade'],
                        'component_breakdown': dashboard_analysis['component_breakdown'],
                        'improvement_suggestions': dashboard_analysis['improvement_suggestions']
                    })
                except Exception:
                    pass  # Use basic scores
            
            return result
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return {
                'overall_score': 0.5,
                'overall_percentage': 50.0,
                'overall_grade': 'C',
                'raw_scores': {'relevance': 0.5, 'accuracy': 0.5, 'completeness': 0.5, 'coherence': 0.5, 'citation': 0.5},
                'improvement_suggestions': ['Basic quality assessment applied']
            }


    def _unified_quality_evaluation(self, question: str, response: str, sources: List[Dict]) -> Dict[str, Any]:
        """Single, unified quality evaluation method - no duplication"""
        try:
            # Check for "no data" responses first and penalize heavily
            no_data_indicators = [
                "no relevant information", "no data available", "cannot find",
                "don't have information", "no sources found", "unable to find",
                "does not contain information", "insufficient information"
            ]
            
            response_lower = response.lower()
            is_no_data_response = any(indicator in response_lower for indicator in no_data_indicators)
            
            if is_no_data_response:
                return self._create_no_data_quality_scores()
            
            # Normal quality evaluation for actual responses
            word_count = len(response.split())
            question_words = set(question.lower().split())
            response_words = set(response.lower().split())
            
            # Calculate component scores
            relevance = min(1.0, len(question_words & response_words) / len(question_words)) if question_words else 0.5
            relevance = max(0.3, relevance)
            
            completeness = min(1.0, word_count / 150)  # 150 words = complete
            completeness = max(0.3, completeness)
            
            accuracy = 0.8 if sources else 0.4
            
            sentence_count = len([s for s in response.split('.') if s.strip()])
            coherence = min(1.0, max(0.4, word_count / (sentence_count * 10))) if sentence_count > 0 else 0.5
            
            citation = 0.7 if sources else 0.3
            
            raw_scores = {
                'relevance': relevance,
                'accuracy': accuracy,
                'completeness': completeness,
                'coherence': coherence,
                'citation': citation
            }
            
            # Use quality dashboard if available
            if hasattr(self, 'quality_dashboard') and self.quality_dashboard:
                try:
                    dashboard_analysis = self.quality_dashboard.calculate_weighted_score(raw_scores)
                    return {
                        'overall_score': dashboard_analysis['overall_score'],
                        'overall_percentage': dashboard_analysis['overall_percentage'],
                        'overall_grade': dashboard_analysis['overall_grade'],
                        'component_breakdown': dashboard_analysis['component_breakdown'],
                        'improvement_suggestions': dashboard_analysis['improvement_suggestions'],
                        'calculation_formula': dashboard_analysis['calculation_formula'],
                        'raw_scores': raw_scores
                    }
                except Exception as e:
                    logger.warning(f"Dashboard analysis failed: {e}")
            
            # Fallback calculation if dashboard not available
            overall_score = sum(raw_scores.values()) / len(raw_scores)
            return {
                'overall_score': overall_score,
                'overall_percentage': overall_score * 100,
                'overall_grade': self._score_to_grade(overall_score),
                'component_breakdown': {},
                'improvement_suggestions': ['Basic quality assessment applied'],
                'raw_scores': raw_scores,
                'calculation_formula': 'Average of component scores'
            }
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return self._get_default_quality_scores()

    def _create_no_data_quality_scores(self) -> Dict[str, Any]:
        """Create appropriate quality scores for 'no data' responses"""
        return {
            'overall_score': 0.25,
            'overall_percentage': 25.0,
            'overall_grade': 'F',
            'component_breakdown': {
                'relevance': {'percentage': 20.0, 'grade': 'F', 'weight': 0.25, 'weighted_contribution': 0.05},
                'accuracy': {'percentage': 30.0, 'grade': 'F', 'weight': 0.30, 'weighted_contribution': 0.09},
                'completeness': {'percentage': 10.0, 'grade': 'F', 'weight': 0.20, 'weighted_contribution': 0.02},
                'coherence': {'percentage': 50.0, 'grade': 'D', 'weight': 0.15, 'weighted_contribution': 0.075},
                'citation': {'percentage': 10.0, 'grade': 'F', 'weight': 0.10, 'weighted_contribution': 0.01}
            },
            'improvement_suggestions': [
                "⚠️ No relevant information found in knowledge base",
                "Consider expanding document corpus with relevant content",
                "Verify that appropriate documents have been uploaded and processed",
                "Try rephrasing the query with different keywords"
            ],
            'raw_scores': {
                'relevance': 0.2, 'accuracy': 0.3, 'completeness': 0.1, 
                'coherence': 0.5, 'citation': 0.1
            },
            'calculation_formula': 'Penalized scoring for no-data responses'
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to letter grade"""
        if score >= 0.95: return 'A+'
        elif score >= 0.90: return 'A'
        elif score >= 0.85: return 'A-'
        elif score >= 0.80: return 'B+'
        elif score >= 0.75: return 'B'
        elif score >= 0.70: return 'B-'
        elif score >= 0.65: return 'C+'
        elif score >= 0.60: return 'C'
        elif score >= 0.55: return 'C-'
        elif score >= 0.50: return 'D'
        else: return 'F'

    def _get_default_quality_scores(self) -> Dict[str, Any]:
        """Return safe default quality scores when evaluation fails"""
        return {
            'overall_score': 0.5,
            'overall_percentage': 50.0,
            'overall_grade': 'C',
            'component_breakdown': {},
            'improvement_suggestions': ['Quality evaluation temporarily unavailable'],
            'raw_scores': {
                'relevance': 0.5, 'accuracy': 0.5, 'completeness': 0.5, 
                'coherence': 0.5, 'citation': 0.5
            },
            'calculation_formula': 'Default scoring applied'
        }



# Global instance with error handling
rag_service = None

try:
    logger.info("Creating global RAG service instance...")
    rag_service = RAGService()
    
    # Verify the instance has all required methods
    required_methods = ['stream_search_and_generate', 'search_and_generate', 'store_document']
    missing_methods = []
    
    for method in required_methods:
        if not hasattr(rag_service, method):
            missing_methods.append(method)
    
    if missing_methods:
        logger.error(f"RAG service missing methods: {missing_methods}")
        rag_service = None
    else:
        logger.info("✅ RAG service created successfully with all methods")
        
except Exception as e:
    logger.exception("Failed to create global RAG service instance")
    rag_service = None
