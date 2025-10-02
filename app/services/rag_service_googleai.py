import logging
from typing import List, Dict, AsyncGenerator, Optional
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
            
            self._initialized = True
            logger.info("=== RAGService Initialization Complete ===")
            
        except Exception as e:
            self._initialization_error = str(e)
            logger.exception("=== RAGService Initialization FAILED ===")
            # Don't raise - let the object exist but track the error
    
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

    async def stream_search_and_generate(self, question: str, max_sources: int = 3, doc_ids: Optional[List[str]] = None, query_id: Optional[str] = None, session_id: Optional[str] = None) -> AsyncGenerator[Dict, None]:
        """Stream search and generate with quality evaluation"""
        logger.info(f"=== STREAMING QUERY START ===")
        logger.info(f"Question: {question[:100]}...")
        logger.info(f"Query ID: {query_id}")
        logger.info(f"Session ID: {session_id}")
        
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
                    if self.quality_evaluator:
                        quality_scores = self.quality_evaluator.evaluate_response(question, full_response, search_results)
                    else:
                        quality_scores = self._get_default_quality_scores()
                except Exception as e:
                    logger.error(f"Quality evaluation failed: {e}")
                    quality_scores = self._get_default_quality_scores()
                
                yield {
                    "type": "complete",
                    "full_response": full_response,
                    "sources": search_results,
                    "response_time": time.time() - start_time,
                    "tokens_used": tokens_used,
                    "chunk_count": chunk_count,
                    "quality_scores": quality_scores,
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
