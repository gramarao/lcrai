import logging
from typing import List, Dict, AsyncGenerator
from sqlalchemy.orm import Session
from app.models.database import SessionLocal, Document, DocumentChunk
from datetime import datetime
import time
import hashlib
from google.cloud import aiplatform
from vertexai.preview.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        try:
            logger.debug("Initializing RAGService...")
            aiplatform.init(project="lcrai-472418", location="us-central1")
            self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
            self.llm_model = GenerativeModel("gemini-2.5-flash")
            logger.info("RAGService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAGService: {str(e)}")
            raise

    def _vector_search(self, question_embedding: List[float], limit: int = 3, doc_ids: list = None) -> List[Dict]:
        db = SessionLocal()
        try:
            logger.debug(f"Vector search with limit={limit}, doc_ids={doc_ids}")
            query = db.query(
                DocumentChunk.content,
                Document.filename,
                DocumentChunk.chunk_index,
                DocumentChunk.id.label('chunk_id')
            ).join(
                Document, Document.id == DocumentChunk.document_id
            ).order_by(
                DocumentChunk.embedding.op('<->')(question_embedding)
            )
            if doc_ids:
                query = query.filter(Document.id.in_(doc_ids))
            results = query.limit(limit).all()
            search_results = [
                {
                    "id": str(row.chunk_id),  # Convert UUID to string
                    "content": row.content,
                    "filename": row.filename or "Unknown",
                    "chunk_index": row.chunk_index
                } for row in results
            ]
            logger.debug(f"Found {len(search_results)} search results")
            return search_results
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            raise
        finally:
            db.close()

    async def search_and_generate(self, question: str, max_sources: int = 3, doc_ids: list = None) -> Dict:
        try:
            start_time = time.time()
            logger.debug(f"Processing query: {question[:50]}... with doc_ids: {doc_ids}")
            question_embedding = self.embedding_model.get_embeddings([question])[0].values
            search_results = self._vector_search(question_embedding, limit=max_sources, doc_ids=doc_ids)
            if not search_results:
                logger.warning("No search results found")
                return {
                    "answer": "No relevant information found in the selected documents.",
                    "sources": [],
                    "response_time": time.time() - start_time,
                    "tokens_used": 0
                }
            context = "\n\n".join(
                f"[Source: {result['filename']} (Chunk {result['chunk_index']})]\n{result['content'][:300]}"
                for result in search_results
            )
            prompt = f"""Answer based on the context below. If the context doesn't contain enough information, say so.
CONTEXT: {context}
QUESTION: {question}
ANSWER:"""
            response = self.llm_model.generate_content(
                contents=prompt,
                generation_config={"temperature": 0.1}
            )
            answer = response.text
            tokens_used = response.usage_metadata.total_token_count if response.usage_metadata else 0
            logger.info(f"Query completed in {time.time() - start_time:.2f}s")
            return {
                "answer": answer,
                "sources": search_results,
                "response_time": time.time() - start_time,
                "tokens_used": tokens_used
            }
        except Exception as e:
            logger.error(f"Search and generate failed: {str(e)}")
            raise

    async def stream_search_and_generate(self, question: str, max_sources: int = 3, doc_ids: list = None) -> AsyncGenerator[Dict, None]:
        logger.debug(f"Entering stream_search_and_generate with question: {question[:50]}...")
        try:
            start_time = time.time()
            yield {
                "type": "metadata",
                "question": question,
                "timestamp": datetime.now().isoformat(),
                "status": "starting"
            }
            question_embedding = self.embedding_model.get_embeddings([question])[0].values
            search_results = self._vector_search(question_embedding, limit=max_sources, doc_ids=doc_ids)
            if not search_results:
                logger.warning("No search results found for streaming")
                yield {
                    "type": "complete",
                    "full_response": "No relevant information found in the selected documents.",
                    "sources": [],
                    "response_time": time.time() - start_time,
                    "tokens_used": 0,
                    "is_complete": True
                }
                yield {
                    "type": "stream_end",
                    "timestamp": datetime.now().isoformat()
                }
                return
            context = "\n\n".join(
                f"[Source: {result['filename']} (Chunk {result['chunk_index']})]\n{result['content'][:300]}"
                for result in search_results
            )
            prompt = f"""Answer based on the context below. If the context doesn't contain enough information, say so.
CONTEXT: {context}
QUESTION: {question}
ANSWER:"""
            response_stream = self.llm_model.generate_content(
                contents=prompt,
                generation_config={"temperature": 0.1},
                stream=True
            )
            full_response = ""
            chunk_count = 0
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
                    logger.debug(f"Streamed chunk {chunk_count}: {chunk.text[:50]}...")
            tokens_used = chunk.usage_metadata.total_token_count if chunk.usage_metadata else 0
            yield {
                "type": "complete",
                "full_response": full_response,
                "sources": search_results,
                "response_time": time.time() - start_time,
                "tokens_used": tokens_used,
                "chunk_count": chunk_count,
                "is_complete": True
            }
            yield {
                "type": "stream_end",
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"Streaming completed in {time.time() - start_time:.2f}s")
        except Exception as e:
            logger.error(f"Stream failed: {str(e)}")
            yield {
                "type": "error",
                "message": f"Stream failed: {str(e)}",
                "is_complete": True
            }
            yield {
                "type": "stream_end",
                "timestamp": datetime.now().isoformat()
            }

    def store_document(self, filename: str, file_content: bytes) -> str:
        db = SessionLocal()
        try:
            text = file_content.decode('utf-8', errors='ignore')
            text_hash = hashlib.md5(text.encode()).hexdigest()
            doc = Document(filename=filename, content=text, content_hash=text_hash)
            db.add(doc)
            db.flush()
            doc_id = doc.id
            chunks = self.chunk_text(text)
            for i, chunk_text in enumerate(chunks):
                embedding_result = self.embedding_model.get_embeddings([chunk_text])[0]
                chunk = DocumentChunk(
                    document_id=doc_id,
                    content=chunk_text,
                    embedding=embedding_result.values,
                    chunk_index=i,
                    metadata={"filename": filename, "chunk_range": f"{i}-{i+len(chunk_text.split())}"}
                )
                db.add(chunk)
            db.commit()
            logger.info(f"Stored {filename} with {len(chunks)} chunks")
            return str(doc_id)
        except Exception as e:
            db.rollback()
            logger.error(f"Error storing document: {str(e)}")
            raise
        finally:
            db.close()

    def chunk_text(self, text: str) -> List[str]:
        return [text[i:i+500] for i in range(0, len(text), 500)]

try:
    rag_service = RAGService()
    logger.info("rag_service instantiated successfully")
except Exception as e:
    logger.error(f"Failed to instantiate rag_service: {str(e)}")
    raise