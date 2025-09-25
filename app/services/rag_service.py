import vertexai
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
from google.cloud import storage
import PyPDF2
from io import BytesIO
from sqlalchemy import text
from app.models.database import SessionLocal, Document, DocumentChunk
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        vertexai.init(project=settings.GCP_PROJECT_ID, location="us-central1")
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        self.llm_model = TextGenerationModel.from_pretrained("text-bison@002")
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(f"{settings.GCP_PROJECT_ID}-docs")
    
    def process_pdf(self, file_content: bytes, filename: str) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages])
            return text.strip()
        except Exception as e:
            raise Exception(f"PDF processing failed: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> list:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
    
    def store_document(self, filename: str, file_content: bytes) -> str:
        """Process and store document"""
        db = SessionLocal()
        try:
            # Extract text
            text = self.process_pdf(file_content, filename)
            if not text:
                raise Exception("No text extracted")
            
            # Store document
            doc = Document(filename=filename, content=text)
            db.add(doc)
            db.flush()
            
            # Create chunks and embeddings
            chunks = self.chunk_text(text)
            embeddings = self.embedding_model.get_embeddings(chunks)
            
            # Store chunks
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings)):
                chunk = DocumentChunk(
                    document_id=doc.id,
                    content=chunk_text,
                    embedding=embedding.values,
                    chunk_index=i
                )
                db.add(chunk)
            
            db.commit()
            logger.info(f"Stored {filename} with {len(chunks)} chunks")
            return str(doc.id)
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
    
    def search_and_generate(self, question: str) -> dict:
        """Optimized search and generation"""
        db = SessionLocal()
        try:
            # Get question embedding (cache this in production)
            question_embedding = self.embedding_model.get_embeddings([question])[0].values
            
            # Faster vector search with similarity threshold
            query = text("""
                SELECT content, (1 - (embedding <-> :embedding)) as similarity
                FROM chunks 
                WHERE (1 - (embedding <-> :embedding)) > 0.6
                ORDER BY embedding <-> :embedding 
                LIMIT 2
            """)
            
            results = db.execute(query, {"embedding": str(question_embedding)}).fetchall()
            
            if not results:
                return {
                    "answer": "I couldn't find relevant information in the uploaded documents.",
                    "sources": []
                }
            
            # Shorter context for faster processing
            context = "\n".join([r.content[:800] for r in results])  # Limit context size
            
            # Optimized prompt for speed
            prompt = f"Answer briefly based on context:\n\nContext: {context}\n\nQ: {question}\nA:"
            
            try:
                # Configure for faster response
                generation_config = {
                    'temperature': 0.1,
                    'top_p': 0.8,
                    'top_k': 20,
                    'max_output_tokens': 300,  # Shorter responses = faster
                }
                
                response = self.llm_model.generate_content(
                    prompt, 
                    generation_config=generation_config
                )
                answer_text = response.text
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                answer_text = "Found relevant info but couldn't generate response quickly. Please try again."
            
            return {
                "answer": answer_text,
                "sources": [{"content": r.content[:150] + "...", "similarity": float(r.similarity)} for r in results]
            }
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {
                "answer": f"Query error: {str(e)}",
                "sources": []
            }
        finally:
            db.close()

rag_service = RAGService()
