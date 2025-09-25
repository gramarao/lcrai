import vertexai
from vertexai.language_models import TextEmbeddingModel, TextGenerationModel
from google.cloud import storage
import logging
from typing import List, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

class GCPService:
    def __init__(self):
        # Initialize Vertex AI
        vertexai.init(project=settings.GCP_PROJECT_ID, location=settings.GCP_REGION)
        
        # Initialize models
        self.embedding_model = TextEmbeddingModel.from_pretrained(settings.EMBEDDING_MODEL)
        self.llm_model = TextGenerationModel.from_pretrained(settings.LLM_MODEL)
        
        # Initialize storage client
        self.storage_client = storage.Client(project=settings.GCP_PROJECT_ID)
        self.bucket = self.storage_client.bucket(settings.GCP_BUCKET_NAME)
    
    def generate_response(self, prompt: str, context: str = "") -> str:
        """Generate response using Vertex AI"""
        try:
            full_prompt = f"""Based on the following context, please answer the question comprehensively.

Context:
{context}

Question: {prompt}

Answer:"""

            response = self.llm_model.predict(
                full_prompt,
                max_output_tokens=1024,
                temperature=0.1,
                top_p=0.8,
                top_k=40
            )
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating the response: {str(e)}"
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Vertex AI"""
        try:
            embeddings = []
            # Process in batches of 5 to optimize costs
            batch_size = 5
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.embedding_model.get_embeddings(batch)
                embeddings.extend([emb.values for emb in batch_embeddings])
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def upload_file(self, file_content: bytes, filename: str) -> str:
        """Upload file to Google Cloud Storage"""
        try:
            blob = self.bucket.blob(filename)
            blob.upload_from_string(file_content)
            return f"gs://{settings.GCP_BUCKET_NAME}/{filename}"
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check GCP services health"""
        try:
            # Test Vertex AI
            test_embedding = self.embedding_model.get_embeddings(["test"])
            vertex_ai_status = "healthy" if test_embedding else "unhealthy"
            
            # Test Storage
            try:
                list(self.bucket.list_blobs(max_results=1))
                storage_status = "healthy"
            except:
                storage_status = "unhealthy"
            
            return {
                "vertex_ai": vertex_ai_status,
                "storage": storage_status,
                "project_id": settings.GCP_PROJECT_ID
            }
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {"error": str(e)}

gcp_service = GCPService()
