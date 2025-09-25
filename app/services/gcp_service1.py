# In app/services/gcp_service.py or a test file
from config.settings import settings
from vertexai.language_models import TextEmbeddingModel
import vertexai
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project=settings.GCP_PROJECT_ID, location="us-central1")

try:
    model = TextEmbeddingModel.from_pretrained(settings.EMBEDDING_MODEL)
    print(f"Successfully loaded model: {settings.EMBEDDING_MODEL}")
    
    # Quick test embedding
    embeddings = model.get_embeddings(["Test sentence"])
    print(f"Embedding shape: {len(embeddings[0].values)} dimensions")
except Exception as e:
    print(f"Error: {e}")
