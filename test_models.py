from google.cloud import aiplatform
aiplatform.init(project="lcrai-472418", location="us-central1")
from vertexai.preview.language_models import TextEmbeddingModel
model = TextEmbeddingModel.from_pretrained("text-embedding-004")
embeddings = model.get_embeddings(["test"])[0].values
print(len(embeddings))  # Should print 768 without error
