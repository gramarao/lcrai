import vertexai
from vertexai.language_models import TextEmbeddingModel

vertexai.init(project="lcrai-472418", location="us-central1")

models_to_try = [
    "text-embedding-004",
    "text-embedding-gecko@003", 
    "textembedding-gecko@001",
    "text-multilingual-embedding-002"
]

for model_name in models_to_try:
    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
        embeddings = model.get_embeddings(["test text"])
        dimension = len(embeddings[0].values)
        print(f"✅ {model_name}: {dimension} dimensions")
    except Exception as e:
        print(f"❌ {model_name}: {e}")
