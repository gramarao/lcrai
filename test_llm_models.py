import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.generative_models import GenerativeModel

vertexai.init(project="lcrai-472418", location="us-central1")

# Test older text generation models
old_models = ["text-bison@001", "text-bison", "text-unicorn@001"]

print("=== Testing Text Generation Models ===")
for model_name in old_models:
    try:
        model = TextGenerationModel.from_pretrained(model_name)
        response = model.predict("Hello", max_output_tokens=10)
        print(f"✅ {model_name}: Works!")
    except Exception as e:
        print(f"❌ {model_name}: {e}")

# Test newer Gemini models
gemini_models = ["gemini-1.5-flash-001", "gemini-1.0-pro", "gemini-pro"]

print("\n=== Testing Gemini Models ===")
for model_name in gemini_models:
    try:
        model = GenerativeModel(model_name)
        response = model.generate_content("Hello")
        print(f"✅ {model_name}: Works!")
    except Exception as e:
        print(f"❌ {model_name}: {e}")
