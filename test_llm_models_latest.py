import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project='lcrai-472418', location='us-central1')

models_to_try = [
    'gemini-2.5-flash',  # Current lightweight Gemini model
    'gemini-2.5-pro',    # Advanced Gemini model
    'gemini-2.0-pro'     # Older Gemini model (may not be available)
]

for model in models_to_try:
    try:
        llm = GenerativeModel(model)
        response = llm.generate_content('Hello', generation_config={'temperature': 0.1})
        print(f'✅ {model}: Works - {response.text[:50]}...')
    except Exception as e:
        print(f'❌ {model}: {str(e)[:100]}...')