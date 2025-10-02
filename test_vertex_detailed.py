# test_vertex_detailed.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

def test_vertex_ai_detailed():
    try:
        print("Testing Vertex AI configuration...")
        
        # Initialize Vertex AI
        vertexai.init(project="lcrai-472418", location="us-central1")
        print("✅ Vertex AI initialized")
        
        # Load model
        embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")
        print("✅ text-embedding-004 model loaded")
        
        # Test with various inputs
        test_cases = [
            "Short text",
            "This is a medium length sentence for testing embedding generation.",
            "This is a much longer text that contains multiple sentences and various words. It should still generate a proper 768-dimensional embedding vector regardless of the input length, as that is the standard output dimension for the text-embedding-004 model from Google's Vertex AI service.",
            "Special characters: @#$%^&*()_+-=[]{}|;:,.<>?",
            "Numbers and text: 123 456 789 mixed with words",
            ""  # Empty string test
        ]
        
        for i, text in enumerate(test_cases):
            print(f"\n--- Test Case {i+1} ---")
            print(f"Input: '{text}' (length: {len(text)})")
            
            try:
                if not text.strip():  # Skip empty strings
                    print("⚠️  Skipping empty string")
                    continue
                
                result = embedding_model.get_embeddings([text])
                
                print(f"Result type: {type(result)}")
                print(f"Result length: {len(result)}")
                
                if result and result[0]:
                    embedding = result[0]
                    print(f"Embedding object type: {type(embedding)}")
                    print(f"Embedding attributes: {[attr for attr in dir(embedding) if not attr.startswith('_')]}")
                    
                    if hasattr(embedding, 'values') and embedding.values:
                        values = embedding.values
                        print(f"Values type: {type(values)}")
                        print(f"Values length: {len(values)}")
                        print(f"First 3 values: {values[:3]}")
                        print(f"Last 3 values: {values[-3:]}")
                        print(f"Value types: {[type(v) for v in values[:3]]}")
                        
                        if len(values) == 768:
                            print("✅ Correct dimensions")
                        else:
                            print(f"❌ Wrong dimensions: got {len(values)}, expected 768")
                    else:
                        print("❌ No values attribute or values is empty")
                        print(f"Embedding object: {embedding}")
                else:
                    print("❌ No result or empty result")
                    
            except Exception as e:
                print(f"❌ Test case failed: {e}")
                import traceback
                traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ Vertex AI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_vertex_ai_detailed()
