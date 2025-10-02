# test_rag_methods.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.rag_service_googleai import rag_service

def check_rag_methods():
    if not rag_service:
        print("❌ RAG service not initialized")
        return False
    
    required_methods = [
        'search_and_generate',
        'stream_search_and_generate',
        'store_document',
        '_vector_search',
        '_build_context',
        '_create_prompt'
    ]
    
    print("Checking RAG service methods:")
    all_good = True
    
    for method in required_methods:
        if hasattr(rag_service, method):
            print(f"✅ {method}")
        else:
            print(f"❌ {method} - MISSING")
            all_good = False
    
    # Check if method is callable
    if hasattr(rag_service, 'stream_search_and_generate'):
        if callable(getattr(rag_service, 'stream_search_and_generate')):
            print("✅ stream_search_and_generate is callable")
        else:
            print("❌ stream_search_and_generate exists but not callable")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    if check_rag_methods():
        print("\n✅ All required methods present")
    else:
        print("\n❌ Some methods are missing")
