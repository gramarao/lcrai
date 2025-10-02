# test_feedback_api.py
import requests
import json
import uuid

def test_feedback_minimal():
    """Test with minimal valid payload"""
    
    payload = {
        "session_id": "test-session-123",
        "query_id": str(uuid.uuid4()),
        "question": "Test question",
        "response": "Test response",
        "rating": 3,
        "thumbs_up": True,
        "feedback_text": "Test feedback",
        "response_time": 1.5,
        "tokens_used": 100,
        "sources_count": 0,
        "sources": []
    }
    
    print("Testing minimal feedback payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            "http://localhost:8000/api/feedback",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Minimal payload works!")
            return True
        else:
            print("❌ Minimal payload failed")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

def test_feedback_with_sources():
    """Test with sources included"""
    
    # You'll need to get real UUIDs from your database
    payload = {
        "session_id": "test-session-456",
        "query_id": str(uuid.uuid4()),
        "question": "Test question with sources",
        "response": "Test response with sources",
        "rating": 4,
        "thumbs_up": True,
        "feedback_text": "",
        "response_time": 2.3,
        "tokens_used": 150,
        "sources_count": 1,
        "sources": [
            {
                "document_id": str(uuid.uuid4()),  # Replace with real UUID from your DB
                "id": str(uuid.uuid4()),  # Replace with real chunk UUID from your DB
                "relevance_score": 0.85,
                "filename": "test.pdf",
                "chunk_index": 0
            }
        ],
        "quality_scores": {
            "relevance": 0.8,
            "accuracy": 0.7,
            "completeness": 0.9,
            "coherence": 0.6,
            "citation": 0.5,
            "overall": 0.7
        }
    }
    
    print("\nTesting payload with sources:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            "http://localhost:8000/api/feedback",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Sources payload works!")
            return True
        else:
            print("❌ Sources payload failed")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Feedback API ===")
    
    # Test minimal payload first
    if test_feedback_minimal():
        # If minimal works, test with sources
        test_feedback_with_sources()
    else:
        print("Fix minimal payload issues first")
